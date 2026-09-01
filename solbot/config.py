"""Central settings.

Every tunable in the build spec lives here, never hardcoded at the call site.
Values are layered:  DEFAULTS  <-  config.json  <-  environment overrides.

The dashboard edits config.json through :meth:`Config.update`, which validates
each change against the bounds in ``SPEC`` before applying it; the caller then
appends a row to ``settings_audit`` so there is a record of what was tuned and
when.

The worker calls :meth:`Config.maybe_reload` once per loop; it re-reads the
file only when its mtime changes, so live edits take effect without a restart.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------
# Defaults. Values come from the build spec; see README for the rationale on
# the few that deliberately differ from the spec's stated numbers.
# --------------------------------------------------------------------------
DEFAULTS: dict[str, Any] = {
    # --- mode -------------------------------------------------------------
    "trading_mode": "paper",              # paper | live (must be set explicitly)
    "paper_starting_balance": 1000.0,

    # --- position sizing --------------------------------------------------
    "max_concurrent_positions": 2,
    "max_position_pct_of_wallet": 0.45,       # 45% of wallet per position
    "max_position_pct_of_liquidity": 0.01,    # 1% of the pool's depth
    "gas_reserve_sol": 0.05,                  # never deployed, always pays fees
    "min_position_usd": 10.0,

    # --- signal -----------------------------------------------------------
    "candle_minutes": 10,                 # spec: 5-15 min, default 10
    "broad_scan_seconds": 15,
    "hot_scan_seconds": 1,
    "hot_list_max": 25,

    # --- reward:risk ------------------------------------------------------
    "rr_min": 2.0,
    "rr_max": 4.0,

    # --- circuit breaker --------------------------------------------------
    "circuit_consecutive_losses": 3,
    "circuit_daily_drawdown_pct": 0.09,   # spec suggested 8-10%

    # --- universe filtering -----------------------------------------------
    "min_liquidity_usd": 50000.0,
    "min_volume_24h_usd": 75000.0,
    "universe_max_tokens": 300,
    "universe_refresh_seconds": 900,

    # --- entry rules ------------------------------------------------------
    "volume_spike_multiple": 2.0,         # vs. its own rolling average
    "volume_spike_lookback": 20,
    "momentum_candles": 3,                # consecutive confirming candles
    "momentum_min_pct": 0.008,            # 0.8% over the momentum window
    "rsi_period": 14,
    # Blow-off-top guard. Deliberately not ~70: the momentum rule already
    # requires several consecutive rising candles, which mechanically drives
    # Wilder RSI into the mid-70s, so a ceiling near 70 double-counts the same
    # condition and leaves almost no window where an entry can fire. 78 still
    # rejects genuine parabolic moves (which read 82+) without fighting the
    # momentum rule. See tests/test_strategy.py::test_rsi_ceiling_leaves_a_
    # workable_entry_window.
    "rsi_max_entry": 78.0,
    "ema_fast": 9,
    "ema_slow": 21,
    "atr_period": 14,
    "min_candles_required": 40,

    # --- exit rules -------------------------------------------------------
    "trailing_activate_r": 1.0,           # arm the trail after +1R
    "trailing_distance_atr": 1.5,
    "signal_invalidation_bars": 2,        # bars of failed signal before exit
    "max_hold_minutes": 240,

    # --- risk -------------------------------------------------------------
    "correlation_lookback": 60,
    "correlation_max": 0.80,
    "volatility_size_floor": 0.35,        # never size below 35% of base
    "volatility_target_atr_pct": 0.03,

    # --- execution --------------------------------------------------------
    "max_slippage_pct": 0.5,              # spec default 0.5%
    "taker_fee_pct": 0.25,                # round-trip cost model (per side)
    "priority_fee_lamports": 200000,
    "congestion_max_priority_fee_lamports": 2000000,
    "congestion_check_enabled": True,

    # --- rug check --------------------------------------------------------
    "rugcheck_enabled": True,
    "rugcheck_require_mint_revoked": True,
    "rugcheck_require_freeze_revoked": True,
    "rugcheck_min_lp_locked_pct": 80.0,
    # Above this many independent pools the LP-lock check is waived: with
    # liquidity spread that widely no single actor can withdraw it, and
    # concentrated-liquidity venues have no LP token to lock in the first
    # place. See SafetyGate.evaluate for the full reasoning.
    "rugcheck_lp_lock_waiver_markets": 20,
    "rugcheck_max_top_holder_pct": 25.0,
    "rugcheck_max_insider_pct": 20.0,
    "rugcheck_min_token_age_hours": 72.0,
    "rugcheck_min_liquidity_usd": 50000.0,
    "rugcheck_pass_ttl_seconds": 21600,       # re-verify a pass every 6h
    "rugcheck_fail_cooldown_seconds": 14400,  # retry a failure after 4h

    # --- rate limits ------------------------------------------------------
    # Jupiter tiers as of 2026-08: keyless 0.5 | free 1 | developer 10
    # | launch 50 | pro 150 requests per second.
    "jupiter_rps": 1.0,
    "jupiter_burst": 3,
    "jupiter_price_batch_size": 50,       # API hard limit is 50 ids per call
    "birdeye_rps": 1.0,
    "birdeye_monthly_cu_budget": 30000,   # free Standard tier
    "rugcheck_rps": 2.0,

    # --- data retention ---------------------------------------------------
    "candle_retention_days": 120,
    "price_tick_retention_hours": 48,
    "event_retention_days": 45,

    # --- backtest ---------------------------------------------------------
    "backtest_daily_enabled": True,
    "backtest_daily_hour_utc": 4,
    "backtest_lookback_days": 90,

    # --- dashboard --------------------------------------------------------
    "session_timeout_minutes": 60,
    "login_rate_limit_attempts": 5,
    "login_rate_limit_window_seconds": 900,
}

# --------------------------------------------------------------------------
# Validation. (type, low, high) - bounds are inclusive. Editing a key from the
# dashboard is refused unless the new value lands inside these.
# --------------------------------------------------------------------------
Bound = tuple[type, float | None, float | None]

SPEC: dict[str, Bound] = {
    "paper_starting_balance": (float, 1.0, 10000000.0),
    "max_concurrent_positions": (int, 1, 10),
    "max_position_pct_of_wallet": (float, 0.01, 0.50),
    "max_position_pct_of_liquidity": (float, 0.0001, 0.05),
    "gas_reserve_sol": (float, 0.0, 10.0),
    "min_position_usd": (float, 1.0, 100000.0),
    "candle_minutes": (int, 5, 15),
    "broad_scan_seconds": (int, 5, 600),
    "hot_scan_seconds": (int, 1, 60),
    "hot_list_max": (int, 1, 100),
    "rr_min": (float, 1.5, 4.0),
    "rr_max": (float, 2.0, 6.0),
    "circuit_consecutive_losses": (int, 2, 20),
    "circuit_daily_drawdown_pct": (float, 0.01, 0.50),
    "min_liquidity_usd": (float, 1000.0, 100000000.0),
    "min_volume_24h_usd": (float, 1000.0, 1000000000.0),
    "universe_max_tokens": (int, 10, 2000),
    "universe_refresh_seconds": (int, 60, 86400),
    "volume_spike_multiple": (float, 1.1, 20.0),
    "volume_spike_lookback": (int, 5, 200),
    "momentum_candles": (int, 2, 10),
    "momentum_min_pct": (float, 0.001, 0.20),
    "rsi_period": (int, 2, 50),
    "rsi_max_entry": (float, 50.0, 95.0),
    "ema_fast": (int, 2, 50),
    "ema_slow": (int, 3, 200),
    "atr_period": (int, 2, 50),
    "min_candles_required": (int, 10, 500),
    "trailing_activate_r": (float, 0.1, 5.0),
    "trailing_distance_atr": (float, 0.2, 10.0),
    "signal_invalidation_bars": (int, 1, 10),
    "max_hold_minutes": (int, 10, 10080),
    "correlation_lookback": (int, 10, 500),
    "correlation_max": (float, 0.1, 1.0),
    "volatility_size_floor": (float, 0.05, 1.0),
    "volatility_target_atr_pct": (float, 0.001, 0.50),
    "max_slippage_pct": (float, 0.05, 5.0),
    "taker_fee_pct": (float, 0.0, 2.0),
    "priority_fee_lamports": (int, 0, 10000000),
    "congestion_max_priority_fee_lamports": (int, 1000, 100000000),
    "rugcheck_min_lp_locked_pct": (float, 0.0, 100.0),
    "rugcheck_lp_lock_waiver_markets": (int, 0, 10000),
    "rugcheck_max_top_holder_pct": (float, 1.0, 100.0),
    "rugcheck_max_insider_pct": (float, 1.0, 100.0),
    "rugcheck_min_token_age_hours": (float, 0.0, 8760.0),
    "rugcheck_min_liquidity_usd": (float, 0.0, 100000000.0),
    "rugcheck_pass_ttl_seconds": (int, 60, 604800),
    "rugcheck_fail_cooldown_seconds": (int, 60, 604800),
    "jupiter_rps": (float, 0.1, 200.0),
    "jupiter_burst": (int, 1, 100),
    "jupiter_price_batch_size": (int, 1, 50),
    "birdeye_rps": (float, 0.1, 50.0),
    "birdeye_monthly_cu_budget": (int, 0, 100000000),
    "rugcheck_rps": (float, 0.1, 50.0),
    "candle_retention_days": (int, 7, 3650),
    "price_tick_retention_hours": (int, 1, 8760),
    "event_retention_days": (int, 1, 3650),
    "backtest_daily_hour_utc": (int, 0, 23),
    "backtest_lookback_days": (int, 7, 1095),
    "session_timeout_minutes": (int, 5, 1440),
    "login_rate_limit_attempts": (int, 1, 100),
    "login_rate_limit_window_seconds": (int, 30, 86400),
}

ENUMS: dict[str, set[str]] = {"trading_mode": {"paper", "live"}}

BOOLS = {
    "congestion_check_enabled",
    "rugcheck_enabled",
    "rugcheck_require_mint_revoked",
    "rugcheck_require_freeze_revoked",
    "backtest_daily_enabled",
}

# Keys the dashboard settings page may change. `trading_mode` is deliberately
# excluded - flipping to live is a separate, guarded action.
EDITABLE = set(SPEC) | BOOLS


class ConfigError(ValueError):
    """A proposed settings change failed validation."""


def _coerce(key: str, value: Any) -> Any:
    if key in BOOLS:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if key in ENUMS:
        v = str(value).strip().lower()
        if v not in ENUMS[key]:
            raise ConfigError(f"{key} must be one of {sorted(ENUMS[key])}")
        return v
    if key not in SPEC:
        return value
    typ, low, high = SPEC[key]
    if isinstance(value, bool):
        raise ConfigError(f"{key} must be {typ.__name__}, not a boolean")
    try:
        v = typ(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{key} must be {typ.__name__}") from exc
    if low is not None and v < low:
        raise ConfigError(f"{key} must be >= {low} (got {v})")
    if high is not None and v > high:
        raise ConfigError(f"{key} must be <= {high} (got {v})")
    return v


def validate(pending: dict[str, Any], merged: dict[str, Any]) -> dict[str, Any]:
    """Coerce and bounds-check `pending`, then run the cross-field rules.

    The cross-field rules run against the state the config *would* be in once
    `pending` is applied, not against the delta alone.
    """
    clean = {k: _coerce(k, v) for k, v in pending.items()}
    after = {**merged, **clean}

    if after["rr_min"] > after["rr_max"]:
        raise ConfigError("reward:risk minimum cannot exceed the maximum")
    if after["ema_fast"] >= after["ema_slow"]:
        raise ConfigError("ema_fast must be shorter than ema_slow")
    if after["hot_scan_seconds"] > after["broad_scan_seconds"]:
        raise ConfigError("hot scan must be at least as frequent as the broad scan")
    if after["min_candles_required"] <= after["volume_spike_lookback"]:
        raise ConfigError("min_candles_required must exceed volume_spike_lookback")
    if after["max_position_pct_of_wallet"] * after["max_concurrent_positions"] > 0.95:
        raise ConfigError(
            "position size x max concurrent positions would deploy more than 95% "
            "of the wallet; the spec requires an untouched reserve"
        )
    if after["rugcheck_pass_ttl_seconds"] < after["broad_scan_seconds"]:
        raise ConfigError("rug check TTL must be longer than one scan cycle")
    return clean


@dataclass
class Secrets:
    """Read from the environment at import; never written to config.json."""

    jupiter_api_key: str = ""
    birdeye_api_key: str = ""
    rugcheck_api_key: str = ""
    solana_private_key: str = ""
    solana_rpc_url: str = "https://api.mainnet-beta.solana.com"
    flask_secret_key: str = ""
    secret_encryption_key: str = ""

    @classmethod
    def from_env(cls) -> "Secrets":
        return cls(
            jupiter_api_key=os.getenv("JUPITER_API_KEY", "").strip(),
            birdeye_api_key=os.getenv("BIRDEYE_API_KEY", "").strip(),
            rugcheck_api_key=os.getenv("RUGCHECK_API_KEY", "").strip(),
            solana_private_key=os.getenv("SOLANA_PRIVATE_KEY", "").strip(),
            solana_rpc_url=os.getenv(
                "SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"
            ).strip(),
            flask_secret_key=os.getenv("FLASK_SECRET_KEY", "").strip(),
            secret_encryption_key=os.getenv("SECRET_ENCRYPTION_KEY", "").strip(),
        )


class Config:
    """Thread-safe settings holder backed by a JSON file."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path or os.getenv("SOLBOT_CONFIG", "config.json"))
        self._lock = threading.RLock()
        self._values: dict[str, Any] = dict(DEFAULTS)
        self._mtime: float = 0.0
        self.secrets = Secrets.from_env()
        self._listeners: list[Callable[[dict[str, Any]], None]] = []
        self.load()

    # -- reading -----------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        with self._lock:
            return self._values[key]

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._values.get(key, default)

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._values)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def is_live(self) -> bool:
        return self.get("trading_mode") == "live"

    def candle_seconds(self) -> int:
        return int(self["candle_minutes"]) * 60

    # -- loading / reloading -----------------------------------------------
    def load(self) -> None:
        with self._lock:
            values = dict(DEFAULTS)
            if self._path.exists():
                try:
                    stored = json.loads(self._path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    stored = {}
                if isinstance(stored, dict):
                    for key, raw in stored.items():
                        if key not in DEFAULTS:
                            continue  # drop keys we no longer recognise
                        try:
                            values[key] = _coerce(key, raw)
                        except ConfigError:
                            values[key] = DEFAULTS[key]  # fall back, never crash
                try:
                    self._mtime = self._path.stat().st_mtime
                except OSError:
                    self._mtime = 0.0
            self._values = values

    def maybe_reload(self) -> bool:
        """Re-read the file if it changed on disk. Returns True if reloaded."""
        try:
            mtime = self._path.stat().st_mtime
        except OSError:
            return False
        if mtime <= self._mtime:
            return False
        self.load()
        snapshot = self.as_dict()
        for fn in list(self._listeners):
            try:
                fn(snapshot)
            except Exception:  # a bad listener must not stall the trading loop
                pass
        return True

    def on_reload(self, fn: Callable[[dict[str, Any]], None]) -> None:
        self._listeners.append(fn)

    # -- writing -----------------------------------------------------------
    def update(
        self, changes: dict[str, Any], *, allow_mode: bool = False
    ) -> dict[str, tuple[Any, Any]]:
        """Validate and persist `changes`. Returns {key: (old, new)} applied."""
        with self._lock:
            pending = {k: v for k, v in changes.items() if k in DEFAULTS}
            unknown = set(changes) - set(pending)
            if unknown:
                raise ConfigError(f"unknown setting(s): {', '.join(sorted(unknown))}")
            if not allow_mode:
                blocked = set(pending) - EDITABLE
                if blocked:
                    raise ConfigError(
                        f"not editable from the dashboard: {', '.join(sorted(blocked))}"
                    )
            clean = validate(pending, self._values)
            applied: dict[str, tuple[Any, Any]] = {}
            for key, value in clean.items():
                old = self._values.get(key)
                if old != value:
                    applied[key] = (old, value)
                    self._values[key] = value
            if applied:
                self._write()
            return applied

    def _write(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._values, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self._path)
        try:
            self._mtime = self._path.stat().st_mtime
        except OSError:
            self._mtime = 0.0

    def set_trading_mode(self, mode: str) -> None:
        self.update({"trading_mode": mode}, allow_mode=True)


_config: Config | None = None
_config_lock = threading.Lock()


def get_config(path: str | Path | None = None) -> Config:
    """Process-wide singleton."""
    global _config
    with _config_lock:
        if _config is None:
            _config = Config(path)
        return _config


def reset_config_for_tests(path: str | Path) -> Config:
    global _config
    with _config_lock:
        _config = Config(path)
        return _config


def utcnow() -> float:
    return time.time()
