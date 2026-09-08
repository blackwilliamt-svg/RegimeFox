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
    # No hardcoded ceiling on how many positions may be open at once - the bot
    # decides that itself, as an output of the same sizing logic, from
    # walk-forward/Monte Carlo evidence. The only hard constraints are the
    # per-position cap, the total-deployed cap, and always retaining enough
    # balance to cover transaction fees (the gas reserve, below).
    "max_position_pct_of_wallet": 0.45,       # 45% of wallet per position
    "max_total_deployed_pct": 0.90,           # 90% of wallet deployed at once, at most
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
    # The universe is Binance's top `binance_top_n` coins by 24h quote volume
    # (spec 2) - established, large-cap coins, not a pure Jupiter-liquidity
    # floor over the whole token list. That Binance shortlist is then
    # intersected with what Jupiter can actually route a swap for on Solana,
    # and the floors below are a filter *on* that shortlist, same as before.
    "binance_top_n": 100,
    "min_liquidity_usd": 50000.0,
    "min_volume_24h_usd": 75000.0,
    "universe_max_tokens": 150,
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

    # Stop distance as a multiple of ATR. Was hardcoded at 1.2 inside the
    # strategy; exposed here because the walk-forward optimizer tunes it, and a
    # tuned value that cannot be applied is no use.
    "stop_atr_mult": 1.2,

    # --- regime detection (spec 4.1) --------------------------------------
    # The efficiency ratio over `regime_lookback` bars separates a trend from a
    # drift; the ATR threshold then splits the non-trending case into an orderly
    # range and genuine chop.
    "regime_lookback": 20,
    "regime_trend_er": 0.35,
    "regime_chop_atr_pct": 0.03,
    # Bitmask of the regimes an entry may fire in: 1 trending | 2 ranging |
    # 4 choppy. The default admits trending and ranging and excludes chop, which
    # is where the overtrading in the earlier backtest came from.
    "regime_allowed": 3,
    "regime_gate_enabled": True,

    # --- multi-timeframe confluence (spec 4.2) ----------------------------
    # Aggregate timeframes as multiples of the base candle. At the 10-minute
    # default, (3, 6) is the 30-minute and hourly view.
    "confluence_timeframes": [3, 6],
    "confluence_required": 1,             # how many must agree before an entry
    "confluence_enabled": True,

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

    # --- correlation-aware portfolio sizing (spec 4.4) --------------------
    # "flat" is the original per-position scaling. "portfolio" solves for the
    # weight that holds *portfolio* volatility at the target given what is
    # already open and how correlated the candidate is with it. Both stay inside
    # the hard caps above; this only ever sizes down.
    "sizing_mode": "portfolio",
    # The budget for the book as a whole: two uncorrelated positions at the 45%
    # cap and the 3% target ATR, i.e. sqrt(2) x 0.45 x 0.03. Anchoring it there
    # leaves a single position governed by the flat volatility scalar (so
    # switching modes changes nothing on its own) while still leaving room for a
    # second one. Anchoring it to a *single* position instead would exhaust the
    # budget on the first trade and silently halve the bot's concurrency.
    "portfolio_vol_target": 0.019,
    # The share of the account the operator will tolerate being underwater in the
    # 5%-worst case. The Monte Carlo tail below is measured against it.
    "drawdown_tolerance": 0.25,
    # 5th-percentile drawdown from the last Monte Carlo run, written by the
    # parameter hand-off. Zero means "not measured yet", which leaves sizing
    # untouched rather than guessing.
    "monte_carlo_p5_drawdown": 0.0,

    # --- live vs backtest drift (spec 4.5) --------------------------------
    "drift_window_days": 14,
    "drift_min_trades": 20,
    "drift_win_rate_tolerance": 0.15,     # absolute, e.g. 0.15 = 15 points
    "drift_expectancy_tolerance": 0.40,   # relative to the backtest expectancy
    "drift_check_seconds": 900,

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
    # Binance's public endpoints are keyless and generously weight-limited for
    # this bot's call volume (a ranking pull once per universe refresh, a
    # klines pull once per coin per day).
    "binance_rps": 5.0,
    "rugcheck_rps": 2.0,

    # --- candle history (spec 3) -------------------------------------------
    # Kept indefinitely - no retention window. Raw candle Parquet files are
    # small (single-digit GB compressed for the whole universe over a year),
    # unlike walk-forward/Monte Carlo output, which does need one (spec 6c/7).
    "bulk_backfill_months": 12,

    # --- data retention ---------------------------------------------------
    "price_tick_retention_hours": 48,
    "event_retention_days": 45,
    # WF/MC output (window scores, parameter bundles, Monte Carlo
    # distributions) is the fastest-growing storage component - unlike raw
    # candles, which are kept indefinitely, this needs its own bound (spec 6c).
    "wfmc_result_retention_days": 180,

    # --- backtest ---------------------------------------------------------
    "backtest_daily_enabled": True,
    "backtest_daily_hour_utc": 4,
    "backtest_lookback_days": 90,

    # --- overtrading brake (spec 5) ---------------------------------------
    # A prior backtest fired 244 trades in seven days and paid $449 in fees for
    # them. These are the hard brakes; the regime gate and confluence check are
    # the soft ones.
    "max_trades_per_day": 8,
    "min_seconds_between_entries": 900,
    "min_seconds_between_entries_same_mint": 3600,

    # --- entry review gate --------------------------------------------------
    # A deterministic rules engine, not a model: see solbot/review.py's RUBRIC.
    "entry_gate_enabled": True,
    "entry_gate_min_strength": 0.0,       # weaker signals skip straight to the rules score
    "daily_review_enabled": True,

    # --- walk-forward / Monte Carlo optimizer (spec 5) ---------------------
    # Daily incremental re-scoring runs on the droplet's own CPU, against a
    # focused set of retained promising parameter combinations. Scheduled for
    # a low-activity hour so it doesn't compete with the trading loop.
    "wfmc_daily_enabled": True,
    "wfmc_daily_hour_utc": 3,
    # The monthly full parameter-space retest runs on a RunPod GPU worker,
    # orchestrated by this droplet: data shipped up, job run, results reported
    # back, worker torn down and the teardown verified.
    "wfmc_monthly_enabled": False,        # off until RUNPOD_API_KEY is set
    "wfmc_monthly_day_utc": 1,
    "wfmc_monthly_hour_utc": 3,
    "runpod_gpu_type": "NVIDIA RTX 4090",  # benchmark on RunPod before trusting this
    "runpod_batch_size": 75,              # coins per parallel RunPod job
    "runpod_callback_url": "",             # this droplet's own public URL
    # Auto-promotion from shadow to live: no manual approval, but a long clean
    # run and a decisive margin over whatever live is doing.
    "auto_promote_enabled": True,
    "promotion_check_interval_seconds": 3600,
    "shadow_min_days": 15,
    "shadow_min_trades": 30,
    "shadow_min_margin": 0.25,
    "shadow_min_confidence": 0.90,

    # --- RunPod worker ingest endpoint --------------------------------------
    "runpod_ingest_enabled": True,
    "runpod_rate_limit_per_minute": 60,

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
    "max_position_pct_of_wallet": (float, 0.01, 0.50),
    "max_total_deployed_pct": (float, 0.10, 0.98),
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
    "binance_top_n": (int, 10, 300),
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
    "stop_atr_mult": (float, 0.3, 5.0),
    "regime_lookback": (int, 5, 200),
    "regime_trend_er": (float, 0.05, 0.95),
    "regime_chop_atr_pct": (float, 0.001, 0.50),
    "regime_allowed": (int, 1, 7),
    "confluence_required": (int, 0, 4),
    "correlation_lookback": (int, 10, 500),
    "correlation_max": (float, 0.1, 1.0),
    "volatility_size_floor": (float, 0.05, 1.0),
    "volatility_target_atr_pct": (float, 0.001, 0.50),
    "portfolio_vol_target": (float, 0.001, 0.50),
    "drawdown_tolerance": (float, 0.01, 1.0),
    "monte_carlo_p5_drawdown": (float, 0.0, 1.0),
    "drift_window_days": (int, 1, 365),
    "drift_min_trades": (int, 5, 1000),
    "drift_win_rate_tolerance": (float, 0.01, 1.0),
    "drift_expectancy_tolerance": (float, 0.01, 5.0),
    "drift_check_seconds": (int, 60, 86400),
    "max_trades_per_day": (int, 1, 500),
    "min_seconds_between_entries": (int, 0, 86400),
    "min_seconds_between_entries_same_mint": (int, 0, 604800),
    "entry_gate_min_strength": (float, 0.0, 1.0),
    "wfmc_daily_hour_utc": (int, 0, 23),
    "wfmc_monthly_day_utc": (int, 1, 28),
    "wfmc_monthly_hour_utc": (int, 0, 23),
    "runpod_batch_size": (int, 10, 100),
    "promotion_check_interval_seconds": (int, 60, 86400),
    "shadow_min_days": (int, 1, 365),
    "shadow_min_trades": (int, 5, 10000),
    "shadow_min_margin": (float, 0.0, 10.0),
    "shadow_min_confidence": (float, 0.5, 1.0),
    "runpod_rate_limit_per_minute": (int, 1, 10000),
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
    "binance_rps": (float, 0.5, 50.0),
    "bulk_backfill_months": (int, 1, 24),
    "rugcheck_rps": (float, 0.1, 50.0),
    "price_tick_retention_hours": (int, 1, 8760),
    "wfmc_result_retention_days": (int, 7, 3650),
    "event_retention_days": (int, 1, 3650),
    "backtest_daily_hour_utc": (int, 0, 23),
    "backtest_lookback_days": (int, 7, 1095),
    "session_timeout_minutes": (int, 5, 1440),
    "login_rate_limit_attempts": (int, 1, 100),
    "login_rate_limit_window_seconds": (int, 30, 86400),
}

ENUMS: dict[str, set[str]] = {
    "trading_mode": {"paper", "live"},
    "sizing_mode": {"flat", "portfolio"},
}

BOOLS = {
    "congestion_check_enabled",
    "rugcheck_enabled",
    "rugcheck_require_mint_revoked",
    "rugcheck_require_freeze_revoked",
    "backtest_daily_enabled",
    "regime_gate_enabled",
    "confluence_enabled",
    "entry_gate_enabled",
    "daily_review_enabled",
    "wfmc_daily_enabled",
    "wfmc_monthly_enabled",
    "auto_promote_enabled",
    "runpod_ingest_enabled",
}

# Free-text settings. (max length,) - validated for length and stripped, never
# interpolated anywhere that would let one become an injection.
STRINGS: dict[str, int] = {
    "runpod_gpu_type": 64,
    "runpod_callback_url": 512,
}

# List settings: (element type, low, high, max length). Rendered on the settings
# page as a comma-separated field.
ListBound = tuple[type, float, float, int]
LISTS: dict[str, ListBound] = {
    "confluence_timeframes": (int, 1, 288, 4),
}

# Keys the dashboard settings page may change. `trading_mode` is deliberately
# excluded - flipping to live is a separate, guarded action.
EDITABLE = set(SPEC) | BOOLS | set(STRINGS) | set(LISTS) | {"sizing_mode"}


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
    if key in STRINGS:
        text = str(value).strip()
        if len(text) > STRINGS[key]:
            raise ConfigError(f"{key} must be at most {STRINGS[key]} characters")
        return text
    if key in LISTS:
        return _coerce_list(key, value)
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


def _coerce_list(key: str, value: Any) -> list[Any]:
    """Parse a list setting from a real list or a comma-separated string."""
    typ, low, high, max_len = LISTS[key]
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise ConfigError(f"{key} must be a list or a comma-separated string")
    if len(items) > max_len:
        raise ConfigError(f"{key} accepts at most {max_len} values")

    out: list[Any] = []
    for item in items:
        try:
            parsed = typ(item)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{key} values must be {typ.__name__}") from exc
        if parsed < low or parsed > high:
            raise ConfigError(f"{key} values must be between {low} and {high}")
        if parsed not in out:
            out.append(parsed)
    return sorted(out)


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
    if after["max_position_pct_of_wallet"] > after["max_total_deployed_pct"]:
        raise ConfigError(
            "a single position cannot be allowed to exceed the total-deployed cap"
        )
    if after["rugcheck_pass_ttl_seconds"] < after["broad_scan_seconds"]:
        raise ConfigError("rug check TTL must be longer than one scan cycle")
    if after["confluence_required"] > len(after["confluence_timeframes"]):
        raise ConfigError(
            f"confluence_required ({after['confluence_required']}) exceeds the "
            f"{len(after['confluence_timeframes'])} timeframe(s) configured, so no "
            "entry could ever satisfy it"
        )
    if after["min_seconds_between_entries_same_mint"] < after["min_seconds_between_entries"]:
        raise ConfigError(
            "the same-token cooldown cannot be shorter than the global one"
        )
    if after["drawdown_tolerance"] <= 0:
        raise ConfigError("drawdown_tolerance must be positive")
    return clean


@dataclass
class Secrets:
    """Read from the environment at import; never written to config.json."""

    jupiter_api_key: str = ""
    rugcheck_api_key: str = ""
    solana_private_key: str = ""
    solana_rpc_url: str = "https://api.mainnet-beta.solana.com"
    flask_secret_key: str = ""
    secret_encryption_key: str = ""
    # Bearer token a RunPod worker presents when it reports a monthly retest's
    # progress back to this droplet (spec 5). Held here rather than in
    # config.json so it never lands in a file the dashboard renders or a
    # backup copies around casually.
    bulk_data_token: str = ""
    # RunPod's own API key, used by this droplet to orchestrate the monthly
    # GPU retest: spin up a worker, ship it data, tear it down afterward.
    runpod_api_key: str = ""
    # RunPod's per-account S3-compatible credentials, for uploading a batch's
    # Parquet chunk onto its network volume (console -> Settings -> S3 API Keys).
    runpod_s3_access_key: str = ""
    runpod_s3_secret_key: str = ""

    @classmethod
    def from_env(cls) -> "Secrets":
        return cls(
            jupiter_api_key=os.getenv("JUPITER_API_KEY", "").strip(),
            rugcheck_api_key=os.getenv("RUGCHECK_API_KEY", "").strip(),
            solana_private_key=os.getenv("SOLANA_PRIVATE_KEY", "").strip(),
            solana_rpc_url=os.getenv(
                "SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"
            ).strip(),
            flask_secret_key=os.getenv("FLASK_SECRET_KEY", "").strip(),
            secret_encryption_key=os.getenv("SECRET_ENCRYPTION_KEY", "").strip(),
            bulk_data_token=os.getenv("BULK_DATA_TOKEN", "").strip(),
            runpod_api_key=os.getenv("RUNPOD_API_KEY", "").strip(),
            runpod_s3_access_key=os.getenv("RUNPOD_S3_ACCESS_KEY", "").strip(),
            runpod_s3_secret_key=os.getenv("RUNPOD_S3_SECRET_KEY", "").strip(),
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

    def ensure_on_disk(self) -> bool:
        """Write the current (default, on a fresh install) values to `path` if
        no file exists yet. Returns True if a file was created.

        Deploy needs this: systemd's `ReadWritePaths=` under `ProtectSystem=strict`
        can only bind-mount a path that already exists, so `config.json` must be
        created before the first `systemctl enable --now`, not on first settings
        edit (which is otherwise the only place `_write()` was ever called from).
        """
        with self._lock:
            if self._path.exists():
                return False
            self._write()
            return True

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
