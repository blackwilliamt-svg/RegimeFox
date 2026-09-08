"""The optimizer must compute exactly what the live bot computes.

These are the most important tests in the suite. The whole value of the
walk-forward optimizer rests on one claim: that the parameters it validates
behave the same way when the droplet trades them. The moment
``solopt.indicators`` and ``solbot.indicators`` disagree - or the vectorized
engine and the scalar backtester disagree - every number the optimizer produces
becomes a statement about a strategy nobody is running.

So the indicators are compared bar for bar against pandas, and the engine is
compared trade for trade against the shipped backtester on the same data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solbot import indicators as scalar
from solopt import indicators as vector
from solopt.frames import frames_from_series
from solopt.schema import CRYPTO, CostModel

# The vectorized path stores prices as float32 - at 200 symbols by 100k bars,
# float64 would double a panel that already runs to hundreds of megabytes, and
# seven significant digits is far more than any trading decision needs. So the
# comparison is `|a - b| <= ATOL + RTOL * |b|`, the same form numpy uses.
#
# ATOL matters for the fraction-valued series. Momentum is the difference of two
# nearly-equal prices divided by one of them, so its relative error explodes as
# the move approaches zero while the absolute error stays around 1e-5 - which is
# three orders of magnitude below the ~0.008 threshold it is compared against.
# A purely relative tolerance would be measuring cancellation, not disagreement.
RTOL = 2e-3
ATOL = 1e-5


@pytest.fixture
def series() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 400
    close = np.cumprod(1 + rng.normal(0, 0.01, n)) * 100
    return pd.DataFrame(
        {
            "ts": np.arange(n) * 600,
            "open": close,
            "high": close * (1 + np.abs(rng.normal(0, 0.004, n))),
            "low": close * (1 - np.abs(rng.normal(0, 0.004, n))),
            "close": close,
            "volume": np.abs(rng.normal(1000, 400, n)),
        }
    )


def _block(df: pd.DataFrame, column: str) -> np.ndarray:
    return df[column].to_numpy(dtype=np.float32)[None, :]


def assert_matches(
    got: np.ndarray, expected: pd.Series, name: str, *, atol: float = ATOL
) -> None:
    a = np.asarray(got, dtype=np.float64).ravel()
    b = expected.to_numpy(dtype=np.float64)
    assert a.shape == b.shape, f"{name}: shape {a.shape} against {b.shape}"

    # A value defined in one implementation and not the other is a real
    # divergence even when the numbers that do exist agree.
    assert np.array_equal(np.isnan(a), np.isnan(b)), f"{name}: NaN positions differ"

    both = ~np.isnan(a)
    if not both.any():
        return
    allowed = atol + RTOL * np.abs(b[both])
    error = np.abs(a[both] - b[both])
    worst = int(np.argmax(error - allowed))
    assert (error <= allowed).all(), (
        f"{name}: worst disagreement {error[worst]:.3e} against an allowance of "
        f"{allowed[worst]:.3e} (values {a[both][worst]:.8g} / {b[both][worst]:.8g})"
    )


@pytest.mark.parametrize("span", [9, 12, 21, 34])
def test_ema_matches_pandas(series, span):
    assert_matches(
        vector.ema_stack(_block(series, "close"), [span])[0],
        scalar.ema(series["close"], span),
        f"ema{span}",
    )


@pytest.mark.parametrize("period", [7, 14, 21])
def test_rsi_matches_pandas_including_the_warmup_convention(series, period):
    """RSI reads 100 before it has enough history in both implementations.

    That is a quirk of the shipped code rather than a considered choice, but it
    is the shipped behaviour, and an optimizer that smoothed it over would find
    entries the live bot rejects.
    """
    assert_matches(
        vector.rsi_stack(_block(series, "close"), [period])[0],
        scalar.rsi(series["close"], period),
        f"rsi{period}",
    )


@pytest.mark.parametrize("period", [7, 14])
def test_atr_matches_pandas(series, period):
    assert_matches(
        vector.atr_stack(
            _block(series, "high"), _block(series, "low"), _block(series, "close"), [period]
        )[0],
        scalar.atr(series, period),
        f"atr{period}",
    )


@pytest.mark.parametrize("lookback", [5, 20, 40])
def test_volume_ratio_matches_pandas(series, lookback):
    assert_matches(
        vector.volume_ratio_stack(_block(series, "volume"), [lookback])[0],
        scalar.volume_ratio(series, lookback),
        f"volume_ratio{lookback}",
    )


@pytest.mark.parametrize("bars", [2, 3, 5])
def test_momentum_matches_pandas(series, bars):
    assert_matches(
        vector.momentum_stack(_block(series, "close"), [bars])[0],
        scalar.momentum_pct(series, bars),
        f"momentum{bars}",
    )


@pytest.mark.parametrize("bars", [2, 3, 5])
def test_consecutive_up_matches_pandas(series, bars):
    got = vector.consecutive_up_stack(_block(series, "close"), [bars])[0].ravel()
    expected = scalar.consecutive_up(series, bars).to_numpy()
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("lookback", [10, 20])
def test_efficiency_ratio_matches_pandas(series, lookback):
    assert_matches(
        vector.efficiency_ratio_stack(_block(series, "close"), [lookback])[0],
        scalar.efficiency_ratio(series["close"], lookback),
        f"efficiency{lookback}",
    )


@pytest.mark.parametrize("multiple", [3, 6])
def test_aggregate_trend_matches_pandas(series, multiple):
    got = vector.aggregate_trend(_block(series, "close"), multiple, 9, 21)[0]
    expected = scalar.aggregate_trend(series["close"], multiple, 9, 21).to_numpy()
    assert np.array_equal(got.ravel(), expected)


def test_regime_classification_matches(series):
    lookback, trend_er, chop = 20, 0.35, 0.03
    computed = scalar.compute(
        series,
        {
            "ema_fast": 9, "ema_slow": 21, "rsi_period": 14, "atr_period": 14,
            "volume_spike_lookback": 20, "momentum_candles": 3,
            "regime_lookback": lookback, "regime_trend_er": trend_er,
            "regime_chop_atr_pct": chop, "confluence_timeframes": [],
        },
    )
    efficiency = vector.efficiency_ratio_stack(_block(series, "close"), [lookback])[0]
    atr_pct = (
        vector.atr_stack(
            _block(series, "high"), _block(series, "low"), _block(series, "close"), [14]
        )[0]
        / _block(series, "close")
    )
    got = vector.classify_regime(
        efficiency, atr_pct, trend_er=trend_er, chop_atr_pct=chop
    )
    assert np.array_equal(got.ravel(), computed["regime"].to_numpy())


# --------------------------------------------------------------------------
# Indicator-combination search (gap-closure item 3)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("fast,slow,signal", [(12, 26, 9), (5, 13, 4)])
def test_macd_matches_pandas(series, fast, slow, signal):
    macd_line, macd_signal = scalar.macd(series["close"], fast, slow, signal)
    got_line = vector.macd_line_stack(_block(series, "close"), [(fast, slow)])[0]
    assert_matches(got_line, macd_line, f"macd_line{fast}/{slow}")
    got_signal = vector.macd_signal_stack(got_line, [signal])[0]
    assert_matches(got_signal, macd_signal, f"macd_signal{fast}/{slow}/{signal}")


@pytest.mark.parametrize("period", [10, 20])
def test_bollinger_matches_pandas(series, period):
    mid, upper, lower = scalar.bollinger_bands(series["close"], period, 2.0)
    expected_pb = scalar.percent_b(series["close"], upper, lower)

    mean, std = vector.bollinger_stack(_block(series, "close"), [period])
    got_upper = mean[0] + 2.0 * std[0]
    got_lower = mean[0] - 2.0 * std[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        got_pb = (_block(series, "close")[0] - got_lower) / (got_upper - got_lower)

    assert_matches(mean[0], mid, f"bb_mid{period}")
    assert_matches(got_pb, expected_pb, f"bb_percent{period}")


@pytest.mark.parametrize("k_period,d_period", [(14, 3), (9, 5)])
def test_stochastic_matches_pandas(series, k_period, d_period):
    # %K's numerator (close - lowest_low) is occasionally near zero - close
    # sitting almost exactly at the window low - so the same float32-input
    # cancellation the module docstring calls out for momentum shows up here
    # too, amplified by %K's own 100x scale. A slightly wider absolute
    # tolerance for this one series, same rationale as momentum's ATOL.
    k, d = scalar.stochastic(series, k_period, d_period)
    got_k = vector.stochastic_k_stack(
        _block(series, "high"), _block(series, "low"), _block(series, "close"), [k_period]
    )[0]
    assert_matches(got_k, k, f"stoch_k{k_period}", atol=2e-3)
    got_d = vector.stochastic_d_stack(got_k, [d_period])[0]
    assert_matches(got_d, d, f"stoch_d{k_period}/{d_period}", atol=2e-3)


@pytest.mark.parametrize("period", [7, 14])
def test_adx_matches_pandas(series, period):
    adx_line, plus_di, minus_di = scalar.adx(series, period)
    got_adx, got_plus, got_minus = vector.adx_stack(
        _block(series, "high"), _block(series, "low"), _block(series, "close"), [period]
    )
    assert_matches(got_plus[0], plus_di, f"plus_di{period}")
    assert_matches(got_minus[0], minus_di, f"minus_di{period}")
    assert_matches(got_adx[0], adx_line, f"adx{period}")


@pytest.mark.parametrize("period", [10, 20])
def test_vwap_matches_pandas(series, period):
    expected = scalar.vwap(series, period)
    got = vector.vwap_stack(
        _block(series, "high"), _block(series, "low"), _block(series, "close"),
        _block(series, "volume"), [period],
    )[0]
    assert_matches(got, expected, f"vwap{period}")


# --------------------------------------------------------------------------
# Engine parity
# --------------------------------------------------------------------------
def _make_market(seed: int, n: int, seconds: int, start: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0002, 0.012, n)
    close = 100 * np.cumprod(1 + ret)
    return np.vstack(
        [
            start + np.arange(n) * seconds,
            np.concatenate(([close[0]], close[:-1])),
            close * (1 + np.abs(rng.normal(0, 0.006, n))),
            close * (1 - np.abs(rng.normal(0, 0.006, n))),
            close,
            np.abs(rng.lognormal(8, 0.8, n)),
        ]
    )


def test_vector_engine_reproduces_the_scalar_backtester(workspace, cfg):
    """The two engines must agree trade for trade on identical data.

    The vectorized engine batches every combination and symbol together; the
    shipped backtester walks one merged timeline. They share no code beyond the
    strategy rules, so agreement here is evidence the batching is faithful
    rather than merely fast.
    """
    from solbot.backtest import Backtester
    from solbot.candlestore import ParquetCandleStore
    from solbot.datastore import DataStore
    from solopt.arrays import Throttle, get_backend
    from solopt.engine import PortfolioSettings, VectorEngine

    import time as _time

    conn = workspace["conn"]
    settings = cfg.as_dict()
    settings.update(
        {
            "candle_minutes": 1,          # matches the fixed 1-minute base; nothing is aggregated
            "regime_gate_enabled": False,  # exercised by their own tests
            "confluence_enabled": False,
            "sizing_mode": "flat",
        }
    )

    seconds, bars = 300, 1500
    start = (int(_time.time()) - (bars + 5) * seconds) // seconds * seconds
    symbols = ["AAA", "BBB", "CCC"]
    series_by_symbol = {}
    candles = ParquetCandleStore()
    for i, symbol in enumerate(symbols):
        rows = _make_market(11 + i, bars, seconds, start)
        series_by_symbol[symbol] = rows
        candles.append(symbol, "1m", list(zip(*rows)))
        conn.execute(
            "INSERT INTO universe(mint, symbol, liquidity_usd, volume_24h_usd, updated_at) "
            "VALUES (?,?,?,?,?)",
            (symbol, symbol, 5_000_000.0, 2_000_000.0, start),
        )

    scalar_result = Backtester(DataStore(object(), settings), settings).run(
        mints=symbols, conn=conn
    )

    frames = frames_from_series(series_by_symbol, seconds=seconds, schema=CRYPTO)
    frames.liquidity = np.full(frames.close.shape, 5_000_000.0, dtype=np.float32)
    combo = {
        key: settings[key]
        for key in (
            "volume_spike_multiple", "volume_spike_lookback", "momentum_candles",
            "momentum_min_pct", "rsi_period", "rsi_max_entry", "ema_fast", "ema_slow",
            "atr_period", "stop_atr_mult", "rr_min", "rr_max", "trailing_activate_r",
            "trailing_distance_atr", "signal_invalidation_bars", "max_hold_minutes",
        )
    }
    combo.update(
        regime_lookback=20, regime_trend_er=0.3, regime_chop_atr_pct=0.03,
        regime_allowed=7, confluence_required=0,
    )

    engine = VectorEngine(get_backend(prefer_gpu=False), Throttle(100.0))
    vector_result = engine.run(
        frames,
        [combo],
        PortfolioSettings(
            starting_balance=settings["paper_starting_balance"],
            max_total_deployed_pct=settings["max_total_deployed_pct"],
            max_position_pct_of_wallet=settings["max_position_pct_of_wallet"],
            max_position_pct_of_liquidity=settings["max_position_pct_of_liquidity"],
            min_position_usd=settings["min_position_usd"],
            min_candles_required=settings["min_candles_required"],
            volatility_target_atr_pct=settings["volatility_target_atr_pct"],
            volatility_size_floor=settings["volatility_size_floor"],
            correlation_lookback=settings["correlation_lookback"],
            correlation_max=settings["correlation_max"],
            costs=CostModel(
                fee_pct=settings["taker_fee_pct"],
                slippage_pct=settings["max_slippage_pct"],
            ),
            confluence_multiples=(),
            sizing_mode="flat",
        ),
    )

    metrics = vector_result.metrics[0]
    assert metrics["trades"] == len(scalar_result.trades), (
        f"{metrics['trades']} vectorized trades against "
        f"{len(scalar_result.trades)} scalar ones"
    )

    # Decisions must match exactly: the same token entered on the same bar and
    # exited on the same bar. This is the part that would break if the batching
    # were unfaithful, and float32 storage cannot move a bar boundary.
    scalar_trades = sorted(
        (t.mint, t.entry_ts, t.exit_ts) for t in scalar_result.trades
    )
    vector_trades = sorted(
        (
            vector_result.symbols[int(row["symbol"])],
            int(row["entry_ts"]),
            int(row["exit_ts"]),
        )
        for row in vector_result.trades
    )
    assert vector_trades == scalar_trades

    # Magnitudes agree to float32 precision. Per trade rather than only in
    # aggregate, so opposite-signed errors cannot cancel into a false pass.
    scalar_pnl = {
        (t.mint, t.entry_ts): t.pnl_usd for t in scalar_result.trades
    }
    for row in vector_result.trades:
        key = (vector_result.symbols[int(row["symbol"])], int(row["entry_ts"]))
        assert float(row["pnl_usd"]) == pytest.approx(scalar_pnl[key], rel=1e-3, abs=1e-4)

    assert metrics["total_pnl"] == pytest.approx(scalar_result.total_pnl, rel=1e-3)
    assert metrics["ending_balance"] == pytest.approx(
        scalar_result.ending_balance, rel=1e-3
    )
