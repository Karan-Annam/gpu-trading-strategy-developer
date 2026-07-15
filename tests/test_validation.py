"""Adversarial validation at data, configuration, and engine boundaries."""

import numpy as np
import pytest

from data.ohlcv import cache_path
from dsl.compiler import compile_source
from dsl.refengine import BacktestConfig
from engine.cpu_engine import _param_matrix, _validate_bars_layout
from tests.util import make_bars


def test_backtest_config_rejects_invalid_costs_and_scales():
    with pytest.raises(ValueError, match="fee_rate"):
        BacktestConfig(fee_rate=-0.001)
    with pytest.raises(ValueError, match="slip"):
        BacktestConfig(slip=1.0)
    with pytest.raises(ValueError, match="equity0"):
        BacktestConfig(equity0=0)
    with pytest.raises(ValueError, match="finite"):
        BacktestConfig(bars_per_year=float("nan"))


def test_bar_validation_rejects_invalid_ohlc_and_alignment():
    bars = make_bars([100, 101, 102])
    bars.high[1] = 99
    with pytest.raises(ValueError, match="high"):
        bars.validate()

    bars = make_bars([100, 101, 102])
    bars.ts[2] += 1
    with pytest.raises(ValueError, match="aligned"):
        bars.validate()

    bars = make_bars([100, 101, 102])
    bars.ts += 1
    with pytest.raises(ValueError, match="aligned"):
        bars.validate()


def test_bar_gaps_are_reported_not_hidden():
    bars = make_bars([100, 101, 102])
    bars.ts[2] += 2 * 60_000
    bars.validate()
    assert bars.missing_intervals() == 2
    with pytest.raises(ValueError, match="missing intervals"):
        _validate_bars_layout(bars)


def test_cache_path_rejects_traversal():
    with pytest.raises(ValueError, match="symbol"):
        cache_path("../BTC", "1m")
    with pytest.raises(ValueError, match="interval"):
        cache_path("BTCUSDT", "../../x")


def test_engine_boundary_rejects_bad_bars_and_params():
    bars = make_bars([100, 101, 102])
    bars.close[1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _validate_bars_layout(bars)

    program = compile_source(
        "param n = 2 in [1, 5] step 1\nbuy when close > sma(close, n)\n")
    with pytest.raises(ValueError, match="non-finite"):
        _param_matrix(program, np.array([[np.nan]], dtype=np.float32))
    with pytest.raises(ValueError, match="declared range"):
        _param_matrix(program, np.array([[6]], dtype=np.float32))
