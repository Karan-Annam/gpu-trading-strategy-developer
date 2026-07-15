"""Python-facing wrapper around the btcpu pybind11 module.

Presents the same interface as dsl.refengine.run so tests and the Lab can
switch engines with one argument.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from dsl.compiler import Program
from dsl.refengine import BacktestConfig, BacktestResult, Trade

_BUILD_DIRS = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "build", "engine", "Release"),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "build", "Release"),
    os.path.join(os.path.dirname(__file__)),
]

_btcpu = None


def _validate_bars_layout(bars) -> None:
    if hasattr(bars, "validate"):
        bars.validate()
    if hasattr(bars, "missing_intervals"):
        missing = bars.missing_intervals()
        if missing:
            raise ValueError(
                f"bar data has {missing} missing intervals; fill or reindex gaps before backtesting")
    close = bars.close
    if close.ndim != 1 or close.dtype != np.float32:
        raise ValueError("close must be a one-dimensional float32 array")
    n = len(close)
    if n == 0:
        raise ValueError("bar data is empty")
    for name in ("open", "high", "low", "close", "volume"):
        values = getattr(bars, name)
        if values.ndim != 1 or values.dtype != np.float32 or len(values) != n:
            raise ValueError(f"{name} must be a length-{n} float32 array")
        if not bool(np.all(np.isfinite(values))):
            raise ValueError(f"{name} contains non-finite values")
    if not bool(np.all(bars.low > 0)):
        raise ValueError("prices must be positive")
    if not bool(np.all(bars.high >= np.maximum(bars.open, bars.close))):
        raise ValueError("high is below open or close")
    if not bool(np.all(bars.low <= np.minimum(bars.open, bars.close))):
        raise ValueError("low is above open or close")
    if not bool(np.all(bars.volume >= 0)):
        raise ValueError("volume is negative")


def _param_matrix(program: Program, values: np.ndarray) -> np.ndarray:
    matrix = np.ascontiguousarray(values, dtype=np.float32)
    expected = len(program.params)
    if matrix.ndim != 2 or matrix.shape[1] != expected:
        raise ValueError(f"param_matrix must have shape (N, {expected})")
    if matrix.shape[0] == 0:
        raise ValueError("param_matrix must contain at least one row")
    if not bool(np.all(np.isfinite(matrix))):
        raise ValueError("param_matrix contains non-finite values")
    for j, spec in enumerate(program.params):
        if bool(np.any(matrix[:, j] < spec["lo"])) or bool(np.any(matrix[:, j] > spec["hi"])):
            raise ValueError(f"parameter {spec['name']} is outside its declared range")
    return matrix


def _param_vector(program: Program, values) -> np.ndarray:
    vector = np.ascontiguousarray(values, dtype=np.float32)
    if vector.ndim != 1 or len(vector) != len(program.params):
        raise ValueError(f"params must contain {len(program.params)} values")
    _param_matrix(program, vector.reshape(1, -1))
    return vector


def _load():
    global _btcpu
    if _btcpu is None:
        for d in _BUILD_DIRS:
            if d not in sys.path and os.path.isdir(d):
                sys.path.insert(0, d)
        import btcpu  # noqa: PLC0415
        _btcpu = btcpu
    return _btcpu


def available() -> bool:
    try:
        _load()
        return True
    except ImportError:
        return False


def _engine_args(program: Program, bars):
    a = program.arrays()
    return (a["code"], a["consts"], a["state_kind"], a["state_off"], a["state_cap"],
            a["state_aux"], int(a["n_locals"]), int(a["state_floats"]),
            bars.open, bars.high, bars.low, bars.close, bars.volume)


def run(program: Program, bars, params: list[float] | None = None,
        config: BacktestConfig | None = None,
        record_locals: bool = False) -> BacktestResult:
    m = _load()
    cfg = config or BacktestConfig()
    _validate_bars_layout(bars)
    pvals = _param_vector(
        program, params if params is not None else program.param_defaults())
    out = m.run_single(*_engine_args(program, bars), pvals,
                       cfg.fee_rate, cfg.slip, cfg.equity0, cfg.bars_per_year,
                       record_locals)
    if (not bool(np.all(np.isfinite(out["equity"]))) or
            not all(np.isfinite(value) for value in out["metrics"].values())):
        raise FloatingPointError("CPU engine produced non-finite results")
    trades = [Trade(**t) for t in out["trades"]]
    metrics = dict(out["metrics"])
    metrics["total_return"] = float(metrics["total_return"])
    return BacktestResult(out["equity"], trades, metrics, out.get("locals"))


def run_batch(program: Program, bars, param_matrix: np.ndarray,
              config: BacktestConfig | None = None) -> np.ndarray:
    """Returns (N, 7): final_eq, total_ret, sharpe, max_dd, n_trades, wins, exposure_bars."""
    m = _load()
    cfg = config or BacktestConfig()
    _validate_bars_layout(bars)
    pm = _param_matrix(program, param_matrix)
    out = m.run_batch(*_engine_args(program, bars), pm,
                      cfg.fee_rate, cfg.slip, cfg.equity0, cfg.bars_per_year)
    if not bool(np.all(np.isfinite(out))):
        raise FloatingPointError("CPU engine produced non-finite metrics")
    return out
