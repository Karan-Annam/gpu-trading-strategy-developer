"""Python-facing wrapper around the btcuda module (GPU parameter sweeps).

Single backtests with trade capture stay on the CPU engine; the GPU exists to
evaluate thousands of parameter rows per launch. Large batches are chunked to
respect device memory.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from dsl.compiler import Program
from dsl.refengine import BacktestConfig
from engine.cpu_engine import _param_matrix, _validate_bars_layout

_BUILD_DIRS = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "build", "engine", "Release"),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "build", "Release"),
    os.path.join(os.path.dirname(__file__)),
]

_btcuda = None
STATE_BUDGET_BYTES = 1_500_000_000   # chunk launches beyond ~1.5 GB of VM state


def _load():
    global _btcuda
    if _btcuda is None:
        for d in _BUILD_DIRS:
            if d not in sys.path and os.path.isdir(d):
                sys.path.insert(0, d)
        import btcuda  # noqa: PLC0415
        _btcuda = btcuda
    return _btcuda


def available() -> bool:
    try:
        _load()
        return True
    except ImportError:
        return False


def device_info() -> dict:
    return dict(_load().device_info())


def run_batch(program: Program, bars, param_matrix: np.ndarray,
              config: BacktestConfig | None = None) -> np.ndarray:
    """Returns (N, 7): final_eq, total_ret, sharpe, max_dd, n_trades, wins, exposure_bars."""
    m = _load()
    cfg = config or BacktestConfig()
    a = program.arrays()
    _validate_bars_layout(bars)
    pm = _param_matrix(program, param_matrix)
    N = pm.shape[0]

    # sort rows so warps see similar window sizes (divergence), then unsort;
    # widest-range params (usually the big windows) become the primary key
    if N > 1 and pm.shape[1] > 0:
        spans = pm.max(axis=0) - pm.min(axis=0)
        col_order = np.argsort(spans)          # lexsort: last key is primary
        order = np.lexsort(pm[:, col_order].T)
        pm_sorted = np.ascontiguousarray(pm[order])
    else:
        order = None
        pm_sorted = pm

    per_thread = max(program.state_floats, 1) * 4
    chunk = max(min(N, STATE_BUDGET_BYTES // per_thread), 1024)

    outs = []
    for s in range(0, N, chunk):
        outs.append(m.run_batch(
            a["code"], a["consts"], a["state_kind"], a["state_off"], a["state_cap"],
            a["state_aux"], int(a["n_locals"]), int(a["state_floats"]),
            bars.open, bars.high, bars.low, bars.close, bars.volume,
            pm_sorted[s:s + chunk],
            cfg.fee_rate, cfg.slip, cfg.equity0, cfg.bars_per_year))
    out = np.concatenate(outs, axis=0) if len(outs) > 1 else outs[0]
    if order is not None:
        inv = np.empty_like(order)
        inv[order] = np.arange(N)
        out = out[inv]
    return out
