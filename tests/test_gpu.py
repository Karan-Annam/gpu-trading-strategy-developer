"""CUDA engine vs C++ CPU engine: metric matrices must match.

Built with --fmad=false, the GPU runs the same float ops in the same order,
so agreement should be near-exact.
"""

import numpy as np
import pytest

from dsl.compiler import compile_source
from engine import cpu_engine, gpu_engine
from tests.test_equivalence import MINI
from tests.util import random_walk_bars

pytestmark = pytest.mark.skipif(
    not (cpu_engine.available() and gpu_engine.available()),
    reason="btcpu/btcuda modules not built")


def param_grid(prog, n_per_axis=6, rng=None):
    """Grid over declared params (or a single default row)."""
    if not prog.params:
        return np.zeros((1, 0), dtype=np.float32)
    axes = [np.linspace(p["lo"], p["hi"], n_per_axis, dtype=np.float32)
            for p in prog.params]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1).astype(np.float32)


def compare(src, bars, pm=None):
    prog = compile_source(src)
    if pm is None:
        pm = param_grid(prog)
    if pm.shape[1] == 0:
        pm = np.zeros((1, 0), dtype=np.float32)
        pm_cpu = np.asarray([prog.param_defaults()], dtype=np.float32).reshape(1, -1)
    else:
        pm_cpu = pm
    cpu = cpu_engine.run_batch(prog, bars, pm_cpu)
    gpu = gpu_engine.run_batch(prog, bars, pm_cpu)
    assert cpu.shape == gpu.shape
    # integer-valued columns: trades, wins, exposure_bars must match exactly
    np.testing.assert_array_equal(cpu[:, 4], gpu[:, 4], err_msg="n_trades differ")
    np.testing.assert_array_equal(cpu[:, 5], gpu[:, 5], err_msg="wins differ")
    np.testing.assert_array_equal(cpu[:, 6], gpu[:, 6], err_msg="exposure differ")
    np.testing.assert_allclose(cpu[:, 0], gpu[:, 0], rtol=1e-5, err_msg="final_eq")
    np.testing.assert_allclose(cpu[:, 3], gpu[:, 3], rtol=1e-4, atol=1e-6, err_msg="max_dd")
    np.testing.assert_allclose(cpu[:, 2], gpu[:, 2], rtol=1e-3, atol=1e-3, err_msg="sharpe")


def test_device_info():
    info = gpu_engine.device_info()
    assert info["sm"] > 0


@pytest.mark.parametrize("i", range(len(MINI)))
def test_mini_strategies_gpu(i):
    bars = random_walk_bars(2500, seed=10 + i)
    compare(MINI[i], bars)


def test_param_sweep_grid_gpu():
    bars = random_walk_bars(4000, seed=42)
    src = ("param fast = 10 in [4, 40]\nparam slow = 50 in [20, 200]\n"
           "buy when crossover(sma(close, fast), sma(close, slow))\n"
           "sell when crossunder(sma(close, fast), sma(close, slow))\n"
           "set stop_loss = 0.03\n")
    prog = compile_source(src)
    pm = param_grid(prog, n_per_axis=12)   # 144 combos
    compare(src, bars, pm)


def test_large_batch_chunking():
    bars = random_walk_bars(1500, seed=43)
    src = ("param n = 20 in [5, 200]\n"
           "buy when close > sma(close, n)\nsell when close < sma(close, n)\n")
    prog = compile_source(src)
    rng = np.random.default_rng(0)
    pm = rng.uniform(5, 200, size=(20000, 1)).astype(np.float32)
    gpu = gpu_engine.run_batch(prog, bars, pm)
    assert gpu.shape == (20000, 7)
    # spot-check a few rows against the CPU engine
    idx = [0, 777, 19999]
    cpu = cpu_engine.run_batch(prog, bars, pm[idx])
    np.testing.assert_allclose(cpu[:, 0], gpu[idx, 0], rtol=1e-5)
    np.testing.assert_array_equal(cpu[:, 4], gpu[idx, 4])
