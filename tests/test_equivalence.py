"""C++ CPU engine vs Python reference: identical trades, matching curves.

The strategy list is chosen to cover every opcode and broker path.
"""

import glob
import os

import numpy as np
import pytest

from dsl import refengine
from dsl.compiler import compile_source
from engine import cpu_engine
from tests.util import random_walk_bars

pytestmark = pytest.mark.skipif(not cpu_engine.available(),
                                reason="btcpu module not built")

STRATS = glob.glob(os.path.join(os.path.dirname(__file__), "..", "strategies", "*.dsl"))

# every opcode / broker path gets exercised somewhere in this list
MINI = [
    "buy when crossover(sma(close, 10), sma(close, 30))\n"
    "sell when crossunder(sma(close, 10), sma(close, 30))\n",

    "let e = ema(close, 21)\nlet r = rsi(close, 14)\n"
    "buy when r < 30 and close > e\nsell when r > 70 or close < e * 0.98\n"
    "set stop_loss = 0.05\n",

    "let a = atr(14)\nlet hi = highest(high, 20)\nlet lo = lowest(low, 20)\n"
    "buy when close > delay(hi, 1)\nsell when close < delay(lo, 1)\n"
    "set trail_stop = 2 * a / close\n",

    "let s = stddev(close, 30)\nlet m = sma(close, 30)\n"
    "buy when close < m - 2 * s\nsell when close > m\n"
    "set take_profit = 0.03\nset size = 0.5\n",

    "short when crossunder(close, sma(close, 50))\n"
    "cover when crossover(close, sma(close, 50))\n"
    "set stop_loss = 0.04\n",

    "let x = roc(close, 10)\nlet y = change(close)\n"
    "buy when x > 0.01 and y > 0\nsell when x < -0.01\n",

    # safe-math and stateless ops
    "let z = sqrt(close - 200) + log(close - 200) + abs(volume) / (close - close)\n"
    "let w = min(open, close) + max(high, low) - z\n"
    "buy when not (w < 0) and close[3] < close\nsell when bar_index > 500 and w > 0\n",

    # ctx vars + position-aware exit
    "buy when position == 0 and close > open\n"
    "sell when position > 0 and equity > 10050 or entry_price > close * 1.01\n",

    "param n = 15 in [5, 60]\nparam k = 2 in [1, 4]\n"
    "let m = sma(close, n * k)\nbuy when close > m\nsell when close < m\n",
]


def both(src, bars, params=None):
    prog = compile_source(src)
    r_ref = refengine.run(prog, bars, params)
    r_cpu = cpu_engine.run(prog, bars, params)
    return r_ref, r_cpu


def assert_equivalent(r_ref, r_cpu):
    assert len(r_ref.trades) == len(r_cpu.trades)
    for a, b in zip(r_ref.trades, r_cpu.trades):
        assert (a.side, a.entry_t, a.exit_t, a.reason) == (b.side, b.entry_t, b.exit_t, b.reason)
        assert a.entry_px == pytest.approx(b.entry_px, rel=1e-6)
        assert a.exit_px == pytest.approx(b.exit_px, rel=1e-6)
        assert a.qty == pytest.approx(b.qty, rel=1e-5)
    np.testing.assert_allclose(np.asarray(r_ref.equity), np.asarray(r_cpu.equity),
                               rtol=1e-5)
    for k in ("final_equity", "max_drawdown"):
        assert r_ref.metrics[k] == pytest.approx(r_cpu.metrics[k], rel=1e-4, abs=1e-6), k
    assert r_ref.metrics["n_trades"] == r_cpu.metrics["n_trades"]
    assert r_ref.metrics["sharpe"] == pytest.approx(r_cpu.metrics["sharpe"],
                                                    rel=1e-3, abs=1e-4)


@pytest.mark.parametrize("path", STRATS, ids=[os.path.basename(p) for p in STRATS])
@pytest.mark.parametrize("seed", [0, 1])
def test_seed_strategies_match(path, seed):
    with open(path, encoding="utf-8") as f:
        src = f.read()
    bars = random_walk_bars(3000, seed=seed, drift=0.0002 * (seed - 0.5))
    assert_equivalent(*both(src, bars))


@pytest.mark.parametrize("i", range(len(MINI)))
@pytest.mark.parametrize("seed", [2, 3])
def test_mini_strategies_match(i, seed):
    bars = random_walk_bars(2500, seed=seed)
    assert_equivalent(*both(MINI[i], bars))


def test_param_overrides_match():
    bars = random_walk_bars(2000, seed=5)
    src = MINI[8]
    for params in ([5.0, 1.0], [33.3, 2.7], [60.0, 4.0]):
        assert_equivalent(*both(src, bars, params))


def test_locals_recording_matches():
    bars = random_walk_bars(1000, seed=6)
    prog = compile_source("let m = sma(close, 20)\nlet r = rsi(close, 14)\n"
                          "buy when close > m\nsell when close < m\n")
    r_ref = refengine.run(prog, bars, record_locals=True)
    r_cpu = cpu_engine.run(prog, bars, record_locals=True)
    np.testing.assert_allclose(r_ref.locals_curve, r_cpu.locals_curve, rtol=1e-5)


def test_batch_rows_match_single():
    bars = random_walk_bars(2000, seed=7)
    prog = compile_source(MINI[8])
    pm = np.array([[10, 1], [20, 2], [40, 3], [60, 4]], dtype=np.float32)
    batch = cpu_engine.run_batch(prog, bars, pm)
    assert batch.shape == (4, 7)
    for i in range(4):
        single = cpu_engine.run(prog, bars, list(map(float, pm[i])))
        assert batch[i, 0] == pytest.approx(single.metrics["final_equity"], rel=1e-6)
        assert int(batch[i, 4]) == single.metrics["n_trades"]
