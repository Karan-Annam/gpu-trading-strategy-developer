"""Golden-model tests: broker math, indicator values, protective exits."""

import numpy as np
import pytest

from dsl import refengine
from dsl.compiler import compile_source
from dsl.refengine import BacktestConfig
from tests.util import make_bars, random_walk_bars

FLAT = "enter_long when bar_index == 0\n"


def run(src, bars, params=None, **cfg):
    prog = compile_source(src)
    return refengine.run(prog, bars, params,
                         BacktestConfig(**cfg) if cfg else None)


def test_buy_and_hold_math():
    n = 10
    bars = make_bars([100.0] * n)
    fee, slip = 0.001, 0.0005
    res = run(FLAT, bars, fee_rate=fee, slip=slip)
    # signal at t=0 -> fill at open[1]*(1+slip); force-close at close[9]*(1-slip)
    entry = 100.0 * (1 + slip)
    qty = 10000.0 / entry * 1.0
    cash_after_entry = 10000.0 - qty * entry - qty * entry * fee
    exit_px = 100.0 * (1 - slip)
    final = cash_after_entry + qty * exit_px - qty * exit_px * fee
    assert len(res.trades) == 1
    tr = res.trades[0]
    assert tr.reason == "eod" and tr.side == 1 and tr.entry_t == 1
    assert res.metrics["final_equity"] == pytest.approx(final, rel=1e-5)
    assert tr.pnl == pytest.approx(final - 10000.0, rel=1e-3, abs=0.05)


def test_no_signals_no_trades():
    res = run("enter_long when close > 1000000\n", random_walk_bars(200))
    assert res.trades == []
    assert np.all(res.equity == np.float32(10000.0))
    assert res.metrics["sharpe"] == 0.0


def test_stop_loss_gap_and_trigger_fills():
    # entry fills at t=1 open=100(1+slip); stop 5% => trigger 99.7...*0.95
    close = [100, 100, 96, 100, 100]
    low = [100, 100, 90, 100, 100]
    open_ = [100, 100, 99, 100, 100]
    bars = make_bars(close, open_, None, low)
    res = run(FLAT + "set stop_loss = 0.05\n", bars, slip=0.0, fee_rate=0.0)
    assert len(res.trades) == 1
    tr = res.trades[0]
    assert tr.reason == "stop" and tr.exit_t == 2
    assert tr.exit_px == pytest.approx(100.0 * 0.95, rel=1e-6)  # min(open=99? no: open 99 > trigger 95)
    # gap case: open below trigger -> fill at open
    open2 = [100, 100, 92, 100, 100]
    res2 = run(FLAT + "set stop_loss = 0.05\n", make_bars(close, open2, None, low),
               slip=0.0, fee_rate=0.0)
    assert res2.trades[0].exit_px == pytest.approx(92.0, rel=1e-6)


def test_take_profit():
    close = [100, 100, 104, 110, 100]
    high = [100, 100, 108, 110, 100]
    bars = make_bars(close, None, high)
    res = run(FLAT + "set take_profit = 0.06\n", bars, slip=0.0, fee_rate=0.0)
    tr = res.trades[0]
    assert tr.reason == "tp" and tr.exit_t == 2
    assert tr.exit_px == pytest.approx(106.0, rel=1e-6)


def test_trailing_stop_uses_prior_peak():
    # peak established at t=2 high=120; trail 10% => trigger 108 at t=3
    close = [100, 100, 118, 109, 109]
    high = [100, 100, 120, 112, 109]
    low = [100, 100, 100, 107, 109]
    bars = make_bars(close, None, high, low)
    res = run(FLAT + "set trail_stop = 0.1\n", bars, slip=0.0, fee_rate=0.0)
    tr = res.trades[0]
    assert tr.reason == "trail" and tr.exit_t == 3
    assert tr.exit_px == pytest.approx(108.0, rel=1e-6)


def test_short_side_profits_on_decline():
    close = list(np.linspace(100, 80, 30))
    bars = make_bars(close)
    res = run("enter_short when bar_index == 0\nexit_short when bar_index == 25\n",
              bars, slip=0.0, fee_rate=0.0)
    assert len(res.trades) == 1
    tr = res.trades[0]
    assert tr.side == -1 and tr.pnl > 0
    assert res.metrics["total_return"] > 0.1


def test_conflicting_entries_ignored():
    res = run("enter_long when bar_index >= 0\nenter_short when bar_index >= 0\n",
              random_walk_bars(50))
    assert res.trades == []


def test_crossover_fires_once():
    a = [1, 1, 1, 5, 5, 5, 5, 5, 5, 5]  # crosses "3" once at t=3
    bars = make_bars(a)
    res = run("enter_long when crossover(close, 3)\n", bars, slip=0.0, fee_rate=0.0)
    assert len(res.trades) == 1
    assert res.trades[0].entry_t == 4  # signal t=3, fill next open


def test_bars_held_times_the_exit():
    # bars_held is 0 on the entry bar, so `>= 3` first holds 3 bars after the
    # fill; the exit signal then fills at the next open
    res = run("enter_long when bar_index == 10\n"
              "exit_long when bars_held >= 3\n",
              random_walk_bars(30), slip=0.0, fee_rate=0.0)
    assert len(res.trades) == 1
    tr = res.trades[0]
    assert tr.reason == "signal"
    assert tr.entry_t == 11   # signal t=10, fill next open
    assert tr.exit_t == 15    # bars_held hits 3 at t=14, fill next open


def indicator_curve(src_let: str, bars, params=None):
    prog = compile_source(src_let + "buy when close > 999999\n")
    res = refengine.run(prog, bars, params, record_locals=True)
    return res.locals_curve[:, 0].astype(np.float64)


def test_sma_matches_numpy():
    bars = random_walk_bars(300, seed=1)
    got = indicator_curve("let m = sma(close, 20)\n", bars)
    c = bars.close.astype(np.float64)
    for t in [0, 5, 19, 100, 299]:
        expect = c[max(0, t - 19):t + 1].mean()
        assert got[t] == pytest.approx(expect, rel=2e-4), t


def test_ema_matches_recurrence():
    bars = random_walk_bars(300, seed=2)
    got = indicator_curve("let m = ema(close, 15)\n", bars)
    c = bars.close.astype(np.float64)
    alpha = 2.0 / 16.0
    e = c[0]
    assert got[0] == pytest.approx(e, rel=1e-5)
    for t in range(1, 300):
        e = e + alpha * (c[t] - e)
    assert got[299] == pytest.approx(e, rel=1e-3)


def test_highest_lowest_delay():
    bars = random_walk_bars(200, seed=3)
    hi = indicator_curve("let x = highest(high, 30)\n", bars)
    lo = indicator_curve("let x = lowest(low, 30)\n", bars)
    dl = indicator_curve("let x = delay(close, 7)\n", bars)
    h = bars.high.astype(np.float64)
    l = bars.low.astype(np.float64)
    c = bars.close.astype(np.float64)
    for t in [0, 3, 29, 150, 199]:
        assert hi[t] == pytest.approx(h[max(0, t - 29):t + 1].max(), rel=1e-6)
        assert lo[t] == pytest.approx(l[max(0, t - 29):t + 1].min(), rel=1e-6)
        assert dl[t] == pytest.approx(c[max(0, t - 7)], rel=1e-6)


def test_stddev_two_pass_precision():
    # large price level is exactly where naive sumsq dies in float32
    bars = random_walk_bars(300, seed=4, start=100000.0, vol=0.0005)
    got = indicator_curve("let s = stddev(close, 50)\n", bars)
    c = bars.close.astype(np.float64)
    t = 250
    expect = c[t - 49:t + 1].std()  # population
    assert got[t] == pytest.approx(expect, rel=5e-2)


def test_rsi_range_and_direction():
    up = make_bars(np.linspace(100, 120, 100))
    down = make_bars(np.linspace(120, 100, 100))
    r_up = indicator_curve("let r = rsi(close, 14)\n", up)
    r_dn = indicator_curve("let r = rsi(close, 14)\n", down)
    assert np.all((r_up >= 0) & (r_up <= 100))
    assert r_up[50] > 90
    assert r_dn[50] < 10


def test_atr_constant_range():
    n = 50
    close = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    bars = make_bars(close, None, high, low)
    got = indicator_curve("let a = atr(14)\n", bars)
    assert got[40] == pytest.approx(2.0, rel=1e-5)


def test_dynamic_size_and_param_override():
    bars = make_bars([100.0] * 10)
    src = ("param frac = 1.0 in [0.1, 1.0]\n" + FLAT + "set size = frac\n")
    full = run(src, bars, params=[1.0], slip=0.0, fee_rate=0.0)
    half = run(src, bars, params=[0.5], slip=0.0, fee_rate=0.0)
    assert full.trades[0].qty == pytest.approx(2 * half.trades[0].qty, rel=1e-5)


def test_equity_curve_marks_to_market():
    close = [100, 100, 110, 120, 120]
    bars = make_bars(close)
    res = run(FLAT, bars, slip=0.0, fee_rate=0.0)
    assert res.equity[2] == pytest.approx(11000.0, rel=1e-5)
    assert res.equity[3] == pytest.approx(12000.0, rel=1e-5)
