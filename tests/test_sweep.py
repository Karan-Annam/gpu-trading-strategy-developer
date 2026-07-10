"""Sweep / walk-forward / deflated-Sharpe utilities."""

import numpy as np
import pytest

from dsl.compiler import compile_source
from engine import cpu_engine
from evolve import sweep
from tests.util import random_walk_bars

SRC = ("param n = 20 in [5, 60] step 5\nparam sl = 0.02 in [0.01, 0.05]\n"
       "let m = sma(close, n)\nbuy when close > m\nsell when close < m\n"
       "set stop_loss = sl\n")


def test_axis_values_step():
    vals = sweep.axis_values({"lo": 5, "hi": 60, "step": 5})
    assert list(vals) == [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    vals2 = sweep.axis_values({"lo": 0.0, "hi": 1.0, "step": None}, steps=11)
    assert len(vals2) == 11


def test_build_grid_and_lhs():
    prog = compile_source(SRC)
    grid = sweep.build_grid(prog, steps_per_axis=7)
    assert grid.shape == (12 * 7, 2)          # stepped axis x linspace axis
    lhs = sweep.lhs_sample(prog, 100, seed=1)
    assert lhs.shape == (100, 2)
    assert lhs[:, 0].min() >= 5 and lhs[:, 0].max() <= 60
    assert lhs[:, 1].min() >= 0.01 and lhs[:, 1].max() <= 0.05


def test_folds_are_ordered_and_cover():
    fs = sweep.folds(1000, n_folds=4, train_frac=0.7)
    assert len(fs) == 4
    for (a0, a1), (b0, b1) in fs:
        assert 0 <= a0 < a1 == b0 < b1 <= 1000
    assert fs[-1][1][1] == 1000


@pytest.mark.skipif(not cpu_engine.available(), reason="btcpu not built")
def test_fold_metrics_and_wfa():
    prog = compile_source(SRC)
    bars = random_walk_bars(6000, seed=11)
    pm = sweep.build_grid(prog, steps_per_axis=3)
    train, test = sweep.fold_metrics(prog, bars, pm, n_folds=3, prefer_gpu=False)
    assert train.shape == (3, len(pm), 7)
    assert test.shape == (3, len(pm), 7)
    score = sweep.robust_score(test)
    assert score.shape == (len(pm),)
    wfa = sweep.walk_forward_select(prog, bars, pm, n_folds=3, prefer_gpu=False)
    assert wfa["picks"].shape == (3, 2)
    assert wfa["oos_metrics"].shape == (3, 7)


def test_deflated_sharpe_penalizes_many_trials():
    rng = np.random.default_rng(0)
    few = rng.normal(0.5, 0.5, 5)
    many = rng.normal(0.5, 0.5, 5000)
    sr_obs = 1.5
    p_few = sweep.deflated_sharpe(sr_obs, few, n_bar_returns=100_000)
    p_many = sweep.deflated_sharpe(sr_obs, many, n_bar_returns=100_000)
    assert 0 <= p_many <= 1 and 0 <= p_few <= 1
    assert p_many < p_few          # more trials -> less credible best


def test_deflated_sharpe_rewards_strong_sr():
    rng = np.random.default_rng(1)
    trials = rng.normal(0.0, 0.3, 1000)
    weak = sweep.deflated_sharpe(0.2, trials, n_bar_returns=500_000)
    strong = sweep.deflated_sharpe(3.0, trials, n_bar_returns=500_000)
    assert strong > weak
    assert strong > 0.9
