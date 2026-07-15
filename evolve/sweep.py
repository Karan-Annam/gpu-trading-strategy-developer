"""Parameter sampling + walk-forward evaluation on top of the batch engines.

Metric-matrix column order (from the engines):
    0 final_equity, 1 total_return, 2 sharpe, 3 max_dd, 4 n_trades,
    5 wins, 6 exposure_bars
"""

from __future__ import annotations

import numpy as np

from dsl.compiler import Program

COL = {"final_equity": 0, "total_return": 1, "sharpe": 2, "max_dd": 3,
       "n_trades": 4, "wins": 5, "exposure_bars": 6}


def pick_engine(prefer_gpu: bool = True):
    from engine import cpu_engine, gpu_engine
    if prefer_gpu and gpu_engine.available():
        return gpu_engine
    return cpu_engine


# ---------- sampling ----------

def axis_values(spec: dict, steps: int | None = None) -> np.ndarray:
    """Values along one param axis, honouring a declared step."""
    lo, hi, step = spec["lo"], spec["hi"], spec.get("step")
    if not np.isfinite([lo, hi]).all() or hi < lo:
        raise ValueError("parameter axis requires finite lo <= hi")
    if step is not None:
        if not np.isfinite(step) or step <= 0:
            raise ValueError("parameter step must be finite and positive")
        vals = np.arange(lo, hi + step * 0.5, step, dtype=np.float64)
    else:
        count = steps or 25
        if count < 2:
            raise ValueError("parameter axis requires at least two steps")
        vals = np.linspace(lo, hi, count)
    return vals.astype(np.float32)


def build_grid(program: Program, steps_per_axis: int = 10) -> np.ndarray:
    """Full cartesian grid over all declared params."""
    if not program.params:
        return np.zeros((1, 0), dtype=np.float32)
    axes = [axis_values(p, steps_per_axis) for p in program.params]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1).astype(np.float32)


def lhs_sample(program: Program, n: int, seed: int = 0) -> np.ndarray:
    """Latin hypercube over the param box."""
    P = len(program.params)
    if n < 1:
        raise ValueError("sample count must be positive")
    if P == 0:
        return np.zeros((1, 0), dtype=np.float32)
    rng = np.random.default_rng(seed)
    u = (rng.permuted(np.tile(np.arange(n), (P, 1)), axis=1).T
         + rng.uniform(size=(n, P))) / n
    lo = np.array([p["lo"] for p in program.params])
    hi = np.array([p["hi"] for p in program.params])
    return (lo + u * (hi - lo)).astype(np.float32)


# ---------- walk-forward ----------

def folds(T: int, n_folds: int = 4, train_frac: float = 0.7):
    """Rolling contiguous (train, test) index-pair windows covering [0, T)."""
    if n_folds < 1:
        raise ValueError("n_folds >= 1")
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be between zero and one")
    seg = T // (n_folds + 1)          # anchored-ish rolling windows
    if seg < 2:
        raise ValueError("not enough bars for the requested folds")
    out = []
    for i in range(n_folds):
        start = i * seg
        end = start + 2 * seg if i < n_folds - 1 else T
        split = start + int((end - start) * train_frac)
        if split <= start or split >= end:
            raise ValueError("fold contains an empty train or validation slice")
        out.append(((start, split), (split, end)))
    return out


def fold_metrics(program: Program, bars, pm: np.ndarray, n_folds: int = 4,
                 train_frac: float = 0.7, config=None, prefer_gpu: bool = True):
    """Per-row train/test metric tensors: (n_folds, N, 7) each."""
    eng = pick_engine(prefer_gpu)
    T = len(bars.close)
    tr, te = [], []
    for (a0, a1), (b0, b1) in folds(T, n_folds, train_frac):
        tr.append(eng.run_batch(program, bars.slice(a0, a1), pm, config))
        te.append(eng.run_batch(program, bars.slice(b0, b1), pm, config))
    return np.stack(tr), np.stack(te)


def validation_metrics(program: Program, bars, pm: np.ndarray, n_folds: int = 4,
                       train_frac: float = 0.7, config=None,
                       prefer_gpu: bool = True) -> np.ndarray:
    """Metrics for fold validation slices only: (n_folds, N, 7).

    Use this when the training slices are not used for per-fold selection. It
    avoids running and discarding half of the backtests from ``fold_metrics``.
    """
    eng = pick_engine(prefer_gpu)
    return np.stack([
        eng.run_batch(program, bars.slice(b0, b1), pm, config)
        for _, (b0, b1) in folds(len(bars.close), n_folds, train_frac)
    ])


def robust_score(test_metrics: np.ndarray, metric: str = "sharpe") -> np.ndarray:
    """Mean minus half a std of the OOS metric across folds. (N,)"""
    m = test_metrics[:, :, COL[metric]]
    return m.mean(axis=0) - 0.5 * m.std(axis=0)


def walk_forward_select(program: Program, bars, pm: np.ndarray, n_folds: int = 4,
                        train_frac: float = 0.7, metric: str = "sharpe",
                        config=None, prefer_gpu: bool = True) -> dict:
    """Classic WFA: per fold, pick the best train-window row, score it OOS."""
    train, test = fold_metrics(program, bars, pm, n_folds, train_frac,
                               config, prefer_gpu)
    c = COL[metric]
    picks = train[:, :, c].argmax(axis=1)                  # (n_folds,)
    oos = np.array([test[f, picks[f], :] for f in range(len(picks))])
    return {
        "picks": pm[picks],
        "pick_rows": picks,
        "oos_metrics": oos,
        "oos_mean_sharpe": float(oos[:, COL["sharpe"]].mean()),
        "oos_mean_return": float(oos[:, COL["total_return"]].mean()),
    }


# ---------- deflated Sharpe (Bailey & Lopez de Prado, normal-returns form) ----------

def _norm_ppf(p: float) -> float:
    # Acklam's rational approximation; avoids a scipy dependency
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p <= 0:
        return -np.inf
    if p >= 1:
        return np.inf
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return ((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1))
    if p > phigh:
        return -_norm_ppf(1 - p)
    q = p - 0.5
    r = q * q
    return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1))


def _norm_cdf(x: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def deflated_sharpe(sr_obs: float, sr_trials: np.ndarray, n_bar_returns: int,
                    bars_per_year: float = 525600.0) -> float:
    """P[true SR > 0] after correcting for having tried len(sr_trials) configs.

    Inputs are annualized Sharpes; internally rescaled to per-bar units.
    Assumes ~normal per-bar returns (skew 0, kurtosis 3) - documented
    simplification of Bailey & Lopez de Prado (2014).
    """
    n = len(sr_trials)
    ann = np.sqrt(bars_per_year)
    sr = sr_obs / ann
    trials = np.asarray(sr_trials, dtype=np.float64) / ann
    if n < 2 or n_bar_returns < 2:
        return float("nan")
    var_trials = trials.var()
    if var_trials <= 0:
        return float("nan")
    gamma = 0.5772156649015329
    sr0 = np.sqrt(var_trials) * ((1 - gamma) * _norm_ppf(1 - 1.0 / n)
                                 + gamma * _norm_ppf(1 - 1.0 / (n * np.e)))
    denom = np.sqrt(1.0 + 0.5 * sr * sr)
    z = (sr - sr0) * np.sqrt(n_bar_returns - 1) / denom
    return _norm_cdf(float(z))
