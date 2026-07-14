"""Reproducible benchmark for the README numbers.

    python bench/bench_engines.py [--bars 100000]

Sweeps a 3-param SMA-cross grid on real BTC 1m data across:
  - btcuda  (GPU, one thread per param combo)
  - btcpu   (C++ OpenMP batch)
  - refengine (pure-python golden model, small sample, extrapolated)
  - vectorbt (if installed; NOTE: different fill semantics - throughput
    comparison only, not result-equivalent)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ohlcv import load_bars                    # noqa: E402
from dsl import refengine                           # noqa: E402
from dsl.compiler import compile_source             # noqa: E402
from engine import cpu_engine, gpu_engine           # noqa: E402

SRC = """\
param fast = 20 in [4, 200]
param slow = 100 in [20, 800]
param sl = 0.02 in [0.005, 0.1]
let f = sma(close, fast)
let s = sma(close, slow)
enter_long when crossover(f, s)
exit_long when crossunder(f, s)
set stop_loss = sl
"""


def grid(prog, n, seed=7):
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(p["lo"], p["hi"], n).astype(np.float32)
                     for p in prog.params], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, default=100_000)
    ap.add_argument("--combos", type=int, default=10_000)
    ap.add_argument("--cpu-combos", type=int, default=0,
                    help="rows to measure on CPU; 0 measures the full sweep")
    ap.add_argument("--trials", type=int, default=3,
                    help="CUDA and OpenMP trials; medians are reported")
    args = ap.parse_args()
    if args.bars < 1 or args.combos < 1 or args.trials < 1:
        ap.error("bars, combos, and trials must be positive")

    bars = load_bars("BTCUSDT").slice(-args.bars, None)
    prog = compile_source(SRC)
    pm = grid(prog, args.combos)
    results = {}

    # GPU (includes H2D/D2H transfers); warm up CUDA context first
    gpu_engine.run_batch(prog, bars.slice(0, 2000), pm[:256])
    gpu_times = []
    gpu_result = None
    for _ in range(args.trials):
        t0 = time.perf_counter()
        gpu_result = gpu_engine.run_batch(prog, bars, pm)
        gpu_times.append(time.perf_counter() - t0)
    dt = float(np.median(gpu_times))
    results["cuda_gpu"] = {"combos": args.combos, "seconds": dt,
                           "backtests_per_s": args.combos / dt,
                           "trial_seconds": gpu_times,
                           "statistic": "median"}

    # CPU C++ OpenMP. The default measures the complete sweep.
    n = args.combos if args.cpu_combos <= 0 else min(args.cpu_combos, args.combos)
    cpu_times = []
    cpu_result = None
    for _ in range(args.trials):
        t0 = time.perf_counter()
        cpu_result = cpu_engine.run_batch(prog, bars, pm[:n])
        cpu_times.append(time.perf_counter() - t0)
    measured_dt = float(np.median(cpu_times))
    dt = measured_dt * args.combos / n
    results["cpp_openmp"] = {"combos": args.combos, "seconds": dt,
                             "backtests_per_s": args.combos / dt,
                             "measured_combos": n,
                             "trial_seconds": cpu_times,
                             "statistic": "median"}
    if n != args.combos:
        results["cpp_openmp"]["note"] = f"extrapolated from {n}"
    np.testing.assert_allclose(cpu_result, gpu_result[:n], rtol=1e-4, atol=1e-5)
    results["cuda_cpu_crosscheck_rows"] = n

    # Python reference (sample 2, extrapolate)
    t0 = time.perf_counter()
    for row in pm[:2]:
        refengine.run(prog, bars, [float(v) for v in row])
    dt = (time.perf_counter() - t0) * args.combos / 2
    results["python_ref"] = {"combos": args.combos, "seconds": dt,
                             "backtests_per_s": args.combos / dt,
                             "note": "extrapolated from 2"}

    # vectorbt (optional; throughput-only comparison, different semantics)
    try:
        import vectorbt as vbt  # noqa: PLC0415
        import pandas as pd     # noqa: PLC0415
        close = pd.Series(bars.close.astype(np.float64))
        n_vbt = min(500, args.combos)
        pairs = [(int(r[0]), int(r[1])) for r in pm[:n_vbt]]

        def vbt_run(k):
            fast_ma = vbt.MA.run(close, window=[p[0] for p in pairs[:k]])
            slow_ma = vbt.MA.run(close, window=[p[1] for p in pairs[:k]])
            entries = fast_ma.ma_crossed_above(slow_ma)
            exits = fast_ma.ma_crossed_below(slow_ma)
            pf = vbt.Portfolio.from_signals(
                close, entries, exits, fees=0.001,
                sl_stop=pm[:k, 2].astype(np.float64))
            return pf.total_return()

        vbt_run(8)   # numba JIT warmup, excluded from timing
        t0 = time.perf_counter()
        vbt_run(n_vbt)
        dt = (time.perf_counter() - t0) * args.combos / n_vbt
        results["vectorbt"] = {"combos": args.combos, "seconds": dt,
                               "backtests_per_s": args.combos / dt,
                               "note": f"extrapolated from {n_vbt}; different "
                                       f"fill semantics - throughput only"}
    except ImportError:
        results["vectorbt"] = {"note": "not installed"}
    except Exception as e:  # vectorbt API drift shouldn't kill the bench
        results["vectorbt"] = {"note": f"failed: {e}"}

    out = {
        "bars": args.bars,
        "combos": args.combos,
        "device": gpu_engine.device_info()["name"] if gpu_engine.available() else None,
        "host": {"system": platform.system(), "processor": platform.processor()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "results": results,
    }
    print(json.dumps(out, indent=2))
    path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
