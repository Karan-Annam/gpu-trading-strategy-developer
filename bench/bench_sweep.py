"""Benchmark: parameter sweep throughput, CPU (OpenMP) vs CUDA.

    python bench/bench_sweep.py [--bars 100000] [--combos 10000]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ohlcv import load_bars                    # noqa: E402
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


def make_grid(prog, n):
    rng = np.random.default_rng(7)
    cols = [rng.uniform(p["lo"], p["hi"], n).astype(np.float32) for p in prog.params]
    return np.stack(cols, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, default=100_000)
    ap.add_argument("--combos", type=int, default=10_000)
    ap.add_argument("--cpu-combos", type=int, default=500,
                    help="CPU sample size (extrapolated)")
    args = ap.parse_args()

    bars = load_bars("BTCUSDT").slice(-args.bars, None)
    prog = compile_source(SRC)
    grid = make_grid(prog, args.combos)
    print(f"{args.bars:,} bars x {args.combos:,} param combos "
          f"({args.bars * args.combos / 1e9:.1f}B bar-evaluations)")

    # GPU (includes upload + launch + download)
    t0 = time.perf_counter()
    g = gpu_engine.run_batch(prog, bars, grid)
    t_gpu = time.perf_counter() - t0
    print(f"GPU : {t_gpu:8.2f}s  {args.combos / t_gpu:10,.0f} backtests/s "
          f"{args.bars * args.combos / t_gpu / 1e9:6.2f}B bar-evals/s")

    # CPU (sample then extrapolate)
    n_cpu = min(args.cpu_combos, args.combos)
    t0 = time.perf_counter()
    c = cpu_engine.run_batch(prog, bars, grid[:n_cpu])
    t_cpu = time.perf_counter() - t0
    est = t_cpu * args.combos / n_cpu
    print(f"CPU : {est:8.2f}s (est from {n_cpu})  {n_cpu / t_cpu:10,.0f} backtests/s")
    print(f"speedup: {est / t_gpu:,.0f}x")

    # sanity: same results on the sampled rows
    np.testing.assert_allclose(c[:, 0], g[:n_cpu, 0], rtol=1e-4)
    print("cross-check vs CPU rows: OK")


if __name__ == "__main__":
    main()
