# GPU Strategy Lab

A CUDA-powered backtesting engine with GPU parameter sweeps and AI strategy
generation — strategies are written in a small DSL, compiled to bytecode, and
interpreted inside a CUDA kernel where **one GPU thread runs one complete
backtest**. A genetic loop (optionally steered by Claude) evolves strategy
programs, using the GPU sweep as its fitness function, and a local web IDE
("Strategy Lab") lets you write, run, sweep, and refine strategies
interactively.

```
 plain English ──► Claude ──┐
 hand-written DSL ──────────┼──► compiler ──► bytecode ──► CUDA kernel
 genetic evolution ─────────┘      (safety gate)            1 thread = 1 backtest
                                                            ▲            │
                                                            └── fitness ──┘
```

## Why this exists

- **vectorbt** and friends are fast but CPU-bound and memory-bound (they
  materialize `T×N` signal arrays). This engine *streams*: each thread holds
  O(1) state and reads each bar once, so a million-combo sweep needs memory
  proportional to `T + N`, not `T×N`.
- **LLM strategy-evolution research** (AlgoEvolve, AlphaAgent, …) is
  bottlenecked on slow backtest evaluation. Here the evaluator is a GPU kernel
  that runs thousands of full backtests per second, which makes
  evolutionary/LLM search loops actually cheap.
- Nobody ships the closed loop — *AI proposes strategy programs → GPU
  evaluates thousands of parameter variants → fitness feeds back* — as an
  open, end-to-end system.

## Benchmark

SMA-cross with 3 tunable params (fast/slow window + stop-loss) on real
BTCUSDT 1-minute bars, RTX 4070 Laptop 8GB. `bench/bench_engines.py`
reproduces every number (JIT/context warmups excluded):

| engine | 100k bars × 10k combos | backtests/s | vs CUDA |
|---|---:|---:|---:|
| CUDA (btcuda) | **1.9 s** | **5,300** | — |
| C++ OpenMP (btcpu) | 50 s | 198 | 27× slower |
| vectorbt 1.1 (numba, warmed) | 470 s | 21 | 250× slower¹ |
| Python reference | ~4.7 days | 0.025 | — |

¹ throughput-only comparison — vectorbt's fill semantics differ, and
per-combo windows + stop-losses are not its vectorization sweet spot.

Full history sweep: 2.5 years of 1m bars (1.3M) × 2,000 combos in **16 s** —
2.6 billion bar-evaluations with full broker semantics (next-bar-open fills,
stop/trail/take-profit, fees, slippage) and streaming indicators.

Three implementation details do the heavy lifting:

1. **No warp divergence by construction** — one *strategy* per block, one
   *parameter combo* per thread. Every thread in a warp interprets the same
   bytecode in lockstep, only the parameter values differ. Batches are
   lexsorted so window sizes cluster within warps.
2. **Raw-series fast path** — `sma(close, n)` needs no per-thread ring buffer:
   the window contents *are* the close series, which lives in L2 and is shared
   by every thread. The compiler detects this case and emits zero-state
   opcodes (bit-identical results to the ring-buffer path).
3. **Column-major VM state** — what per-thread state remains (EMA/RSI/ATR
   accumulators, rings for computed expressions) is laid out strided across
   the launch so state access coalesces across warps.

## The golden-model chain

Three engines share one specification ([dsl/SPEC.md](dsl/SPEC.md)), and two of
them share the literal implementation (`engine/vm_core.h` compiles under both
MSVC and nvcc):

```
dsl/refengine.py     slow, obviously-correct python  (the specification)
       ▲ trades must match EXACTLY
engine/cpu  (btcpu)  C++17 + OpenMP via pybind11
       ▲ metrics must match (fmad=false)
engine/cuda (btcuda) same vm_core.h inside a CUDA kernel
```

87 pytest tests enforce the chain — exact trade-sequence equality between
Python and C++, and exact trade/win/exposure counts between C++ and CUDA.

## The DSL

```
param fast = 20 in [5, 100] step 5
param slow = 100 in [20, 400] step 20
param sl = 0.02 in [0.005, 0.1]

let f = sma(close, fast)
let s = sma(close, slow)

enter_long when crossover(f, s)
exit_long  when crossunder(f, s)

set stop_loss = sl
```

Streaming indicators (`sma ema rsi atr highest lowest stddev delay crossover
crossunder change roc`), longs and shorts, dynamic risk config (`stop_loss /
take_profit / trail_stop / size` can be expressions), and tunable `param`s
that become GPU sweep axes. Lookahead is impossible *by construction* — the
language can only express `series[t - k]` access, which also makes it safe to
compile whatever an LLM emits: **generated code is compiled, never executed**.
Strategies cannot lower their own fees or slippage; those are backtester
configuration.

## Strategy Lab (web IDE)

```
python -m uvicorn lab.server:app --port 8321     # then open localhost:8321
```

- CodeMirror editor with DSL highlighting and inline compile errors
- Run on cached Binance data → price chart with trade markers, indicator
  overlays, equity curve, metrics, trade blotter
- **GPU sweeps** from the UI: pick 1–2 params → heatmap (diverging palette
  around 0) → click a cell to apply those params; optional walk-forward mode
  scores each cell out-of-sample, and the best cell gets a
  **deflated Sharpe ratio** (Bailey & López de Prado) that discounts for the
  number of combos tried
- **Evolution**: launch a genetic run, watch generations tick, click any
  evolved strategy to open its source in the editor
- **Claude integration** (set `ANTHROPIC_API_KEY`):
  - *idea box* — type a strategy idea in English, get compilable DSL
  - *improve* — conversational refinement with an accept/reject diff view
    and version history
  - *guided evolution* — Claude injects novel strategies into the population
    every few generations, optionally steered by a hint

## Evolution loop

```
python -m evolve.loop --bars 200000 --pop 32 --gens 15
```

Genomes are param-free ASTs; a deterministic pass re-promotes the most
sweep-worthy literals to `param`s, so the GPU explores a *region* around each
genome (48 Latin-hypercube samples × 3 walk-forward folds), not a single
point. Fitness = 75th-percentile out-of-sample robust score − complexity
penalty − a dead-strategy penalty (no-trade strategies would otherwise win by
paying no fees). Mutation/crossover operate on typed condition trees; the
compiler is the validity gate, exactly as for LLM output.

## Repo layout

```
data/     Binance archive fetcher + float32 SoA cache (BTC/ETH 1m included via script)
dsl/      SPEC.md, lexer/parser/compiler, python reference engine, opcode table
engine/   vm_core.h (shared VM), cpu/ pybind11 module, cuda/ kernel, wrappers
evolve/   genome ops, evolution loop, walk-forward/sweep utils, Claude client
lab/      FastAPI backend + static frontend (vendored CodeMirror + uPlot)
bench/    reproducible benchmark scripts
tests/    87 tests: compiler, broker math, indicators, cross-engine equivalence
```

## Build

Requirements: Python 3.11+, CMake 3.24+, MSVC (VS 2022) or equivalent,
CUDA Toolkit 12/13 (optional — everything but `btcuda` works without it).

```
pip install -r requirements.txt
python data/fetch_binance.py --symbol BTCUSDT --start 2024-01 --end 2026-06
python data/fetch_binance.py --symbol ETHUSDT --start 2024-01 --end 2026-06
cmake -S . -B build -DPython3_EXECUTABLE=$(python -c "import sys;print(sys.executable)")
cmake --build build --config Release
python -m pytest tests/ -q
```

The CUDA module builds with `--fmad=false` by default so results are
bit-comparable with the CPU engines; configure with `-DBT_CUDA_FMAD=ON` for a
slightly faster benchmark build.

## Honest limitations

- **Bar-based OHLCV only.** No order-book microstructure, queue position, or
  intra-bar path modeling beyond conservative stop/TP ordering (both hit in
  one bar → the stop wins).
- **Backtest ≠ live.** Fills at next-bar open with fixed fee + slippage are a
  model; real execution differs.
- **Most strategies lose.** On 1-minute crypto data with 10 bps taker fees,
  nearly every classic indicator strategy is net-negative — the engine's job
  is to report that honestly, and the walk-forward + deflated-Sharpe layer
  exists precisely because a big enough sweep always finds something that
  *looks* good in-sample.
- vectorbt comparisons are throughput-only; its fill semantics differ.
