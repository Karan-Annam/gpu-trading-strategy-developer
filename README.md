# GPU Strategy Lab

A backtesting engine for sweeping parameterized strategies across CPU and CUDA
implementations. Strategies are written in a small DSL, compiled to bytecode,
and executed by a Python reference engine or a shared C++/CUDA VM. A local web
app provides editing, charts, parameter heatmaps, and optional strategy
suggestions from Claude.

The simulator uses bar-based OHLCV data, next-bar-open signal fills, configurable
fees and slippage, long/short positions, position sizing, and stop-loss,
take-profit, and trailing-stop exits. It is an engineering project, not a live
trading system or a claim of strategy profitability.

## Measured performance

`bench/bench_engines.py` sweeps 10,000 SMA-cross parameter rows over 100,000
cached BTCUSDT one-minute bars on an RTX 4070 Laptop GPU. CUDA and OpenMP numbers
are medians of three trials after warmup; every OpenMP result row is compared
against CUDA. The Python and vectorbt rows are extrapolated and are included as
context, not result-equivalent comparisons.

| Engine | Time | Backtests/s | Measurement |
|---|---:|---:|---|
| CUDA | **1.56 s** | **6,414** | 10,000 rows, median of 3 |
| C++17 + OpenMP | 68.6 s | 146 | 10,000 rows, median of 3 |
| vectorbt | 579 s | 17.3 | 500 rows, throughput only |
| Python reference | ~5.7 days | 0.0203 | 2 rows |

Raw timings, trial values, versions, and device metadata are stored in
[`bench/results.json`](bench/results.json).

## Architecture

```text
strategy.dsl -> lexer/parser -> compiler -> bytecode
                                      |-> Python reference engine
OHLCV float32 SoA --------------------|-> C++17/OpenMP VM
                                      `-> CUDA VM (one parameter row/thread)

FastAPI + browser IDE -> run / sweep / walk-forward / evolve / inspect
```

The C++ and CUDA modules compile the same `engine/vm_core.h`. The CUDA launch
stages bytecode in shared memory per block, keeps opcode dispatch uniform for a
single strategy, and lays per-row VM state out across threads for coalesced
access. A compiler-selected raw-series path avoids per-thread ring buffers for
window functions applied directly to OHLCV columns. Parameter rows are sorted
by window spans before launch to reduce loop-length divergence, then restored to
their original order.

## Strategy DSL

```text
param fast = 20 in [5, 100] step 5
param slow = 100 in [20, 400] step 20
param stop = 0.02 in [0.005, 0.1]

let f = sma(close, fast)
let s = sma(close, slow)
enter_long when crossover(f, s)
exit_long when crossunder(f, s)
set stop_loss = stop
```

The language supports arithmetic and Boolean expressions, OHLCV series and
lags, streaming indicators, long/short signals, and dynamic risk expressions.
Window bounds, stack depth, names, and state allocation are checked at compile
time. Generated strategy text is compiled as DSL source; it is not evaluated as
Python or shell code. See [`dsl/SPEC.md`](dsl/SPEC.md) for exact semantics.

Compiled programs can also be exported as the dependency-free `OBP1` binary
format used by the companion order-book project:

```powershell
python -m dsl.export strategies/sma_cross.dsl strategy.obp
```

## Verification

The test suite covers parsing and compilation, broker and indicator behavior,
invalid market data, binary export, walk-forward statistics, the evolution
loop, exact Python/C++ trade sequences, and CPU/CUDA metric equivalence.

```powershell
python -m pytest tests -q
```

Current local result: **113 passed** on Python 3.13, including the compiled
C++ and CUDA extensions. CUDA is built with `--fmad=false` for the equivalence
configuration.

## Build and run

Requirements: Python 3.11+, CMake 3.24+, a C++17 compiler, and optionally the
CUDA toolkit.

```powershell
pip install -r requirements.txt
python data/fetch_binance.py --symbol BTCUSDT --start 2024-01 --end 2026-06
cmake -S . -B build -DPython3_EXECUTABLE=<python>
cmake --build build --config Release
python -m pytest tests -q
python -m uvicorn lab.server:app --port 8321
```

The web app includes a DSL editor, trade/equity charts, parameter-sweep
heatmaps, walk-forward scoring, and a genetic search. Setting
`ANTHROPIC_API_KEY` enables idea, refinement, and guided-evolution actions.
Evolution searches the first 80% of the selected bars and evaluates the final
selected program and parameters once on a held-out final 20%.

## Repository map

```text
data/       Binance fetcher and validated float32 OHLCV cache
dsl/        language, compiler, reference engine, and OBP1 export
engine/     shared VM, OpenMP module, CUDA kernel, and Python wrappers
evolve/     walk-forward utilities, genetic operators, and Claude client
lab/        FastAPI backend and browser UI
bench/      reproducible engine benchmark and saved results
tests/      unit, integration, and cross-engine tests
```

Technical notes:

- [`docs/01_ARCHITECTURE.md`](docs/01_ARCHITECTURE.md)
- [`docs/02_CUDA_AND_PERFORMANCE.md`](docs/02_CUDA_AND_PERFORMANCE.md)
- [`docs/03_VALIDATION_EVOLUTION_AI.md`](docs/03_VALIDATION_EVOLUTION_AI.md)
- [`docs/04_STATE_AND_ROADMAP.md`](docs/04_STATE_AND_ROADMAP.md)

## AI use and tooling

AI tools assisted with implementation, debugging, test generation, and
documentation. Their output was reviewed against the code and verified with the
test and benchmark commands above. Performance figures come from saved local
runs rather than generated estimates.

## License

MIT
