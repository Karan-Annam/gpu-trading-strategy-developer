# CUDA and performance

## Launch design

One CUDA thread executes one complete backtest for one parameter row. A launch
contains one compiled strategy, so threads encounter the same bytecode opcode at
the same time. This removes interpreter-dispatch divergence, although branches
caused by different positions, signals, and indicator window lengths can still
diverge.

Each block cooperatively copies bytecode into shared memory. OHLCV arrays,
constants, parameter rows, VM state, and metrics remain in global memory.

## State layout

Logical per-thread state is stored column-major across the launch:

```text
state[element * row_count + thread]
```

Threads in a warp therefore access adjacent addresses for the same VM state
element. Parameter rows are sorted by their widest parameter spans before
launch so similar window lengths tend to share warps; results are unsorted on
return.

For `sma`, `highest`, `lowest`, and `stddev` applied directly to a raw OHLCV
series, the compiler emits a raw-series opcode. The VM scans the source series
instead of copying the same window into every thread's ring buffer. Indicators
over computed expressions still allocate per-row state.

Large sweeps are split when estimated VM state would exceed 1.5 GB. Current
calls allocate/copy device buffers per batch; persistent bar/program buffers are
a useful future optimization for repeated evolution folds.

## Benchmark method

Run:

```powershell
python bench/bench_engines.py
```

The default workload is 100,000 BTCUSDT one-minute bars by 10,000 parameter
rows. CUDA is warmed before timing. CUDA and OpenMP each run three trials and
report the median. OpenMP processes the full grid, and its complete output is
compared against CUDA with `numpy.testing.assert_allclose`.

vectorbt is warmed on a small input, then measured on 500 rows and extrapolated.
Its fill behavior differs, so the row is a throughput comparison only. The
Python reference is measured on two rows and extrapolated because its role is
correctness, not batch performance.

Current results and trial-level data are in [`../bench/results.json`](../bench/results.json).

## Memory comparison

A dense `100,000 x 10,000` float32 matrix occupies about 4 GB per matrix.
Vectorized workflows may need several such signal, position, and result
matrices. This engine instead stores the five OHLCV columns plus per-row VM
state and metrics; the exact footprint depends on the compiled strategy.

## Build configuration

The local RTX 4070 build targets CUDA architecture 89. CMake leaves CUDA
optional so the reference and OpenMP paths remain usable on CPU-only systems.
`BT_CUDA_FMAD=ON` can enable contraction for performance experiments, but that
build is outside the default cross-engine equivalence configuration.
