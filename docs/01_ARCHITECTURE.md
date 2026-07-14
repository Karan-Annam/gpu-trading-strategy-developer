# Architecture

## Data boundary

`data/ohlcv.py` stores timestamps as `int64` and OHLCV columns as contiguous
`float32` arrays. Loading validates dimensions, lengths, finite values,
strictly increasing timestamps, positive prices, bar envelopes, and
non-negative volume. Engine wrappers additionally validate array layout and
parameter-matrix width before entering native code.

The engines consume structure-of-arrays data directly:

```text
open[T] high[T] low[T] close[T] volume[T]
```

## Compiler

The DSL pipeline is:

```text
source -> lexer -> parser -> typed AST -> state allocation -> bytecode
```

The compiler rejects unresolved names, invalid parameter ranges, excessive
stack/state use, and unbounded or oversized indicator windows. Each stateful
call site receives private state. Instructions use an 8-bit opcode and 24-bit
argument; Python generates the matching C++ header so opcode values stay
synchronized.

`dsl/export.py` serializes compiled programs to the versioned little-endian
`OBP1` format. The parser rejects truncated data, trailing bytes, invalid UTF-8,
unknown versions, and resource-limit violations.

## Execution engines

The Python reference engine is intentionally direct and is used to specify
broker behavior and trade sequences. The native implementations are:

- `engine/cpu/module.cpp`: a single-run recorder and an OpenMP batch loop.
- `engine/cuda/cuda_module.cu`: one full parameterized backtest per CUDA
  thread.

Both native engines call the same templated VM in `engine/vm_core.h`. The GPU
uses a metrics-only sink; detailed trade/equity recording stays on CPU.

Batch output columns are:

```text
final_equity, total_return, sharpe, max_drawdown,
n_trades, wins, exposure_bars
```

## Bar and fill order

For each bar the simulator applies pending signal orders at the open, evaluates
protective exits using the bar range, marks equity, evaluates the strategy, and
schedules new signal orders for the next bar. If stop and take-profit levels are
both crossed within one bar, the conservative stop result wins. Fees and
slippage are engine configuration and cannot be changed from DSL source.

These semantics model bar data; they do not model spread dynamics, queue
position, partial fills, exchange latency, funding, or an intra-bar tick path.

## Numerical equivalence

The DSL uses float32 operations. CUDA equivalence builds disable fused
multiply-add contraction. Tests compare Python and C++ trade records, then
compare CPU and CUDA metric matrices with tight float tolerances and exact
integer-valued counters. The benchmark also cross-checks all 10,000 measured
CPU rows against the corresponding CUDA rows.
