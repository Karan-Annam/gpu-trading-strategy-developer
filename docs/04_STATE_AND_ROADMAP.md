# State and roadmap

## Verified state

- DSL lexer, parser, compiler, bytecode VM, and OBP1 export are implemented.
- Python, C++17/OpenMP, and CUDA engines are implemented.
- CPU/CUDA equivalence is tested, including large parameter batches.
- OHLCV input validation rejects malformed array/value data.
- The Lab can run, sweep, chart, and evolve strategies locally.
- The evolution loop separates search/validation bars from a final 20%
  holdout.
- The benchmark records trial timings, environment metadata, and a full
  CPU/CUDA result cross-check.
- GitHub Actions builds the CPU extension and runs the CPU-capable test suite.

The optional Claude flows require credentials and are not part of automated
tests. vectorbt measurements use different fill semantics and are not
correctness comparisons.

## Engineering priorities

1. Cache bars and program tables on the GPU across repeated sweep/evolution
   calls; overlap independent folds with CUDA streams.
2. Use Nsight Compute to record occupancy, register pressure, and memory
   throughput before further kernel tuning.
3. Replace scan-based window functions with deterministic running/block sums
   and monotonic queues, mirrored across all three engines.
4. Add property-based random-program differential tests and sanitizers for the
   native CPU module.
5. Package the CPU extension with a portable wheel/build backend.

## Simulation and statistics priorities

1. Add funding/borrow costs, spread-aware slippage, and explicit intra-bar path
   policies.
2. Add purged/embargoed validation and measured higher moments for deflated
   Sharpe.
3. Report confidence intervals and parameter-neighborhood stability rather
   than only the best grid cell.
4. Support multi-symbol portfolios with shared equity and exposure limits.
5. Add diversity preservation or multi-objective selection to the evolution
   loop.
