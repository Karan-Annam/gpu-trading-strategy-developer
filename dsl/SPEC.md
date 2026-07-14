# Strategy DSL + VM Specification (v1)

This document is the contract between the compiler and all three engines
(Python reference, C++ CPU, CUDA). Any behavioral change bumps the version.

## 1. Language

Line-oriented, `#` comments. Four statement kinds:

```
param fast = 20 in [5, 100] step 5        # tunable, swept on GPU
let mid = sma(close, fast)                # per-bar named value
enter_long when crossover(close, mid)     # rules (aliases: buy/sell/short/cover)
set stop_loss = 0.02                      # risk config, may reference params
```

- **param** `name = default in [lo, hi] (step s)?` — numeric. Every param becomes
  one axis of a GPU sweep; each CUDA thread gets its own values.
- **let** `name = expr` — evaluated every bar, in source order. May reference
  series, params, and earlier lets only.
- **rules** — `enter_long | exit_long | enter_short | exit_short when expr`
  (aliases `buy | sell | short | cover`). Multiple statements for the same
  signal OR together. Omitted signals are never true.
- **set** `cfg = expr` where cfg ∈ `stop_loss, take_profit, trail_stop, size`.
  Evaluated every bar (so params/dynamic values are allowed). Fractions:
  `stop_loss = 0.02` is 2%. `size` is fraction of equity in [0, 1], default 1.
  Defaults: no stop, no take-profit, no trail (value ≤ 0 disables), size = 1.
  Fees and slippage are **backtest configuration**, not DSL — strategies cannot
  lower their own costs.

### Expressions

- Series (current bar): `open high low close volume`; lagged: `close[k]`,
  integer literal k ≥ 0, clamped to the first bar (`close[5]` at t=2 reads t=0).
- Context scalars: `bar_index`, `position` (−1/0/+1), `entry_price` (0 when
  flat), `equity`.
- Operators by precedence (low→high): `or`, `and`, `not`, comparisons
  (`> < >= <= == !=`, non-chaining), `+ -`, `* /`, unary `-`, call/index.
- Booleans are floats: comparisons yield 1.0/0.0, truthy means ≠ 0. `and`/`or`
  evaluate both sides (no short-circuit — keeps state updates unconditional).
- **Guarded domains**: `x / 0 → 0`, `sqrt(x<0) → 0`, and `log(x≤0) → 0`.
  Other arithmetic follows float32 behavior; validated market data is finite,
  but a strategy can still overflow with extreme expressions or parameters.

### Builtins

Stateless: `abs(x) min(a,b) max(a,b) sqrt(x) log(x)`.

Stateful (each *call site* owns a private state slot, updated exactly once per
bar in program order):

| call | semantics |
|---|---|
| `sma(x, n)` | mean of last min(t+1, n) values of x |
| `ema(x, n)` | α = 2/(n+1), seeded with first x |
| `rsi(x, n)` | Wilder. Seeds averages with first delta; flat → 50 |
| `atr(n)` | Wilder over true range; TR₀ = high−low |
| `highest(x, n)` / `lowest(x, n)` | extreme of last min(t+1, n) values |
| `stddev(x, n)` | population σ of last min(t+1, n) values |
| `delay(x, k)` | x from k bars ago (clamped to oldest seen) |
| `crossover(a, b)` | 1 if a>b and prevₐ ≤ prev_b (0 on bar 0) |
| `crossunder(a, b)` | mirror image |
| `change(x)` | sugar: `x − delay(x, 1)` |
| `roc(x, n)` | sugar: `x / delay(x, n) − 1` (safe divide) |

Window arguments (`n`, `k`) must be **statically boundable**: the compiler
evaluates them with interval arithmetic over param ranges and errors if the
upper bound is unknown or > 4096. At runtime n is truncated to int and clamped
to [1, bound]. Windowed state uses a ring buffer of `bound` floats per call
site; per-strategy state is capped at 8192 floats (compile error above that).

Windowed aggregates are computed by **scanning the ring buffer oldest→newest
in float32** each bar. Slower than incremental updates but drift-free and
bit-reproducible across engines; optimizing this is an engine concern only if
it preserves results.

## 2. Bytecode

Stack machine, float32 operands, max depth 32 (compiler-verified). One
instruction = one uint32: `op << 24 | arg`. `PUSH_SERIES_LAG` packs
`sid << 16 | lag`. Stateful ops carry a state-slot index; slot offsets/capacities
live in side tables (`state_kind[] state_off[] state_cap[]`).

Program = `consts f32[]`, `code u32[]`, param specs, `n_locals`, state tables,
`state_floats` (total per-thread state), `max_stack`.

Opcode list is generated from `dsl/opcodes.py` (single source of truth;
`python dsl/gen_opcodes.py` emits `engine/cpu/opcodes.h`).

Indicator stack contracts (args pushed left-to-right, so n is on top):
`sma/ema/rsi/highest/lowest/stddev: pop n, pop x` · `atr: pop n` ·
`delay: pop k, pop x` · `crossover/crossunder: pop b, pop a`.

Signal/config ops pop one value: `SIG_EL SIG_XL SIG_ES SIG_XS` (OR-accumulate
truthiness into the bar's signals), `SET_STOP SET_TP SET_TRAIL SET_SIZE`
(last write wins). `HALT` ends the bar program.

## 3. Broker semantics (per bar t, in this order)

1. **Fill pending orders at open[t]** (signals were raised at the end of bar
   t−1 — next-bar-open fills, no lookahead):
   - exits first: long exit fills at `open·(1−slip)`, short at `open·(1+slip)`.
   - then entry if flat: long fills at `open·(1+slip)`, short at `open·(1−slip)`.
   - qty = `equity · size / fill_price` (size sampled when the order was
     raised); fee = `qty · fill_price · fee_rate`, deducted from cash.
   - conflicting enter_long + enter_short while flat → both ignored.
   - exit + enter of the same side on the same bar → exit wins, no re-entry.
2. **Protective exits intra-bar** (only while a position is open, including one
   just filled this bar — checked in this order, first hit wins; pessimistic):
   - stop_loss: long triggers if `low ≤ entry·(1−sl)`, fills at
     `min(open, trigger)·(1−slip)`; short mirrored.
   - trail_stop: peak = highest high since entry (initialized to fill price,
     updated at end of bar, so today's high can't trigger today's trail);
     long triggers if `low ≤ peak·(1−trail)`, same fill rule.
   - take_profit: long triggers if `high ≥ entry·(1+tp)`, fills at
     `max(open, trigger)·(1−slip)`.
3. **Run the bar program** (indicators update, signals + configs for t+1).
4. **Mark to market**: equity = cash + qty·close[t]; record equity curve;
   update trail peak/trough.

Final bar: any open position is force-closed at `close·(1∓slip)` with fee, and
recorded as a trade.

Backtest config (host side): `fee_rate` default 0.001, `slip` default 0.0005,
`bars_per_year` default 525600 (1m crypto), initial equity 10000.

## 4. Metrics

From the float32 equity curve, streaming (Welford for mean/σ of simple per-bar
returns): `total_return`, `sharpe` = mean/σ · √bars_per_year (0 if σ = 0),
`max_drawdown` (peak-to-trough fraction), `n_trades`, `win_rate`
(trade net pnl > 0), `exposure` (fraction of bars with a position).

## 5. Equivalence

All engines use float32 with identical operation order. CPU engines must match
the Python reference exactly on trade sequences; equity curves within
rel 1e-5. CUDA is compiled with `-fmad=false` for equivalence runs (fmad
allowed for benchmark builds).
