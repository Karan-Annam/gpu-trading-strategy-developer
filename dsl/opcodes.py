"""VM opcode table — single source of truth.

engine/cpu/opcodes.h is generated from this file (python dsl/gen_opcodes.py);
never edit the header by hand.
"""

OPCODES = [
    # pushes
    "PUSH_CONST",       # arg = const index
    "PUSH_PARAM",       # arg = param index
    "PUSH_SERIES",      # arg = series id (0 open, 1 high, 2 low, 3 close, 4 volume)
    "PUSH_SERIES_LAG",  # arg = sid << 16 | lag
    "PUSH_CTX",         # arg = 0 bar_index, 1 position, 2 entry_price, 3 equity, 4 bars_held
    "LOAD",             # arg = local index
    "STORE",
    # arithmetic / logic (pop 2 or 1, push 1)
    "ADD", "SUB", "MUL", "DIV", "NEG",
    "ABS", "MIN2", "MAX2", "SQRT", "LOG",
    "GT", "LT", "GE", "LE", "EQ", "NE",
    "AND", "OR", "NOT",
    # stateful indicators, arg = state slot index
    "SMA", "EMA", "RSI", "ATR", "HIGHEST", "LOWEST", "STDDEV",
    "DELAY", "CROSSOVER", "CROSSUNDER",
    # raw-series variants: window contents come straight from the bar series
    # (bit-identical to the ring version, zero per-thread storage). arg is a
    # state slot of kind SK_RAW whose aux packs sid << 16 | lag.
    "SMA_RAW", "HIGHEST_RAW", "LOWEST_RAW", "STDDEV_RAW", "DELAY_RAW",
    # signal / config sinks (pop 1)
    "SIG_EL", "SIG_XL", "SIG_ES", "SIG_XS",
    "SET_STOP", "SET_TP", "SET_TRAIL", "SET_SIZE",
    "HALT",
]

OP = {name: i for i, name in enumerate(OPCODES)}

SERIES = {"open": 0, "high": 1, "low": 2, "close": 3, "volume": 4}
CTX = {"bar_index": 0, "position": 1, "entry_price": 2, "equity": 3, "bars_held": 4}

# state slot kinds (state_kind[] values)
SK_EMA = 0        # [count, value]
SK_WILDER1 = 1    # atr: [count, value]
SK_WILDER2 = 2    # rsi: [count, prev_x, avg_gain, avg_loss]
SK_RING = 3       # [count, buf...] (capacity = state_cap)
SK_PREV2 = 4      # crossover: [count, prev_a, prev_b]
SK_RAW = 5        # no storage; cap + aux(sid<<16|lag) describe a series window

STATE_HEADER = {SK_EMA: 2, SK_WILDER1: 2, SK_WILDER2: 4, SK_RING: 1,
                SK_PREV2: 3, SK_RAW: 0}

MAX_STACK = 32
MAX_WINDOW = 4096
MAX_STATE_FLOATS = 8192


def encode(op: str, arg: int = 0) -> int:
    assert 0 <= arg < (1 << 24), (op, arg)
    return (OP[op] << 24) | arg
