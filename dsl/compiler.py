"""AST -> bytecode Program.

Responsibilities:
- name resolution (params / lets / series / context scalars)
- desugaring (change, roc)
- static bounding of window arguments via interval arithmetic over param ranges
- state-slot allocation for stateful call sites
- stack-depth verification
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from dsl import ast_nodes as A
from dsl.errors import CompileError
from dsl.opcodes import (
    CTX, MAX_STACK, MAX_STATE_FLOATS, MAX_WINDOW, OP, SERIES,
    SK_EMA, SK_PREV2, SK_RAW, SK_RING, SK_WILDER1, SK_WILDER2, STATE_HEADER, encode,
)
from dsl.parser import parse

STATELESS = {"abs": 1, "sqrt": 1, "log": 1, "min": 2, "max": 2}
STATELESS_OP = {"abs": "ABS", "sqrt": "SQRT", "log": "LOG", "min": "MIN2", "max": "MAX2"}
BIN_OP = {
    "+": "ADD", "-": "SUB", "*": "MUL", "/": "DIV",
    ">": "GT", "<": "LT", ">=": "GE", "<=": "LE", "==": "EQ", "!=": "NE",
    "and": "AND", "or": "OR",
}
SIG_OP = {"enter_long": "SIG_EL", "exit_long": "SIG_XL",
          "enter_short": "SIG_ES", "exit_short": "SIG_XS"}
CFG_OP = {"stop_loss": "SET_STOP", "take_profit": "SET_TP",
          "trail_stop": "SET_TRAIL", "size": "SET_SIZE"}

# stack effect per opcode name (delta after execution)
STACK_EFFECT = {
    "PUSH_CONST": 1, "PUSH_PARAM": 1, "PUSH_SERIES": 1, "PUSH_SERIES_LAG": 1,
    "PUSH_CTX": 1, "LOAD": 1, "STORE": -1,
    "ADD": -1, "SUB": -1, "MUL": -1, "DIV": -1, "NEG": 0,
    "ABS": 0, "MIN2": -1, "MAX2": -1, "SQRT": 0, "LOG": 0,
    "GT": -1, "LT": -1, "GE": -1, "LE": -1, "EQ": -1, "NE": -1,
    "AND": -1, "OR": -1, "NOT": 0,
    "SMA": -1, "EMA": -1, "RSI": -1, "ATR": 0, "HIGHEST": -1, "LOWEST": -1,
    "STDDEV": -1, "DELAY": -1, "CROSSOVER": -1, "CROSSUNDER": -1,
    "SMA_RAW": 0, "HIGHEST_RAW": 0, "LOWEST_RAW": 0, "STDDEV_RAW": 0, "DELAY_RAW": 0,
    "SIG_EL": -1, "SIG_XL": -1, "SIG_ES": -1, "SIG_XS": -1,
    "SET_STOP": -1, "SET_TP": -1, "SET_TRAIL": -1, "SET_SIZE": -1,
    "HALT": 0,
}


@dataclass
class Program:
    source: str
    consts: list[float]
    code: list[int]
    params: list[dict]           # {name, default, lo, hi, step}
    n_locals: int
    state_kind: list[int]
    state_off: list[int]
    state_cap: list[int]         # ring capacity (0 for non-ring slots)
    state_aux: list[int]         # SK_RAW: sid << 16 | lag
    state_floats: int
    max_stack: int
    local_names: list[str] = field(default_factory=list)

    def param_defaults(self) -> list[float]:
        return [p["default"] for p in self.params]

    def arrays(self) -> dict:
        """Engine-facing packed form (contiguous numpy arrays)."""
        return {
            "code": np.asarray(self.code, dtype=np.uint32),
            "consts": np.asarray(self.consts, dtype=np.float32),
            "state_kind": np.asarray(self.state_kind, dtype=np.int32),
            "state_off": np.asarray(self.state_off, dtype=np.int32),
            "state_cap": np.asarray(self.state_cap, dtype=np.int32),
            "state_aux": np.asarray(self.state_aux, dtype=np.int32),
            "n_locals": self.n_locals,
            "state_floats": self.state_floats,
        }


# ---- interval arithmetic for static window bounds ----

class Unbounded(Exception):
    pass


def _bounds(node: A.Node, params: dict[str, tuple[float, float]],
            lets: dict[str, tuple[float, float] | None]) -> tuple[float, float]:
    if isinstance(node, A.Num):
        return node.value, node.value
    if isinstance(node, A.Name):
        if node.id in params:
            return params[node.id]
        if node.id in lets:
            b = lets[node.id]
            if b is None:
                raise Unbounded()
            return b
        raise Unbounded()
    if isinstance(node, A.UnaryOp) and node.op == "-":
        lo, hi = _bounds(node.operand, params, lets)
        return -hi, -lo
    if isinstance(node, A.BinOp) and node.op in ("+", "-", "*"):
        a, b = _bounds(node.left, params, lets)
        c, d = _bounds(node.right, params, lets)
        if node.op == "+":
            return a + c, b + d
        if node.op == "-":
            return a - d, b - c
        prods = (a * c, a * d, b * c, b * d)
        return min(prods), max(prods)
    if isinstance(node, A.Call) and node.func in ("min", "max") and len(node.args) == 2:
        a, b = _bounds(node.args[0], params, lets)
        c, d = _bounds(node.args[1], params, lets)
        if node.func == "min":
            return min(a, c), min(b, d)
        return max(a, c), max(b, d)
    if isinstance(node, A.Call) and node.func == "abs" and len(node.args) == 1:
        lo, hi = _bounds(node.args[0], params, lets)
        if lo >= 0:
            return lo, hi
        if hi <= 0:
            return -hi, -lo
        return 0.0, max(-lo, hi)
    raise Unbounded()


class Compiler:
    def __init__(self, strat: A.Strategy, source: str):
        self.strat = strat
        self.source = source
        self.consts: list[float] = []
        self.const_ix: dict[float, int] = {}
        self.code: list[int] = []
        self.params = {p.name: i for i, p in enumerate(strat.params)}
        self.param_bounds = {p.name: (p.lo, p.hi) for p in strat.params}
        self.locals: dict[str, int] = {}
        self.let_bounds: dict[str, tuple[float, float] | None] = {}
        self.local_names: list[str] = []
        self.state_kind: list[int] = []
        self.state_off: list[int] = []
        self.state_cap: list[int] = []
        self.state_aux: list[int] = []
        self.state_floats = 0

    # ---- helpers ----

    def emit(self, op: str, arg: int = 0) -> None:
        self.code.append(encode(op, arg))

    def const(self, v: float) -> int:
        v = float(np.float32(v))
        if v not in self.const_ix:
            self.const_ix[v] = len(self.consts)
            self.consts.append(v)
        return self.const_ix[v]

    def alloc_state(self, kind: int, cap: int = 0, aux: int = 0) -> int:
        size = STATE_HEADER[kind] + (cap if kind == SK_RING else 0)
        idx = len(self.state_kind)
        self.state_kind.append(kind)
        self.state_off.append(self.state_floats)
        self.state_cap.append(cap)
        self.state_aux.append(aux)
        self.state_floats += size
        return idx

    def raw_series_of(self, node: A.Node) -> int | None:
        """sid << 16 | lag when the expression is a plain (possibly lagged)
        series reference; such windowed calls need no ring buffer."""
        if isinstance(node, A.Name) and node.id in SERIES:
            return SERIES[node.id] << 16
        if isinstance(node, A.SeriesLag) and node.id in SERIES and 0 <= node.lag <= 0xFFFF:
            return (SERIES[node.id] << 16) | node.lag
        return None

    def window_cap(self, node: A.Node, what: str, extra: int = 0) -> int:
        try:
            _, hi = _bounds(node, self.param_bounds, self.let_bounds)
        except Unbounded:
            raise CompileError(
                f"{what} must be statically boundable (a literal, param, or "
                f"simple arithmetic over them)", node.line, node.col) from None
        cap = int(math.ceil(hi)) + extra
        if cap < 1:
            cap = 1
        if cap > MAX_WINDOW:
            raise CompileError(
                f"{what} upper bound {cap} exceeds the {MAX_WINDOW} limit",
                node.line, node.col)
        return cap

    # ---- expression emission ----

    def expr(self, node: A.Node) -> None:
        if isinstance(node, A.Num):
            self.emit("PUSH_CONST", self.const(node.value))
        elif isinstance(node, A.Name):
            self.name(node)
        elif isinstance(node, A.SeriesLag):
            if node.id not in SERIES:
                raise CompileError(f"only series can be lagged, not {node.id!r}",
                                   node.line, node.col)
            if not (0 <= node.lag <= 0xFFFF):
                raise CompileError("lag out of range", node.line, node.col)
            self.emit("PUSH_SERIES_LAG", (SERIES[node.id] << 16) | node.lag)
        elif isinstance(node, A.UnaryOp):
            self.expr(node.operand)
            self.emit("NEG" if node.op == "-" else "NOT")
        elif isinstance(node, A.BinOp):
            self.expr(node.left)
            self.expr(node.right)
            self.emit(BIN_OP[node.op])
        elif isinstance(node, A.Call):
            self.call(node)
        else:
            raise CompileError(f"cannot compile node {type(node).__name__}",
                               node.line, node.col)

    def name(self, node: A.Name) -> None:
        nid = node.id
        if nid in self.params:
            self.emit("PUSH_PARAM", self.params[nid])
        elif nid in self.locals:
            self.emit("LOAD", self.locals[nid])
        elif nid in SERIES:
            self.emit("PUSH_SERIES", SERIES[nid])
        elif nid in CTX:
            self.emit("PUSH_CTX", CTX[nid])
        else:
            raise CompileError(f"unknown name {nid!r}", node.line, node.col)

    def argc(self, node: A.Call, n: int) -> None:
        if len(node.args) != n:
            raise CompileError(
                f"{node.func}() takes {n} argument{'s' if n != 1 else ''}, "
                f"got {len(node.args)}", node.line, node.col)

    def call(self, node: A.Call) -> None:
        f = node.func
        if f in STATELESS:
            self.argc(node, STATELESS[f])
            for a in node.args:
                self.expr(a)
            self.emit(STATELESS_OP[f])
        elif f == "change":          # x - delay(x, 1)
            self.argc(node, 1)
            x = node.args[0]
            self.call_delay(x, A.Num(1.0, line=node.line, col=node.col))
            # stack: [delayed]; need x - delayed -> emit x, swapless: compute as -(delayed - x)
            self.expr(x)
            self.emit("SUB")         # delayed - x
            self.emit("NEG")         # x - delayed
        elif f == "roc":             # x / delay(x, n) - 1
            self.argc(node, 2)
            x, n = node.args
            self.expr(x)
            self.call_delay(x, n)
            self.emit("DIV")
            self.emit("PUSH_CONST", self.const(1.0))
            self.emit("SUB")
        elif f == "delay":
            self.argc(node, 2)
            self.call_delay(node.args[0], node.args[1])
        elif f in ("sma", "highest", "lowest", "stddev"):
            self.argc(node, 2)
            x, n = node.args
            cap = self.window_cap(n, f"{f}() window")
            raw = self.raw_series_of(x)
            if raw is not None:
                si = self.alloc_state(SK_RAW, cap, raw)
                self.expr(n)
                self.emit(f.upper() + "_RAW", si)
            else:
                si = self.alloc_state(SK_RING, cap)
                self.expr(x)
                self.expr(n)
                self.emit(f.upper(), si)
        elif f in ("ema", "rsi"):
            self.argc(node, 2)
            si = self.alloc_state(SK_WILDER2 if f == "rsi" else SK_EMA)
            self.expr(node.args[0])
            self.expr(node.args[1])
            self.emit(f.upper(), si)
        elif f == "atr":
            self.argc(node, 1)
            si = self.alloc_state(SK_WILDER1)
            self.expr(node.args[0])
            self.emit("ATR", si)
        elif f in ("crossover", "crossunder"):
            self.argc(node, 2)
            si = self.alloc_state(SK_PREV2)
            self.expr(node.args[0])
            self.expr(node.args[1])
            self.emit(f.upper(), si)
        else:
            raise CompileError(f"unknown function {f!r}", node.line, node.col)

    def call_delay(self, x: A.Node, k: A.Node) -> None:
        # ring must hold k+1 values to reach k bars back
        cap = self.window_cap(k, "delay() lag", extra=1)
        raw = self.raw_series_of(x)
        if raw is not None:
            si = self.alloc_state(SK_RAW, cap, raw)
            self.expr(k)
            self.emit("DELAY_RAW", si)
        else:
            si = self.alloc_state(SK_RING, cap)
            self.expr(x)
            self.expr(k)
            self.emit("DELAY", si)

    # ---- top level ----

    def compile(self) -> Program:
        seen = set()
        for p in self.strat.params:
            if p.name in seen:
                raise CompileError(f"duplicate param {p.name!r}", p.line, p.col)
            if p.name in SERIES or p.name in CTX:
                raise CompileError(f"param {p.name!r} shadows a builtin name", p.line, p.col)
            seen.add(p.name)

        for stmt in self.strat.body:
            if isinstance(stmt, A.LetDecl):
                if stmt.name in seen or stmt.name in self.locals:
                    raise CompileError(f"duplicate name {stmt.name!r}", stmt.line, stmt.col)
                if stmt.name in SERIES or stmt.name in CTX:
                    raise CompileError(f"let {stmt.name!r} shadows a builtin name",
                                       stmt.line, stmt.col)
                self.expr(stmt.expr)
                li = len(self.local_names)
                self.locals[stmt.name] = li
                self.local_names.append(stmt.name)
                try:
                    self.let_bounds[stmt.name] = _bounds(
                        stmt.expr, self.param_bounds, self.let_bounds)
                except Unbounded:
                    self.let_bounds[stmt.name] = None
                self.emit("STORE", li)
            elif isinstance(stmt, A.Rule):
                self.expr(stmt.cond)
                self.emit(SIG_OP[stmt.kind])
            elif isinstance(stmt, A.SetCfg):
                self.expr(stmt.expr)
                self.emit(CFG_OP[stmt.key])
            else:
                raise CompileError("unexpected statement", stmt.line, stmt.col)
        self.emit("HALT")

        if self.state_floats > MAX_STATE_FLOATS:
            raise CompileError(
                f"strategy needs {self.state_floats} floats of per-thread state, "
                f"limit is {MAX_STATE_FLOATS} (reduce window sizes)")

        max_stack = self.verify_stack()
        return Program(
            source=self.source, consts=self.consts, code=self.code,
            params=[{"name": p.name, "default": p.default, "lo": p.lo,
                     "hi": p.hi, "step": p.step} for p in self.strat.params],
            n_locals=len(self.local_names), state_kind=self.state_kind,
            state_off=self.state_off, state_cap=self.state_cap,
            state_aux=self.state_aux,
            state_floats=self.state_floats, max_stack=max_stack,
            local_names=self.local_names,
        )

    def verify_stack(self) -> int:
        names = {v: k for k, v in OP.items()}
        depth = max_depth = 0
        for ins in self.code:
            op = names[ins >> 24]
            depth += STACK_EFFECT[op]
            if depth < 0:
                raise CompileError(f"internal: stack underflow at {op}")
            max_depth = max(max_depth, depth)
        if max_depth > MAX_STACK:
            raise CompileError(
                f"expression too deep: needs stack {max_depth}, limit {MAX_STACK}")
        return max_depth


def compile_source(src: str) -> Program:
    return Compiler(parse(src), src).compile()
