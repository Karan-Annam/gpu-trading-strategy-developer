"""AST node definitions.

Kept as plain dataclasses so evolve/ can mutate trees directly and printer.py
can turn any tree back into DSL source.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Node:
    line: int = field(default=0, kw_only=True)
    col: int = field(default=0, kw_only=True)


# ---- expressions ----

@dataclass
class Num(Node):
    value: float


@dataclass
class Name(Node):
    """Reference to a param, let, series (unlagged), or context scalar."""
    id: str


@dataclass
class SeriesLag(Node):
    """close[3] — series name with a literal lag."""
    id: str
    lag: int


@dataclass
class UnaryOp(Node):
    op: str          # '-', 'not'
    operand: Node


@dataclass
class BinOp(Node):
    op: str          # + - * / > < >= <= == != and or
    left: Node
    right: Node


@dataclass
class Call(Node):
    func: str
    args: list[Node]


# ---- statements ----

@dataclass
class ParamDecl(Node):
    name: str
    default: float
    lo: float
    hi: float
    step: float | None = None


@dataclass
class LetDecl(Node):
    name: str
    expr: Node


@dataclass
class Rule(Node):
    kind: str        # enter_long | exit_long | enter_short | exit_short
    cond: Node


@dataclass
class SetCfg(Node):
    key: str         # stop_loss | take_profit | trail_stop | size
    expr: Node


@dataclass
class Strategy(Node):
    params: list[ParamDecl] = field(default_factory=list)
    body: list[Node] = field(default_factory=list)   # LetDecl | Rule | SetCfg in source order
