"""AST -> DSL source. Used to render evolved/mutated strategies as
human-readable code for the Lab editor."""

from __future__ import annotations

from dsl import ast_nodes as A

_PREC = {"or": 1, "and": 2, "not": 3,
         ">": 4, "<": 4, ">=": 4, "<=": 4, "==": 4, "!=": 4,
         "+": 5, "-": 5, "*": 6, "/": 6, "u-": 7}


def _num(v: float) -> str:
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return repr(round(v, 10))


def expr_str(node: A.Node, parent_prec: int = 0) -> str:
    if isinstance(node, A.Num):
        s = _num(node.value)
        return f"({s})" if node.value < 0 and parent_prec >= 5 else s
    if isinstance(node, A.Name):
        return node.id
    if isinstance(node, A.SeriesLag):
        return f"{node.id}[{node.lag}]"
    if isinstance(node, A.UnaryOp):
        if node.op == "not":
            s = f"not {expr_str(node.operand, _PREC['not'])}"
            return f"({s})" if parent_prec > _PREC["not"] else s
        s = f"-{expr_str(node.operand, _PREC['u-'])}"
        return f"({s})" if parent_prec >= _PREC["u-"] else s
    if isinstance(node, A.BinOp):
        p = _PREC[node.op]
        s = f"{expr_str(node.left, p)} {node.op} {expr_str(node.right, p + 1)}"
        return f"({s})" if parent_prec > p else s
    if isinstance(node, A.Call):
        return f"{node.func}({', '.join(expr_str(a) for a in node.args)})"
    raise TypeError(f"cannot print {type(node).__name__}")


def strategy_str(strat: A.Strategy) -> str:
    lines: list[str] = []
    for p in strat.params:
        s = f"param {p.name} = {_num(p.default)} in [{_num(p.lo)}, {_num(p.hi)}]"
        if p.step is not None:
            s += f" step {_num(p.step)}"
        lines.append(s)
    if strat.params:
        lines.append("")
    for stmt in strat.body:
        if isinstance(stmt, A.LetDecl):
            lines.append(f"let {stmt.name} = {expr_str(stmt.expr)}")
        elif isinstance(stmt, A.Rule):
            lines.append(f"{stmt.kind} when {expr_str(stmt.cond)}")
        elif isinstance(stmt, A.SetCfg):
            lines.append(f"set {stmt.key} = {expr_str(stmt.expr)}")
    return "\n".join(lines) + "\n"
