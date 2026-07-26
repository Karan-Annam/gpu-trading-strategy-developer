"""Genetic representation of strategies.

Genetics operate on *param-free* ASTs (every number is a literal); a
deterministic `parameterize` pass then promotes the most sweep-worthy literals
to `param` declarations so the GPU explores a region around each genome, not a
single point. The compiler is the validity gate: any op that yields an
uncompilable tree is retried, mirroring how LLM output is handled.
"""

from __future__ import annotations

import copy
import random

from dsl import ast_nodes as A
from dsl.compiler import compile_source
from dsl.errors import CompileError
from dsl.parser import parse
from dsl.printer import strategy_str

WINDOW_RANGE = (5, 400)
CMP_SWAP = {">": "<", "<": ">", ">=": "<=", "<=": ">=", "and": "or", "or": "and",
            "crossover": "crossunder", "crossunder": "crossover"}


# ---------- typed random generation ----------

def _rand_window(rng: random.Random) -> A.Num:
    return A.Num(float(rng.choice([5, 8, 13, 20, 30, 50, 80, 120, 200, 300])))


def _price_expr(rng: random.Random, depth: int = 0) -> A.Node:
    """Price-scale value expression."""
    roll = rng.random()
    if roll < 0.30:
        return A.Name(rng.choice(["close", "open", "high", "low"]))
    if roll < 0.55:
        return A.Call("sma", [A.Name("close"), _rand_window(rng)])
    if roll < 0.70:
        return A.Call("ema", [A.Name("close"), _rand_window(rng)])
    if roll < 0.80:
        return A.Call("highest", [A.Name("high"), _rand_window(rng)])
    if roll < 0.90:
        return A.Call("lowest", [A.Name("low"), _rand_window(rng)])
    return A.Call("delay", [A.Name("close"), A.Num(float(rng.randint(1, 30)))])


def _condition(rng: random.Random) -> A.Node:
    """A boolean expression of one of the sane scale-free families."""
    kind = rng.random()
    if kind < 0.40:                       # price vs price cross/compare
        a, b = _price_expr(rng), _price_expr(rng)
        f = rng.choice(["crossover", "crossunder", ">", "<"])
        if f in (">", "<"):
            return A.BinOp(f, a, b)
        return A.Call(f, [a, b])
    if kind < 0.65:                       # oscillator vs threshold
        osc = rng.choice(["rsi", "stoch_k", "willr"])
        if osc == "rsi":
            r = A.Call("rsi", [A.Name("close"), _rand_window(rng)])
        else:                             # stoch_k / willr take only a window
            r = A.Call(osc, [_rand_window(rng)])
        thr = A.Num(float(-rng.randint(15, 85) if osc == "willr"
                          else rng.randint(15, 85)))
        if rng.random() < 0.5:
            return A.BinOp(rng.choice([">", "<"]), r, thr)
        return A.Call(rng.choice(["crossover", "crossunder"]), [r, thr])
    if kind < 0.85:                       # momentum vs small threshold
        roc = A.Call("roc", [A.Name("close"), _rand_window(rng)])
        thr = A.Num(round(rng.uniform(-0.03, 0.03), 4))
        return A.BinOp(rng.choice([">", "<"]), roc, thr)
    # volatility regime: price distance vs ATR multiple
    dist = A.BinOp("-", A.Name("close"), A.Call("sma", [A.Name("close"), _rand_window(rng)]))
    mult = A.BinOp("*", A.Num(round(rng.uniform(0.5, 3.0), 2)),
                   A.Call("atr", [_rand_window(rng)]))
    return A.BinOp(rng.choice([">", "<"]), dist, mult)


def _bool_expr(rng: random.Random, max_terms: int = 3) -> A.Node:
    node = _condition(rng)
    while rng.random() < 0.35 and max_terms > 1:
        node = A.BinOp(rng.choice(["and", "or"]), node, _condition(rng))
        max_terms -= 1
    return node


def random_strategy(rng: random.Random, allow_short: bool = False) -> A.Strategy:
    body: list[A.Node] = [
        A.Rule("enter_long", _bool_expr(rng)),
        A.Rule("exit_long", _bool_expr(rng)),
    ]
    if allow_short and rng.random() < 0.3:
        body.append(A.Rule("enter_short", _bool_expr(rng)))
        body.append(A.Rule("exit_short", _bool_expr(rng)))
    if rng.random() < 0.6:
        body.append(A.SetCfg("stop_loss", A.Num(round(rng.uniform(0.005, 0.08), 4))))
    if rng.random() < 0.3:
        body.append(A.SetCfg("take_profit", A.Num(round(rng.uniform(0.01, 0.15), 4))))
    if rng.random() < 0.25:
        body.append(A.SetCfg("trail_stop", A.Num(round(rng.uniform(0.01, 0.1), 4))))
    return A.Strategy(params=[], body=body)


# ---------- param handling ----------

def strip_params(strat: A.Strategy) -> A.Strategy:
    """Inline every param reference as its default value."""
    out = copy.deepcopy(strat)
    defaults = {p.name: p.default for p in out.params}

    def sub(node: A.Node) -> A.Node:
        if isinstance(node, A.Name) and node.id in defaults:
            return A.Num(defaults[node.id])
        if isinstance(node, A.UnaryOp):
            node.operand = sub(node.operand)
        elif isinstance(node, A.BinOp):
            node.left = sub(node.left)
            node.right = sub(node.right)
        elif isinstance(node, A.Call):
            node.args = [sub(a) for a in node.args]
        return node

    for stmt in out.body:
        if isinstance(stmt, A.LetDecl):
            stmt.expr = sub(stmt.expr)
        elif isinstance(stmt, A.Rule):
            stmt.cond = sub(stmt.cond)
        elif isinstance(stmt, A.SetCfg):
            stmt.expr = sub(stmt.expr)
    out.params = []
    return out


WINDOWED = {"sma": 1, "ema": 1, "rsi": 1, "highest": 1, "lowest": 1,
            "stddev": 1, "atr": 0, "delay": 1, "roc": 1,
            "stoch_k": 0, "willr": 0, "vwap": 0,
            "macd": 1, "bb_upper": 1, "bb_lower": 1}


def parameterize(strat: A.Strategy, max_params: int = 4) -> A.Strategy:
    """Promote the most sweep-worthy literals to params (windows first, then
    thresholds and risk settings)."""
    out = copy.deepcopy(strat)
    out.params = []
    candidates: list[tuple[int, A.Call | A.SetCfg | A.BinOp, int | None, str]] = []

    def walk(node: A.Node) -> None:
        if isinstance(node, A.Call):
            if node.func in WINDOWED:
                ai = WINDOWED[node.func]
                if ai < len(node.args) and isinstance(node.args[ai], A.Num):
                    candidates.append((0, node, ai, "window"))
            for a in node.args:
                walk(a)
        elif isinstance(node, A.BinOp):
            # oscillator / momentum thresholds: cmp with numeric right side
            if node.op in (">", "<", ">=", "<=") and isinstance(node.right, A.Num):
                candidates.append((1, node, None, "threshold"))
            walk(node.left)
            walk(node.right)
        elif isinstance(node, A.UnaryOp):
            walk(node.operand)

    for stmt in out.body:
        if isinstance(stmt, A.LetDecl):
            walk(stmt.expr)
        elif isinstance(stmt, A.Rule):
            walk(stmt.cond)
        elif isinstance(stmt, A.SetCfg):
            if isinstance(stmt.expr, A.Num):
                candidates.append((2, stmt, None, "risk"))
            else:
                walk(stmt.expr)

    candidates.sort(key=lambda c: c[0])
    names = iter(["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"])
    for _, holder, ai, kind in candidates[:max_params]:
        name = next(names)
        if kind == "window":
            v = holder.args[ai].value
            lo, hi = max(2.0, round(v / 3)), min(float(4096), round(v * 3))
            holder.args[ai] = A.Name(name)
        elif kind == "threshold":
            v = holder.right.value
            span = max(abs(v) * 0.6, 1e-3 if abs(v) < 1 else 5.0)
            lo, hi = v - span, v + span
            holder.right = A.Name(name)
        else:  # risk
            v = holder.expr.value
            lo, hi = max(0.002, v / 4), min(0.5, v * 4)
            holder.expr = A.Name(name)
        v = min(max(v, lo), hi)
        out.params.append(A.ParamDecl(name, float(v), float(lo), float(hi)))
    return out


# ---------- genetic operators ----------

def _bool_slots(strat: A.Strategy) -> list[A.Rule]:
    return [s for s in strat.body if isinstance(s, A.Rule)]


def _numbers(node: A.Node, acc: list[A.Num]) -> None:
    if isinstance(node, A.Num):
        acc.append(node)
    elif isinstance(node, A.UnaryOp):
        _numbers(node.operand, acc)
    elif isinstance(node, A.BinOp):
        _numbers(node.left, acc)
        _numbers(node.right, acc)
    elif isinstance(node, A.Call):
        for a in node.args:
            _numbers(a, acc)


def _swap_ops(node: A.Node, rng: random.Random, prob: float = 0.5) -> None:
    if isinstance(node, A.BinOp):
        if node.op in CMP_SWAP and rng.random() < prob:
            node.op = CMP_SWAP[node.op]
        _swap_ops(node.left, rng, prob * 0.7)
        _swap_ops(node.right, rng, prob * 0.7)
    elif isinstance(node, A.Call):
        if node.func in CMP_SWAP and rng.random() < prob:
            node.func = CMP_SWAP[node.func]
        for a in node.args:
            _swap_ops(a, rng, prob * 0.7)
    elif isinstance(node, A.UnaryOp):
        _swap_ops(node.operand, rng, prob * 0.7)


def mutate(strat: A.Strategy, rng: random.Random) -> A.Strategy:
    """One random structural or numeric mutation on a param-free tree."""
    out = copy.deepcopy(strat)
    rules = _bool_slots(out)
    op = rng.random()

    if op < 0.35:                                   # jitter numbers
        nums: list[A.Num] = []
        for s in out.body:
            src = s.expr if isinstance(s, (A.LetDecl, A.SetCfg)) else s.cond
            _numbers(src, nums)
        if nums:
            for tgt in rng.sample(nums, k=max(1, len(nums) // 3)):
                tgt.value = round(tgt.value * rng.uniform(0.6, 1.6), 6)
    elif op < 0.50 and rules:                        # swap an operator
        _swap_ops(rng.choice(rules).cond, rng)
    elif op < 0.70 and rules:                        # regrow a rule condition
        rng.choice(rules).cond = _bool_expr(rng)
    elif op < 0.85 and rules:                        # add / strip a conjunct
        r = rng.choice(rules)
        if isinstance(r.cond, A.BinOp) and r.cond.op in ("and", "or") and rng.random() < 0.5:
            r.cond = r.cond.left
        else:
            r.cond = A.BinOp(rng.choice(["and", "or"]), r.cond, _condition(rng))
    else:                                            # risk-config mutation
        cfgs = [s for s in out.body if isinstance(s, A.SetCfg)]
        if cfgs and rng.random() < 0.5:
            out.body.remove(rng.choice(cfgs))
        else:
            key = rng.choice(["stop_loss", "take_profit", "trail_stop"])
            out.body = [s for s in out.body
                        if not (isinstance(s, A.SetCfg) and s.key == key)]
            out.body.append(A.SetCfg(key, A.Num(round(rng.uniform(0.005, 0.1), 4))))
    return out


def crossover(a: A.Strategy, b: A.Strategy, rng: random.Random) -> A.Strategy:
    """Child takes entry/exit/risk blocks from either parent."""
    child = copy.deepcopy(a)
    donor = copy.deepcopy(b)

    def rules_of(s: A.Strategy, kind: str) -> list[A.Rule]:
        return [r for r in _bool_slots(s) if r.kind == kind]

    new_body: list[A.Node] = []
    for kind in ("enter_long", "exit_long", "enter_short", "exit_short"):
        src = child if rng.random() < 0.5 else donor
        new_body.extend(rules_of(src, kind))
    src = child if rng.random() < 0.5 else donor
    new_body.extend(s for s in src.body if isinstance(s, A.SetCfg))
    # lets are not generated by the genome; drop any inherited ones safely
    child.body = new_body
    child.params = []
    return child


# ---------- compile gate ----------

def to_source(strat: A.Strategy, max_params: int = 4) -> str | None:
    """Parameterize, print, and verify compilability. None if invalid."""
    try:
        src = strategy_str(parameterize(strat, max_params))
        compile_source(src)
        return src
    except (CompileError, Exception):
        return None


def from_source(src: str) -> A.Strategy:
    return strip_params(parse(src))


def spawn_valid(make, rng: random.Random, tries: int = 12):
    """Repeat a generator until the compile gate passes."""
    for _ in range(tries):
        strat = make()
        src = to_source(strat)
        if src is not None:
            return strat, src
    return None, None
