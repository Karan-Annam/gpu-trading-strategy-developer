"""Recursive-descent parser: DSL source -> Strategy AST."""

from __future__ import annotations

from dsl import ast_nodes as A
from dsl.errors import CompileError
from dsl.lexer import Tok, tokenize

RULE_ALIASES = {
    "buy": "enter_long", "sell": "exit_long",
    "short": "enter_short", "cover": "exit_short",
    "enter_long": "enter_long", "exit_long": "exit_long",
    "enter_short": "enter_short", "exit_short": "exit_short",
}
CONFIG_KEYS = {"stop_loss", "take_profit", "trail_stop", "size"}
CMP_OPS = {">", "<", ">=", "<=", "==", "!="}


class Parser:
    def __init__(self, toks: list[Tok]):
        self.toks = toks
        self.i = 0

    # ---- token plumbing ----

    def peek(self) -> Tok:
        return self.toks[self.i]

    def next(self) -> Tok:
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, kind: str, text: str | None = None) -> Tok:
        t = self.peek()
        if t.kind != kind or (text is not None and t.text != text):
            want = text or kind
            got = t.text or t.kind
            raise CompileError(f"expected {want!r}, got {got!r}", t.line, t.col)
        return self.next()

    def at(self, kind: str, text: str | None = None) -> bool:
        t = self.peek()
        return t.kind == kind and (text is None or t.text == text)

    def skip_newlines(self) -> None:
        while self.at("NEWLINE"):
            self.next()

    # ---- statements ----

    def parse(self) -> A.Strategy:
        strat = A.Strategy(line=1, col=1)
        self.skip_newlines()
        while not self.at("EOF"):
            t = self.peek()
            if self.at("KEYWORD", "param"):
                strat.params.append(self.param_decl())
            elif self.at("KEYWORD", "let"):
                strat.body.append(self.let_decl())
            elif self.at("KEYWORD", "set"):
                strat.body.append(self.set_cfg())
            elif t.kind == "KEYWORD" and t.text in RULE_ALIASES:
                strat.body.append(self.rule())
            else:
                got = t.text or t.kind
                raise CompileError(
                    f"expected a statement (param/let/set/rule), got {got!r}", t.line, t.col)
            if not self.at("EOF"):
                self.expect("NEWLINE")
            self.skip_newlines()
        return strat

    def number(self) -> float:
        neg = False
        if self.at("OP", "-"):
            self.next()
            neg = True
        t = self.expect("NUMBER")
        v = float(t.text)
        return -v if neg else v

    def param_decl(self) -> A.ParamDecl:
        kw = self.expect("KEYWORD", "param")
        name = self.expect("NAME").text
        self.expect("OP", "=")
        default = self.number()
        self.expect("KEYWORD", "in")
        self.expect("OP", "[")
        lo = self.number()
        self.expect("OP", ",")
        hi = self.number()
        self.expect("OP", "]")
        step = None
        if self.at("KEYWORD", "step"):
            self.next()
            step = self.number()
        if hi < lo:
            raise CompileError(f"param {name}: hi < lo", kw.line, kw.col)
        if not (lo <= default <= hi):
            raise CompileError(f"param {name}: default outside [lo, hi]", kw.line, kw.col)
        return A.ParamDecl(name, default, lo, hi, step, line=kw.line, col=kw.col)

    def let_decl(self) -> A.LetDecl:
        kw = self.expect("KEYWORD", "let")
        name = self.expect("NAME").text
        self.expect("OP", "=")
        return A.LetDecl(name, self.expr(), line=kw.line, col=kw.col)

    def set_cfg(self) -> A.SetCfg:
        kw = self.expect("KEYWORD", "set")
        t = self.next()
        if t.kind != "NAME" or t.text not in CONFIG_KEYS:
            raise CompileError(
                f"unknown config {t.text!r} (expected one of {sorted(CONFIG_KEYS)})",
                t.line, t.col)
        self.expect("OP", "=")
        return A.SetCfg(t.text, self.expr(), line=kw.line, col=kw.col)

    def rule(self) -> A.Rule:
        kw = self.next()
        self.expect("KEYWORD", "when")
        return A.Rule(RULE_ALIASES[kw.text], self.expr(), line=kw.line, col=kw.col)

    # ---- expressions (precedence climbing) ----

    def expr(self) -> A.Node:
        return self.or_expr()

    def or_expr(self) -> A.Node:
        left = self.and_expr()
        while self.at("KEYWORD", "or"):
            t = self.next()
            left = A.BinOp("or", left, self.and_expr(), line=t.line, col=t.col)
        return left

    def and_expr(self) -> A.Node:
        left = self.not_expr()
        while self.at("KEYWORD", "and"):
            t = self.next()
            left = A.BinOp("and", left, self.not_expr(), line=t.line, col=t.col)
        return left

    def not_expr(self) -> A.Node:
        if self.at("KEYWORD", "not"):
            t = self.next()
            return A.UnaryOp("not", self.not_expr(), line=t.line, col=t.col)
        return self.cmp_expr()

    def cmp_expr(self) -> A.Node:
        left = self.add_expr()
        if self.peek().kind == "OP" and self.peek().text in CMP_OPS:
            t = self.next()
            right = self.add_expr()
            node = A.BinOp(t.text, left, right, line=t.line, col=t.col)
            nxt = self.peek()
            if nxt.kind == "OP" and nxt.text in CMP_OPS:
                raise CompileError("chained comparisons are not allowed", nxt.line, nxt.col)
            return node
        return left

    def add_expr(self) -> A.Node:
        left = self.mul_expr()
        while self.peek().kind == "OP" and self.peek().text in ("+", "-"):
            t = self.next()
            left = A.BinOp(t.text, left, self.mul_expr(), line=t.line, col=t.col)
        return left

    def mul_expr(self) -> A.Node:
        left = self.unary()
        while self.peek().kind == "OP" and self.peek().text in ("*", "/"):
            t = self.next()
            left = A.BinOp(t.text, left, self.unary(), line=t.line, col=t.col)
        return left

    def unary(self) -> A.Node:
        if self.at("OP", "-"):
            t = self.next()
            return A.UnaryOp("-", self.unary(), line=t.line, col=t.col)
        return self.postfix()

    def postfix(self) -> A.Node:
        node = self.primary()
        if self.at("OP", "["):
            t = self.next()
            lag_tok = self.expect("NUMBER")
            if "." in lag_tok.text:
                raise CompileError("lag must be an integer literal", lag_tok.line, lag_tok.col)
            self.expect("OP", "]")
            if not isinstance(node, A.Name):
                raise CompileError("only series can be indexed, e.g. close[1]", t.line, t.col)
            return A.SeriesLag(node.id, int(lag_tok.text), line=node.line, col=node.col)
        return node

    def primary(self) -> A.Node:
        t = self.peek()
        if t.kind == "NUMBER":
            self.next()
            return A.Num(float(t.text), line=t.line, col=t.col)
        if t.kind == "NAME":
            self.next()
            if self.at("OP", "("):
                self.next()
                args: list[A.Node] = []
                if not self.at("OP", ")"):
                    args.append(self.expr())
                    while self.at("OP", ","):
                        self.next()
                        args.append(self.expr())
                self.expect("OP", ")")
                return A.Call(t.text, args, line=t.line, col=t.col)
            return A.Name(t.text, line=t.line, col=t.col)
        if self.at("OP", "("):
            self.next()
            node = self.expr()
            self.expect("OP", ")")
            return node
        got = t.text or t.kind
        raise CompileError(f"expected an expression, got {got!r}", t.line, t.col)


def parse(src: str) -> A.Strategy:
    return Parser(tokenize(src)).parse()
