"""Tokenizer for the strategy DSL."""

from __future__ import annotations

from dataclasses import dataclass

from dsl.errors import CompileError

KEYWORDS = {
    "param", "let", "in", "when", "set", "step",
    "and", "or", "not",
    "enter_long", "exit_long", "enter_short", "exit_short",
    "buy", "sell", "short", "cover",
}

TWO_CHAR = {">=", "<=", "==", "!="}
ONE_CHAR = set("+-*/()[]=<>,")


@dataclass
class Tok:
    kind: str      # NAME KEYWORD NUMBER OP NEWLINE EOF
    text: str
    line: int
    col: int


def tokenize(src: str) -> list[Tok]:
    toks: list[Tok] = []
    for lineno, raw in enumerate(src.splitlines(), start=1):
        line = raw.split("#", 1)[0]
        i, n = 0, len(line)
        while i < n:
            c = line[i]
            if c in " \t\r":
                i += 1
                continue
            col = i + 1
            if c.isdigit() or (c == "." and i + 1 < n and line[i + 1].isdigit()):
                j = i
                seen_dot = False
                while j < n and (line[j].isdigit() or (line[j] == "." and not seen_dot)):
                    seen_dot = seen_dot or line[j] == "."
                    j += 1
                # exponent form deliberately unsupported (keep grammar tiny)
                toks.append(Tok("NUMBER", line[i:j], lineno, col))
                i = j
            elif c.isalpha() or c == "_":
                j = i
                while j < n and (line[j].isalnum() or line[j] == "_"):
                    j += 1
                word = line[i:j]
                toks.append(Tok("KEYWORD" if word in KEYWORDS else "NAME", word, lineno, col))
                i = j
            elif line[i:i + 2] in TWO_CHAR:
                toks.append(Tok("OP", line[i:i + 2], lineno, col))
                i += 2
            elif c in ONE_CHAR:
                toks.append(Tok("OP", c, lineno, col))
                i += 1
            else:
                raise CompileError(f"unexpected character {c!r}", lineno, col)
        if toks and toks[-1].kind != "NEWLINE":
            toks.append(Tok("NEWLINE", "", lineno, n + 1))
    toks.append(Tok("EOF", "", len(src.splitlines()) + 1, 1))
    return toks
