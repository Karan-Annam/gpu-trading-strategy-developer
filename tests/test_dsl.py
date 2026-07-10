"""Compiler tests: parsing, roundtrip, errors, bounds."""

import glob
import os

import pytest

from dsl.compiler import compile_source
from dsl.errors import CompileError
from dsl.parser import parse
from dsl.printer import strategy_str

STRATS = glob.glob(os.path.join(os.path.dirname(__file__), "..", "strategies", "*.dsl"))


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


@pytest.mark.parametrize("path", STRATS, ids=[os.path.basename(p) for p in STRATS])
def test_compile_seed(path):
    prog = compile_source(read(path))
    assert prog.code
    assert prog.max_stack <= 32
    assert prog.state_floats < 8192


@pytest.mark.parametrize("path", STRATS, ids=[os.path.basename(p) for p in STRATS])
def test_print_roundtrip(path):
    src = read(path)
    printed = strategy_str(parse(src))
    p1 = compile_source(src)
    p2 = compile_source(printed)
    assert p1.code == p2.code
    assert p1.consts == p2.consts


def test_simple_program_shape():
    prog = compile_source(
        "param n = 10 in [2, 50]\n"
        "let m = sma(close, n)\n"
        "enter_long when close > m\n"
        "exit_long when close < m\n")
    assert prog.n_locals == 1
    assert len(prog.params) == 1
    # sma over a raw series takes the RAW path: no per-thread ring storage
    assert prog.state_floats == 0
    assert prog.state_cap[0] == 50


def test_ring_vs_raw_paths():
    # raw series input -> zero storage; computed input -> ring buffer
    raw = compile_source("let m = sma(close, 20)\nbuy when close > m\n")
    ring = compile_source("let x = close * 2\nlet m = sma(x, 20)\nbuy when close > m\n")
    assert raw.state_floats == 0
    assert ring.state_floats == 1 + 20


@pytest.mark.parametrize("src,frag", [
    ("let x = nosuch(close, 3)\nbuy when x > 0\n", "unknown function"),
    ("buy when qqq > 1\n", "unknown name"),
    ("let x = sma(close, volume)\nbuy when x > 0\n", "statically boundable"),
    ("param n = 10 in [2, 9999]\nlet x = sma(close, n)\nbuy when x > 0\n", "exceeds"),
    ("param a = 1 in [0, 2]\nparam a = 1 in [0, 2]\n", "duplicate"),
    ("buy when 1 < close < 2\n", "chained"),
    ("let close = 1\nbuy when close > 0\n", "shadows"),
    ("param n = 99 in [1, 10]\n", "default outside"),
    ("let x = sma(close)\nbuy when x > 0\n", "takes 2 arguments"),
    ("set fee = 0\n", "unknown config"),
])
def test_compile_errors(src, frag):
    with pytest.raises(CompileError) as ei:
        compile_source(src)
    assert frag in str(ei.value)


def test_error_carries_location():
    with pytest.raises(CompileError) as ei:
        compile_source("let a = close\nlet b = zzz + 1\n")
    assert ei.value.line == 2
    assert ei.value.col > 0


def test_window_bound_via_params():
    # bound = hi(a) * hi(b) = 20 * 4 = 80
    prog = compile_source(
        "param a = 10 in [5, 20]\n"
        "param b = 2 in [1, 4]\n"
        "let m = sma(close, a * b)\n"
        "buy when close > m\n")
    assert prog.state_cap[0] == 80


def test_alias_keywords():
    p1 = compile_source("buy when close > 1\nsell when close < 1\n")
    p2 = compile_source("enter_long when close > 1\nexit_long when close < 1\n")
    assert p1.code == p2.code
