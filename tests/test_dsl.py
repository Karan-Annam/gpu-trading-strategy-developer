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


def test_parameter_step_must_be_positive():
    with pytest.raises(CompileError, match="step must be positive"):
        compile_source("param n = 1 in [1, 5] step 0\nbuy when close > open\n")


def test_float32_overflow_is_rejected():
    huge = "9" * 100
    with pytest.raises(CompileError, match="finite float32"):
        compile_source(f"buy when close > {huge}\n")


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


# ---- sugar builtins ----

SUGAR_CASES = [
    ("stoch_k(14)",
     "100 * (close - lowest(low, 14)) / (highest(high, 14) - lowest(low, 14))"),
    ("willr(14)",
     "-(100 * (highest(high, 14) - close) / (highest(high, 14) - lowest(low, 14)))"),
    ("macd(close, 12, 26)", "ema(close, 12) - ema(close, 26)"),
    ("bb_upper(close, 20, 2)", "sma(close, 20) + 2 * stddev(close, 20)"),
    ("bb_lower(close, 20, 2)", "sma(close, 20) - 2 * stddev(close, 20)"),
    ("vwap(20)", "sma(close * volume, 20) / sma(volume, 20)"),
]


@pytest.mark.parametrize("sugar,manual", SUGAR_CASES,
                         ids=[c[0].split("(")[0] for c in SUGAR_CASES])
def test_sugar_matches_manual(sugar, manual):
    ps = compile_source(f"let v = {sugar}\nbuy when v > 0\n")
    pm = compile_source(f"let v = {manual}\nbuy when v > 0\n")
    assert ps.code == pm.code
    assert ps.consts == pm.consts
    assert ps.state_kind == pm.state_kind
    assert ps.state_cap == pm.state_cap
    assert ps.state_off == pm.state_off
    assert ps.state_aux == pm.state_aux


def test_sugar_state_cost():
    # every duplicated window in stoch_k reads a raw series -> storage-free
    assert compile_source("let k = stoch_k(14)\nbuy when k > 80\n").state_floats == 0
    # vwap's close*volume numerator is computed -> one ring of 1 + n floats
    assert compile_source("let v = vwap(20)\nbuy when close > v\n").state_floats == 21


@pytest.mark.parametrize("src", [
    "let x = stoch_k(14, 3)\nbuy when x > 0\n",
    "let x = macd(close, 12)\nbuy when x > 0\n",
    "let x = bb_upper(close, 20)\nbuy when x > 0\n",
    "let x = vwap()\nbuy when x > 0\n",
])
def test_sugar_argc_errors(src):
    with pytest.raises(CompileError, match="argument"):
        compile_source(src)
