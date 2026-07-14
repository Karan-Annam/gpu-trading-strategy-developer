"""OBP1 export round-trip tests."""

import glob
import os
import struct

import numpy as np
import pytest

from dsl.compiler import compile_source
from dsl.export import MAGIC, from_bytes, to_bytes

STRATS = glob.glob(os.path.join(os.path.dirname(__file__), "..", "strategies", "*.dsl"))


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


@pytest.mark.parametrize("path", STRATS, ids=[os.path.basename(p) for p in STRATS])
def test_roundtrip(path):
    prog = compile_source(read(path))
    d = from_bytes(to_bytes(prog))
    assert d["code"] == prog.code
    assert d["state_kind"] == prog.state_kind
    assert d["state_off"] == prog.state_off
    assert d["state_cap"] == prog.state_cap
    assert d["state_aux"] == prog.state_aux
    assert d["n_locals"] == prog.n_locals
    assert d["state_floats"] == prog.state_floats
    assert d["max_stack"] == prog.max_stack
    assert d["param_names"] == [p["name"] for p in prog.params]
    # consts/defaults survive the f64 -> f32 narrowing exactly as the
    # engines see them (they consume float32 too)
    np.testing.assert_array_equal(
        np.asarray(d["consts"], np.float32), np.asarray(prog.consts, np.float32))
    np.testing.assert_array_equal(
        np.asarray(d["param_defaults"], np.float32),
        np.asarray(prog.param_defaults(), np.float32))


def test_bad_magic():
    prog = compile_source("buy when close > open\n")
    blob = bytearray(to_bytes(prog))
    blob[:4] = b"NOPE"
    with pytest.raises(ValueError, match="magic"):
        from_bytes(bytes(blob))


def test_trailing_bytes_rejected():
    prog = compile_source("buy when close > open\n")
    with pytest.raises(ValueError, match="trailing"):
        from_bytes(to_bytes(prog) + b"\x00")


@pytest.mark.parametrize("cut", [0, 4, 35, 36, 40])
def test_truncated_payload_rejected(cut):
    prog = compile_source("buy when close > open\n")
    with pytest.raises(ValueError, match="truncated|magic"):
        from_bytes(to_bytes(prog)[:cut])


def test_resource_limits_rejected():
    prog = compile_source("buy when close > open\n")
    blob = bytearray(to_bytes(prog))
    # max_stack is the final u32 in the fixed header.
    struct.pack_into("<I", blob, 32, 10_000)
    with pytest.raises(ValueError, match="resource limits"):
        from_bytes(bytes(blob))


def test_magic_constant():
    assert MAGIC == b"OBP1"
