"""Serialize a compiled Program to the OBP1 flat-binary format.

OBP1 is the transfer format consumed by orderbook_project's trading engine
(sw/trade/program.cpp): compile strategies here with the full Python
toolchain, load them there with a ~100-line dependency-free C++ reader.

Layout (little-endian, no padding):

    header   9 x u32   magic 'OBP1', version, n_code, n_consts, n_state,
                       n_params, n_locals, state_floats, max_stack
    code     u32[n_code]
    consts   f32[n_consts]
    state    i32[n_state] x 4   kind, off, cap, aux (arrays, not interleaved)
    defaults f32[n_params]
    names    n_params x { u16 len, bytes }   param names for CLI overrides
"""

from __future__ import annotations

import struct

from dsl.compiler import Program, compile_source
from dsl.opcodes import MAX_STACK, MAX_STATE_FLOATS

MAGIC = b"OBP1"
VERSION = 1


def to_bytes(prog: Program) -> bytes:
    for param in prog.params:
        if len(param["name"].encode("utf-8")) > 0xFFFF:
            raise ValueError("parameter name is too long for OBP1")
    out = bytearray()
    out += MAGIC
    out += struct.pack(
        "<8I", VERSION, len(prog.code), len(prog.consts), len(prog.state_kind),
        len(prog.params), prog.n_locals, prog.state_floats, prog.max_stack)
    out += struct.pack(f"<{len(prog.code)}I", *prog.code)
    out += struct.pack(f"<{len(prog.consts)}f", *prog.consts)
    for arr in (prog.state_kind, prog.state_off, prog.state_cap, prog.state_aux):
        out += struct.pack(f"<{len(arr)}i", *arr)
    out += struct.pack(f"<{len(prog.params)}f", *prog.param_defaults())
    for p in prog.params:
        name = p["name"].encode("utf-8")
        out += struct.pack("<H", len(name)) + name
    return bytes(out)


def from_bytes(data: bytes) -> dict:
    """Parse OBP1 back into plain fields (test round-trip / debugging)."""
    if len(data) < 36:
        raise ValueError("truncated OBP1 header")
    if data[:4] != MAGIC:
        raise ValueError(f"bad magic {data[:4]!r}")
    ver, n_code, n_consts, n_state, n_params, n_locals, state_floats, max_stack = \
        struct.unpack_from("<8I", data, 4)
    if ver != VERSION:
        raise ValueError(f"unsupported version {ver}")
    if max_stack > MAX_STACK or state_floats > MAX_STATE_FLOATS:
        raise ValueError("OBP1 resource limits exceeded")
    if n_params > 65535:
        raise ValueError("too many OBP1 parameters")
    minimum = 36 + 4 * (n_code + n_consts + 4 * n_state + n_params)
    if minimum > len(data):
        raise ValueError("truncated OBP1 payload")
    off = 36
    code = list(struct.unpack_from(f"<{n_code}I", data, off)); off += 4 * n_code
    consts = list(struct.unpack_from(f"<{n_consts}f", data, off)); off += 4 * n_consts
    state = []
    for _ in range(4):
        state.append(list(struct.unpack_from(f"<{n_state}i", data, off)))
        off += 4 * n_state
    defaults = list(struct.unpack_from(f"<{n_params}f", data, off)); off += 4 * n_params
    names = []
    for _ in range(n_params):
        if off + 2 > len(data):
            raise ValueError("truncated OBP1 parameter name")
        (ln,) = struct.unpack_from("<H", data, off); off += 2
        if off + ln > len(data):
            raise ValueError("truncated OBP1 parameter name")
        try:
            names.append(data[off:off + ln].decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ValueError("invalid UTF-8 parameter name") from exc
        off += ln
    if off != len(data):
        raise ValueError(f"trailing bytes: {len(data) - off}")
    return {
        "code": code, "consts": consts,
        "state_kind": state[0], "state_off": state[1],
        "state_cap": state[2], "state_aux": state[3],
        "param_defaults": defaults, "param_names": names,
        "n_locals": n_locals, "state_floats": state_floats,
        "max_stack": max_stack,
    }


def export_file(dsl_path: str, obp_path: str) -> Program:
    with open(dsl_path, encoding="utf-8") as f:
        prog = compile_source(f.read())
    with open(obp_path, "wb") as f:
        f.write(to_bytes(prog))
    return prog


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="compile .dsl -> .obp bytecode")
    ap.add_argument("dsl", help="input strategy source")
    ap.add_argument("obp", help="output OBP1 binary")
    args = ap.parse_args()
    prog = export_file(args.dsl, args.obp)
    print(f"{args.obp}: {len(prog.code)} ops, {len(prog.consts)} consts, "
          f"{len(prog.params)} params, {prog.state_floats} state floats")


if __name__ == "__main__":
    main()
