"""Python reference engine — the golden model.

Deliberately slow and explicit: a per-bar interpreter loop in which every
arithmetic operation goes through np.float32, in the exact order the C++ and
CUDA engines perform it. All engine work is validated against this.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dsl.compiler import Program
from dsl.opcodes import OP, SK_RING

F = np.float32
F0 = F(0.0)
F1 = F(1.0)

_OPN = {v: k for k, v in OP.items()}


@dataclass
class BacktestConfig:
    fee_rate: float = 0.001
    slip: float = 0.0005
    bars_per_year: float = 525600.0
    equity0: float = 10000.0


@dataclass
class Trade:
    side: int          # +1 long, -1 short
    entry_t: int
    exit_t: int
    entry_px: float
    exit_px: float
    qty: float
    pnl: float         # net of fees
    reason: str        # signal | stop | trail | tp | eod


@dataclass
class BacktestResult:
    equity: np.ndarray
    trades: list[Trade]
    metrics: dict = field(default_factory=dict)
    locals_curve: np.ndarray | None = None   # (T, n_locals) when recorded


def _truthy(x) -> bool:
    return x != F0


def run(program: Program, bars, params: list[float] | None = None,
        config: BacktestConfig | None = None,
        record_locals: bool = False) -> BacktestResult:
    cfg = config or BacktestConfig()
    pvals = [F(v) for v in (params if params is not None else program.param_defaults())]
    if len(pvals) != len(program.params):
        raise ValueError("wrong number of params")

    op_arr = np.asarray(program.code, dtype=np.uint64)
    ops = [(int(c) >> 24, int(c) & 0xFFFFFF) for c in op_arr]
    consts = [F(c) for c in program.consts]
    state = np.zeros(program.state_floats, dtype=np.float32)
    locals_ = np.zeros(max(program.n_locals, 1), dtype=np.float32)
    scap = program.state_cap
    soff = program.state_off
    saux = program.state_aux

    o, h, l, c, v = bars.open, bars.high, bars.low, bars.close, bars.volume
    T = len(c)

    fee = F(cfg.fee_rate)
    slip = F(cfg.slip)

    # broker state
    cash = F(cfg.equity0)
    qty = F0                 # unsigned
    side = 0                 # -1 / 0 / +1
    entry_px = F0
    entry_t = -1
    anchor = F0              # trail anchor: peak (long) / trough (short)
    pend_action = 0          # 0 none, +1 enter long, -1 enter short, 2 exit
    pend_size = F1
    # configs persist from the previous bar's program run
    cfg_stop = F0
    cfg_tp = F0
    cfg_trail = F0
    cfg_size = F1

    equity_curve = np.zeros(T, dtype=np.float32)
    trades: list[Trade] = []
    exposure_bars = 0
    locals_curve = (np.zeros((T, program.n_locals), dtype=np.float32)
                    if record_locals and program.n_locals else None)

    # names bound locally for speed
    OPC = _OPN

    def qsigned() -> np.float32:
        return qty if side >= 0 else F(-qty)

    def open_pos(t: int, direction: int, px: np.float32) -> None:
        nonlocal cash, qty, side, entry_px, entry_t, anchor
        eq = cash  # flat, so equity == cash
        q = F(eq * pend_size / px) if px > F0 else F0
        if q <= F0:
            return
        side_ = direction
        cash = F(cash - F(side_) * q * px)
        cash = F(cash - q * px * fee)
        qty, side, entry_px, entry_t, anchor = q, side_, px, t, px

    def close_pos(t: int, px: np.float32, reason: str) -> None:
        nonlocal cash, qty, side, entry_px, entry_t, anchor
        cash = F(cash + F(side) * qty * px)
        fee_exit = F(qty * px * fee)
        cash = F(cash - fee_exit)
        gross = F(F(side) * qty * F(px - entry_px))
        fee_entry = F(qty * entry_px * fee)
        trades.append(Trade(side, entry_t, t, float(entry_px), float(px),
                            float(qty), float(F(gross - F(fee_entry + fee_exit))), reason))
        qty, side, entry_px, entry_t, anchor = F0, 0, F0, -1, F0

    for t in range(T):
        # 1. pending orders fill at open[t]
        if pend_action == 2 and side != 0:
            px = F(o[t] * F(F1 - slip)) if side > 0 else F(o[t] * F(F1 + slip))
            close_pos(t, px, "signal")
        elif pend_action in (1, -1) and side == 0:
            px = F(o[t] * F(F1 + slip)) if pend_action > 0 else F(o[t] * F(F1 - slip))
            open_pos(t, pend_action, px)
        pend_action = 0

        # 2. protective exits, worst-first: stop, trail, tp
        if side > 0:
            if cfg_stop > F0:
                trig = F(entry_px * F(F1 - cfg_stop))
                if l[t] <= trig:
                    close_pos(t, F(min(o[t], trig) * F(F1 - slip)), "stop")
            if side > 0 and cfg_trail > F0:
                trig = F(anchor * F(F1 - cfg_trail))
                if l[t] <= trig:
                    close_pos(t, F(min(o[t], trig) * F(F1 - slip)), "trail")
            if side > 0 and cfg_tp > F0:
                trig = F(entry_px * F(F1 + cfg_tp))
                if h[t] >= trig:
                    close_pos(t, F(max(o[t], trig) * F(F1 - slip)), "tp")
        elif side < 0:
            if cfg_stop > F0:
                trig = F(entry_px * F(F1 + cfg_stop))
                if h[t] >= trig:
                    close_pos(t, F(max(o[t], trig) * F(F1 + slip)), "stop")
            if side < 0 and cfg_trail > F0:
                trig = F(anchor * F(F1 + cfg_trail))
                if h[t] >= trig:
                    close_pos(t, F(max(o[t], trig) * F(F1 + slip)), "trail")
            if side < 0 and cfg_tp > F0:
                trig = F(entry_px * F(F1 - cfg_tp))
                if l[t] <= trig:
                    close_pos(t, F(min(o[t], trig) * F(F1 + slip)), "tp")

        # 3. run the bar program
        eq_now = F(cash + qsigned() * c[t])
        stack: list[np.float32] = []
        push = stack.append
        pop = stack.pop
        sig_el = sig_xl = sig_es = sig_xs = False
        # config defaults re-established each bar; omitted `set` keeps defaults
        cfg_stop, cfg_tp, cfg_trail, cfg_size = F0, F0, F0, F1

        for opcode, arg in ops:
            name = OPC[opcode]
            if name == "PUSH_CONST":
                push(consts[arg])
            elif name == "PUSH_PARAM":
                push(pvals[arg])
            elif name == "PUSH_SERIES":
                push((o, h, l, c, v)[arg][t])
            elif name == "PUSH_SERIES_LAG":
                sid, lag = arg >> 16, arg & 0xFFFF
                push((o, h, l, c, v)[sid][max(t - lag, 0)])
            elif name == "PUSH_CTX":
                push((F(t), F(side), entry_px, eq_now)[arg])
            elif name == "LOAD":
                push(locals_[arg])
            elif name == "STORE":
                locals_[arg] = pop()
            elif name == "ADD":
                b = pop(); push(F(pop() + b))
            elif name == "SUB":
                b = pop(); push(F(pop() - b))
            elif name == "MUL":
                b = pop(); push(F(pop() * b))
            elif name == "DIV":
                b = pop(); a = pop()
                push(F(a / b) if b != F0 else F0)
            elif name == "NEG":
                push(F(-pop()))
            elif name == "ABS":
                push(F(abs(pop())))
            elif name == "MIN2":
                b = pop(); push(F(min(pop(), b)))
            elif name == "MAX2":
                b = pop(); push(F(max(pop(), b)))
            elif name == "SQRT":
                a = pop()
                push(F(np.sqrt(a)) if a > F0 else F0)
            elif name == "LOG":
                a = pop()
                push(F(np.log(a)) if a > F0 else F0)
            elif name in ("GT", "LT", "GE", "LE", "EQ", "NE"):
                b = pop(); a = pop()
                r = {"GT": a > b, "LT": a < b, "GE": a >= b,
                     "LE": a <= b, "EQ": a == b, "NE": a != b}[name]
                push(F1 if r else F0)
            elif name == "AND":
                b = pop(); a = pop()
                push(F1 if (_truthy(a) and _truthy(b)) else F0)
            elif name == "OR":
                b = pop(); a = pop()
                push(F1 if (_truthy(a) or _truthy(b)) else F0)
            elif name == "NOT":
                push(F1 if pop() == F0 else F0)
            elif name == "EMA":
                n = pop(); x = pop()
                off = soff[arg]
                n_eff = max(int(n), 1)
                if state[off] == F0:
                    state[off + 1] = x
                else:
                    alpha = F(F(2.0) / F(n_eff + 1))
                    state[off + 1] = F(state[off + 1] + F(alpha * F(x - state[off + 1])))
                state[off] = F(state[off] + F1)
                push(state[off + 1])
            elif name == "RSI":
                n = pop(); x = pop()
                off = soff[arg]
                n_eff = max(int(n), 1)
                count = int(state[off])
                if count == 0:
                    res = F(50.0)
                else:
                    delta = F(x - state[off + 1])
                    g = F(max(delta, F0))
                    lo_ = F(max(F(-delta), F0))
                    if count == 1:
                        state[off + 2] = g
                        state[off + 3] = lo_
                    else:
                        state[off + 2] = F(state[off + 2] + F(F(g - state[off + 2]) / F(n_eff)))
                        state[off + 3] = F(state[off + 3] + F(F(lo_ - state[off + 3]) / F(n_eff)))
                    denom = F(state[off + 2] + state[off + 3])
                    res = F(50.0) if denom == F0 else F(F(100.0) * F(state[off + 2] / denom))
                state[off + 1] = x
                state[off] = F(state[off] + F1)
                push(res)
            elif name == "ATR":
                n = pop()
                off = soff[arg]
                n_eff = max(int(n), 1)
                if t == 0:
                    tr = F(h[t] - l[t])
                else:
                    pc = c[t - 1]
                    tr = F(max(F(h[t] - l[t]), max(F(abs(F(h[t] - pc))), F(abs(F(l[t] - pc))))))
                if state[off] == F0:
                    state[off + 1] = tr
                else:
                    state[off + 1] = F(state[off + 1] + F(F(tr - state[off + 1]) / F(n_eff)))
                state[off] = F(state[off] + F1)
                push(state[off + 1])
            elif name in ("SMA", "HIGHEST", "LOWEST", "STDDEV"):
                n = pop(); x = pop()
                off, cap = soff[arg], scap[arg]
                n_eff = min(max(int(n), 1), cap)
                count = int(state[off])
                state[off + 1 + count % cap] = x
                count += 1
                state[off] = F(count)
                k = min(count, n_eff)
                base = count - k
                if name == "SMA":
                    acc = F0
                    for i in range(k):
                        acc = F(acc + state[off + 1 + (base + i) % cap])
                    push(F(acc / F(k)))
                elif name == "HIGHEST":
                    acc = state[off + 1 + base % cap]
                    for i in range(1, k):
                        acc = F(max(acc, state[off + 1 + (base + i) % cap]))
                    push(acc)
                elif name == "LOWEST":
                    acc = state[off + 1 + base % cap]
                    for i in range(1, k):
                        acc = F(min(acc, state[off + 1 + (base + i) % cap]))
                    push(acc)
                else:  # STDDEV: two-pass for precision
                    acc = F0
                    for i in range(k):
                        acc = F(acc + state[off + 1 + (base + i) % cap])
                    mean = F(acc / F(k))
                    ss = F0
                    for i in range(k):
                        d = F(state[off + 1 + (base + i) % cap] - mean)
                        ss = F(ss + F(d * d))
                    var = F(ss / F(k))
                    push(F(np.sqrt(var)) if var > F0 else F0)
            elif name in ("SMA_RAW", "HIGHEST_RAW", "LOWEST_RAW", "STDDEV_RAW"):
                # bit-identical to the ring version, but the window contents
                # come straight from the (possibly lagged) series
                n = pop()
                cap = scap[arg]
                sid, lag = saux[arg] >> 16, saux[arg] & 0xFFFF
                s = (o, h, l, c, v)[sid]
                n_eff = min(max(int(n), 1), cap)
                k = min(t + 1, n_eff)
                xs = [s[max(j - lag, 0)] for j in range(t - k + 1, t + 1)]
                if name == "SMA_RAW":
                    acc = F0
                    for xi in xs:
                        acc = F(acc + xi)
                    push(F(acc / F(k)))
                elif name == "HIGHEST_RAW":
                    acc = xs[0]
                    for xi in xs[1:]:
                        acc = F(max(acc, xi))
                    push(acc)
                elif name == "LOWEST_RAW":
                    acc = xs[0]
                    for xi in xs[1:]:
                        acc = F(min(acc, xi))
                    push(acc)
                else:
                    acc = F0
                    for xi in xs:
                        acc = F(acc + xi)
                    mean = F(acc / F(k))
                    ss = F0
                    for xi in xs:
                        d = F(xi - mean)
                        ss = F(ss + F(d * d))
                    var = F(ss / F(k))
                    push(F(np.sqrt(var)) if var > F0 else F0)
            elif name == "DELAY_RAW":
                kk = pop()
                cap = scap[arg]
                sid, lag = saux[arg] >> 16, saux[arg] & 0xFFFF
                s = (o, h, l, c, v)[sid]
                k_eff = min(max(int(kk), 0), cap - 1)
                avail = min(t + 1, cap)
                back = min(k_eff, avail - 1)
                push(s[max(t - back - lag, 0)])
            elif name == "DELAY":
                kk = pop(); x = pop()
                off, cap = soff[arg], scap[arg]
                count = int(state[off])
                state[off + 1 + count % cap] = x
                count += 1
                state[off] = F(count)
                k_eff = min(max(int(kk), 0), cap - 1)
                avail = min(count, cap)
                back = min(k_eff, avail - 1)
                push(state[off + 1 + (count - 1 - back) % cap])
            elif name in ("CROSSOVER", "CROSSUNDER"):
                b = pop(); a = pop()
                off = soff[arg]
                count = int(state[off])
                if name == "CROSSOVER":
                    r = count >= 1 and a > b and state[off + 1] <= state[off + 2]
                else:
                    r = count >= 1 and a < b and state[off + 1] >= state[off + 2]
                state[off + 1] = a
                state[off + 2] = b
                state[off] = F(count + 1)
                push(F1 if r else F0)
            elif name == "SIG_EL":
                sig_el = sig_el or _truthy(pop())
            elif name == "SIG_XL":
                sig_xl = sig_xl or _truthy(pop())
            elif name == "SIG_ES":
                sig_es = sig_es or _truthy(pop())
            elif name == "SIG_XS":
                sig_xs = sig_xs or _truthy(pop())
            elif name == "SET_STOP":
                cfg_stop = pop()
            elif name == "SET_TP":
                cfg_tp = pop()
            elif name == "SET_TRAIL":
                cfg_trail = pop()
            elif name == "SET_SIZE":
                cfg_size = F(min(max(pop(), F0), F1))
            elif name == "HALT":
                break
            else:
                raise RuntimeError(f"bad opcode {name}")

        if locals_curve is not None:
            locals_curve[t, :] = locals_[:program.n_locals]

        # signals -> pending orders for bar t+1
        if side > 0 and sig_xl:
            pend_action = 2
        elif side < 0 and sig_xs:
            pend_action = 2
        elif side == 0:
            if sig_el and not sig_es:
                pend_action = 1
                pend_size = cfg_size
            elif sig_es and not sig_el:
                pend_action = -1
                pend_size = cfg_size

        # 4. mark to market, update trail anchor, record
        equity_curve[t] = F(cash + qsigned() * c[t])
        if side > 0:
            anchor = F(max(anchor, h[t]))
            exposure_bars += 1
        elif side < 0:
            anchor = F(min(anchor, l[t]))
            exposure_bars += 1

    # force-close at the last bar's close
    if side != 0:
        px = F(c[T - 1] * F(F1 - slip)) if side > 0 else F(c[T - 1] * F(F1 + slip))
        close_pos(T - 1, px, "eod")
        equity_curve[T - 1] = cash

    return BacktestResult(equity_curve, trades,
                          compute_metrics(equity_curve, trades, exposure_bars, cfg),
                          locals_curve)


def compute_metrics(equity: np.ndarray, trades: list[Trade],
                    exposure_bars: int, cfg: BacktestConfig) -> dict:
    """Streaming float32 Welford over per-bar simple returns (matches engines)."""
    T = len(equity)
    count = 0
    mean = F0
    m2 = F0
    peak = equity[0]
    max_dd = F0
    for t in range(1, T):
        prev = equity[t - 1]
        r = F(F(equity[t] - prev) / prev) if prev != F0 else F0
        count += 1
        d = F(r - mean)
        mean = F(mean + F(d / F(count)))
        m2 = F(m2 + F(d * F(r - mean)))
        if equity[t] > peak:
            peak = equity[t]
        if peak > F0:
            dd = F(F(peak - equity[t]) / peak)
            if dd > max_dd:
                max_dd = dd
    var = float(m2) / count if count > 1 else 0.0
    sharpe = 0.0
    if var > 0:
        sharpe = float(mean) / (var ** 0.5) * (cfg.bars_per_year ** 0.5)
    wins = sum(1 for tr in trades if tr.pnl > 0)
    return {
        "total_return": float(equity[T - 1] / equity[0] - 1.0) if equity[0] else 0.0,
        "sharpe": sharpe,
        "max_drawdown": float(max_dd),
        "n_trades": len(trades),
        "win_rate": wins / len(trades) if trades else 0.0,
        "exposure": exposure_bars / T if T else 0.0,
        "final_equity": float(equity[T - 1]),
    }
