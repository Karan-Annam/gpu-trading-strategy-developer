// Shared VM + broker core. Compiled by MSVC for the CPU engine and by nvcc
// for the CUDA kernel (BT_HD expands to __host__ __device__ under nvcc), so
// both engines are the same code by construction. Must mirror
// dsl/refengine.py operation-for-operation in float32.
#pragma once

#include <cmath>
#include <cstdint>

#include "cpu/opcodes.h"

#ifdef __CUDACC__
#define BT_HD __host__ __device__ __forceinline__
#else
#define BT_HD inline
#endif

namespace bt {

struct ProgramView {
    const uint32_t* code;
    int n_code;
    const float* consts;
    const int32_t* state_kind;   // unused at runtime, kept for debugging
    const int32_t* state_off;
    const int32_t* state_cap;
    const int32_t* state_aux;    // SK_RAW: sid << 16 | lag
    int n_state;
    int n_locals;
    int n_params;
    int state_floats;
};

struct BarsView {
    const float* o;
    const float* h;
    const float* l;
    const float* c;
    const float* v;
    int T;
};

struct RunConfig {
    float fee;
    float slip;
    float equity0;
    double bars_per_year;
};

struct Metrics {
    float final_equity;
    float total_return;
    float sharpe;
    float max_dd;
    int n_trades;
    int wins;
    int exposure_bars;
};

// trade exit reasons
enum { RSN_SIGNAL = 0, RSN_STOP = 1, RSN_TRAIL = 2, RSN_TP = 3, RSN_EOD = 4 };

struct TradeRec {
    int side;
    int entry_t;
    int exit_t;
    float entry_px;
    float exit_px;
    float qty;
    float pnl;
    int reason;
};

// Strided float view: CPU uses stride 1; the GPU lays per-thread state out
// column-major across the launch (index j lives at base[j * stride]) so ring
// scans coalesce across a warp.
struct StateRef {
    float* base;
    size_t stride;
    BT_HD float& operator[](int i) const { return base[(size_t)i * stride]; }
    BT_HD StateRef at(int off) const { return StateRef{base + (size_t)off * stride, stride}; }
};

// Recorder policies: CPU records trades/curves, GPU uses NullSink.
struct NullSink {
    BT_HD void trade(const TradeRec&) {}
    BT_HD void equity(int, float) {}
    BT_HD void locals(int, const StateRef&, int) {}
};

// ---- broker state ----
struct Broker {
    float cash;
    float qty;        // unsigned
    int side;         // -1/0/+1
    float entry_px;
    int entry_t;
    float anchor;     // trail peak (long) / trough (short)
    int pend_action;  // 0 none, +1 long, -1 short, 2 exit
    float pend_size;
    float cfg_stop, cfg_tp, cfg_trail, cfg_size;
    int n_trades;
    int wins;
};

template <typename Sink>
BT_HD void close_pos(Broker& b, Sink& sink, int t, float px, int reason, float fee) {
    b.cash = b.cash + (float)b.side * b.qty * px;
    float fee_exit = b.qty * px * fee;
    b.cash = b.cash - fee_exit;
    float gross = (float)b.side * b.qty * (px - b.entry_px);
    float fee_entry = b.qty * b.entry_px * fee;
    float pnl = gross - (fee_entry + fee_exit);
    TradeRec tr{b.side, b.entry_t, t, b.entry_px, px, b.qty, pnl, reason};
    sink.trade(tr);
    b.n_trades += 1;
    if (pnl > 0.0f) b.wins += 1;
    b.qty = 0.0f;
    b.side = 0;
    b.entry_px = 0.0f;
    b.entry_t = -1;
    b.anchor = 0.0f;
}

BT_HD void open_pos(Broker& b, int t, int direction, float px, float fee) {
    float eq = b.cash;  // flat, so equity == cash
    if (!(px > 0.0f)) return;
    float q = eq * b.pend_size / px;
    if (!(q > 0.0f)) return;
    b.cash = b.cash - (float)direction * q * px;
    b.cash = b.cash - q * px * fee;
    b.qty = q;
    b.side = direction;
    b.entry_px = px;
    b.entry_t = t;
    b.anchor = px;
}

// ---- full backtest over one (program, param-set). ----
// state:  state_floats floats, zero-initialized by caller
// locals: n_locals floats (>=1), zero-initialized by caller
template <typename Sink>
BT_HD Metrics run_backtest(const ProgramView& prog, const BarsView& bars,
                           const float* params, const RunConfig& cfg,
                           StateRef state, StateRef locals, Sink& sink) {
    const float fee = cfg.fee;
    const float slip = cfg.slip;
    const int T = bars.T;

    Broker b;
    b.cash = cfg.equity0;
    b.qty = 0.0f; b.side = 0; b.entry_px = 0.0f; b.entry_t = -1; b.anchor = 0.0f;
    b.pend_action = 0; b.pend_size = 1.0f;
    b.cfg_stop = 0.0f; b.cfg_tp = 0.0f; b.cfg_trail = 0.0f; b.cfg_size = 1.0f;
    b.n_trades = 0; b.wins = 0;

    // streaming metrics (must match refengine.compute_metrics order)
    float prev_eq = 0.0f;
    int r_count = 0;
    float r_mean = 0.0f, r_m2 = 0.0f;
    float peak = 0.0f, max_dd = 0.0f;
    int exposure_bars = 0;
    float first_eq = 0.0f, last_eq = 0.0f;

    float stack[VM_MAX_STACK];

    for (int t = 0; t < T; ++t) {
        // 1. pending orders fill at open[t]
        if (b.pend_action == 2 && b.side != 0) {
            float px = (b.side > 0) ? bars.o[t] * (1.0f - slip) : bars.o[t] * (1.0f + slip);
            close_pos(b, sink, t, px, RSN_SIGNAL, fee);
        } else if ((b.pend_action == 1 || b.pend_action == -1) && b.side == 0) {
            float px = (b.pend_action > 0) ? bars.o[t] * (1.0f + slip)
                                           : bars.o[t] * (1.0f - slip);
            open_pos(b, t, b.pend_action, px, fee);
        }
        b.pend_action = 0;

        // 2. protective exits: stop, trail, tp (worst first)
        if (b.side > 0) {
            if (b.cfg_stop > 0.0f) {
                float trig = b.entry_px * (1.0f - b.cfg_stop);
                if (bars.l[t] <= trig) {
                    float fill = (bars.o[t] < trig ? bars.o[t] : trig) * (1.0f - slip);
                    close_pos(b, sink, t, fill, RSN_STOP, fee);
                }
            }
            if (b.side > 0 && b.cfg_trail > 0.0f) {
                float trig = b.anchor * (1.0f - b.cfg_trail);
                if (bars.l[t] <= trig) {
                    float fill = (bars.o[t] < trig ? bars.o[t] : trig) * (1.0f - slip);
                    close_pos(b, sink, t, fill, RSN_TRAIL, fee);
                }
            }
            if (b.side > 0 && b.cfg_tp > 0.0f) {
                float trig = b.entry_px * (1.0f + b.cfg_tp);
                if (bars.h[t] >= trig) {
                    float fill = (bars.o[t] > trig ? bars.o[t] : trig) * (1.0f - slip);
                    close_pos(b, sink, t, fill, RSN_TP, fee);
                }
            }
        } else if (b.side < 0) {
            if (b.cfg_stop > 0.0f) {
                float trig = b.entry_px * (1.0f + b.cfg_stop);
                if (bars.h[t] >= trig) {
                    float fill = (bars.o[t] > trig ? bars.o[t] : trig) * (1.0f + slip);
                    close_pos(b, sink, t, fill, RSN_STOP, fee);
                }
            }
            if (b.side < 0 && b.cfg_trail > 0.0f) {
                float trig = b.anchor * (1.0f + b.cfg_trail);
                if (bars.h[t] >= trig) {
                    float fill = (bars.o[t] > trig ? bars.o[t] : trig) * (1.0f + slip);
                    close_pos(b, sink, t, fill, RSN_TRAIL, fee);
                }
            }
            if (b.side < 0 && b.cfg_tp > 0.0f) {
                float trig = b.entry_px * (1.0f - b.cfg_tp);
                if (bars.l[t] <= trig) {
                    float fill = (bars.o[t] < trig ? bars.o[t] : trig) * (1.0f + slip);
                    close_pos(b, sink, t, fill, RSN_TP, fee);
                }
            }
        }

        // 3. run the bar program
        float qsig = (b.side >= 0) ? b.qty : -b.qty;
        float eq_now = b.cash + qsig * bars.c[t];
        float bars_held = (b.side != 0) ? (float)(t - b.entry_t) : 0.0f;
        int sp = 0;
        bool sig_el = false, sig_xl = false, sig_es = false, sig_xs = false;
        b.cfg_stop = 0.0f; b.cfg_tp = 0.0f; b.cfg_trail = 0.0f; b.cfg_size = 1.0f;

        for (int pc = 0; pc < prog.n_code; ++pc) {
            uint32_t ins = prog.code[pc];
            int op = (int)(ins >> 24);
            int arg = (int)(ins & 0xFFFFFF);
            switch (op) {
            case OP_PUSH_CONST:  stack[sp++] = prog.consts[arg]; break;
            case OP_PUSH_PARAM:  stack[sp++] = params[arg]; break;
            case OP_PUSH_SERIES: {
                const float* s = (arg == 0) ? bars.o : (arg == 1) ? bars.h
                                : (arg == 2) ? bars.l : (arg == 3) ? bars.c : bars.v;
                stack[sp++] = s[t];
                break;
            }
            case OP_PUSH_SERIES_LAG: {
                int sid = arg >> 16, lag = arg & 0xFFFF;
                const float* s = (sid == 0) ? bars.o : (sid == 1) ? bars.h
                                : (sid == 2) ? bars.l : (sid == 3) ? bars.c : bars.v;
                int idx = t - lag;
                stack[sp++] = s[idx > 0 ? idx : 0];
                break;
            }
            case OP_PUSH_CTX:
                stack[sp++] = (arg == 0) ? (float)t : (arg == 1) ? (float)b.side
                             : (arg == 2) ? b.entry_px : (arg == 3) ? eq_now
                             : bars_held;
                break;
            case OP_LOAD:  stack[sp++] = locals[arg]; break;
            case OP_STORE: locals[arg] = stack[--sp]; break;
            case OP_ADD: { float y = stack[--sp]; stack[sp - 1] = stack[sp - 1] + y; break; }
            case OP_SUB: { float y = stack[--sp]; stack[sp - 1] = stack[sp - 1] - y; break; }
            case OP_MUL: { float y = stack[--sp]; stack[sp - 1] = stack[sp - 1] * y; break; }
            case OP_DIV: {
                float y = stack[--sp];
                stack[sp - 1] = (y != 0.0f) ? stack[sp - 1] / y : 0.0f;
                break;
            }
            case OP_NEG: stack[sp - 1] = -stack[sp - 1]; break;
            case OP_ABS: stack[sp - 1] = fabsf(stack[sp - 1]); break;
            case OP_MIN2: { float y = stack[--sp]; float x = stack[sp - 1];
                            stack[sp - 1] = (x < y) ? x : y; break; }
            case OP_MAX2: { float y = stack[--sp]; float x = stack[sp - 1];
                            stack[sp - 1] = (x > y) ? x : y; break; }
            case OP_SQRT: { float x = stack[sp - 1];
                            stack[sp - 1] = (x > 0.0f) ? sqrtf(x) : 0.0f; break; }
            case OP_LOG:  { float x = stack[sp - 1];
                            stack[sp - 1] = (x > 0.0f) ? logf(x) : 0.0f; break; }
            case OP_GT: { float y = stack[--sp]; stack[sp - 1] = (stack[sp - 1] > y) ? 1.0f : 0.0f; break; }
            case OP_LT: { float y = stack[--sp]; stack[sp - 1] = (stack[sp - 1] < y) ? 1.0f : 0.0f; break; }
            case OP_GE: { float y = stack[--sp]; stack[sp - 1] = (stack[sp - 1] >= y) ? 1.0f : 0.0f; break; }
            case OP_LE: { float y = stack[--sp]; stack[sp - 1] = (stack[sp - 1] <= y) ? 1.0f : 0.0f; break; }
            case OP_EQ: { float y = stack[--sp]; stack[sp - 1] = (stack[sp - 1] == y) ? 1.0f : 0.0f; break; }
            case OP_NE: { float y = stack[--sp]; stack[sp - 1] = (stack[sp - 1] != y) ? 1.0f : 0.0f; break; }
            case OP_AND: { float y = stack[--sp];
                           stack[sp - 1] = (stack[sp - 1] != 0.0f && y != 0.0f) ? 1.0f : 0.0f; break; }
            case OP_OR:  { float y = stack[--sp];
                           stack[sp - 1] = (stack[sp - 1] != 0.0f || y != 0.0f) ? 1.0f : 0.0f; break; }
            case OP_NOT: stack[sp - 1] = (stack[sp - 1] == 0.0f) ? 1.0f : 0.0f; break;

            case OP_EMA: {
                float n = stack[--sp];
                float x = stack[--sp];
                StateRef st = state.at(prog.state_off[arg]);   // [count, value]
                int n_eff = (int)n; if (n_eff < 1) n_eff = 1;
                if (st[0] == 0.0f) {
                    st[1] = x;
                } else {
                    float alpha = 2.0f / (float)(n_eff + 1);
                    st[1] = st[1] + alpha * (x - st[1]);
                }
                st[0] = st[0] + 1.0f;
                stack[sp++] = st[1];
                break;
            }
            case OP_RSI: {
                float n = stack[--sp];
                float x = stack[--sp];
                StateRef st = state.at(prog.state_off[arg]);   // [count, prev_x, ag, al]
                int n_eff = (int)n; if (n_eff < 1) n_eff = 1;
                int count = (int)st[0];
                float res;
                if (count == 0) {
                    res = 50.0f;
                } else {
                    float delta = x - st[1];
                    float g = (delta > 0.0f) ? delta : 0.0f;
                    float lo = (-delta > 0.0f) ? -delta : 0.0f;
                    if (count == 1) {
                        st[2] = g;
                        st[3] = lo;
                    } else {
                        st[2] = st[2] + (g - st[2]) / (float)n_eff;
                        st[3] = st[3] + (lo - st[3]) / (float)n_eff;
                    }
                    float denom = st[2] + st[3];
                    res = (denom == 0.0f) ? 50.0f : 100.0f * (st[2] / denom);
                }
                st[1] = x;
                st[0] = (float)(count + 1);
                stack[sp++] = res;
                break;
            }
            case OP_ATR: {
                float n = stack[--sp];
                StateRef st = state.at(prog.state_off[arg]);   // [count, value]
                int n_eff = (int)n; if (n_eff < 1) n_eff = 1;
                float tr;
                if (t == 0) {
                    tr = bars.h[t] - bars.l[t];
                } else {
                    float pc_ = bars.c[t - 1];
                    float a = bars.h[t] - bars.l[t];
                    float b1 = fabsf(bars.h[t] - pc_);
                    float c1 = fabsf(bars.l[t] - pc_);
                    float m = (b1 > c1) ? b1 : c1;
                    tr = (a > m) ? a : m;
                }
                if (st[0] == 0.0f) {
                    st[1] = tr;
                } else {
                    st[1] = st[1] + (tr - st[1]) / (float)n_eff;
                }
                st[0] = st[0] + 1.0f;
                stack[sp++] = st[1];
                break;
            }
            case OP_SMA: case OP_HIGHEST: case OP_LOWEST: case OP_STDDEV: {
                float n = stack[--sp];
                float x = stack[--sp];
                StateRef st = state.at(prog.state_off[arg]);   // [count, ring...]
                int cap = prog.state_cap[arg];
                int n_eff = (int)n;
                if (n_eff < 1) n_eff = 1;
                if (n_eff > cap) n_eff = cap;
                int count = (int)st[0];
                st[1 + count % cap] = x;
                count += 1;
                st[0] = (float)count;
                int k = (count < n_eff) ? count : n_eff;
                int base = count - k;
                if (op == OP_SMA) {
                    float acc = 0.0f;
                    for (int i = 0; i < k; ++i) acc = acc + st[1 + (base + i) % cap];
                    stack[sp++] = acc / (float)k;
                } else if (op == OP_HIGHEST) {
                    float acc = st[1 + base % cap];
                    for (int i = 1; i < k; ++i) {
                        float xi = st[1 + (base + i) % cap];
                        if (xi > acc) acc = xi;
                    }
                    stack[sp++] = acc;
                } else if (op == OP_LOWEST) {
                    float acc = st[1 + base % cap];
                    for (int i = 1; i < k; ++i) {
                        float xi = st[1 + (base + i) % cap];
                        if (xi < acc) acc = xi;
                    }
                    stack[sp++] = acc;
                } else {  // STDDEV, two-pass
                    float acc = 0.0f;
                    for (int i = 0; i < k; ++i) acc = acc + st[1 + (base + i) % cap];
                    float mean = acc / (float)k;
                    float ss = 0.0f;
                    for (int i = 0; i < k; ++i) {
                        float d = st[1 + (base + i) % cap] - mean;
                        ss = ss + d * d;
                    }
                    float var = ss / (float)k;
                    stack[sp++] = (var > 0.0f) ? sqrtf(var) : 0.0f;
                }
                break;
            }
            case OP_SMA_RAW: case OP_HIGHEST_RAW: case OP_LOWEST_RAW:
            case OP_STDDEV_RAW: {
                // window contents read straight from the (lagged) series;
                // bit-identical to the ring version, zero per-thread state
                float n = stack[--sp];
                int cap = prog.state_cap[arg];
                int aux = prog.state_aux[arg];
                int sid = aux >> 16, lag = aux & 0xFFFF;
                const float* s = (sid == 0) ? bars.o : (sid == 1) ? bars.h
                                : (sid == 2) ? bars.l : (sid == 3) ? bars.c : bars.v;
                int n_eff = (int)n;
                if (n_eff < 1) n_eff = 1;
                if (n_eff > cap) n_eff = cap;
                int k = (t + 1 < n_eff) ? t + 1 : n_eff;
                int j0 = t - k + 1;
                if (op == OP_SMA_RAW) {
                    float acc = 0.0f;
                    for (int j = j0; j <= t; ++j) {
                        int idx = j - lag;
                        acc = acc + s[idx > 0 ? idx : 0];
                    }
                    stack[sp++] = acc / (float)k;
                } else if (op == OP_HIGHEST_RAW) {
                    int idx0 = j0 - lag;
                    float acc = s[idx0 > 0 ? idx0 : 0];
                    for (int j = j0 + 1; j <= t; ++j) {
                        int idx = j - lag;
                        float xi = s[idx > 0 ? idx : 0];
                        if (xi > acc) acc = xi;
                    }
                    stack[sp++] = acc;
                } else if (op == OP_LOWEST_RAW) {
                    int idx0 = j0 - lag;
                    float acc = s[idx0 > 0 ? idx0 : 0];
                    for (int j = j0 + 1; j <= t; ++j) {
                        int idx = j - lag;
                        float xi = s[idx > 0 ? idx : 0];
                        if (xi < acc) acc = xi;
                    }
                    stack[sp++] = acc;
                } else {  // STDDEV_RAW, two-pass
                    float acc = 0.0f;
                    for (int j = j0; j <= t; ++j) {
                        int idx = j - lag;
                        acc = acc + s[idx > 0 ? idx : 0];
                    }
                    float mean = acc / (float)k;
                    float ss = 0.0f;
                    for (int j = j0; j <= t; ++j) {
                        int idx = j - lag;
                        float d = s[idx > 0 ? idx : 0] - mean;
                        ss = ss + d * d;
                    }
                    float var = ss / (float)k;
                    stack[sp++] = (var > 0.0f) ? sqrtf(var) : 0.0f;
                }
                break;
            }
            case OP_DELAY_RAW: {
                float kk = stack[--sp];
                int cap = prog.state_cap[arg];
                int aux = prog.state_aux[arg];
                int sid = aux >> 16, lag = aux & 0xFFFF;
                const float* s = (sid == 0) ? bars.o : (sid == 1) ? bars.h
                                : (sid == 2) ? bars.l : (sid == 3) ? bars.c : bars.v;
                int k_eff = (int)kk;
                if (k_eff < 0) k_eff = 0;
                if (k_eff > cap - 1) k_eff = cap - 1;
                int avail = (t + 1 < cap) ? t + 1 : cap;
                int back = (k_eff < avail - 1) ? k_eff : avail - 1;
                int idx = t - back - lag;
                stack[sp++] = s[idx > 0 ? idx : 0];
                break;
            }
            case OP_DELAY: {
                float kk = stack[--sp];
                float x = stack[--sp];
                StateRef st = state.at(prog.state_off[arg]);
                int cap = prog.state_cap[arg];
                int count = (int)st[0];
                st[1 + count % cap] = x;
                count += 1;
                st[0] = (float)count;
                int k_eff = (int)kk;
                if (k_eff < 0) k_eff = 0;
                if (k_eff > cap - 1) k_eff = cap - 1;
                int avail = (count < cap) ? count : cap;
                int back = (k_eff < avail - 1) ? k_eff : avail - 1;
                stack[sp++] = st[1 + (count - 1 - back) % cap];
                break;
            }
            case OP_CROSSOVER: case OP_CROSSUNDER: {
                float y = stack[--sp];
                float x = stack[--sp];
                StateRef st = state.at(prog.state_off[arg]);   // [count, pa, pb]
                int count = (int)st[0];
                bool r;
                if (op == OP_CROSSOVER)
                    r = count >= 1 && x > y && st[1] <= st[2];
                else
                    r = count >= 1 && x < y && st[1] >= st[2];
                st[1] = x;
                st[2] = y;
                st[0] = (float)(count + 1);
                stack[sp++] = r ? 1.0f : 0.0f;
                break;
            }
            case OP_SIG_EL: sig_el = sig_el || (stack[--sp] != 0.0f); break;
            case OP_SIG_XL: sig_xl = sig_xl || (stack[--sp] != 0.0f); break;
            case OP_SIG_ES: sig_es = sig_es || (stack[--sp] != 0.0f); break;
            case OP_SIG_XS: sig_xs = sig_xs || (stack[--sp] != 0.0f); break;
            case OP_SET_STOP:  b.cfg_stop = stack[--sp]; break;
            case OP_SET_TP:    b.cfg_tp = stack[--sp]; break;
            case OP_SET_TRAIL: b.cfg_trail = stack[--sp]; break;
            case OP_SET_SIZE: {
                float s = stack[--sp];
                if (s < 0.0f) s = 0.0f;
                if (s > 1.0f) s = 1.0f;
                b.cfg_size = s;
                break;
            }
            case OP_HALT: pc = prog.n_code; break;
            default: break;
            }
        }

        sink.locals(t, locals, prog.n_locals);

        // signals -> pending orders for bar t+1
        if (b.side > 0 && sig_xl) {
            b.pend_action = 2;
        } else if (b.side < 0 && sig_xs) {
            b.pend_action = 2;
        } else if (b.side == 0) {
            if (sig_el && !sig_es) { b.pend_action = 1;  b.pend_size = b.cfg_size; }
            else if (sig_es && !sig_el) { b.pend_action = -1; b.pend_size = b.cfg_size; }
        }

        // 4. mark to market, trail anchor, exposure
        qsig = (b.side >= 0) ? b.qty : -b.qty;
        float eq_t = b.cash + qsig * bars.c[t];
        if (b.side > 0) {
            b.anchor = (b.anchor > bars.h[t]) ? b.anchor : bars.h[t];
            exposure_bars += 1;
        } else if (b.side < 0) {
            b.anchor = (b.anchor < bars.l[t]) ? b.anchor : bars.l[t];
            exposure_bars += 1;
        }

        // force-close on the final bar (metrics stream sees post-close equity)
        if (t == T - 1 && b.side != 0) {
            float px = (b.side > 0) ? bars.c[t] * (1.0f - slip) : bars.c[t] * (1.0f + slip);
            close_pos(b, sink, t, px, RSN_EOD, fee);
            eq_t = b.cash;
        }
        sink.equity(t, eq_t);

        // streaming metrics (matches refengine.compute_metrics)
        if (t == 0) {
            first_eq = eq_t;
            peak = eq_t;
        } else {
            float r = (prev_eq != 0.0f) ? (eq_t - prev_eq) / prev_eq : 0.0f;
            r_count += 1;
            float d = r - r_mean;
            r_mean = r_mean + d / (float)r_count;
            r_m2 = r_m2 + d * (r - r_mean);
            if (eq_t > peak) peak = eq_t;
            if (peak > 0.0f) {
                float dd = (peak - eq_t) / peak;
                if (dd > max_dd) max_dd = dd;
            }
        }
        prev_eq = eq_t;
        last_eq = eq_t;
    }

    Metrics m;
    m.final_equity = last_eq;
    m.total_return = (first_eq != 0.0f) ? last_eq / first_eq - 1.0f : 0.0f;
    double var = (r_count > 1) ? (double)r_m2 / r_count : 0.0;
    m.sharpe = (var > 0.0)
        ? (float)((double)r_mean / sqrt(var) * sqrt(cfg.bars_per_year)) : 0.0f;
    m.max_dd = max_dd;
    m.n_trades = b.n_trades;
    m.wins = b.wins;
    m.exposure_bars = exposure_bars;
    return m;
}

}  // namespace bt
