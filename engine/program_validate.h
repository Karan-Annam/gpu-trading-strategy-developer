// Host-side validation shared by the CPU and CUDA Python extension boundaries.
#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "cpu/opcodes.h"

namespace bt {

inline int required_stack(uint32_t op) {
    switch (op) {
    case OP_PUSH_CONST: case OP_PUSH_PARAM: case OP_PUSH_SERIES:
    case OP_PUSH_SERIES_LAG: case OP_PUSH_CTX: case OP_LOAD: case OP_HALT:
        return 0;
    case OP_STORE: case OP_NEG: case OP_ABS: case OP_SQRT: case OP_LOG:
    case OP_ATR: case OP_SMA_RAW: case OP_HIGHEST_RAW: case OP_LOWEST_RAW:
    case OP_STDDEV_RAW: case OP_DELAY_RAW: case OP_SIG_EL: case OP_SIG_XL:
    case OP_SIG_ES: case OP_SIG_XS: case OP_SET_STOP: case OP_SET_TP:
    case OP_SET_TRAIL: case OP_SET_SIZE: case OP_NOT:
        return 1;
    default:
        return 2;
    }
}

inline int stack_delta(uint32_t op) {
    switch (op) {
    case OP_PUSH_CONST: case OP_PUSH_PARAM: case OP_PUSH_SERIES:
    case OP_PUSH_SERIES_LAG: case OP_PUSH_CTX: case OP_LOAD:
        return 1;
    case OP_STORE: case OP_SIG_EL: case OP_SIG_XL: case OP_SIG_ES:
    case OP_SIG_XS: case OP_SET_STOP: case OP_SET_TP: case OP_SET_TRAIL:
    case OP_SET_SIZE:
        return -1;
    case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV: case OP_MIN2:
    case OP_MAX2: case OP_GT: case OP_LT: case OP_GE: case OP_LE: case OP_EQ:
    case OP_NE: case OP_AND: case OP_OR: case OP_SMA: case OP_EMA: case OP_RSI:
    case OP_HIGHEST: case OP_LOWEST: case OP_STDDEV: case OP_DELAY:
    case OP_CROSSOVER: case OP_CROSSUNDER:
        return -1;
    default:
        return 0;
    }
}

inline int expected_kind(uint32_t op) {
    switch (op) {
    case OP_EMA: return SK_EMA;
    case OP_ATR: return SK_WILDER1;
    case OP_RSI: return SK_WILDER2;
    case OP_SMA: case OP_HIGHEST: case OP_LOWEST: case OP_STDDEV:
    case OP_DELAY: return SK_RING;
    case OP_CROSSOVER: case OP_CROSSUNDER: return SK_PREV2;
    case OP_SMA_RAW: case OP_HIGHEST_RAW: case OP_LOWEST_RAW:
    case OP_STDDEV_RAW: case OP_DELAY_RAW: return SK_RAW;
    default: return -1;
    }
}

inline int state_size(int kind, int cap) {
    switch (kind) {
    case SK_EMA: case SK_WILDER1: return 2;
    case SK_WILDER2: return 4;
    case SK_PREV2: return 3;
    case SK_RING:
        if (cap < 1 || cap > 4096) throw std::runtime_error("invalid ring state capacity");
        return 1 + cap;
    case SK_RAW:
        if (cap < 1 || cap > 4096) throw std::runtime_error("invalid raw state capacity");
        return 0;
    default: throw std::runtime_error("invalid state kind");
    }
}

inline void validate_program(const uint32_t* code, int n_code,
                             const float* consts, int n_consts,
                             const int32_t* kind, const int32_t* off,
                             const int32_t* cap, const int32_t* aux, int n_state,
                             int n_locals, int state_floats, int n_params) {
    if (n_code < 1 || n_code > 6144 || n_consts < 0 || n_consts > 6144 ||
        n_state < 0 || n_state > 6144 || n_locals < 0 || n_locals > 6144 ||
        state_floats < 0 || state_floats > 8192 ||
        n_params < 0 || n_params > 16)
        throw std::runtime_error("invalid program resource counts");
    if ((code[n_code - 1] >> 24) != OP_HALT)
        throw std::runtime_error("program does not end in HALT");
    for (int i = 0; i < n_consts; ++i)
        if (!std::isfinite(consts[i])) throw std::runtime_error("non-finite program constant");

    std::vector<uint8_t> occupied(static_cast<size_t>(state_floats));
    for (int i = 0; i < n_state; ++i) {
        int size = state_size(kind[i], cap[i]);
        if (off[i] < 0 || off[i] > state_floats - size)
            throw std::runtime_error("state slot is out of bounds");
        for (int j = 0; j < size; ++j) {
            if (occupied[static_cast<size_t>(off[i] + j)])
                throw std::runtime_error("state slots overlap");
            occupied[static_cast<size_t>(off[i] + j)] = 1;
        }
        if (kind[i] == SK_RAW && (static_cast<uint32_t>(aux[i]) >> 16) >= 5)
            throw std::runtime_error("raw state has invalid series");
    }

    int depth = 0;
    for (int i = 0; i < n_code; ++i) {
        uint32_t op = code[i] >> 24;
        uint32_t arg = code[i] & 0xFFFFFFu;
        if (op > OP_HALT) throw std::runtime_error("invalid opcode");
        if ((op == OP_PUSH_CONST && arg >= static_cast<uint32_t>(n_consts)) ||
            (op == OP_PUSH_PARAM && arg >= static_cast<uint32_t>(n_params)) ||
            (op == OP_PUSH_SERIES && arg >= 5) ||
            (op == OP_PUSH_SERIES_LAG && (arg >> 16) >= 5) ||
            (op == OP_PUSH_CTX && arg >= 4) ||
            ((op == OP_LOAD || op == OP_STORE) && arg >= static_cast<uint32_t>(n_locals)))
            throw std::runtime_error("bytecode operand is out of bounds");
        int wanted = expected_kind(op);
        if (wanted >= 0) {
            if (arg >= static_cast<uint32_t>(n_state) || kind[arg] != wanted)
                throw std::runtime_error("bytecode state operand is invalid");
        }
        if (op == OP_HALT && i + 1 != n_code)
            throw std::runtime_error("HALT must be the final instruction");
        int need = required_stack(op);
        if (depth < need) throw std::runtime_error("bytecode stack underflow");
        depth += stack_delta(op);
        if (depth > VM_MAX_STACK) throw std::runtime_error("bytecode stack overflow");
    }
    if (depth != 0) throw std::runtime_error("bytecode leaves a non-empty stack");
}

inline void validate_run_inputs(const float* o, const float* h, const float* l,
                                const float* c, const float* v, int n_bars,
                                const float* params, size_t n_param_values,
                                double fee, double slip, double equity0,
                                double bars_per_year) {
    if (!std::isfinite(static_cast<float>(fee)) || fee < 0.0 ||
        !std::isfinite(static_cast<float>(slip)) || slip < 0.0 || slip >= 1.0 ||
        !std::isfinite(static_cast<float>(equity0)) || equity0 <= 0.0 ||
        !std::isfinite(bars_per_year) || bars_per_year <= 0.0)
        throw std::runtime_error("invalid backtest configuration");
    for (int i = 0; i < n_bars; ++i) {
        if (!std::isfinite(o[i]) || !std::isfinite(h[i]) ||
            !std::isfinite(l[i]) || !std::isfinite(c[i]) ||
            !std::isfinite(v[i]) || l[i] <= 0.0f || v[i] < 0.0f ||
            h[i] < o[i] || h[i] < c[i] || l[i] > o[i] || l[i] > c[i])
            throw std::runtime_error("invalid OHLCV input");
    }
    for (size_t i = 0; i < n_param_values; ++i)
        if (!std::isfinite(params[i]))
            throw std::runtime_error("non-finite parameter input");
}

}  // namespace bt
