// pybind11 module `btcpu`: single backtests (with trade/equity/locals capture)
// and OpenMP-parallel parameter-batch runs returning a metrics matrix.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

#include "../vm_core.h"

namespace py = pybind11;
using namespace bt;

using farr = py::array_t<float, py::array::c_style | py::array::forcecast>;
using u32arr = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;
using i32arr = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

namespace {

struct CpuSink {
    std::vector<TradeRec> trades;
    float* equity_out = nullptr;
    float* locals_out = nullptr;   // row-major T x n_locals
    void trade(const TradeRec& tr) { trades.push_back(tr); }
    void equity(int t, float eq) { if (equity_out) equity_out[t] = eq; }
    void locals(int t, const StateRef& loc, int n) {
        if (locals_out) for (int i = 0; i < n; ++i) locals_out[(size_t)t * n + i] = loc[i];
    }
};

ProgramView make_prog(const u32arr& code, const farr& consts, const i32arr& kind,
                      const i32arr& off, const i32arr& cap, const i32arr& aux,
                      int n_locals, int state_floats, int n_params) {
    ProgramView p;
    p.code = code.data();
    p.n_code = (int)code.size();
    p.consts = consts.data();
    p.state_kind = kind.data();
    p.state_off = off.data();
    p.state_cap = cap.data();
    p.state_aux = aux.data();
    p.n_state = (int)off.size();
    p.n_locals = n_locals;
    p.n_params = n_params;
    p.state_floats = state_floats;
    return p;
}

BarsView make_bars(const farr& o, const farr& h, const farr& l, const farr& c,
                   const farr& v) {
    BarsView b{o.data(), h.data(), l.data(), c.data(), v.data(), (int)c.size()};
    return b;
}

const char* reason_str(int r) {
    switch (r) {
    case RSN_SIGNAL: return "signal";
    case RSN_STOP: return "stop";
    case RSN_TRAIL: return "trail";
    case RSN_TP: return "tp";
    default: return "eod";
    }
}

}  // namespace

py::dict run_single(u32arr code, farr consts, i32arr kind, i32arr off, i32arr cap,
                    i32arr aux, int n_locals, int state_floats, farr o, farr h,
                    farr l, farr c, farr v, farr params, double fee, double slip,
                    double equity0, double bars_per_year, bool want_locals) {
    ProgramView prog = make_prog(code, consts, kind, off, cap, aux, n_locals,
                                 state_floats, (int)params.size());
    BarsView bars = make_bars(o, h, l, c, v);
    RunConfig cfg{(float)fee, (float)slip, (float)equity0, bars_per_year};

    py::array_t<float> equity(bars.T);
    py::array_t<float> locals_curve;
    CpuSink sink;
    sink.equity_out = equity.mutable_data();
    if (want_locals && n_locals > 0) {
        locals_curve = py::array_t<float>({(size_t)bars.T, (size_t)n_locals});
        sink.locals_out = locals_curve.mutable_data();
    }

    std::vector<float> state((size_t)std::max(state_floats, 1), 0.0f);
    std::vector<float> locals_((size_t)std::max(n_locals, 1), 0.0f);

    Metrics m;
    {
        py::gil_scoped_release rel;
        m = run_backtest(prog, bars, params.data(), cfg,
                         StateRef{state.data(), 1}, StateRef{locals_.data(), 1}, sink);
    }

    py::list trades;
    for (const auto& tr : sink.trades) {
        py::dict d;
        d["side"] = tr.side;
        d["entry_t"] = tr.entry_t;
        d["exit_t"] = tr.exit_t;
        d["entry_px"] = tr.entry_px;
        d["exit_px"] = tr.exit_px;
        d["qty"] = tr.qty;
        d["pnl"] = tr.pnl;
        d["reason"] = reason_str(tr.reason);
        trades.append(d);
    }

    py::dict metrics;
    metrics["final_equity"] = m.final_equity;
    metrics["total_return"] = m.total_return;
    metrics["sharpe"] = m.sharpe;
    metrics["max_drawdown"] = m.max_dd;
    metrics["n_trades"] = m.n_trades;
    metrics["win_rate"] = m.n_trades ? (double)m.wins / m.n_trades : 0.0;
    metrics["exposure"] = bars.T ? (double)m.exposure_bars / bars.T : 0.0;

    py::dict out;
    out["equity"] = equity;
    out["trades"] = trades;
    out["metrics"] = metrics;
    if (sink.locals_out) out["locals"] = locals_curve;
    return out;
}

// batch over N param rows -> (N, 7) float64 metric matrix
// cols: final_equity, total_return, sharpe, max_dd, n_trades, wins, exposure_bars
py::array_t<double> run_batch(u32arr code, farr consts, i32arr kind, i32arr off,
                              i32arr cap, i32arr aux, int n_locals, int state_floats,
                              farr o, farr h, farr l, farr c, farr v,
                              farr param_matrix, double fee, double slip,
                              double equity0, double bars_per_year) {
    if (param_matrix.ndim() != 2)
        throw std::runtime_error("param_matrix must be 2-D (N, n_params)");
    int N = (int)param_matrix.shape(0);
    int P = (int)param_matrix.shape(1);
    ProgramView prog = make_prog(code, consts, kind, off, cap, aux, n_locals,
                                 state_floats, P);
    BarsView bars = make_bars(o, h, l, c, v);
    RunConfig cfg{(float)fee, (float)slip, (float)equity0, bars_per_year};

    py::array_t<double> out({(size_t)N, (size_t)7});
    double* om = out.mutable_data();
    const float* pm = param_matrix.data();

    {
        py::gil_scoped_release rel;
#pragma omp parallel
        {
            std::vector<float> state((size_t)std::max(state_floats, 1));
            std::vector<float> locals_((size_t)std::max(n_locals, 1));
            NullSink sink;
#pragma omp for schedule(dynamic, 16)
            for (int i = 0; i < N; ++i) {
                std::fill(state.begin(), state.end(), 0.0f);
                std::fill(locals_.begin(), locals_.end(), 0.0f);
                Metrics m = run_backtest(prog, bars, pm + (size_t)i * P, cfg,
                                         StateRef{state.data(), 1},
                                         StateRef{locals_.data(), 1}, sink);
                double* row = om + (size_t)i * 7;
                row[0] = m.final_equity;
                row[1] = m.total_return;
                row[2] = m.sharpe;
                row[3] = m.max_dd;
                row[4] = m.n_trades;
                row[5] = m.wins;
                row[6] = m.exposure_bars;
            }
        }
    }
    return out;
}

PYBIND11_MODULE(btcpu, mod) {
    mod.doc() = "C++ CPU backtest engine (shared VM core with CUDA)";
    mod.def("run_single", &run_single);
    mod.def("run_batch", &run_batch);
}
