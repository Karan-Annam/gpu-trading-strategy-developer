// pybind11 module `btcuda`: GPU parameter sweeps.
//
// Layout choices (the whole point of the design):
//  - one thread = one full backtest over its own parameter row
//  - all threads run the SAME bytecode -> no interpreter divergence in a warp
//  - bytecode is staged into shared memory once per block
//  - per-thread VM state is column-major across the launch (StateRef stride =
//    launch width) so ring-buffer scans coalesce across the warp
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "../vm_core.h"

namespace py = pybind11;
using namespace bt;

using farr = py::array_t<float, py::array::c_style | py::array::forcecast>;
using u32arr = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;
using i32arr = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

#define CUDA_CHECK(x) do { cudaError_t err__ = (x); if (err__ != cudaSuccess) \
    throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(err__) + \
                             " at " __FILE__ ":" + std::to_string(__LINE__)); } while (0)

namespace {

constexpr int kBlock = 128;
constexpr int kMaxSharedCode = 6144;   // instructions staged to shared memory (24 KB)

__global__ void sweep_kernel(ProgramView prog, BarsView bars,
                             const float* __restrict__ param_matrix, int N, int P,
                             RunConfig cfg, float* state_pool, float* locals_pool,
                             double* __restrict__ metrics_out) {
    extern __shared__ uint32_t scode[];
    for (int i = threadIdx.x; i < prog.n_code; i += blockDim.x)
        scode[i] = prog.code[i];
    __syncthreads();
    ProgramView p = prog;
    p.code = scode;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    // column-major per-thread state: element j of thread tid at pool[j*N + tid]
    StateRef state{state_pool + tid, (size_t)N};
    StateRef locals{locals_pool + tid, (size_t)N};
    for (int j = 0; j < p.state_floats; ++j) state[j] = 0.0f;
    for (int j = 0; j < p.n_locals; ++j) locals[j] = 0.0f;

    float params[16];
    for (int j = 0; j < P; ++j) params[j] = param_matrix[(size_t)tid * P + j];

    NullSink sink;
    Metrics m = run_backtest(p, bars, params, cfg, state, locals, sink);

    double* row = metrics_out + (size_t)tid * 7;
    row[0] = m.final_equity;
    row[1] = m.total_return;
    row[2] = m.sharpe;
    row[3] = m.max_dd;
    row[4] = m.n_trades;
    row[5] = m.wins;
    row[6] = m.exposure_bars;
}

struct DeviceBuf {
    void* p = nullptr;
    ~DeviceBuf() { if (p) cudaFree(p); }
    void alloc(size_t bytes) { CUDA_CHECK(cudaMalloc(&p, bytes)); }
    template <typename T> T* as() { return (T*)p; }
};

void upload(DeviceBuf& buf, const void* src, size_t bytes) {
    buf.alloc(bytes ? bytes : 4);
    if (bytes)
        CUDA_CHECK(cudaMemcpy(buf.p, src, bytes, cudaMemcpyHostToDevice));
}

}  // namespace

py::dict device_info() {
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    py::dict d;
    d["name"] = std::string(prop.name);
    d["sm"] = prop.multiProcessorCount;
    d["cc"] = std::to_string(prop.major) + "." + std::to_string(prop.minor);
    d["mem_free_mb"] = (double)free_b / 1048576.0;
    d["mem_total_mb"] = (double)total_b / 1048576.0;
    return d;
}

py::array_t<double> run_batch(u32arr code, farr consts, i32arr kind, i32arr off,
                              i32arr cap, i32arr aux, int n_locals, int state_floats,
                              farr o, farr h, farr l, farr c, farr v,
                              farr param_matrix, double fee, double slip,
                              double equity0, double bars_per_year) {
    if (param_matrix.ndim() != 2)
        throw std::runtime_error("param_matrix must be 2-D (N, n_params)");
    const int N = (int)param_matrix.shape(0);
    const int P = (int)param_matrix.shape(1);
    if (N <= 0) throw std::runtime_error("param_matrix must not be empty");
    if (P > 16) throw std::runtime_error("max 16 params");
    if (code.ndim() != 1 || consts.ndim() != 1 || kind.ndim() != 1 ||
        off.ndim() != 1 || cap.ndim() != 1 || aux.ndim() != 1)
        throw std::runtime_error("program arrays must be one-dimensional");
    if (code.size() == 0)
        throw std::runtime_error("program bytecode is empty");
    if (kind.size() != off.size() || cap.size() != off.size() ||
        aux.size() != off.size())
        throw std::runtime_error("program state arrays must have equal lengths");
    if (n_locals < 0 || state_floats < 0)
        throw std::runtime_error("invalid program resource counts");
    if (o.ndim() != 1 || h.ndim() != 1 || l.ndim() != 1 || c.ndim() != 1 ||
        v.ndim() != 1)
        throw std::runtime_error("OHLCV arrays must be one-dimensional");
    if (c.size() == 0)
        throw std::runtime_error("bar data is empty");
    if (o.size() != c.size() || h.size() != c.size() || l.size() != c.size() ||
        v.size() != c.size())
        throw std::runtime_error("OHLCV arrays must have equal lengths");
    const int n_code = (int)code.size();
    if (n_code > kMaxSharedCode) throw std::runtime_error("program too large");
    const int T = (int)c.size();

    py::array_t<double> out({(size_t)N, (size_t)7});

    {
        py::gil_scoped_release rel;

        DeviceBuf d_code, d_consts, d_kind, d_off, d_cap, d_aux;
        upload(d_code, code.data(), n_code * 4);
        upload(d_consts, consts.data(), consts.size() * 4);
        upload(d_kind, kind.data(), kind.size() * 4);
        upload(d_off, off.data(), off.size() * 4);
        upload(d_cap, cap.data(), cap.size() * 4);
        upload(d_aux, aux.data(), aux.size() * 4);

        DeviceBuf d_o, d_h, d_l, d_c, d_v;
        upload(d_o, o.data(), (size_t)T * 4);
        upload(d_h, h.data(), (size_t)T * 4);
        upload(d_l, l.data(), (size_t)T * 4);
        upload(d_c, c.data(), (size_t)T * 4);
        upload(d_v, v.data(), (size_t)T * 4);

        DeviceBuf d_params, d_state, d_locals, d_metrics;
        upload(d_params, param_matrix.data(), (size_t)N * P * 4);
        d_state.alloc((size_t)N * (state_floats > 0 ? state_floats : 1) * 4);
        d_locals.alloc((size_t)N * (n_locals > 0 ? n_locals : 1) * 4);
        d_metrics.alloc((size_t)N * 7 * 8);

        ProgramView prog;
        prog.code = d_code.as<uint32_t>();
        prog.n_code = n_code;
        prog.consts = d_consts.as<float>();
        prog.state_kind = d_kind.as<int32_t>();
        prog.state_off = d_off.as<int32_t>();
        prog.state_cap = d_cap.as<int32_t>();
        prog.state_aux = d_aux.as<int32_t>();
        prog.n_state = (int)off.size();
        prog.n_locals = n_locals;
        prog.n_params = P;
        prog.state_floats = state_floats;

        BarsView bars{d_o.as<float>(), d_h.as<float>(), d_l.as<float>(),
                      d_c.as<float>(), d_v.as<float>(), T};
        RunConfig cfg{(float)fee, (float)slip, (float)equity0, bars_per_year};

        int blocks = (N + kBlock - 1) / kBlock;
        size_t shmem = (size_t)n_code * sizeof(uint32_t);
        sweep_kernel<<<blocks, kBlock, shmem>>>(
            prog, bars, d_params.as<float>(), N, P, cfg,
            d_state.as<float>(), d_locals.as<float>(), d_metrics.as<double>());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(out.mutable_data(), d_metrics.p,
                              (size_t)N * 7 * 8, cudaMemcpyDeviceToHost));
    }
    return out;
}

PYBIND11_MODULE(btcuda, mod) {
    mod.doc() = "CUDA backtest sweep engine (shared VM core with CPU)";
    mod.def("run_batch", &run_batch);
    mod.def("device_info", &device_info);
}
