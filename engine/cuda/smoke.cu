// M0 toolchain smoke test: prove nvcc + MSVC + driver all agree.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void iota(int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = i * 2;
}

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    return 1; } } while (0)

int main() {
    int dev = 0;
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("device: %s, SM %d.%d, %d SMs, %.1f GB\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, prop.totalGlobalMem / 1073741824.0);

    const int N = 1 << 20;
    int* d = nullptr;
    CHECK(cudaMalloc(&d, N * sizeof(int)));
    iota<<<(N + 255) / 256, 256>>>(d, N);
    CHECK(cudaGetLastError());
    int host[4];
    CHECK(cudaMemcpy(host, d + N - 4, sizeof(host), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d));
    bool ok = host[3] == (N - 1) * 2;
    printf("kernel check: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
