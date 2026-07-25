#include "kernels.cuh"
#include "lsm_gpu.h"
#include "cuda_check.h"
#include "../core/hd_math.hpp"
#include "../core/math_utils.hpp"
#include "american_cuda.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdint>
#include <cmath>

// ---------------- Path generation ----------------
// One thread per path, writing the whole path to global memory point-major.
//
// LSM regresses across paths at each exercise date, so the path set has to be
// materialised; the previous kernel kept it in a 64-double per-thread stack
// array, which was possible only because the naive recursion consumed each path
// in isolation.
__global__ void gen_paths_lcg_kernel(
    double* __restrict__ S,
    double S0, double drift, double vol_sqdt,
    int m, int stride, int N)
{
    int path = blockIdx.x * blockDim.x + threadIdx.x;
    if (path >= N) return;

    uint32_t seed = static_cast<uint32_t>(path + 1) * 1234567u;
    double* row = S + static_cast<size_t>(path) * stride;

    row[0] = S0;
    for (int i = 1; i <= m; ++i) {
        const double u = lcg_next(seed);
        const double z = moro_inv_cnd_device(u);
        row[i] = row[i - 1] * exp(drift + vol_sqdt * z);
    }
}

// ---------------- Host launcher ----------------
double price_american_call_cuda(const OptionParams& p, int threads_per_block,
                                double* out_stderr, CudaTiming* timing) {
    if (out_stderr) *out_stderr = 0.0;

    if (p.m < 1 || p.m > CUDA_MAX_M) {
        fprintf(stderr, "price_american_call_cuda: m=%d out of range (1..%d)\n",
                p.m, CUDA_MAX_M);
        return 0.0;
    }
    if (p.N <= 0 || threads_per_block <= 0) return 0.0;

    const int    stride   = p.m + 1;
    const double dt       = p.T / static_cast<double>(p.m + 1);
    const double drift    = (p.r - 0.5 * p.v * p.v) * dt;
    const double vol_sqdt = p.v * std::sqrt(dt);

    const auto t_start = std::chrono::high_resolution_clock::now();

    double* d_S = nullptr;
    const size_t bytes = static_cast<size_t>(p.N) * stride * sizeof(double);
    CUDA_TRY(cudaMalloc(&d_S, bytes));

    const auto t_setup = std::chrono::high_resolution_clock::now();

    cudaEvent_t ev_begin, ev_end;
    CUDA_TRY(cudaEventCreate(&ev_begin));
    CUDA_TRY(cudaEventCreate(&ev_end));
    CUDA_TRY(cudaEventRecord(ev_begin));

    const int blocks = (p.N + threads_per_block - 1) / threads_per_block;
    gen_paths_lcg_kernel<<<blocks, threads_per_block>>>(
        d_S, p.S0, drift, vol_sqdt, p.m, stride, p.N);
    CUDA_TRY_KERNEL();

    const double price = lsm_price_device(d_S, p, threads_per_block, out_stderr);

    CUDA_TRY(cudaEventRecord(ev_end));
    CUDA_TRY(cudaEventSynchronize(ev_end));

    if (timing) {
        float kernel_ms = 0.0f;
        cudaEventElapsedTime(&kernel_ms, ev_begin, ev_end);
        const auto t_end = std::chrono::high_resolution_clock::now();
        timing->setup_ms  = std::chrono::duration<double, std::milli>(t_setup - t_start).count();
        timing->kernel_ms = kernel_ms;
        timing->total_ms  = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    }
    cudaEventDestroy(ev_begin);
    cudaEventDestroy(ev_end);

    cudaFree(d_S);
    return price;
}
