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
// One thread per path, writing TIME-MAJOR: S[i*N + path].
//
// LSM regresses across paths at each exercise date, so the path set has to be
// materialised; the previous kernel kept it in a 64-double per-thread stack
// array, which was possible only because the naive recursion consumed each path
// in isolation.
//
// Every thread in a warp is at the same i on the same iteration, so writes land
// on consecutive addresses and coalesce. The point-major layout this replaced
// scattered them (m+1)*8 bytes apart, wasting 24 of every 32 bytes moved.
template <typename T>
__global__ void gen_paths_lcg_kernel(
    T* __restrict__ S,
    T S0, T drift, T vol_sqdt,
    int m, int N)
{
    int path = blockIdx.x * blockDim.x + threadIdx.x;
    if (path >= N) return;

    uint32_t seed = static_cast<uint32_t>(path + 1) * 1234567u;

    T s = S0;
    S[path] = s;                                   // i = 0
    for (int i = 1; i <= m; ++i) {
        const T u = PathMath<T>::uniform(seed);
        const T z = PathMath<T>::inv_cnd(u);
        s *= PathMath<T>::expo(drift + vol_sqdt * z);
        S[static_cast<size_t>(i) * N + path] = s;
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

    // Path storage precision. Float halves the footprint and, on a consumer GPU
    // where FP64 runs at 1/64 rate, removes most of the path-generation cost.
    // The regression itself stays double either way.
    const bool use_float = (p.precision == MC_PRECISION_FLOAT);
    const size_t elem = use_float ? sizeof(float) : sizeof(double);

    void* d_S = nullptr;
    const size_t bytes = static_cast<size_t>(p.N) * stride * elem;
    CUDA_TRY(cudaMalloc(&d_S, bytes));

    const auto t_setup = std::chrono::high_resolution_clock::now();

    cudaEvent_t ev_begin, ev_gen, ev_end;
    CUDA_TRY(cudaEventCreate(&ev_begin));
    CUDA_TRY(cudaEventCreate(&ev_gen));
    CUDA_TRY(cudaEventCreate(&ev_end));
    CUDA_TRY(cudaEventRecord(ev_begin));

    const int blocks = (p.N + threads_per_block - 1) / threads_per_block;
    if (use_float) {
        gen_paths_lcg_kernel<float><<<blocks, threads_per_block>>>(
            static_cast<float*>(d_S), static_cast<float>(p.S0),
            static_cast<float>(drift), static_cast<float>(vol_sqdt), p.m, p.N);
    } else {
        gen_paths_lcg_kernel<double><<<blocks, threads_per_block>>>(
            static_cast<double*>(d_S), p.S0, drift, vol_sqdt, p.m, p.N);
    }
    CUDA_TRY_KERNEL();
    CUDA_TRY(cudaEventRecord(ev_gen));

    double lsm_ms = 0.0;
    const double price = use_float
        ? lsm_price_device(static_cast<const float*>(d_S), p, threads_per_block,
                           out_stderr, &lsm_ms)
        : lsm_price_device(static_cast<const double*>(d_S), p, threads_per_block,
                           out_stderr, &lsm_ms);

    CUDA_TRY(cudaEventRecord(ev_end));
    CUDA_TRY(cudaEventSynchronize(ev_end));

    if (timing) {
        float kernel_ms = 0.0f, gen_ms = 0.0f;
        cudaEventElapsedTime(&kernel_ms, ev_begin, ev_end);
        cudaEventElapsedTime(&gen_ms, ev_begin, ev_gen);
        const auto t_end = std::chrono::high_resolution_clock::now();
        timing->setup_ms   = std::chrono::duration<double, std::milli>(t_setup - t_start).count();
        timing->pathgen_ms = gen_ms;
        timing->lsm_ms     = lsm_ms;
        timing->kernel_ms  = kernel_ms;
        timing->total_ms   = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    }
    cudaEventDestroy(ev_begin);
    cudaEventDestroy(ev_gen);
    cudaEventDestroy(ev_end);

    cudaFree(d_S);
    return price;
}
