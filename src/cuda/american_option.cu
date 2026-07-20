#include "kernels.cuh"
#include "reduction.cuh"
#include "../core/math_utils.hpp"
#include "american_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>

// ---------------- Main pricing kernel ----------------
__global__ void american_option_kernel(
    double* __restrict__ d_partial,
    double S0, double X, double T,
    double r,  double v,
    int m, int N)
{
    extern __shared__ double sdata[];

    int path = blockIdx.x * blockDim.x + threadIdx.x;

    const double dt       = T / static_cast<double>(m + 1);
    const double sqdt     = sqrt(dt);
    const double drift    = (r - 0.5 * v * v) * dt;
    const double discount = exp(-r * dt);

    double payoff = 0.0;

    if (path < N) {
        uint32_t seed = static_cast<uint32_t>(path + 1) * 1234567u;

        // Per-thread path buffer. Safe for m <= 63.
        double S_path[64];
        S_path[0] = S0;

        #pragma unroll 1
        for (int i = 1; i <= m; ++i) {
            float  u = lcg_next(seed);
            double z = static_cast<double>(moro_inv_cnd_device(u));
            S_path[i] = S_path[i-1] * exp(drift + v * sqdt * z);
        }

        double c = bs_call_device(S_path[m - 1], X, dt, v, r);
        for (int i = m - 1; i >= 1; --i) {
            double continuation = c * discount;
            double intrinsic    = S_path[i] - X;
            c = fmax(intrinsic, continuation);
        }

        payoff = c;
    }

    double block_sum = block_reduce_sum(payoff, sdata);

    if (threadIdx.x == 0) {
        d_partial[blockIdx.x] = block_sum;
    }
}

// ---------------- Host launcher ----------------
double price_american_call_cuda(const OptionParams& p, int threads_per_block = 512) {
    int blocks = (p.N + threads_per_block - 1) / threads_per_block;

    double* d_partial = nullptr;
    cudaMalloc(&d_partial, blocks * sizeof(double));
    cudaMemset(d_partial, 0, blocks * sizeof(double));

    int num_warps = (threads_per_block + 31) / 32;
    int shared_mem_bytes = num_warps * sizeof(double);

    american_option_kernel<<<blocks, threads_per_block, shared_mem_bytes>>>(
        d_partial,
        p.S0, p.X, p.T,
        p.r, p.v,
        p.m, p.N
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_partial);
        return 0.0;
    }
    cudaDeviceSynchronize();

    std::vector<double> h_partial(blocks);
    cudaMemcpy(h_partial.data(), d_partial, blocks * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_partial);

    double total = 0.0;
    for (double x : h_partial) total += x;

    return (total / static_cast<double>(p.N)) * std::exp(-p.r * p.T);
}