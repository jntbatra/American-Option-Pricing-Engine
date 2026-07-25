#include "sobol_gpu.cuh"
#include "kernels.cuh"
#include "lsm_gpu.h"
#include "cuda_check.h"
#include "../core/hd_math.hpp"
#include "../core/math_utils.hpp"
#include "american_cuda.h"
#include "../core/scramble.hpp"
#include "../core/brownian_bridge.hpp"
#include "../core/sobol_joe_kuo.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>

unsigned int* get_d_V_ptr();

// Mirrors simulate_path_bb() in core/brownian_bridge.cpp step for step, so the
// CPU and GPU QMC backends produce identical paths from identical Sobol points.
__global__ void gen_paths_qmc_kernel(
    double* __restrict__       S,
    const unsigned int* __restrict__ d_shift,
    const unsigned int* __restrict__ d_directions,
    const double* __restrict__ d_bb_wl,
    const double* __restrict__ d_bb_wr,
    const double* __restrict__ d_bb_std,
    const int*   __restrict__  d_bb_mid,
    const int*   __restrict__  d_bb_left,
    const int*   __restrict__  d_bb_right,
    double S0, double r, double v, double dt,
    int m, int stride, int N)
{
    unsigned int path = static_cast<unsigned int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (path >= static_cast<unsigned int>(N)) return;

    // Sobol point, then normal shocks in place.
    double z[GPU_SOBOL_DIM];
    sobol_point_device(path, m, d_shift, d_directions, z);
    for (int d = 0; d < m; ++d) {
        // Same clamp as the host backend, so the two agree in the tails.
        const double u = fmax(1e-10, fmin(z[d], 1.0 - 1e-10));
        z[d] = moro_inv_cnd_device(u);
    }

    // Brownian bridge: fill the Wiener path in variance-decreasing order.
    double W[GPU_SOBOL_DIM + 1];
    W[0] = 0.0;
    for (int bb = 0; bb < m; ++bb) {
        const int mid   = d_bb_mid[bb];
        const int left  = d_bb_left[bb];
        const int right = d_bb_right[bb];
        double left_val  = 0.0;
        double right_val = 0.0;
        if (d_bb_wl[bb] != 0.0 && left  >= 0) left_val  = W[left + 1];
        if (d_bb_wr[bb] != 0.0 && right >= 0) right_val = W[right + 1];
        W[mid + 1] = d_bb_wl[bb] * left_val
                   + d_bb_wr[bb] * right_val
                   + d_bb_std[bb] * z[bb];
    }

    const double drift = r - 0.5 * v * v;
    double* row = S + static_cast<size_t>(path) * stride;
    row[0] = S0;
    for (int i = 1; i <= m; ++i) {
        const double dW = W[i] - W[i - 1];
        row[i] = row[i - 1] * exp(drift * dt + v * dW);
    }
}

// Host-side launcher for the QMC backend.
double price_american_call_qmc_cuda(const OptionParams& p,
                                     int threads_per_block,
                                     uint32_t seed)
{
    // The kernel holds the Sobol point, the bridge and the path in fixed-size
    // per-thread arrays sized by GPU_SOBOL_DIM, and the direction-number table
    // only has that many dimensions. Larger m would smash the stack silently.
    static_assert(CUDA_QMC_MAX_M == GPU_SOBOL_DIM,
                  "host-visible limit must mirror the kernel-side one");
    if (p.m < 1 || p.m > CUDA_QMC_MAX_M) {
        fprintf(stderr,
                "price_american_call_qmc_cuda: m=%d out of range (1..%d)\n",
                p.m, CUDA_QMC_MAX_M);
        return 0.0;
    }
    if (p.N <= 0 || threads_per_block <= 0) return 0.0;

    // Direction numbers, built exactly as SobolGenerator does on the host.
    unsigned int V_host[GPU_SOBOL_DIM][GPU_SOBOL_BITS];
    for (int k = 0; k < GPU_SOBOL_BITS; ++k) V_host[0][k] = 1u << (31 - k);

    for (int dim = 1; dim < GPU_SOBOL_DIM; ++dim) {
        const SobolInitData& init = SOBOL_INIT[dim - 1];
        const int s = init.s;
        const uint32_t a = init.a;

        for (int k = 0; k < s; ++k)
            V_host[dim][k] = init.m[k] << (31 - k);

        for (int k = s; k < GPU_SOBOL_BITS; ++k) {
            V_host[dim][k] = V_host[dim][k - s] ^ (V_host[dim][k - s] >> s);
            for (int l = 1; l < s; ++l) {
                if ((a >> (s - 1 - l)) & 1u)
                    V_host[dim][k] ^= V_host[dim][k - l];
            }
        }
    }
    sobol_gpu_init(V_host);
    unsigned int* d_dir_ptr = get_d_V_ptr();

    const int    stride = p.m + 1;
    const double dt     = p.T / static_cast<double>(p.m + 1);

    // Same digital shift the host backend uses, so the point sets match.
    const std::vector<uint32_t> shifts_vec = make_digital_shift(p.m, seed);
    const std::vector<BBNode> bridge = build_brownian_bridge(p.m, dt);

    std::vector<double> bb_wl(p.m), bb_wr(p.m), bb_std(p.m);
    std::vector<int> bb_mid(p.m), bb_left(p.m), bb_right(p.m);
    for (int i = 0; i < p.m; ++i) {
        bb_wl[i]    = bridge[i].w_l;
        bb_wr[i]    = bridge[i].w_r;
        bb_std[i]   = bridge[i].std;
        bb_mid[i]   = bridge[i].mid;
        bb_left[i]  = bridge[i].left;
        bb_right[i] = bridge[i].right;
    }

    unsigned int* d_shift = nullptr;
    double *d_bb_wl = nullptr, *d_bb_wr = nullptr, *d_bb_std = nullptr;
    int *d_bb_mid = nullptr, *d_bb_left = nullptr, *d_bb_right = nullptr;
    double* d_S = nullptr;

    CUDA_TRY(cudaMalloc(&d_shift,    p.m * sizeof(unsigned int)));
    CUDA_TRY(cudaMalloc(&d_bb_wl,    p.m * sizeof(double)));
    CUDA_TRY(cudaMalloc(&d_bb_wr,    p.m * sizeof(double)));
    CUDA_TRY(cudaMalloc(&d_bb_std,   p.m * sizeof(double)));
    CUDA_TRY(cudaMalloc(&d_bb_mid,   p.m * sizeof(int)));
    CUDA_TRY(cudaMalloc(&d_bb_left,  p.m * sizeof(int)));
    CUDA_TRY(cudaMalloc(&d_bb_right, p.m * sizeof(int)));
    CUDA_TRY(cudaMalloc(&d_S, static_cast<size_t>(p.N) * stride * sizeof(double)));

    CUDA_TRY(cudaMemcpy(d_shift, shifts_vec.data(), p.m * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(d_bb_wl,    bb_wl.data(),    p.m * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(d_bb_wr,    bb_wr.data(),    p.m * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(d_bb_std,   bb_std.data(),   p.m * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(d_bb_mid,   bb_mid.data(),   p.m * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(d_bb_left,  bb_left.data(),  p.m * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(d_bb_right, bb_right.data(), p.m * sizeof(int), cudaMemcpyHostToDevice));

    const int blocks = (p.N + threads_per_block - 1) / threads_per_block;
    gen_paths_qmc_kernel<<<blocks, threads_per_block>>>(
        d_S, d_shift, d_dir_ptr,
        d_bb_wl, d_bb_wr, d_bb_std, d_bb_mid, d_bb_left, d_bb_right,
        p.S0, p.r, p.v, dt, p.m, stride, p.N);
    CUDA_TRY_KERNEL();

    const double price = lsm_price_device(d_S, p, threads_per_block);

    cudaFree(d_S);
    cudaFree(d_shift);
    cudaFree(d_bb_wl);
    cudaFree(d_bb_wr);
    cudaFree(d_bb_std);
    cudaFree(d_bb_mid);
    cudaFree(d_bb_left);
    cudaFree(d_bb_right);

    return price;
}
