#include "sobol_gpu.cuh"
#include "kernels.cuh"
#include "reduction.cuh"
#include "../core/backward_induction.hpp"
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

__global__ void american_option_qmc_kernel(
    double* __restrict__       d_result,
    const unsigned int* __restrict__ d_shift_u,
    const unsigned int* __restrict__ d_directions,
    const double* __restrict__ d_bb_wl,
    const double* __restrict__ d_bb_wr,
    const double* __restrict__ d_bb_std,
    const int*   __restrict__  d_bb_mid,
    const int*   __restrict__  d_bb_left,
    const int*   __restrict__  d_bb_right,
    double S0, double X, double T,
    double r,  double v,
    int m, int N)
{
    extern __shared__ double sdata[];

    unsigned int path_id = static_cast<unsigned int>(blockIdx.x * blockDim.x + threadIdx.x);

    const double dt       = T / static_cast<double>(m + 1);
    const double drift    = (r - 0.5 * v * v) * dt;
    const double discount = exp(-r * dt);

    double payoff = 0.0;

    if (path_id < static_cast<unsigned int>(N)) {
        float u_sobol[GPU_SOBOL_DIM];  
        sobol_point_device(path_id, m, d_shift_u, d_directions, u_sobol);

        double z[GPU_SOBOL_DIM];
        for (int d = 0; d < m; ++d) {
            float u = fmaxf(fminf(u_sobol[d], 1.0f - 1e-7f), 1e-7f);
            z[d] = static_cast<double>(moro_inv_cnd_device(u));
        }

        double W[GPU_SOBOL_DIM + 1];   
        W[0] = 0.0;
        for (int bb = 0; bb < m; ++bb) {
            int mid   = d_bb_mid[bb];
            int left  = d_bb_left[bb];
            int right = d_bb_right[bb];
            double left_val = 0.0;
            double right_val = 0.0;
            if (d_bb_wl[bb] != 0.0 && left >= 0) {
                left_val = W[left + 1];
            }
            if (d_bb_wr[bb] != 0.0 && right >= 0) {
                right_val = W[right + 1];
            }
            W[mid + 1] = d_bb_wl[bb] * left_val
                       + d_bb_wr[bb] * right_val
                       + d_bb_std[bb] * z[bb];
        }

        double S[GPU_SOBOL_DIM + 1];
        S[0] = S0;
        for (int i = 1; i <= m; ++i) {
            double dW = W[i] - W[i - 1];
            S[i] = S[i-1] * exp(drift + v * dW);
        }

        payoff = american_call_path_value(S, m, X, dt, v, r, discount);
    }

    double block_sum = block_reduce_sum(payoff, sdata);
    if (threadIdx.x == 0) d_result[blockIdx.x] = block_sum;
}

// Host-side launcher for QMC kernel
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

    // Direction numbers
    unsigned int V_host[GPU_SOBOL_DIM][GPU_SOBOL_BITS];
    for (int k = 0; k < 32; ++k) {
        V_host[0][k] = 1u << (31 - k);
    }
    
    for (int dim = 1; dim < GPU_SOBOL_DIM; ++dim) {
        const SobolInitData& init = SOBOL_INIT[dim - 1];
        int s = init.s;
        uint32_t a = init.a;

        for (int k = 0; k < s; ++k)
            V_host[dim][k] = init.m[k] << (31 - k);

        for (int k = s; k < 32; ++k) {
            V_host[dim][k] = V_host[dim][k - s] ^ (V_host[dim][k - s] >> s);
            for (int l = 1; l < s; ++l) {
                if ((a >> (s - 1 - l)) & 1u)
                    V_host[dim][k] ^= V_host[dim][k - l];
            }
        }
    }

    sobol_gpu_init(V_host);

    unsigned int* d_dir_ptr = get_d_V_ptr();

    auto shifts_vec = make_digital_shift(p.m, seed);
    unsigned int* d_shift = nullptr;
    cudaMalloc(&d_shift, p.m * sizeof(unsigned int));
    cudaMemcpy(d_shift, shifts_vec.data(), p.m * sizeof(unsigned int), cudaMemcpyHostToDevice);

    double dt = p.T / static_cast<double>(p.m + 1);
    auto bridge = build_brownian_bridge(p.m, dt);
    
    std::vector<double> bb_wl(p.m), bb_wr(p.m), bb_std(p.m);
    std::vector<int> bb_mid(p.m), bb_left(p.m), bb_right(p.m);
    for (int i = 0; i < p.m; ++i) {
        bb_wl[i] = bridge[i].w_l;
        bb_wr[i] = bridge[i].w_r;
        bb_std[i] = bridge[i].std;
        bb_mid[i] = bridge[i].mid;
        bb_left[i] = bridge[i].left;
        bb_right[i] = bridge[i].right;
    }

    double *d_bb_wl, *d_bb_wr, *d_bb_std;
    int *d_bb_mid, *d_bb_left, *d_bb_right;
    cudaMalloc(&d_bb_wl, p.m * sizeof(double));
    cudaMalloc(&d_bb_wr, p.m * sizeof(double));
    cudaMalloc(&d_bb_std, p.m * sizeof(double));
    cudaMalloc(&d_bb_mid, p.m * sizeof(int));
    cudaMalloc(&d_bb_left, p.m * sizeof(int));
    cudaMalloc(&d_bb_right, p.m * sizeof(int));

    cudaMemcpy(d_bb_wl, bb_wl.data(), p.m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bb_wr, bb_wr.data(), p.m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bb_std, bb_std.data(), p.m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bb_mid, bb_mid.data(), p.m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bb_left, bb_left.data(), p.m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bb_right, bb_right.data(), p.m * sizeof(int), cudaMemcpyHostToDevice);

    int blocks = (p.N + threads_per_block - 1) / threads_per_block;
    int shared  = (threads_per_block / 32) * sizeof(double);

    double* d_result = nullptr;
    cudaMalloc(&d_result, blocks * sizeof(double));
    cudaMemset(d_result, 0, blocks * sizeof(double));

    american_option_qmc_kernel<<<blocks, threads_per_block, shared>>>(
        d_result, d_shift, d_dir_ptr,
        d_bb_wl, d_bb_wr, d_bb_std, d_bb_mid, d_bb_left, d_bb_right,
        p.S0, p.X, p.T, p.r, p.v, p.m, p.N
    );
    cudaDeviceSynchronize();

    std::vector<double> h_result(blocks);
    cudaMemcpy(h_result.data(), d_result, blocks * sizeof(double), cudaMemcpyDeviceToHost);

    double total = 0.0;
    for (double v : h_result) total += v;

    cudaFree(d_result);
    cudaFree(d_shift);
    cudaFree(d_bb_wl);
    cudaFree(d_bb_wr);
    cudaFree(d_bb_std);
    cudaFree(d_bb_mid);
    cudaFree(d_bb_left);
    cudaFree(d_bb_right);

    // american_call_path_value() already discounts to t = 0.
    return total / static_cast<double>(p.N);
}
