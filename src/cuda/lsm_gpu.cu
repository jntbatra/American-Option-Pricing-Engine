#include "lsm_gpu.h"
#include "cuda_check.h"
#include "reduction.cuh"
#include "../core/hd_math.hpp"
#include "../core/lsm.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <vector>

// Per-node regression statistics reduced out of each block: the six unique
// entries of the symmetric 3x3 A^T A, then A^T y, then the in-the-money count.
static const int LSM_STATS = 10;

// ---------------------------------------------------------------------------
// Node m sits at T - dt. The value of holding there is the European value over
// the remaining step -- an exact conditional expectation, so this node needs no
// regression.
__global__ void lsm_init_terminal(double* __restrict__ C,
                                  const double* __restrict__ S,
                                  int m, int stride, int N,
                                  double X, double dt, double v, double r,
                                  int type)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const double s    = S[static_cast<size_t>(n) * stride + m];
    const double cont = mc_bs_european(type, s, X, dt, v, r);
    const double intr = mc_intrinsic(type, s, X);
    C[n] = intr > cont ? intr : cont;
}

// ---------------------------------------------------------------------------
// Discounts every path one step back and, in the same sweep, accumulates the
// regression over the in-the-money subset.
//
// Partial sums are written per block and finished on the host rather than
// atomically accumulated on the device: atomicAdd on doubles would make the
// result depend on block completion order, so two runs of the same pricer could
// disagree. Blocks are summed in index order instead, which is reproducible.
__global__ void lsm_accumulate(double* __restrict__ C,
                               const double* __restrict__ S,
                               double* __restrict__ block_stats,
                               int i, int stride, int N,
                               double X, int type, double discount)
{
    extern __shared__ double sdata[];

    int n = blockIdx.x * blockDim.x + threadIdx.x;

    double acc[LSM_STATS];
    #pragma unroll
    for (int k = 0; k < LSM_STATS; ++k) acc[k] = 0.0;

    if (n < N) {
        const double c = C[n] * discount;
        C[n] = c;

        const double s    = S[static_cast<size_t>(n) * stride + i];
        const double intr = mc_intrinsic(type, s, X);
        if (intr > 0.0) {                       // out of the money: no decision
            const double b0 = 1.0;
            const double b1 = s / X;
            const double b2 = b1 * b1;

            acc[0] = b0 * b0; acc[1] = b0 * b1; acc[2] = b0 * b2;
            acc[3] = b1 * b1; acc[4] = b1 * b2; acc[5] = b2 * b2;
            acc[6] = b0 * c;  acc[7] = b1 * c;  acc[8] = b2 * c;
            acc[9] = 1.0;
        }
    }

    for (int k = 0; k < LSM_STATS; ++k) {
        __syncthreads();            // sdata is reused across iterations
        double v = block_reduce_sum(acc[k], sdata);
        if (threadIdx.x == 0)
            block_stats[static_cast<size_t>(blockIdx.x) * LSM_STATS + k] = v;
    }
}

// ---------------------------------------------------------------------------
// Applies the exercise decision. The regression supplies the decision only;
// the value recorded is always the realised intrinsic, never the fitted one.
__global__ void lsm_decide(double* __restrict__ C,
                           const double* __restrict__ S,
                           int i, int stride, int N,
                           double X, int type,
                           double beta0, double beta1, double beta2)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const double s    = S[static_cast<size_t>(n) * stride + i];
    const double intr = mc_intrinsic(type, s, X);
    if (intr <= 0.0) return;

    const double x   = s / X;
    const double fit = beta0 + beta1 * x + beta2 * x * x;
    if (intr > fit) C[n] = intr;
}

// ---------------------------------------------------------------------------
// Reduces both the sum and the sum of squares of the discounted cashflows, so
// the caller can report a standard error alongside the price.
__global__ void lsm_reduce_cashflows(const double* __restrict__ C,
                                     double* __restrict__ block_sums,
                                     double discount, int N)
{
    extern __shared__ double sdata[];
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    const double x = (n < N) ? C[n] * discount : 0.0;

    double s = block_reduce_sum(x, sdata);
    if (threadIdx.x == 0) block_sums[blockIdx.x] = s;

    __syncthreads();
    double sq = block_reduce_sum(x * x, sdata);
    if (threadIdx.x == 0) block_sums[gridDim.x + blockIdx.x] = sq;
}

// ---------------------------------------------------------------------------
double lsm_price_device(const double* d_S, const OptionParams& p,
                        int threads_per_block, double* out_stderr)
{
    if (out_stderr) *out_stderr = 0.0;

    const int    N        = p.N;
    const int    m        = p.m;
    const int    stride   = m + 1;
    const double dt       = p.T / static_cast<double>(m + 1);
    const double discount = std::exp(-p.r * dt);

    if (N <= 0 || m < 1) return 0.0;

    const int    blocks    = (N + threads_per_block - 1) / threads_per_block;
    const int    num_warps = (threads_per_block + 31) / 32;
    const size_t shmem     = static_cast<size_t>(num_warps) * sizeof(double);

    double* d_C = nullptr;
    double* d_stats = nullptr;
    CUDA_TRY(cudaMalloc(&d_C, static_cast<size_t>(N) * sizeof(double)));
    CUDA_TRY(cudaMalloc(&d_stats,
                        static_cast<size_t>(blocks) * LSM_STATS * sizeof(double)));

    std::vector<double> h_stats(static_cast<size_t>(blocks) * LSM_STATS);

    lsm_init_terminal<<<blocks, threads_per_block>>>(
        d_C, d_S, m, stride, N, p.X, dt, p.v, p.r, p.type);
    CUDA_TRY_KERNEL();

    double beta[LSM_BASIS];

    for (int i = m - 1; i >= 1; --i) {
        lsm_accumulate<<<blocks, threads_per_block, shmem>>>(
            d_C, d_S, d_stats, i, stride, N, p.X, p.type, discount);
        CUDA_TRY_KERNEL();

        CUDA_TRY(cudaMemcpy(h_stats.data(), d_stats,
                            h_stats.size() * sizeof(double),
                            cudaMemcpyDeviceToHost));

        double acc[LSM_STATS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int b = 0; b < blocks; ++b)
            for (int k = 0; k < LSM_STATS; ++k)
                acc[k] += h_stats[static_cast<size_t>(b) * LSM_STATS + k];

        // Unpack the symmetric upper triangle into the full 3x3.
        LsmNormalEq eq;
        eq.ata[0] = acc[0]; eq.ata[1] = acc[1]; eq.ata[2] = acc[2];
        eq.ata[3] = acc[1]; eq.ata[4] = acc[3]; eq.ata[5] = acc[4];
        eq.ata[6] = acc[2]; eq.ata[7] = acc[4]; eq.ata[8] = acc[5];
        eq.atb[0] = acc[6]; eq.atb[1] = acc[7]; eq.atb[2] = acc[8];
        eq.count  = static_cast<long long>(acc[9]);

        // Too few in-the-money paths or a degenerate fit: hold everywhere.
        // Note the discounting in lsm_accumulate has already happened, so
        // skipping the decision kernel is the correct "never exercise" branch.
        if (!lsm_solve(eq, beta)) continue;

        lsm_decide<<<blocks, threads_per_block>>>(
            d_C, d_S, i, stride, N, p.X, p.type, beta[0], beta[1], beta[2]);
        CUDA_TRY_KERNEL();
    }

    // Two halves: block sums, then block sums of squares.
    double* d_sums = nullptr;
    CUDA_TRY(cudaMalloc(&d_sums, static_cast<size_t>(blocks) * 2 * sizeof(double)));

    lsm_reduce_cashflows<<<blocks, threads_per_block, shmem>>>(
        d_C, d_sums, discount, N);
    CUDA_TRY_KERNEL();

    std::vector<double> h_sums(static_cast<size_t>(blocks) * 2);
    CUDA_TRY(cudaMemcpy(h_sums.data(), d_sums, h_sums.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));

    double total = 0.0, total_sq = 0.0;
    for (int b = 0; b < blocks; ++b) {
        total    += h_sums[b];
        total_sq += h_sums[blocks + b];
    }

    cudaFree(d_sums);
    cudaFree(d_stats);
    cudaFree(d_C);

    const double mean = total / static_cast<double>(N);
    if (out_stderr) {
        const double var = total_sq / static_cast<double>(N) - mean * mean;
        *out_stderr = (N > 1 && var > 0.0)
                          ? std::sqrt(var / static_cast<double>(N - 1))
                          : 0.0;
    }
    // The discount to t = 0 is applied inside the reduction kernel.
    return mean;
}
