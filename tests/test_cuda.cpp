// tests/test_cuda.cpp
#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "core/math_utils.hpp"
#include "core/black_scholes.hpp"
#include "cuda/american_cuda.h"
#include "tests/kernel_helpers.h"

double price_american_call_serial(const OptionParams&);

static OptionParams reference_case(int m, int N) {
    OptionParams p;
    p.S0 = 100.0; p.X = 100.0; p.T = 1.0; p.r = 0.05; p.v = 0.20;
    p.m = m; p.N = N;
    return p;
}

// ---------------------------------------------------------------
// Dummy kernel launch test.
TEST(CudaBasic, KernelLaunch) {
    int *d_out = nullptr;
    int h_out = 0;
    cudaMalloc(&d_out, sizeof(int));
    launch_dummy_kernel(d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    EXPECT_EQ(h_out, 42);
}

// ---------------------------------------------------------------
// Small-problem pricing test using the standard (LCG) CUDA backend.
TEST(CudaPricing, SmallProblemLCG) {
    OptionParams p;
    p.S0 = 100.0;
    p.X  = 100.0;
    p.T  = 1.0;
    p.r  = 0.05;
    p.v  = 0.20;
    p.m  = 10;
    p.N  = 256;

    double price = price_american_call_cuda(p, 512);
    EXPECT_GT(price, 0.0);
    EXPECT_FALSE(std::isnan(price));
}

// ---------------------------------------------------------------
// Small-problem pricing test using the QMC CUDA backend.
TEST(CudaPricing, SmallProblemQMC) {
    OptionParams p;
    p.S0 = 100.0;
    p.X  = 100.0;
    p.T  = 1.0;
    p.r  = 0.05;
    p.v  = 0.20;
    p.m  = 10;
    p.N  = 256;

    double price = price_american_call_qmc_cuda(p, 256, 42);
    EXPECT_GT(price, 0.0);
    EXPECT_FALSE(std::isnan(price));
}

// ---------------------------------------------------------------
// The GPU LCG backend draws the same per-path stream as the serial backend, so
// the two must produce the same price to within floating-point summation
// order. This is the check that keeps the CPU and GPU implementations of the
// backward induction from drifting apart.
TEST(CudaPricing, MatchesSerialBackend) {
    OptionParams p = reference_case(10, 200000);

    double gpu = price_american_call_cuda(p, 512);
    double cpu = price_american_call_serial(p);

    EXPECT_NEAR(gpu, cpu, 1e-3) << "gpu=" << gpu << " cpu=" << cpu;
}

// ---------------------------------------------------------------
// Both GPU backends price the same option, so they must agree to within the
// difference between a pseudo-random and a quasi-random estimator.
TEST(CudaPricing, LcgAgreesWithQmc) {
    OptionParams p = reference_case(10, 200000);

    double lcg = price_american_call_cuda(p, 512);
    double qmc = price_american_call_qmc_cuda(p, 256, 42);

    EXPECT_NEAR(lcg, qmc, 0.25) << "lcg=" << lcg << " qmc=" << qmc;
}

// ---------------------------------------------------------------
// m is bounded by the fixed-size per-thread arrays in both kernels. Out-of-
// range values must be rejected, not silently smash the stack.
TEST(CudaPricing, RejectsOutOfRangeM) {
    OptionParams p = reference_case(CUDA_MAX_M + 1, 1024);
    EXPECT_DOUBLE_EQ(price_american_call_cuda(p, 256), 0.0);

    p.m = CUDA_QMC_MAX_M + 1;
    EXPECT_DOUBLE_EQ(price_american_call_qmc_cuda(p, 256, 42), 0.0);
}

// ---------------------------------------------------------------
// Acceptance test, currently expected to fail: an American call on a
// non-dividend-paying stock is worth exactly its European counterpart. The
// naive per-path recursion exercises with perfect foresight and prices it far
// too high. Enable this once Longstaff-Schwartz replaces the recursion.
TEST(CudaPricing, DISABLED_EqualsEuropeanForNonDividendCall) {
    OptionParams p = reference_case(10, 500000);

    double gpu      = price_american_call_cuda(p, 512);
    double european = bs_call(p.S0, p.X, p.T, p.v, p.r);

    EXPECT_NEAR(gpu, european, 0.05);
}
