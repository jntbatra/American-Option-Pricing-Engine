// tests/test_cuda.cpp
#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "core/math_utils.hpp"
#include "core/black_scholes.hpp"
#include "core/binomial.hpp"
#include "cuda/american_cuda.h"
#include "tests/kernel_helpers.h"

#include "core/backends.hpp"


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
// An American call on a non-dividend-paying stock is worth exactly its
// European counterpart, because early exercise is never optimal. Any gap is
// bias or bug and does not shrink with N.
TEST(CudaPricing, EqualsEuropeanForNonDividendCall) {
    OptionParams p = reference_case(10, 500000);

    double gpu      = price_american_call_cuda(p, 512);
    double european = bs_call(p.S0, p.X, p.T, p.v, p.r);

    EXPECT_NEAR(gpu, european, 0.05) << "gpu=" << gpu << " bs=" << european;
}

// ---------------------------------------------------------------
// The put is what actually exercises the LSM boundary. Checked against the
// binomial tree restricted to the same m exercise dates, so both price the
// identical Bermudan contract.
TEST(CudaPricing, PutMatchesBinomialTree) {
    OptionParams p = reference_case(20, 500000);
    p.type = OPTION_PUT;

    double gpu       = price_american_call_cuda(p, 512);
    double reference = binomial_bermudan(p, 250);

    // LSM is low-biased, so allow more room below than above.
    EXPECT_LT(gpu - reference,  0.05) << "gpu=" << gpu << " tree=" << reference;
    EXPECT_GT(gpu - reference, -0.20) << "gpu=" << gpu << " tree=" << reference;
}

// ---------------------------------------------------------------
// Both QMC backends now draw the same Sobol points in double precision and run
// the same bridge, so they must agree far more tightly than two independent
// estimators would. A gap here means the GPU Sobol, the GPU bridge, or the
// device LSM has diverged from its host counterpart.
TEST(CudaPricing, QmcMatchesHostQmc) {
    OptionParams p = reference_case(10, 200000);

    double gpu = price_american_call_qmc_cuda(p, 256, 42);
    double cpu = price_american_call_qmc_omp(p, 0, 42);

    EXPECT_NEAR(gpu, cpu, 1e-6) << "gpu=" << gpu << " cpu=" << cpu;
}

// ---------------------------------------------------------------
// Regression: the float LCG can emit exactly 1.0, because
// __uint2float_rn(0xFFFFFFFF) rounds up to 2^32. Moro then evaluates
// log(-log(0)) and the price comes back inf. It first appeared at N = 4e6 --
// small runs never draw a state in the top 128 -- so this test needs the paths
// to be plentiful enough to hit it.
TEST(CudaPrecision, Fp32PathsStayFinite) {
    OptionParams p = reference_case(20, 4000000);
    p.precision = MC_PRECISION_FLOAT;

    double price = price_american_call_cuda(p, 512);
    EXPECT_TRUE(std::isfinite(price)) << "price=" << price;
    EXPECT_NEAR(price, bs_call(p.S0, p.X, p.T, p.v, p.r), 0.05);
}

// ---------------------------------------------------------------
// Single precision must not move the price by anything approaching the Monte
// Carlo standard error -- that is the whole justification for offering it.
TEST(CudaPrecision, Fp32AgreesWithFp64) {
    OptionParams p = reference_case(20, 1000000);

    double se = 0.0;
    p.precision = MC_PRECISION_DOUBLE;
    const double fp64 = price_american_call_cuda(p, 512, &se);
    p.precision = MC_PRECISION_FLOAT;
    const double fp32 = price_american_call_cuda(p, 512);

    ASSERT_GT(se, 0.0);
    // Two orders of magnitude inside the statistical noise, at minimum.
    EXPECT_LT(std::fabs(fp32 - fp64), 0.01 * se)
        << "fp64=" << fp64 << " fp32=" << fp32 << " stderr=" << se;
}
