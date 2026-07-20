// tests/test_cuda.cpp
#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "core/math_utils.hpp"
#include "cuda/american_cuda.h"
#include "tests/kernel_helpers.h"

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
