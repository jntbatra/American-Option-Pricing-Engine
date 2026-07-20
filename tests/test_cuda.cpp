// tests/test_cuda.cpp
#include <gtest/gtest.h>
#include <cuda_runtime.h>

// Adjust include path to where the CUDA header resides in the project.
#include "cuda/american_cuda.cuh"

// ---------------------------------------------------------------
// Dummy kernel – just to prove a launch works.
__global__ void dummy_kernel(int *out) {
    if (threadIdx.x == 0) *out = 42;
}

TEST(CudaBasic, KernelLaunch) {
    int *d_out = nullptr;
    int h_out = 0;
    cudaMalloc(&d_out, sizeof(int));
    dummy_kernel<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    EXPECT_EQ(h_out, 42);
}

// ---------------------------------------------------------------
// Small‑problem pricing test.
// Expose a thin wrapper in `american_cuda.cuh`:
//   double run_american_cuda(int num_paths, int exercise_points);
// The test only checks that the price is a finite, positive number.
TEST(CudaPricing, SmallProblem) {
    const int N = 256; // tiny number of Monte‑Carlo paths – fast
    const int M = 10;  // exercise points
    double price = run_american_cuda(N, M);
    EXPECT_GT(price, 0.0);
    EXPECT_FALSE(std::isnan(price));
}
