// tests/kernel_helpers.cu
#include <cuda_runtime.h>

// Dummy kernel for verifying a basic launch works.
__global__ void dummy_kernel(int *out) {
    if (threadIdx.x == 0) *out = 42;
}

void launch_dummy_kernel(int *d_out) {
    dummy_kernel<<<1, 32>>>(d_out);
}
