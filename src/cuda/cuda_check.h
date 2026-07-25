#pragma once
#include <cuda_runtime.h>
#include <cstdio>

// Every CUDA call in this project goes through one of these. Silently ignoring
// a failed cudaMalloc or an out-of-range kernel launch produces a plausible
// wrong price, which is the worst possible failure mode for a pricer.

#define CUDA_TRY(call)                                                        \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err_), #call);     \
            return 0.0;                                                       \
        }                                                                     \
    } while (0)

// Checks that the most recent kernel launch was accepted and completed.
#define CUDA_TRY_KERNEL()                                                     \
    do {                                                                      \
        CUDA_TRY(cudaGetLastError());                                         \
        CUDA_TRY(cudaDeviceSynchronize());                                    \
    } while (0)
