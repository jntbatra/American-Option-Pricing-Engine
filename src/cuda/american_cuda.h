// src/cuda/american_cuda.h
#pragma once
#include "../core/math_utils.hpp"

// Largest number of exercise points each kernel supports. Both hold the path
// in a fixed-size per-thread array; the QMC one is additionally capped by the
// number of Sobol dimensions in the direction-number table. Callers passing a
// larger m get 0.0 and a message on stderr rather than a corrupted stack.
// Host-visible mirrors of the kernel-side limits (see sobol_gpu.cuh).
static const int CUDA_MAX_M     = 63;
static const int CUDA_QMC_MAX_M = 21;

// Optional breakdown of where GPU wall-clock actually goes.
//
// Quoting the whole launcher call as "GPU time" overstates the kernel cost
// badly at small N, because setup -- allocating the path buffer, building and
// uploading Sobol direction numbers and the Brownian bridge -- is redone on
// every call and dominates until N gets large. Kernel time is measured with
// cudaEvents around the device work only.
struct CudaTiming {
    double setup_ms   = 0.0;  // allocation + host-to-device transfers
    double pathgen_ms = 0.0;  // path-generation kernel alone
    double lsm_ms     = 0.0;  // Longstaff-Schwartz backward pass alone
    double kernel_ms  = 0.0;  // pathgen + lsm
    double total_ms   = 0.0;  // everything the launcher does
};

// Standard pseudo-random CUDA (LCG)
double price_american_call_cuda(const OptionParams& p, int threads_per_block,
                                double* out_stderr = nullptr,
                                CudaTiming* timing = nullptr);

// QMC CUDA (Sobol + Brownian Bridge)
double price_american_call_qmc_cuda(const OptionParams& p,
                                    int threads_per_block,
                                    unsigned int seed,
                                    double* out_stderr = nullptr,
                                    CudaTiming* timing = nullptr);
