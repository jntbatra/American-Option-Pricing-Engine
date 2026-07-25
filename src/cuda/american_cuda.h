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

// Standard pseudo-random CUDA (LCG)
double price_american_call_cuda(const OptionParams& p, int threads_per_block);

// QMC CUDA (Sobol + Brownian Bridge)
double price_american_call_qmc_cuda(const OptionParams& p,
                                    int threads_per_block,
                                    unsigned int seed);
