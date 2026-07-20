// src/cuda/american_cuda.h
#pragma once
#include "../core/math_utils.hpp"

// Standard pseudo-random CUDA (LCG)
double price_american_call_cuda(const OptionParams& p, int threads_per_block = 512);

// QMC CUDA (Sobol + Brownian Bridge)
double price_american_call_qmc_cuda(const OptionParams& p,
                                    int threads_per_block = 256,
                                    unsigned int seed      = 42);
