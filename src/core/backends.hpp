#pragma once
#include "math_utils.hpp"
#include <cstdint>

// Entry points for the CPU pricing backends.
//
// These used to be hand-declared at the top of every caller (validate.cpp,
// test_cuda.cpp, benchmark/main.cpp), which meant a signature change had to be
// mirrored in four places and a missed one would be an ODR violation the
// linker cannot see. Declare them once.
//
// Every backend prices the option described by `p` (call or put, per p.type)
// and returns the value discounted to t = 0. Passing `out_stderr` also yields
// the standard error of the Monte Carlo estimate.

// Serial reference. LCG pseudo-random shocks, Longstaff-Schwartz valuation.
double price_american_call_serial(const OptionParams& p,
                                  double* out_stderr = nullptr);

// The naive per-path recursion from Cvetanoska & Stojanovski, retained only so
// its perfect-foresight bias can be measured. Call-only. Not for pricing.
double price_american_call_serial_naive(const OptionParams& p);

// OpenMP. num_threads = 0 leaves the current setting alone.
double price_american_call_omp(const OptionParams& p, int num_threads = 0,
                               double* out_stderr = nullptr);

// OpenMP + QMC (Sobol with a digital shift, through a Brownian bridge).
double price_american_call_qmc_omp(const OptionParams& p, int num_threads = 0,
                                   uint32_t seed = 42,
                                   double* out_stderr = nullptr);
