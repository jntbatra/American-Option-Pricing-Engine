#pragma once
#include "../core/math_utils.hpp"

// Longstaff-Schwartz backward pass over paths already resident in device
// memory. Shared by both CUDA backends, which differ only in how they fill
// d_S -- LCG shocks or a Sobol sequence through a Brownian bridge.
//
//   d_S : device pointer to (m+1) x N doubles, TIME-MAJOR, so path n at time
//         i*dt is d_S[i*N + n].
//
// Time-major, unlike the host side's point-major layout, because every kernel
// here sweeps all paths at one fixed timestep. Point-major put consecutive
// threads (m+1)*8 = 168 bytes apart, so each one pulled its own 32-byte sector
// and used 8 bytes of it -- the backward pass ran about 10x off the card's
// bandwidth. Time-major makes both the sweep and the path-generation writes
// fully coalesced.
//
// Returns the price discounted to t = 0, or 0.0 on a CUDA error. When
// `out_stderr` is non-null it receives the standard error of the estimate.
// `out_ms`, if given, receives the device time for the backward pass alone.
// Overloaded on the stored path precision. Regression accumulation is double
// in both cases -- A^T A entries scale with N, so summing a million terms in
// float would lose more than the path precision ever could.
double lsm_price_device(const double* d_S, const OptionParams& p,
                        int threads_per_block, double* out_stderr = nullptr,
                        double* out_ms = nullptr);

double lsm_price_device(const float* d_S, const OptionParams& p,
                        int threads_per_block, double* out_stderr = nullptr,
                        double* out_ms = nullptr);
