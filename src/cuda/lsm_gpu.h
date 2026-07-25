#pragma once
#include "../core/math_utils.hpp"

// Longstaff-Schwartz backward pass over paths already resident in device
// memory. Shared by both CUDA backends, which differ only in how they fill
// d_S -- LCG shocks or a Sobol sequence through a Brownian bridge.
//
//   d_S : device pointer to N x (m+1) doubles, point-major, so path n at time
//         i*dt is d_S[n*(m+1) + i].
//
// Returns the price discounted to t = 0, or 0.0 on a CUDA error. When
// `out_stderr` is non-null it receives the standard error of the estimate.
double lsm_price_device(const double* d_S, const OptionParams& p,
                        int threads_per_block, double* out_stderr = nullptr);
