#pragma once
#include "math_utils.hpp"
#include <cmath>

// Option math compiled identically for host and device, so the CPU and CUDA
// backends compute the same function rather than two similar ones. Unqualified
// sqrt/log/exp/erfc resolve to the double overloads on both sides.

#if defined(__CUDACC__)
  #define MC_HD __host__ __device__
#else
  #define MC_HD
#endif

MC_HD inline double mc_max(double a, double b) { return a > b ? a : b; }

MC_HD inline double mc_cnd(double d) {
    return 0.5 * erfc(-d * 0.70710678118654752440);
}

MC_HD inline double mc_bs_call(double S, double X, double t, double v, double r) {
    if (t <= 0.0) return mc_max(S - X, 0.0);
    const double sqrt_t = sqrt(t);
    const double d1 = (log(S / X) + (r + 0.5 * v * v) * t) / (v * sqrt_t);
    const double d2 = d1 - v * sqrt_t;
    return S * mc_cnd(d1) - X * exp(-r * t) * mc_cnd(d2);
}

MC_HD inline double mc_bs_put(double S, double X, double t, double v, double r) {
    if (t <= 0.0) return mc_max(X - S, 0.0);
    // Put-call parity.
    return mc_bs_call(S, X, t, v, r) - S + X * exp(-r * t);
}

MC_HD inline double mc_bs_european(int type, double S, double X, double t,
                                   double v, double r) {
    return type == OPTION_PUT ? mc_bs_put(S, X, t, v, r)
                              : mc_bs_call(S, X, t, v, r);
}

MC_HD inline double mc_intrinsic(int type, double S, double X) {
    return type == OPTION_PUT ? X - S : S - X;
}
