#pragma once
#include <cmath>

// Shared backward-induction step, used by every backend (serial, OpenMP,
// OpenMP+QMC, CUDA, CUDA+QMC) so the discounting convention cannot drift
// between them.
//
// TIME GRID
//   dt   = T / (m + 1)
//   S[i] = underlying at time i*dt, for i = 0..m
//   The last grid point S[m] sits at T - dt, NOT at T. The m exercise
//   opportunities are at times dt, 2dt, ..., m*dt; the option still has dt of
//   life left after the final one, which is worth the European (Black-Scholes)
//   value.
//
// DISCOUNTING
//   The returned value is already discounted to t = 0. Callers must average it
//   over paths and NOT apply a further exp(-r*T) -- doing both was worth about
//   a 4.5% under-price at m = 10.
//
// NOTE: this is still the naive per-path recursion from Cvetanoska &
// Stojanovski, which exercises with knowledge of its own future and is
// therefore biased high. Replacing it with Longstaff-Schwartz is tracked
// separately; the interface here is what LSM will slot into.

#if defined(__CUDACC__)
  #define MC_HD __host__ __device__
#else
  #define MC_HD
#endif

MC_HD inline double mc_max(double a, double b) { return a > b ? a : b; }

// Black-Scholes call, callable from host and device.
MC_HD inline double mc_bs_call(double S, double X, double t, double v, double r) {
    if (t <= 0.0) return mc_max(S - X, 0.0);
    const double sqrt_t = sqrt(t);
    const double d1 = (log(S / X) + (r + 0.5 * v * v) * t) / (v * sqrt_t);
    const double d2 = d1 - v * sqrt_t;
    const double cnd1 = 0.5 * erfc(-d1 * 0.70710678118654752440);
    const double cnd2 = 0.5 * erfc(-d2 * 0.70710678118654752440);
    return S * cnd1 - X * exp(-r * t) * cnd2;
}

// Values one simulated path S[0..m] and returns its contribution discounted to
// t = 0. `discount` must equal exp(-r*dt).
MC_HD inline double american_call_path_value(const double* S, int m,
                                             double X, double dt,
                                             double v, double r,
                                             double discount) {
    // Node m (t = T - dt): exercise now, or hold to maturity for the European
    // value of the remaining dt.
    double c = mc_max(S[m] - X, mc_bs_call(S[m], X, dt, v, r));

    // Nodes m-1 .. 1: exercise now, or discount the next node back one step.
    for (int i = m - 1; i >= 1; --i) {
        c = mc_max(S[i] - X, c * discount);
    }

    // Node 1 sits at t = dt; one more discount factor reaches t = 0.
    return c * discount;
}
