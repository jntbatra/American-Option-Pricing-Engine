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
// NOTE: this is the naive per-path recursion from Cvetanoska & Stojanovski,
// which exercises with knowledge of its own future and is therefore biased
// high -- see the diagnostic table printed by `validate`. It is retained only
// so the bias can be measured against the real pricer. Production pricing goes
// through Longstaff-Schwartz in core/lsm.hpp.

#include "hd_math.hpp"

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
