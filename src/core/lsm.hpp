#pragma once
#include "math_utils.hpp"

// Longstaff-Schwartz least-squares Monte Carlo.
//
// The naive per-path recursion in backward_induction.hpp lets each path decide
// whether to exercise using its own realised future, which is perfect foresight
// and biases the price high without bound as m grows. LSM replaces that
// decision with a cross-sectional estimate: at each exercise date, regress the
// realised discounted future cashflow on functions of the current spot across
// all in-the-money paths, and exercise when the intrinsic value beats the
// FITTED continuation.
//
// Two details matter for correctness:
//
//  * The regression supplies the exercise DECISION only. Valuation always uses
//    the realised cashflow, never the fitted value. Substituting the fitted
//    value back in reintroduces bias.
//  * Only in-the-money paths enter the regression. Out-of-the-money paths carry
//    no exercise decision and including them degrades the fit exactly where the
//    boundary lives.
//
// The result is low-biased (the estimated policy is suboptimal, and any
// suboptimal policy under-values), which is the safe direction and converges
// from below as N and the basis grow.

// Polynomial basis in moneyness x = S/X: 1, x, x^2. Longstaff & Schwartz use
// three basis functions in their worked examples; normalising by the strike
// keeps the normal equations well conditioned across strike scales.
inline constexpr int LSM_BASIS = 3;

// Minimum in-the-money paths at a node before the regression is trusted. Below
// this the fit is noise, so the node is treated as "never exercise" -- the
// conservative choice, since it can only under-value.
inline constexpr int LSM_MIN_ITM = 8;

// Accumulates the normal equations A^T A and A^T y for one exercise date.
struct LsmNormalEq {
    double ata[LSM_BASIS * LSM_BASIS];
    double atb[LSM_BASIS];
    long long count;

    void reset();
    void add(double x, double y);          // x = moneyness, y = discounted cashflow
    void merge(const LsmNormalEq& other);
};

inline void lsm_basis(double x, double* b) {
    b[0] = 1.0;
    b[1] = x;
    b[2] = x * x;
}

// Solves the accumulated system into beta[LSM_BASIS]. Returns false when the
// system is unusable (too few points, or singular even after ridging), in which
// case the caller must not exercise at this node.
bool lsm_solve(const LsmNormalEq& eq, double* beta);

inline double lsm_eval(const double* beta, double x) {
    double b[LSM_BASIS];
    lsm_basis(x, b);
    double acc = 0.0;
    for (int k = 0; k < LSM_BASIS; ++k) acc += beta[k] * b[k];
    return acc;
}

inline double option_intrinsic(int type, double S, double X) {
    return type == OPTION_PUT ? X - S : S - X;
}

// Backward pass over pre-simulated paths.
//   S : N x (m+1) doubles, point-major -- S[n*(m+1) + i] is path n at time i*dt.
// Returns the price discounted to t = 0.
//
// When `out_stderr` is non-null it receives the standard error of the estimate,
// sd(discounted cashflow)/sqrt(N). A price quoted without it is unfalsifiable:
// there is no way to tell a converged result from a noisy one.
//
// Compiled once and used by both the serial and OpenMP backends: the loops
// carry OpenMP pragmas, which the serial target simply ignores because it is
// built without -fopenmp.
double lsm_price_from_paths(const double* S, int N, const OptionParams& p,
                            double* out_stderr = nullptr);
