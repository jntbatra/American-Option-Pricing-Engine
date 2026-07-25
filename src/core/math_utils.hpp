#pragma once
#include <cmath>

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

// Kept as a plain int rather than an enum class so the value can be passed
// straight into a CUDA kernel argument list.
enum OptionType { OPTION_CALL = 0, OPTION_PUT = 1 };

// Precision used to STORE and generate simulated paths on the GPU. The
// Longstaff-Schwartz regression accumulates in double regardless.
//
// Double is the default because it makes the CPU and GPU backends bit-comparable,
// which is what lets the test suite tell a real GPU bug from expected drift.
// Float exists because on a consumer card FP64 runs at 1/64 the FP32 rate; the
// resulting price difference is measured, not assumed.
enum McPrecision { MC_PRECISION_DOUBLE = 0, MC_PRECISION_FLOAT = 1 };

struct OptionParams {
    double S0;       // current underlying price
    double X;        // strike price
    double T;        // time to maturity (years)
    double r;        // risk-free rate
    double v;        // volatility
    int    m;        // number of discrete exercise points
    int    N;        // number of Monte Carlo paths
    int    type = OPTION_CALL;  // OPTION_CALL or OPTION_PUT
    int    precision = MC_PRECISION_DOUBLE;  // GPU path storage; see McPrecision
};