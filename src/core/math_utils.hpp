#pragma once
#include <cmath>

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

// Kept as a plain int rather than an enum class so the value can be passed
// straight into a CUDA kernel argument list.
enum OptionType { OPTION_CALL = 0, OPTION_PUT = 1 };

struct OptionParams {
    double S0;       // current underlying price
    double X;        // strike price
    double T;        // time to maturity (years)
    double r;        // risk-free rate
    double v;        // volatility
    int    m;        // number of discrete exercise points
    int    N;        // number of Monte Carlo paths
    int    type = OPTION_CALL;  // OPTION_CALL or OPTION_PUT
};