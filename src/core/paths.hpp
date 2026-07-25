#pragma once
#include "math_utils.hpp"
#include <vector>

// Simulates all N GBM paths into `S`, point-major: S[n*(m+1) + i] is path n at
// time i*dt, with S[n*(m+1)] = S0.
//
// LSM needs the whole path set resident, because the regression at each
// exercise date is cross-sectional. The naive recursion could consume one path
// at a time; this cannot.
//
// The loop carries an OpenMP pragma. Targets built without -fopenmp (the serial
// backend) simply ignore it, so there is one implementation rather than two.
void simulate_paths_lcg(const OptionParams& p, std::vector<double>& S);
