#pragma once
#include "math_utils.hpp"

// Cox-Ross-Rubinstein binomial tree, used purely as an independent oracle for
// the Monte Carlo backends: it shares no code with them and converges to the
// true price as the step count grows.
//
// A call on a non-dividend-paying stock can be checked against Black-Scholes,
// but a put has no closed form, so this is the only way to tell whether the
// Longstaff-Schwartz exercise boundary is actually right.

// Prices the SAME Bermudan contract the Monte Carlo backends do: maturity
// p.T = (p.m + 1)*dt, with exercise allowed only at the p.m dates dt, 2dt, ...,
// m*dt. The tree runs (p.m + 1) * steps_per_interval steps so that every
// exercise date lands exactly on a layer.
//
// Comparing against a fully American tree instead would be an apples-to-oranges
// test: continuous exercise is worth strictly more than m discrete dates.
double binomial_bermudan(const OptionParams& p, int steps_per_interval);

// Exercise allowed at every one of `steps` layers, i.e. the continuous-exercise
// American limit. Provided for reference; not what the MC backends price.
double binomial_american(const OptionParams& p, int steps);
