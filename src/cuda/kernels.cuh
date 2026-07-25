#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "moro_device.cuh"
#include "../core/hd_math.hpp"

// Linear congruential generator, in double precision to match
// src/core/quasi_rng.cpp exactly. The state update is integer arithmetic and so
// is already exact; returning a double rather than a float is what makes the
// GPU draw the same uniforms as the CPU.
__device__ __forceinline__
double lcg_next(uint32_t& state) {
    state = 1664525u * state + 1013904223u;
    return static_cast<double>(state) * 2.3283064365386963e-10;
}

// Black-Scholes and payoff helpers now live in core/hd_math.hpp, compiled for
// both host and device: mc_bs_call, mc_bs_put, mc_bs_european, mc_intrinsic.
