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

// Precision policy for path generation. The integer LCG state is identical
// either way; only the conversion and the downstream transcendentals differ.
template <typename T> struct PathMath;

template <> struct PathMath<double> {
    __device__ static double uniform(uint32_t& s) { return lcg_next(s); }
    __device__ static double inv_cnd(double u)    { return moro_inv_cnd_device(u); }
    __device__ static double expo(double x)       { return exp(x); }
};

template <> struct PathMath<float> {
    __device__ static float uniform(uint32_t& s) {
        s = 1664525u * s + 1013904223u;
        return __uint2float_rn(s) * 2.3283064365386963e-10f;
    }
    __device__ static float inv_cnd(float u) { return moro_inv_cnd_device(u); }
    __device__ static float expo(float x)    { return __expf(x); }
};

// Black-Scholes and payoff helpers now live in core/hd_math.hpp, compiled for
// both host and device: mc_bs_call, mc_bs_put, mc_bs_european, mc_intrinsic.
