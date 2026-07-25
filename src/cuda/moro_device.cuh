#pragma once
#include <cuda_runtime.h>

// Moro's inverse cumulative normal, in double precision.
//
// This deliberately mirrors src/core/moro_inv_cnd.cpp coefficient for
// coefficient. The earlier single-precision version carried around 1e-7 of tail
// error -- larger than the QMC convergence advantage it was meant to help
// measure -- and made the CPU and GPU backends incomparable, so no test could
// tell a real GPU bug from expected precision drift. The cost is a slower
// kernel; a trustworthy comparison is the point of the project.
// Single-precision twin, used only by the float path-generation variant. Its
// tail error is ~1e-7, five orders of magnitude below the Monte Carlo standard
// error at any path count this project runs, and `profile_gpu` measures the
// resulting price difference rather than assuming it is negligible.
__device__ __forceinline__ float moro_inv_cnd_device(float u) {
    const float a0 =   2.50662823884f;
    const float a1 = -18.61500062529f;
    const float a2 =  41.39119773534f;
    const float a3 = -25.44106049637f;
    const float b0 =  -8.47351093090f;
    const float b1 =  23.08336743743f;
    const float b2 = -21.06224101826f;
    const float b3 =   3.13082909833f;
    const float c0 = 0.3374754822726147f;
    const float c1 = 0.9761690190917186f;
    const float c2 = 0.1607979714918209f;
    const float c3 = 0.0276438810333863f;
    const float c4 = 0.0038405729373609f;
    const float c5 = 0.0003951896511349f;
    const float c6 = 0.0000321767881768f;
    const float c7 = 0.0000002888167364f;
    const float c8 = 0.0000003960315187f;

    // __uint2float_rn(0xFFFFFFFF) rounds up to exactly 2^32, so the float LCG
    // CAN return exactly 1.0 -- observed as inf prices at N = 4e6 before this
    // clamp. 0.99999994f is nextafterf(1.0f, 0.0f); a tighter bound such as
    // 1 - 1e-10 is not representable and rounds straight back to 1.0f.
    u = fminf(fmaxf(u, 1.0e-7f), 0.99999994f);

    float x = u - 0.5f;
    float r;
    if (fabsf(x) < 0.42f) {
        r = x * x;
        r = x * (((a3 * r + a2) * r + a1) * r + a0) /
               ((((b3 * r + b2) * r + b1) * r + b0) * r + 1.0f);
    } else {
        r = (x > 0.0f) ? logf(-logf(1.0f - u)) : logf(-logf(u));
        r = c0 + r * (c1 + r * (c2 + r * (c3 + r * (c4 +
            r * (c5 + r * (c6 + r * (c7 + r * c8)))))));
        if (x < 0.0f) r = -r;
    }
    return r;
}

__device__ __forceinline__ double moro_inv_cnd_device(double u) {
    const double a0 =   2.50662823884;
    const double a1 = -18.61500062529;
    const double a2 =  41.39119773534;
    const double a3 = -25.44106049637;
    const double b0 =  -8.47351093090;
    const double b1 =  23.08336743743;
    const double b2 = -21.06224101826;
    const double b3 =   3.13082909833;
    const double c0 = 0.3374754822726147;
    const double c1 = 0.9761690190917186;
    const double c2 = 0.1607979714918209;
    const double c3 = 0.0276438810333863;
    const double c4 = 0.0038405729373609;
    const double c5 = 0.0003951896511349;
    const double c6 = 0.0000321767881768;
    const double c7 = 0.0000002888167364;
    const double c8 = 0.0000003960315187;

    // Matches the clamp in core/moro_inv_cnd.cpp so host and device agree.
    u = fmin(fmax(u, 1e-10), 1.0 - 1e-10);

    double x = u - 0.5;
    double r;
    if (fabs(x) < 0.42) {
        r = x * x;
        r = x * (((a3 * r + a2) * r + a1) * r + a0) /
               ((((b3 * r + b2) * r + b1) * r + b0) * r + 1.0);
    } else {
        r = (x > 0.0) ? log(-log(1.0 - u)) : log(-log(u));
        r = c0 + r * (c1 + r * (c2 + r * (c3 + r * (c4 +
            r * (c5 + r * (c6 + r * (c7 + r * c8)))))));
        if (x < 0.0) r = -r;
    }
    return r;
}
