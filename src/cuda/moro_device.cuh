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
