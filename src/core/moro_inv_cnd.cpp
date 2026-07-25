#include "moro_inv_cnd.hpp"
#include <cmath>

double moro_inv_cnd(double u) {
    static const double a[] = {
        2.50662823884, -18.61500062529,  41.39119773534, -25.44106049637
    };
    static const double b[] = {
        -8.47351093090,  23.08336743743, -21.06224101826,   3.13082909833
    };
    static const double c[] = {
        0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
        0.0276438810333863, 0.0038405729373609, 0.0003951896511349,
        0.0000321767881768, 0.0000002888167364, 0.0000003960315187
    };

    // Phi^-1 is undefined at the endpoints, and the tail branch below evaluates
    // log(-log(u)), so u = 0 or u = 1 produces +/-inf and poisons the whole
    // path. The LCG reaches state 0 within its period, so clamp here rather
    // than at each of the four call sites.
    //
    // This perturbs nothing legitimate: the generator's outputs lie in
    // [2^-32, 1 - 2^-32] = [2.3e-10, 1 - 2.3e-10], strictly inside the clamp,
    // so only the pathological endpoints move.
    if (u < 1e-10) u = 1e-10;
    else if (u > 1.0 - 1e-10) u = 1.0 - 1e-10;

    double x = u - 0.5;
    double r;
    if (std::fabs(x) < 0.42) {
        r = x * x;
        r = x * (((a[3]*r + a[2])*r + a[1])*r + a[0]) /
               ((((b[3]*r + b[2])*r + b[1])*r + b[0])*r + 1.0);
    } else {
        r = (x > 0.0) ? std::log(-std::log(1.0 - u))
                      : std::log(-std::log(u));
        r = c[0] + r*(c[1] + r*(c[2] + r*(c[3] + r*(c[4] +
            r*(c[5] + r*(c[6] + r*(c[7] + r*c[8])))))));
        if (x < 0.0) r = -r;
    }
    return r;
}