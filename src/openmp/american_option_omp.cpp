#include "../core/black_scholes.hpp"
#include "../core/backward_induction.hpp"
#include "../core/quasi_rng.hpp"
#include "../core/moro_inv_cnd.hpp"
#include "../core/math_utils.hpp"
#include <omp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

double price_american_call_omp(const OptionParams& p, int num_threads = 0) {
    if (num_threads > 0) omp_set_num_threads(num_threads);

    const double dt    = p.T / static_cast<double>(p.m + 1);
    const double sqdt  = std::sqrt(dt);
    const double drift = (p.r - 0.5 * p.v * p.v) * dt;
    const double discount = std::exp(-p.r * dt);

    double total_sum = 0.0;

    #pragma omp parallel reduction(+:total_sum)
    {
        std::vector<double> S(p.m + 1);

        #pragma omp for schedule(dynamic, 64)
        for (int path = 0; path < p.N; ++path) {
            uint32_t seed = static_cast<uint32_t>(path + 1) * 1234567u;

            S[0] = p.S0;
            for (int i = 1; i <= p.m; ++i) {
                double u = lcg_next_uniform(seed);
                double z = moro_inv_cnd(u);
                S[i] = S[i-1] * std::exp(drift + p.v * sqdt * z);
            }

            total_sum += american_call_path_value(S.data(), p.m, p.X, dt,
                                                  p.v, p.r, discount);
        }
    }

    // american_call_path_value() already discounts to t = 0.
    return total_sum / static_cast<double>(p.N);
}