#include "../core/black_scholes.hpp"
#include "../core/backward_induction.hpp"
#include "../core/lsm.hpp"
#include "../core/paths.hpp"
#include "../core/quasi_rng.hpp"
#include "../core/moro_inv_cnd.hpp"
#include "../core/backends.hpp"
#include "../core/math_utils.hpp"
#include <vector>
#include <cmath>
#include <cstdint>

double price_american_call_serial(const OptionParams& p, double* out_stderr) {
    std::vector<double> S;
    simulate_paths_lcg(p, S);
    return lsm_price_from_paths(S.data(), p.N, p, out_stderr);
}

// The original per-path recursion from Cvetanoska & Stojanovski, kept so the
// validation suite can quantify its perfect-foresight bias against LSM. Not
// used for pricing, and call-only.
double price_american_call_serial_naive(const OptionParams& p) {
    const double dt       = p.T / static_cast<double>(p.m + 1);
    const double sqdt     = std::sqrt(dt);
    const double drift    = (p.r - 0.5 * p.v * p.v) * dt;
    const double discount = std::exp(-p.r * dt);

    double sum = 0.0;
    std::vector<double> S(p.m + 1);

    for (int path = 0; path < p.N; ++path) {
        uint32_t seed = static_cast<uint32_t>(path + 1) * 1234567u;

        S[0] = p.S0;
        for (int i = 1; i <= p.m; ++i) {
            double u = lcg_next_uniform(seed);
            double z = moro_inv_cnd(u);
            S[i] = S[i - 1] * std::exp(drift + p.v * sqdt * z);
        }

        sum += american_call_path_value(S.data(), p.m, p.X, dt, p.v, p.r, discount);
    }

    // american_call_path_value() already discounts to t = 0.
    return sum / static_cast<double>(p.N);
}
