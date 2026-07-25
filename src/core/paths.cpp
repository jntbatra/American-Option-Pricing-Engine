#include "paths.hpp"
#include "quasi_rng.hpp"
#include "moro_inv_cnd.hpp"
#include <cmath>
#include <cstdint>

void simulate_paths_lcg(const OptionParams& p, std::vector<double>& S) {
    const int    m      = p.m;
    const int    stride = m + 1;
    const double dt     = p.T / static_cast<double>(m + 1);
    const double sqdt   = std::sqrt(dt);
    const double drift  = (p.r - 0.5 * p.v * p.v) * dt;

    S.resize(static_cast<size_t>(p.N) * stride);

    #pragma omp parallel for schedule(static)
    for (int path = 0; path < p.N; ++path) {
        uint32_t seed = static_cast<uint32_t>(path + 1) * 1234567u;
        double* row = &S[static_cast<size_t>(path) * stride];

        row[0] = p.S0;
        for (int i = 1; i <= m; ++i) {
            const double u = lcg_next_uniform(seed);
            const double z = moro_inv_cnd(u);
            row[i] = row[i - 1] * std::exp(drift + p.v * sqdt * z);
        }
    }
}
