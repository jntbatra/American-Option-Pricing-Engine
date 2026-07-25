#include "core/sobol.hpp"
#include "core/scramble.hpp"
#include "core/moro_inv_cnd.hpp"
#include "core/black_scholes.hpp"
#include "core/lsm.hpp"
#include "core/brownian_bridge.hpp"
#include "core/backends.hpp"
#include "core/math_utils.hpp"
#include <omp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

double price_american_call_qmc_omp(const OptionParams& p,
                                    int num_threads,
                                    uint32_t seed,
                                    double* out_stderr)
{
    if (num_threads > 0) omp_set_num_threads(num_threads);

    const int    m      = p.m;
    const int    stride = m + 1;
    const double dt     = p.T / static_cast<double>(m + 1);

    SobolGenerator gen(m);
    gen.set_digital_shift(make_digital_shift(m, seed));
    std::vector<double> u_flat;
    gen.generate(p.N, u_flat);
    // gen.generate() stores POINT-MAJOR: the m coordinates of path n occupy
    // u_flat[n*m .. n*m + m - 1]. (An earlier revision assumed dimension-major
    // and transposed, which shuffled coordinates between paths.)

    std::vector<double> z_flat(u_flat.size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(u_flat.size()); ++i) {
        double u = std::max(1e-10, std::min(u_flat[i], 1.0 - 1e-10));
        z_flat[i] = moro_inv_cnd(u);
    }
    // z_flat inherits the point-major layout: z_flat[n * m + d].

    auto bridge = build_brownian_bridge(m, dt);

    // LSM regresses across paths at each exercise date, so the whole path set
    // has to be materialised rather than consumed one path at a time.
    std::vector<double> S(static_cast<size_t>(p.N) * stride);

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < p.N; ++n) {
        // Point-major layout: this path's m coordinates are contiguous.
        const double* z = &z_flat[static_cast<size_t>(n) * m];
        simulate_path_bb(z, bridge, p.S0, p.r, p.v, dt, m,
                         &S[static_cast<size_t>(n) * stride]);
    }

    return lsm_price_from_paths(S.data(), p.N, p, out_stderr);
}