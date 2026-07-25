#include "binomial.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace {

// Shared CRR sweep. `exercise_every` = 1 allows exercise at every layer;
// larger values allow it only on layers that are multiples of it.
double crr(const OptionParams& p, int steps, int exercise_every) {
    if (steps < 1) return 0.0;

    const double dt   = p.T / static_cast<double>(steps);
    const double u    = std::exp(p.v * std::sqrt(dt));
    const double d    = 1.0 / u;
    const double disc = std::exp(-p.r * dt);
    const double q    = (std::exp(p.r * dt) - d) / (u - d);

    // spot[j] for the current layer, rebuilt by scaling rather than pow().
    std::vector<double> value(steps + 1);
    std::vector<double> spot(steps + 1);

    spot[0] = p.S0 * std::pow(d, steps);
    for (int j = 1; j <= steps; ++j) spot[j] = spot[j - 1] * u * u;

    for (int j = 0; j <= steps; ++j) {
        value[j] = (p.type == OPTION_PUT) ? std::max(p.X - spot[j], 0.0)
                                          : std::max(spot[j] - p.X, 0.0);
    }

    for (int i = steps - 1; i >= 0; --i) {
        // Layer i spot grid: shift the previous one up by one down-move.
        spot[0] = p.S0 * std::pow(d, i);
        for (int j = 1; j <= i; ++j) spot[j] = spot[j - 1] * u * u;

        const bool exercisable = (i > 0) && (i % exercise_every == 0);

        for (int j = 0; j <= i; ++j) {
            double v = disc * (q * value[j + 1] + (1.0 - q) * value[j]);
            if (exercisable) {
                const double intrinsic = (p.type == OPTION_PUT)
                                             ? p.X - spot[j]
                                             : spot[j] - p.X;
                v = std::max(v, intrinsic);
            }
            value[j] = v;
        }
    }
    return value[0];
}

}  // namespace

double binomial_bermudan(const OptionParams& p, int steps_per_interval) {
    if (p.m < 1 || steps_per_interval < 1) return 0.0;
    return crr(p, (p.m + 1) * steps_per_interval, steps_per_interval);
}

double binomial_american(const OptionParams& p, int steps) {
    return crr(p, steps, 1);
}
