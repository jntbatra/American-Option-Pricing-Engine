#include "../core/black_scholes.hpp"
#include "../core/lsm.hpp"
#include "../core/paths.hpp"
#include "../core/backends.hpp"
#include "../core/math_utils.hpp"
#include <omp.h>
#include <vector>

// Both simulate_paths_lcg() and lsm_price_from_paths() carry OpenMP pragmas, so
// this backend differs from the serial one only in being compiled with
// -fopenmp. There is no second copy of the algorithm to keep in sync.
double price_american_call_omp(const OptionParams& p, int num_threads,
                               double* out_stderr) {
    if (num_threads > 0) omp_set_num_threads(num_threads);

    std::vector<double> S;
    simulate_paths_lcg(p, S);
    return lsm_price_from_paths(S.data(), p.N, p, out_stderr);
}
