// Where does GPU time actually go?
//
// Splits each CUDA backend into setup / path generation / LSM backward pass so
// optimisation targets the measured bottleneck rather than the assumed one.

#include "core/math_utils.hpp"
#include "cuda/american_cuda.h"
#include <algorithm>
#include <cstdio>
#include <vector>

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

struct Run { double price, se, setup, gen, lsm, total; };

static Run measure(OptionParams p, bool qmc, int reps) {
    std::vector<double> st, gen, lsm, tot;
    double price = 0.0, se = 0.0;
    for (int r = 0; r <= reps; ++r) {
        CudaTiming t;
        price = qmc ? price_american_call_qmc_cuda(p, 256, 42u, &se, &t)
                    : price_american_call_cuda(p, 512, &se, &t);
        if (r == 0) continue;                    // warmup
        st.push_back(t.setup_ms);
        gen.push_back(t.pathgen_ms);
        lsm.push_back(t.lsm_ms);
        tot.push_back(t.total_ms);
    }
    return { price, se, median(st), median(gen), median(lsm), median(tot) };
}

int main() {
    OptionParams p;
    p.S0 = 100.0; p.X = 100.0; p.T = 1.0; p.r = 0.05; p.v = 0.20; p.m = 20;

    const int Ns[] = {100000, 1000000, 4000000};
    const int REPS = 5;
    const double EXACT = 10.450583588;

    printf("=== Where GPU time goes (fp64 paths) ===\n");
    printf("%-10s %-10s %9s %9s %9s %9s %7s %7s\n",
           "backend", "N", "setup", "pathgen", "lsm", "total", "gen%", "lsm%");
    for (int q = 0; q < 2; ++q) {
        for (int N : Ns) {
            p.N = N; p.precision = MC_PRECISION_DOUBLE;
            Run r = measure(p, q == 1, REPS);
            printf("%-10s %-10d %9.3f %9.3f %9.3f %9.3f %6.1f%% %6.1f%%\n",
                   q ? "CUDA QMC" : "CUDA LCG", N,
                   r.setup, r.gen, r.lsm, r.total,
                   100 * r.gen / r.total, 100 * r.lsm / r.total);
            fflush(stdout);
        }
    }

    // Does dropping the paths to single precision cost any accuracy that
    // matters? The comparison is against the exact Black-Scholes price, with
    // the Monte Carlo standard error alongside for scale.
    printf("\n=== fp64 vs fp32 path storage (exact = %.6f) ===\n", EXACT);
    printf("%-10s %-10s %12s %12s %11s %11s %9s %9s %8s\n",
           "backend", "N", "fp64 price", "fp32 price", "fp64 err",
           "fp32 err", "mc stderr", "fp32 ms", "speedup");
    for (int q = 0; q < 2; ++q) {
        for (int N : Ns) {
            p.N = N;
            p.precision = MC_PRECISION_DOUBLE;
            Run d = measure(p, q == 1, REPS);
            p.precision = MC_PRECISION_FLOAT;
            Run f = measure(p, q == 1, REPS);
            printf("%-10s %-10d %12.6f %12.6f %+11.6f %+11.6f %9.6f %9.3f %7.2fx\n",
                   q ? "CUDA QMC" : "CUDA LCG", N,
                   d.price, f.price, d.price - EXACT, f.price - EXACT,
                   d.se, f.total, d.total / f.total);
            fflush(stdout);
        }
    }
    return 0;
}
