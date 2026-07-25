// Benchmark driver. One backend per executable, selected by a BACKEND_* define.
//
// Reports, per path count:
//   price   the Monte Carlo estimate
//   stderr  its standard error -- a price without one is unfalsifiable, since
//           there is no way to tell a converged result from a noisy one
//   err     signed error against the exact reference, where one exists
//   time    median of REPS runs after a warmup, not a single sample
//
// The CUDA backends additionally split wall-clock into setup and kernel.
// Quoting the whole launcher call as "GPU time" overstates the kernel cost
// badly at small N, because setup is redone on every call.

#include "../core/backends.hpp"
#include "../core/black_scholes.hpp"
#include "../core/math_utils.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#if defined(BACKEND_CUDA) || defined(BACKEND_QMC_CUDA)
#include <cuda_runtime.h>
#include "../cuda/american_cuda.h"
#define IS_CUDA_BACKEND 1
#endif

static const int  REPS = 5;
static const int  WARMUP = 1;

#if defined(BACKEND_SERIAL)
static const char* BACKEND_NAME = "Serial (LCG)";
#elif defined(BACKEND_OMP)
static const char* BACKEND_NAME = "OpenMP (LCG)";
#elif defined(BACKEND_QMC_OMP)
static const char* BACKEND_NAME = "OpenMP QMC (Sobol+BB)";
#elif defined(BACKEND_CUDA)
static const char* BACKEND_NAME = "CUDA (LCG)";
#elif defined(BACKEND_QMC_CUDA)
static const char* BACKEND_NAME = "CUDA QMC (Sobol+BB)";
#else
#error "no BACKEND_* selected"
#endif

static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char** argv) {
    OptionParams base;
    base.S0 = 100.0;
    base.X  = 100.0;
    base.T  = 1.0;
    base.r  = 0.05;
    base.v  = 0.20;
    base.m  = 20;
    base.N  = 0;
    base.type = (argc > 1 && argv[1][0] == 'p') ? OPTION_PUT : OPTION_CALL;

    const int path_counts[] = {1000, 10000, 100000,
                               200000, 500000, 1000000};

    // A call on a non-dividend-paying stock is worth exactly its European
    // counterpart, which gives an exact yardstick. A put has no closed form.
    const bool   have_exact = (base.type == OPTION_CALL);
    const double exact = bs_call(base.S0, base.X, base.T, base.v, base.r);

    printf("# %s   m=%d  %s\n", BACKEND_NAME, base.m,
           base.type == OPTION_PUT ? "put" : "call");
    if (have_exact) printf("# exact (Black-Scholes) = %.6f\n", exact);
    printf("# median of %d runs after %d warmup\n", REPS, WARMUP);
#if defined(BACKEND_QMC_OMP) || defined(BACKEND_QMC_CUDA)
    // The textbook sd/sqrt(N) formula assumes independent draws. A Sobol
    // sequence is deterministic and correlated by construction, so the number
    // below is the standard error the SAME payoffs would have had under
    // pseudo-random sampling -- an upper bound, and usually a wildly
    // pessimistic one. Compare the `err` column: it is typically an order of
    // magnitude smaller. A real QMC error bar needs randomised QMC (average
    // over several independent digital shifts and take the sd of the means).
    printf("# NOTE: stderr is NOT a valid error estimate for QMC -- see `err`.\n");
#endif
    printf("\n");

#ifdef IS_CUDA_BACKEND
    printf("%-10s %10s %9s %9s %11s %11s %11s\n",
           "N", "price", "stderr", "err", "total(ms)", "setup(ms)", "kernel(ms)");
#else
    printf("%-10s %10s %9s %9s %11s\n",
           "N", "price", "stderr", "err", "time(ms)");
#endif

    for (int N : path_counts) {
        base.N = N;

        double price = 0.0, se = 0.0;
        std::vector<double> totals, setups, kernels;

        for (int rep = 0; rep < REPS + WARMUP; ++rep) {
            const auto t0 = std::chrono::high_resolution_clock::now();

#if defined(BACKEND_SERIAL)
            price = price_american_call_serial(base, &se);
#elif defined(BACKEND_OMP)
            price = price_american_call_omp(base, 0, &se);
#elif defined(BACKEND_QMC_OMP)
            price = price_american_call_qmc_omp(base, 0, 42u, &se);
#elif defined(BACKEND_CUDA)
            CudaTiming t;
            price = price_american_call_cuda(base, 512, &se, &t);
#elif defined(BACKEND_QMC_CUDA)
            CudaTiming t;
            price = price_american_call_qmc_cuda(base, 256, 42u, &se, &t);
#endif
            const auto t1 = std::chrono::high_resolution_clock::now();
            if (rep < WARMUP) continue;

            totals.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
#ifdef IS_CUDA_BACKEND
            setups.push_back(t.setup_ms);
            kernels.push_back(t.kernel_ms);
#endif
        }

        char err_buf[16];
        if (have_exact) snprintf(err_buf, sizeof err_buf, "%+.4f", price - exact);
        else            snprintf(err_buf, sizeof err_buf, "%9s", "-");

#ifdef IS_CUDA_BACKEND
        printf("%-10d %10.4f %9.4f %9s %11.3f %11.3f %11.3f\n",
               N, price, se, err_buf,
               median(totals), median(setups), median(kernels));
#else
        printf("%-10d %10.4f %9.4f %9s %11.3f\n",
               N, price, se, err_buf, median(totals));
#endif
        fflush(stdout);
    }
    return 0;
}
