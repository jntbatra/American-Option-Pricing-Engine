// Validation suite for the CPU backends. Runs without a GPU.
//
// Checks are reported rather than assert()ed so that one failure does not hide
// the rest, and the process exits non-zero if any hard check fails.

#include "core/math_utils.hpp"
#include "core/black_scholes.hpp"
#include "core/binomial.hpp"
#include "core/moro_inv_cnd.hpp"
#include "core/sobol.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

double price_american_call_serial(const OptionParams&);
double price_american_call_serial_naive(const OptionParams&);
#ifdef HAVE_OPENMP
double price_american_call_omp(const OptionParams&, int);
double price_american_call_qmc_omp(const OptionParams&, int, uint32_t);
#endif

static int g_failures = 0;
static int g_xfail    = 0;

static void check(bool ok, const char* name, const char* detail) {
    printf("%-7s %-34s %s\n", ok ? "[ ok ]" : "[FAIL]", name, detail);
    if (!ok) ++g_failures;
}

// A check that is broken for a understood reason. Reported loudly, but does not
// fail the build -- it is the acceptance criterion for the pending fix.
static void check_known_broken(bool ok, const char* name, const char* detail,
                               const char* blocked_on) {
    if (ok) {
        printf("[ ok ]  %-34s %s\n        ^ known-broken check now PASSES, promote it to check()\n",
               name, detail);
    } else {
        printf("[XFAIL] %-34s %s\n        blocked on: %s\n",
               name, detail, blocked_on);
        ++g_xfail;
    }
}

// ---------------------------------------------------------------- primitives

static void test_bs_formula() {
    // Hull: S=42, X=40, T=0.5, r=0.1, v=0.2 -> 4.7594
    double c = bs_call(42.0, 40.0, 0.5, 0.2, 0.1);
    char buf[128];
    snprintf(buf, sizeof buf, "bs_call=%.4f expected 4.7594", c);
    check(std::fabs(c - 4.7594) < 1e-3, "black_scholes_hull", buf);
}

static void test_moro() {
    double z = moro_inv_cnd(0.975);
    char buf[128];
    snprintf(buf, sizeof buf, "moro(0.975)=%.5f expected 1.95996", z);
    check(std::fabs(z - 1.959964) < 1e-4, "moro_inv_cnd", buf);
}

// Regression test for the bit-reversed direction-number indexing. The first
// points of an unshifted Sobol sequence are exact dyadic rationals.
static void test_sobol_known_values() {
    static const double expect_dim0[8] =
        {0.0, 0.5, 0.75, 0.25, 0.375, 0.875, 0.625, 0.125};

    SobolGenerator gen(3);
    std::vector<double> pts;
    gen.generate(8, pts);

    double worst = 0.0;
    for (int n = 0; n < 8; ++n)
        worst = std::fmax(worst, std::fabs(pts[n * 3 + 0] - expect_dim0[n]));

    char buf[128];
    snprintf(buf, sizeof buf, "max err vs van der Corput = %.3g", worst);
    check(worst < 1e-9, "sobol_dim0_known_values", buf);

    // generate() and point() must agree, and both must be point-major.
    double worst_api = 0.0;
    for (uint32_t n = 0; n < 8; ++n) {
        double out[3];
        gen.point(n, out);
        for (int d = 0; d < 3; ++d)
            worst_api = std::fmax(worst_api, std::fabs(out[d] - pts[n * 3 + d]));
    }
    snprintf(buf, sizeof buf, "max |point() - generate()| = %.3g", worst_api);
    check(worst_api < 1e-12, "sobol_generate_matches_point", buf);
}

// Every dimension must be uniform on [0,1). The bit-reversed version pinned
// dimension 0 near zero, which this catches independently of exact values.
static void test_sobol_uniformity() {
    const int d = 8, N = 4096;
    SobolGenerator gen(d);
    std::vector<double> pts;
    gen.generate(N, pts);

    double worst = 0.0;
    for (int dim = 0; dim < d; ++dim) {
        double sum = 0.0;
        for (int n = 0; n < N; ++n) sum += pts[n * d + dim];
        worst = std::fmax(worst, std::fabs(sum / N - 0.5));
    }
    char buf[128];
    snprintf(buf, sizeof buf, "max |mean - 0.5| over 8 dims = %.3g", worst);
    check(worst < 1e-3, "sobol_dimension_uniformity", buf);
}

// ------------------------------------------------------------------- pricing

// Reference case: call on a non-dividend-paying stock.
static OptionParams reference_case(int m, int N) {
    OptionParams p;
    p.S0 = 100.0; p.X = 100.0; p.T = 1.0; p.r = 0.05; p.v = 0.20;
    p.m = m; p.N = N;
    return p;
}

// It is a theorem that early exercise of an American call on a
// non-dividend-paying stock is never optimal, so the American price must equal
// the European one. This is the primary acceptance test for the pricer: any gap
// is pure bias or bug, and unlike Monte Carlo noise it does not shrink with N.
static void test_american_call_equals_european() {
    const double european = bs_call(100.0, 100.0, 1.0, 0.20, 0.05);

    for (int m : {5, 10, 20}) {
        OptionParams p = reference_case(m, 200000);
        double american = price_american_call_serial(p);
        double err = american - european;

        char name[64], buf[160];
        snprintf(name, sizeof name, "american_eq_european_m%d", m);
        snprintf(buf, sizeof buf,
                 "american=%.4f european=%.4f err=%+.4f (%+.2f%%)",
                 american, european, err, 100.0 * err / european);
        check(std::fabs(err) < 0.05, name, buf);
    }
}

// The call test above cannot distinguish "LSM works" from "LSM never
// exercises", because never exercising is the right answer there. A put is the
// real test: early exercise binds, there is no closed form, and the binomial
// tree is an independent oracle sharing no code with the Monte Carlo path.
//
// LSM is low-biased -- the fitted exercise policy is suboptimal, and a
// suboptimal policy under-values -- so the check is one-sided below and tight
// above.
static void test_american_put_vs_binomial() {
    struct Case { double S0, X, v; const char* label; };
    static const Case cases[] = {
        {100.0,  100.0, 0.20, "atm"},
        { 90.0,  100.0, 0.20, "itm"},
        {110.0,  100.0, 0.20, "otm"},
        {100.0,  100.0, 0.40, "high_vol"},
    };

    for (const Case& c : cases) {
        OptionParams p;
        p.S0 = c.S0; p.X = c.X; p.T = 1.0; p.r = 0.05; p.v = c.v;
        p.m = 20; p.N = 200000; p.type = OPTION_PUT;

        // Same exercise dates as the Monte Carlo grid, so the tree prices the
        // same Bermudan contract rather than a finer-grained American one.
        double reference = binomial_bermudan(p, 250);
        double lsm       = price_american_call_serial(p);
        double err       = lsm - reference;

        char name[64], buf[176];
        snprintf(name, sizeof name, "american_put_vs_tree_%s", c.label);
        snprintf(buf, sizeof buf,
                 "lsm=%.4f tree=%.4f err=%+.4f (%+.2f%%)",
                 lsm, reference, err, 100.0 * err / reference);
        // Low-biased: allow more room below than above.
        check(err < 0.05 && err > -0.20, name, buf);
    }
}

// A put must be worth strictly more than its European counterpart -- that
// difference is the early-exercise premium, and if LSM never exercised it would
// be zero. Guards against a regression that silently disables exercise.
static void test_put_early_exercise_premium() {
    OptionParams p;
    p.S0 = 90.0; p.X = 100.0; p.T = 1.0; p.r = 0.05; p.v = 0.20;
    p.m = 20; p.N = 200000; p.type = OPTION_PUT;

    double american = price_american_call_serial(p);
    double european = bs_put(p.S0, p.X, p.T, p.v, p.r);

    char buf[160];
    snprintf(buf, sizeof buf, "american=%.4f european=%.4f premium=%+.4f",
             american, european, american - european);
    check(american - european > 0.10, "put_early_exercise_premium", buf);
}

// Side-by-side with the original paper's per-path recursion, which decides
// exercise using each path's own realised future.
static void test_lsm_vs_naive_bias() {
    const double european = bs_call(100.0, 100.0, 1.0, 0.20, 0.05);
    printf("  call on non-dividend stock, exact price = %.4f\n", european);
    printf("    %-5s %10s %10s %10s %10s\n",
           "m", "naive", "bias", "lsm", "bias");
    for (int m = 1; m <= 21; m += 4) {
        OptionParams p = reference_case(m, 100000);
        double naive = price_american_call_serial_naive(p);
        double lsm   = price_american_call_serial(p);
        printf("    %-5d %10.4f %+10.4f %10.4f %+10.4f\n",
               m, naive, naive - european, lsm, lsm - european);
    }
}

#ifdef HAVE_OPENMP
// Cross-backend agreement. Note that this binary compiles the serial backend
// WITH -fopenmp (it links OpenMP for the other two), so the serial and OpenMP
// entry points run the same parallel code here and comparing them proves
// nothing. What does prove something is running the parallel backend at one
// thread versus many: the LSM regression accumulates A^T A by reduction, so a
// thread-count-dependent answer would mean the reduction is wrong.
//
// The results are not bit-identical -- floating-point summation order differs,
// and a path sitting exactly on the exercise boundary can flip -- so this is a
// tolerance check, not an equality one.
static void test_backends_agree() {
    OptionParams p = reference_case(10, 200000);

    // Capture the default before the single-threaded run, since passing a
    // thread count calls omp_set_num_threads() and that setting persists.
    const int max_threads = omp_get_max_threads();

    double one  = price_american_call_omp(p, 1);
    double many = price_american_call_omp(p, max_threads);

    char buf[176];
    snprintf(buf, sizeof buf, "1 thread=%.6f %d threads=%.6f diff=%.2g",
             one, max_threads, many, std::fabs(one - many));
    check(std::fabs(one - many) < 1e-4, "omp_thread_count_invariant", buf);

    // QMC draws a different point set, so it converges to the same value but
    // not path-for-path. This is what would have caught the Sobol
    // point-major/dimension-major mix-up.
    double qmc = price_american_call_qmc_omp(p, 0, 42u);
    snprintf(buf, sizeof buf, "lcg=%.4f qmc_omp=%.4f diff=%.4f",
             many, qmc, std::fabs(many - qmc));
    check(std::fabs(many - qmc) < 0.25, "lcg_vs_qmc_omp_agree", buf);
}
#endif

int main() {
    printf("=== primitives ===\n");
    test_bs_formula();
    test_moro();
    test_sobol_known_values();
    test_sobol_uniformity();

    printf("\n=== pricing ===\n");
    test_american_call_equals_european();
    test_american_put_vs_binomial();
    test_put_early_exercise_premium();
#ifdef HAVE_OPENMP
    test_backends_agree();
#else
    printf("[skip]  cross-backend agreement (built without OpenMP)\n");
#endif

    printf("\n=== diagnostics ===\n");
    test_lsm_vs_naive_bias();

    printf("\n%d failure(s), %d known-broken.\n", g_failures, g_xfail);
    return g_failures == 0 ? 0 : 1;
}
