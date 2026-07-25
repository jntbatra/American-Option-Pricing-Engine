// Validation suite for the CPU backends. Runs without a GPU.
//
// Checks are reported rather than assert()ed so that one failure does not hide
// the rest, and the process exits non-zero if any hard check fails.

#include "core/math_utils.hpp"
#include "core/black_scholes.hpp"
#include "core/moro_inv_cnd.hpp"
#include "core/sobol.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

double price_american_call_serial(const OptionParams&);
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
// the European one. This is the real acceptance test for the pricer: any gap is
// pure bias or bug, and unlike Monte Carlo noise it does not shrink with N.
static void test_american_call_equals_european() {
    const double european = bs_call(100.0, 100.0, 1.0, 0.20, 0.05);

    for (int m : {5, 10, 20}) {
        OptionParams p = reference_case(m, 200000);
        double american = price_american_call_serial(p);
        double err = american - european;

        char name[64], buf[160];
        snprintf(name, sizeof name, "american_eq_european_m%d", m);
        snprintf(buf, sizeof buf,
                 "american=%.4f european=%.4f err=%+.4f (%+.1f%%)",
                 american, european, err, 100.0 * err / european);
        check_known_broken(std::fabs(err) < 0.05, name, buf,
                           "naive per-path recursion exercises with perfect "
                           "foresight; needs Longstaff-Schwartz");
    }
}

// The bias above must at least not explode with m. Under perfect foresight it
// grows without bound; once LSM lands this column should be flat.
static void test_bias_growth_in_m() {
    const double european = bs_call(100.0, 100.0, 1.0, 0.20, 0.05);
    printf("  bias vs European = %.4f as m grows (flat once LSM lands):\n",
           european);
    for (int m = 1; m <= 21; m += 4) {
        OptionParams p = reference_case(m, 100000);
        double price = price_american_call_serial(p);
        printf("    m=%-3d price=%8.4f  bias=%+7.4f\n",
               m, price, price - european);
    }
}

#ifdef HAVE_OPENMP
// All CPU backends must agree. This is what would have caught the Sobol
// point-major/dimension-major mix-up in the QMC OpenMP backend.
static void test_backends_agree() {
    OptionParams p = reference_case(10, 200000);

    double serial = price_american_call_serial(p);
    double omp    = price_american_call_omp(p, 0);

    char buf[160];
    snprintf(buf, sizeof buf, "serial=%.6f omp=%.6f diff=%.2g",
             serial, omp, std::fabs(serial - omp));
    // Same LCG stream per path, so these must match to summation rounding.
    check(std::fabs(serial - omp) < 1e-6, "serial_vs_omp_identical", buf);

    // QMC uses a different point set, so it converges to the same value but
    // not path-for-path.
    double qmc = price_american_call_qmc_omp(p, 0, 42u);
    snprintf(buf, sizeof buf, "serial=%.4f qmc_omp=%.4f diff=%.4f",
             serial, qmc, std::fabs(serial - qmc));
    check(std::fabs(serial - qmc) < 0.25, "serial_vs_qmc_omp_agree", buf);
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
#ifdef HAVE_OPENMP
    test_backends_agree();
#else
    printf("[skip]  cross-backend agreement (built without OpenMP)\n");
#endif

    printf("\n=== diagnostics ===\n");
    test_bias_growth_in_m();

    printf("\n%d failure(s), %d known-broken.\n", g_failures, g_xfail);
    return g_failures == 0 ? 0 : 1;
}
