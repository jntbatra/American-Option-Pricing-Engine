#include "lsm.hpp"
#include "black_scholes.hpp"
#include <cmath>
#include <cstring>
#include <vector>

static_assert(LSM_BASIS == 3,
              "the OpenMP array-section reductions below hardcode 3 and 9");

// ------------------------------------------------------------ normal equations

void LsmNormalEq::reset() {
    std::memset(ata, 0, sizeof ata);
    std::memset(atb, 0, sizeof atb);
    count = 0;
}

void LsmNormalEq::add(double x, double y) {
    double b[LSM_BASIS];
    lsm_basis(x, b);
    for (int i = 0; i < LSM_BASIS; ++i) {
        for (int j = 0; j < LSM_BASIS; ++j) ata[i * LSM_BASIS + j] += b[i] * b[j];
        atb[i] += b[i] * y;
    }
    ++count;
}

void LsmNormalEq::merge(const LsmNormalEq& other) {
    for (int i = 0; i < LSM_BASIS * LSM_BASIS; ++i) ata[i] += other.ata[i];
    for (int i = 0; i < LSM_BASIS; ++i) atb[i] += other.atb[i];
    count += other.count;
}

bool lsm_solve(const LsmNormalEq& eq, double* beta) {
    if (eq.count < LSM_MIN_ITM) return false;

    double trace = 0.0;
    for (int i = 0; i < LSM_BASIS; ++i) trace += eq.ata[i * LSM_BASIS + i];
    if (!(trace > 0.0)) return false;

    // Augmented [A | b] with a token Tikhonov term. The basis is only
    // quadratic, so this is insurance against a degenerate node (e.g. every
    // in-the-money path sitting at the same spot), not real regularisation.
    const double ridge = 1e-12 * trace;
    double A[LSM_BASIS][LSM_BASIS + 1];
    for (int i = 0; i < LSM_BASIS; ++i) {
        for (int j = 0; j < LSM_BASIS; ++j) A[i][j] = eq.ata[i * LSM_BASIS + j];
        A[i][i] += ridge;
        A[i][LSM_BASIS] = eq.atb[i];
    }

    // Gaussian elimination with partial pivoting.
    for (int col = 0; col < LSM_BASIS; ++col) {
        int piv = col;
        for (int row = col + 1; row < LSM_BASIS; ++row)
            if (std::fabs(A[row][col]) > std::fabs(A[piv][col])) piv = row;

        if (std::fabs(A[piv][col]) <= 1e-12 * trace) return false;
        if (piv != col)
            for (int j = col; j <= LSM_BASIS; ++j) std::swap(A[piv][j], A[col][j]);

        for (int row = col + 1; row < LSM_BASIS; ++row) {
            const double f = A[row][col] / A[col][col];
            if (f == 0.0) continue;
            for (int j = col; j <= LSM_BASIS; ++j) A[row][j] -= f * A[col][j];
        }
    }

    for (int i = LSM_BASIS - 1; i >= 0; --i) {
        double acc = A[i][LSM_BASIS];
        for (int j = i + 1; j < LSM_BASIS; ++j) acc -= A[i][j] * beta[j];
        beta[i] = acc / A[i][i];
        if (!std::isfinite(beta[i])) return false;
    }
    return true;
}

// ------------------------------------------------------------- backward pass

double lsm_price_from_paths(const double* S, int N, const OptionParams& p) {
    const int    m        = p.m;
    const int    stride   = m + 1;
    const double dt       = p.T / static_cast<double>(m + 1);
    const double discount = std::exp(-p.r * dt);

    if (N <= 0 || m < 1) return 0.0;

    // C[n] is path n's realised cashflow, always expressed as a value at the
    // exercise date currently being considered. Sweeping backwards, one
    // multiplication by `discount` per step moves the whole vector to the next
    // node, so no per-path exercise-time bookkeeping is needed.
    std::vector<double> C(static_cast<size_t>(N));

    // Node m sits at T - dt, with dt of life left. The value of not exercising
    // there is the European value over the remaining step -- an exact
    // conditional expectation, so no regression is required at this node.
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const double s    = S[static_cast<size_t>(n) * stride + m];
        const double cont = bs_european(p.type, s, p.X, dt, p.v, p.r);
        const double intr = option_intrinsic(p.type, s, p.X);
        C[n] = intr > cont ? intr : cont;
    }

    double beta[LSM_BASIS];

    for (int i = m - 1; i >= 1; --i) {
        double ata[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        double atb[3] = {0, 0, 0};
        long long cnt = 0;

        // Discount every path one step and, in the same sweep, accumulate the
        // regression over the in-the-money subset.
        #pragma omp parallel for schedule(static) \
                reduction(+ : ata[:9], atb[:3], cnt)
        for (int n = 0; n < N; ++n) {
            C[n] *= discount;

            const double s    = S[static_cast<size_t>(n) * stride + i];
            const double intr = option_intrinsic(p.type, s, p.X);
            if (intr <= 0.0) continue;          // out of the money: no decision

            double b[LSM_BASIS];
            lsm_basis(s / p.X, b);
            for (int a = 0; a < LSM_BASIS; ++a) {
                for (int c = 0; c < LSM_BASIS; ++c)
                    ata[a * LSM_BASIS + c] += b[a] * b[c];
                atb[a] += b[a] * C[n];
            }
            ++cnt;
        }

        LsmNormalEq eq;
        std::memcpy(eq.ata, ata, sizeof ata);
        std::memcpy(eq.atb, atb, sizeof atb);
        eq.count = cnt;

        // Too few in-the-money paths, or a degenerate fit: hold everywhere.
        // Under-exercising can only under-value, which is the safe direction.
        if (!lsm_solve(eq, beta)) continue;

        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const double s    = S[static_cast<size_t>(n) * stride + i];
            const double intr = option_intrinsic(p.type, s, p.X);
            if (intr <= 0.0) continue;
            // Decision from the fitted continuation; value from the intrinsic.
            if (intr > lsm_eval(beta, s / p.X)) C[n] = intr;
        }
    }

    double total = 0.0;
    #pragma omp parallel for schedule(static) reduction(+ : total)
    for (int n = 0; n < N; ++n) total += C[n];

    // C is now a value at node 1 (t = dt); one more step reaches t = 0.
    return (total / static_cast<double>(N)) * discount;
}
