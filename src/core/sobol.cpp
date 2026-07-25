#include "sobol.hpp"
#include <stdexcept>
#include <cmath>

SobolGenerator::SobolGenerator(int num_dimensions)
    : d_(num_dimensions)
{
    if (num_dimensions > SOBOL_MAX_DIM)
        throw std::runtime_error("Sobol: requested dimensions exceed table size");
    shift_.assign(d_, 0u);
    init_direction_numbers();
}

void SobolGenerator::init_direction_numbers() {
    V_.resize(d_);

    for (int k = 0; k < 32; ++k)
        V_[0][k] = 1u << (31 - k);

    for (int dim = 1; dim < d_; ++dim) {
        const SobolInitData& init = SOBOL_INIT[dim - 1];
        int s         = init.s;
        uint32_t a    = init.a;

        for (int k = 0; k < s; ++k)
            V_[dim][k] = init.m[k] << (31 - k);

        for (int k = s; k < 32; ++k) {
            V_[dim][k] = V_[dim][k - s] ^ (V_[dim][k - s] >> s);
            for (int l = 1; l < s; ++l) {
                if ((a >> (s - 1 - l)) & 1u)
                    V_[dim][k] ^= V_[dim][k - l];
            }
        }
    }
}

void SobolGenerator::point(uint32_t n, double* out) const {
    static constexpr double SCALE = 1.0 / (1ULL << 32);

    uint32_t gray = n ^ (n >> 1);
    for (int dim = 0; dim < d_; ++dim) {
        uint32_t result = shift_[dim];
        // V_[dim][k] is the direction number for bit k, already left-aligned as
        // m[k] << (31-k). Index it with k directly -- indexing [31-k] reverses
        // the bit order and collapses the sequence (dim 0 degenerates to
        // n * 2^-32). This must stay consistent with sobol_point_device().
        for (int k = 0; k < 32; ++k) {
            if ((gray >> k) & 1u)
                result ^= V_[dim][k];
        }
        out[dim] = static_cast<double>(result) * SCALE;
    }
}

#if defined(_MSC_VER)
#include <intrin.h>
static inline int count_trailing_zeros(uint32_t x) {
    unsigned long index;
    if (_BitScanForward(&index, x)) {
        return index;
    }
    return 32;
}
#else
static inline int count_trailing_zeros(uint32_t x) {
    return x == 0 ? 32 : __builtin_ctz(x);
}
#endif

void SobolGenerator::generate(int N, std::vector<double>& points) const {
    static constexpr double SCALE = 1.0 / (1ULL << 32);
    points.resize(static_cast<size_t>(N) * d_);

    std::vector<uint32_t> x(d_, 0);
    for (int dim = 0; dim < d_; ++dim) x[dim] = shift_[dim];

    if (N > 0) {
        for (int dim = 0; dim < d_; ++dim)
            points[0 * d_ + dim] = static_cast<double>(x[dim]) * SCALE;
    }

    for (int n = 1; n < N; ++n) {
        uint32_t prev = static_cast<uint32_t>(n - 1);
        int c = count_trailing_zeros(~prev);  // index of rightmost zero bit

        if (c >= 32) c = 31;

        for (int dim = 0; dim < d_; ++dim) {
            // See point(): direction numbers are indexed by bit position
            // directly, not reversed.
            x[dim] ^= V_[dim][c];
            points[n * d_ + dim] = static_cast<double>(x[dim]) * SCALE;
        }
    }
}

void SobolGenerator::set_digital_shift(const std::vector<uint32_t>& shifts) {
    if (static_cast<int>(shifts.size()) < d_)
        throw std::runtime_error("Sobol: shift vector too short");
    shift_ = shifts;
}
