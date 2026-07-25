#pragma once
#include <cuda_runtime.h>

static constexpr int GPU_SOBOL_DIM  = 21;
static constexpr int GPU_SOBOL_BITS = 32;

void sobol_gpu_init(const unsigned int V_host[GPU_SOBOL_DIM][GPU_SOBOL_BITS]);

// Writes the first `dim` coordinates of Sobol point n into out[].
//
// Direction numbers are indexed by bit position directly -- the same convention
// as SobolGenerator on the host. Indexing them reversed collapses the sequence,
// which is what the CPU generator used to do before it was fixed.
//
// Output is double, matching the host generator, so the CPU and GPU QMC
// backends draw the same point set and can be compared against each other.
__device__ __forceinline__ void sobol_point_device(
    unsigned int n, int dim,
    const unsigned int* __restrict__ d_shift,
    const unsigned int* __restrict__ d_directions,
    double* out)
{
    unsigned int gray = n ^ (n >> 1);

    for (int d = 0; d < dim; ++d) {
        unsigned int x = d_shift[d];
        for (int bit = 0; bit < GPU_SOBOL_BITS; ++bit) {
            if (gray & (1u << bit)) {
                x ^= d_directions[d * GPU_SOBOL_BITS + bit];
            }
        }
        out[d] = static_cast<double>(x) * 2.3283064365386963e-10;
    }
}
