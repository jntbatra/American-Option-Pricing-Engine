#include "../core/math_utils.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>

#if defined(BACKEND_CUDA) || defined(BACKEND_QMC_CUDA)
#include <cuda_runtime.h>
#endif

#if defined(BACKEND_SERIAL)
double price_american_call_serial(const OptionParams&);
#elif defined(BACKEND_OMP)
double price_american_call_omp(const OptionParams&, int);
#elif defined(BACKEND_CUDA)
double price_american_call_cuda(const OptionParams&, int);
#elif defined(BACKEND_QMC_OMP)
double price_american_call_qmc_omp(const OptionParams& p, int num_threads = 0, uint32_t seed = 42);
#elif defined(BACKEND_QMC_CUDA)
double price_american_call_qmc_cuda(const OptionParams& p, int threads_per_block = 256, uint32_t seed = 42);
#endif

int main() {
    OptionParams base;
    base.S0 = 100.0;
    base.X  = 100.0;
    base.T  = 1.0;
    base.r  = 0.05;
    base.v  = 0.20;
    base.m  = 10;
    base.N  = 0;

    const int path_counts[] = {10, 100, 1000, 10000, 100000,
                                200000, 300000, 500000, 1000000};

#if defined(BACKEND_SERIAL)
    printf("%-12s  %-14s  %-12s\n", "Paths", "Serial(s)", "Price");
#elif defined(BACKEND_OMP)
    printf("%-12s  %-14s  %-12s\n", "Paths", "OMP(s)", "Price");
#elif defined(BACKEND_CUDA)
    printf("%-12s  %-14s  %-12s\n", "Paths", "CUDA(ms)", "Price");
#elif defined(BACKEND_QMC_OMP)
    printf("%-12s  %-14s  %-12s\n", "Paths", "QMC OMP(s)", "Price");
#elif defined(BACKEND_QMC_CUDA)
    printf("%-12s  %-14s  %-12s\n", "Paths", "QMC CUDA(ms)", "Price");
#endif

#if defined(BACKEND_CUDA) || defined(BACKEND_QMC_CUDA)
        {
                double* warmup_buf = nullptr;
                cudaMalloc(&warmup_buf, sizeof(double));
                cudaFree(warmup_buf);
                cudaDeviceSynchronize();
        }
#endif

    for (int N : path_counts) {
        base.N = N;
        double price = 0.0;

        auto t0 = std::chrono::high_resolution_clock::now();
#if defined(BACKEND_SERIAL)
        price = price_american_call_serial(base);
#elif defined(BACKEND_OMP)
        price = price_american_call_omp(base, 0);
#elif defined(BACKEND_CUDA)
        price = price_american_call_cuda(base, 512);
#elif defined(BACKEND_QMC_OMP)
        price = price_american_call_qmc_omp(base, 0);
#elif defined(BACKEND_QMC_CUDA)
        price = price_american_call_qmc_cuda(base, 256);
#endif
        auto t1 = std::chrono::high_resolution_clock::now();

#if defined(BACKEND_CUDA) || defined(BACKEND_QMC_CUDA)
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("%-12d  %-14.4f  %-12.4f\n", N, ms, price);
#else
        double s = std::chrono::duration<double>(t1 - t0).count();
        printf("%-12d  %-14.4f  %-12.4f\n", N, s, price);
#endif
    }
    return 0;
}