// src/benchmark/cuda_benchmark.cpp
#include <iostream>
#include <chrono>
#include <iomanip>

#include "core/math_utils.hpp"
#include "cuda/american_cuda.h"

int main() {
    const int M  = 20;
    const int Ns[] = {10, 100, 1000, 10000, 100000, 500000, 1000000};

    OptionParams p;
    p.S0 = 100.0;
    p.X  = 100.0;
    p.T  = 1.0;
    p.r  = 0.05;
    p.v  = 0.20;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== Standard CUDA (LCG) ===" << std::endl;
    std::cout << "N\t\tTime(s)\t\tPrice" << std::endl;
    for (int N : Ns) {
        p.N = N;
        p.m = M;
        auto start = std::chrono::high_resolution_clock::now();
        double price = price_american_call_cuda(p);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << N << "\t\t" << elapsed.count() << "\t" << price << std::endl;
    }

    std::cout << std::endl << "=== QMC CUDA (Sobol + BB) ===" << std::endl;
    std::cout << "N\t\tTime(s)\t\tPrice" << std::endl;
    for (int N : Ns) {
        p.N = N;
        p.m = M;
        auto start = std::chrono::high_resolution_clock::now();
        double price = price_american_call_qmc_cuda(p);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << N << "\t\t" << elapsed.count() << "\t" << price << std::endl;
    }

    return 0;
}
