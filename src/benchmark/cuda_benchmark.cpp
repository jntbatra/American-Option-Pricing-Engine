// src/benchmark/cuda_benchmark.cpp
#include <iostream>
#include <chrono>

// Adjust include path as needed.
#include "src/cuda/american_cuda.cuh"

int main() {
    std::cout << "N\tTime(s)\tPrice" << std::endl;
    const int M = 20; // exercise points (reasonable default)
    const int Ns[] = {10, 100, 1000, 10000, 100000, 1000000};
    for (int N : Ns) {
        auto start = std::chrono::high_resolution_clock::now();
        double price = run_american_cuda(N, M);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << N << '\t' << elapsed.count() << '\t' << price << std::endl;
    }
    return 0;
}
