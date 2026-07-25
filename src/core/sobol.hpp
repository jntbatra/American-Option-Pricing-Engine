#pragma once
#include "sobol_joe_kuo.hpp"
#include <cstdint>
#include <vector>
#include <array>

class SobolGenerator {
public:
    explicit SobolGenerator(int num_dimensions);

    // Fills `points` with the first N points of the sequence in POINT-MAJOR
    // order: coordinate `dim` of point `n` lives at points[n * d + dim], so the
    // d coordinates belonging to one point are contiguous. Callers that want
    // "all values for dimension d" must stride, not slice.
    void generate(int N, std::vector<double>& points) const;

    // Writes the d coordinates of point n into out[0..d-1]. Agrees with
    // generate() element for element.
    void point(uint32_t n, double* out) const;

    void set_digital_shift(const std::vector<uint32_t>& shifts);

private:
    int                              d_;
    std::vector<std::array<uint32_t, 32>> V_;  
    std::vector<uint32_t>            shift_;    

    void init_direction_numbers();
};
