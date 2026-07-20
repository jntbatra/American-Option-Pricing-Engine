cmake_minimum_required(VERSION 3.18)
project(MonteCarloCuda LANGUAGES CXX CUDA)

# Use C++17 (required by the project: inline variables, structured bindings)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default CUDA architecture (80 = A100, change for your GPU)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

# Find CUDA toolkit (modern CMake way)
find_package(CUDAToolkit REQUIRED)

# -----------------------------------------------------------------
# Source files.
set(CORE_SRC
    src/core/black_scholes.cpp
    src/core/moro_inv_cnd.cpp
    src/core/brownian_bridge.cpp
    src/core/scramble.cpp
    src/core/sobol.cpp
    src/core/halton.cpp
    src/core/quasi_rng.cpp
)

set(CUDA_SRC
    src/cuda/american_option.cu
    src/cuda/american_option_qmc.cu
    src/cuda/sobol_gpu.cu
)

# -----------------------------------------------------------------
# Enable testing and fetch GoogleTest.
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/release-1.12.1.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# -----------------------------------------------------------------
# Unit test executable.
add_executable(run_cuda_tests
    tests/test_cuda.cpp
    tests/kernel_helpers.cu
    ${CORE_SRC}
    ${CUDA_SRC}
)

target_include_directories(run_cuda_tests PRIVATE src ${CMAKE_SOURCE_DIR})
target_link_libraries(run_cuda_tests gtest_main CUDA::cudart)

add_test(NAME CudaTests COMMAND run_cuda_tests)

# -----------------------------------------------------------------
# Benchmark executable.
add_executable(cuda_benchmark
    src/benchmark/cuda_benchmark.cpp
    ${CORE_SRC}
    ${CUDA_SRC}
)

target_include_directories(cuda_benchmark PRIVATE src)
target_link_libraries(cuda_benchmark CUDA::cudart)
