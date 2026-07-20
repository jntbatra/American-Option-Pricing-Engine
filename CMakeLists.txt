cmake_minimum_required(VERSION 3.14)
project(MonteCarloCuda LANGUAGES CXX CUDA)

# Use C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA (required for CUDA builds)
find_package(CUDA REQUIRED)

# -----------------------------------------------------------------
# Core source files – adjust glob patterns if needed.
file(GLOB_RECURSE CORE_SRC src/core/**/*.cpp src/core/**/*.c src/core/**/*.cu)
file(GLOB_RECURSE CUDA_SRC src/cuda/**/*.cu src/cuda/**/*.cpp)

# -----------------------------------------------------------------
# Enable testing and fetch GoogleTest.
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/release-1.12.1.zip
)
# Ensure we link against the same CRT on Windows.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# -----------------------------------------------------------------
# Unit test executable.
add_executable(run_cuda_tests
    tests/test_cuda.cpp
    ${CORE_SRC}
    ${CUDA_SRC}
)

target_include_directories(run_cuda_tests PRIVATE src)

target_link_libraries(run_cuda_tests gtest_main ${CUDA_LIBRARIES})

add_test(NAME CudaTests COMMAND run_cuda_tests)

# -----------------------------------------------------------------
# Benchmark executable.
add_executable(cuda_benchmark
    src/benchmark/cuda_benchmark.cpp
    ${CORE_SRC}
    ${CUDA_SRC}
)

target_include_directories(cuda_benchmark PRIVATE src)

target_link_libraries(cuda_benchmark ${CUDA_LIBRARIES})
