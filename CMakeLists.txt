cmake_minimum_required(VERSION 3.18)
project(MonteCarloCuda LANGUAGES CXX)

# Use C++17 (required by the project: inline variables, structured bindings)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# -----------------------------------------------------------------
# Optional CUDA. The README promises the CUDA targets are skipped when no
# toolkit is present, so detect rather than hard-require.
option(USE_CUDA "Build the CUDA backends if a toolkit is available" ON)

include(CheckLanguage)
if(USE_CUDA)
    check_language(CUDA)
else()
    set(CMAKE_CUDA_COMPILER NOTFOUND)
endif()

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    # Must be set before enable_language(CUDA), which otherwise fills it in
    # from the compiler default (often sm_52). No FORCE, so -D on the command
    # line still wins. 80 = A100; PTX is emitted alongside so newer GPUs JIT.
    set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures to build")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    set(HAVE_CUDA TRUE)
    message(STATUS "CUDA found -- building GPU targets (arch ${CMAKE_CUDA_ARCHITECTURES})")
else()
    set(HAVE_CUDA FALSE)
    if(USE_CUDA)
        message(STATUS "CUDA not found -- skipping GPU targets")
    else()
        message(STATUS "USE_CUDA=OFF -- skipping GPU targets")
    endif()
endif()

# nvcc pins its own host compiler (CUDA 12.4 supports GCC <= 13). If
# CMAKE_CXX_COMPILER is newer, nvcc's GCC lib directory lands on the link line
# ahead of the newer libstdc++ and linking GoogleTest fails on an undefined
# __cxa_call_terminate. Diagnose that here instead of at link time.
if(HAVE_CUDA AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    foreach(_dir IN LISTS CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES)
        if(_dir MATCHES "/lib/gcc/[^/]+/([0-9]+)$")
            set(_nvcc_gcc ${CMAKE_MATCH_1})
        endif()
    endforeach()
    string(REGEX MATCH "^[0-9]+" _cxx_gcc "${CMAKE_CXX_COMPILER_VERSION}")
    if(_nvcc_gcc AND NOT _nvcc_gcc VERSION_EQUAL _cxx_gcc)
        message(WARNING
            "nvcc uses GCC ${_nvcc_gcc} but CMAKE_CXX_COMPILER is GCC "
            "${_cxx_gcc}. Linking run_cuda_tests will likely fail. Reconfigure "
            "with -DCMAKE_CXX_COMPILER=g++-${_nvcc_gcc}, or use a CUDA toolkit "
            "that supports GCC ${_cxx_gcc}.")
    endif()
endif()

find_package(OpenMP)

# -----------------------------------------------------------------
# Source files.
set(CORE_SRC
    src/core/black_scholes.cpp
    src/core/binomial.cpp
    src/core/lsm.cpp
    src/core/paths.cpp
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
    src/cuda/lsm_gpu.cu
    src/cuda/sobol_gpu.cu
)

# -----------------------------------------------------------------
# Backend benchmark executables. src/benchmark/main.cpp selects a backend via
# a BACKEND_* define; each target compiles it with a different one.
function(add_backend_benchmark target define)
    add_executable(${target} src/benchmark/main.cpp ${ARGN} ${CORE_SRC})
    target_include_directories(${target} PRIVATE src)
    target_compile_definitions(${target} PRIVATE ${define})
endfunction()

add_backend_benchmark(american_serial BACKEND_SERIAL src/cpu/american_option_serial.cpp)

if(OpenMP_CXX_FOUND)
    add_backend_benchmark(american_omp BACKEND_OMP src/openmp/american_option_omp.cpp)
    target_link_libraries(american_omp PRIVATE OpenMP::OpenMP_CXX)

    add_backend_benchmark(american_qmc_omp BACKEND_QMC_OMP src/openmp/american_option_qmc_omp.cpp)
    target_link_libraries(american_qmc_omp PRIVATE OpenMP::OpenMP_CXX)
else()
    message(STATUS "OpenMP not found -- skipping american_omp / american_qmc_omp")
endif()

if(HAVE_CUDA)
    add_backend_benchmark(american_cuda BACKEND_CUDA ${CUDA_SRC})
    target_link_libraries(american_cuda PRIVATE CUDA::cudart)

    add_backend_benchmark(american_qmc_cuda BACKEND_QMC_CUDA ${CUDA_SRC})
    target_link_libraries(american_qmc_cuda PRIVATE CUDA::cudart)

    # Standalone CUDA-only benchmark (both GPU backends in one run).
    add_executable(cuda_benchmark src/benchmark/cuda_benchmark.cpp ${CORE_SRC} ${CUDA_SRC})
    target_include_directories(cuda_benchmark PRIVATE src)
    target_link_libraries(cuda_benchmark PRIVATE CUDA::cudart)

    # Setup / path-generation / LSM breakdown.
    add_executable(profile_gpu src/benchmark/profile_gpu.cpp ${CORE_SRC} ${CUDA_SRC})
    target_include_directories(profile_gpu PRIVATE src)
    target_link_libraries(profile_gpu PRIVATE CUDA::cudart)
endif()

# -----------------------------------------------------------------
# Validation suite. No GoogleTest dependency, runs without a GPU.
enable_testing()

add_executable(validate
    tests/validate.cpp
    src/cpu/american_option_serial.cpp
    ${CORE_SRC}
)
target_include_directories(validate PRIVATE src)
if(OpenMP_CXX_FOUND)
    target_sources(validate PRIVATE
        src/openmp/american_option_omp.cpp
        src/openmp/american_option_qmc_omp.cpp)
    target_compile_definitions(validate PRIVATE HAVE_OPENMP)
    target_link_libraries(validate PRIVATE OpenMP::OpenMP_CXX)
endif()
add_test(NAME Validate COMMAND validate)

# -----------------------------------------------------------------
# GoogleTest-based CUDA tests.
if(HAVE_CUDA)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
      URL_HASH SHA256=f179ec217f9b3b3f3c6e8b02d3e7eda997b49e4ce26d6b235c9053bec9c0bf9f
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(run_cuda_tests
        tests/test_cuda.cpp
        tests/kernel_helpers.cu
        src/cpu/american_option_serial.cpp
        ${CORE_SRC}
        ${CUDA_SRC}
    )
    target_include_directories(run_cuda_tests PRIVATE src ${CMAKE_SOURCE_DIR})
    target_link_libraries(run_cuda_tests PRIVATE gtest_main CUDA::cudart)
    if(OpenMP_CXX_FOUND)
        # Needed for the GPU-vs-host QMC comparison.
        target_sources(run_cuda_tests PRIVATE src/openmp/american_option_qmc_omp.cpp)
        target_link_libraries(run_cuda_tests PRIVATE OpenMP::OpenMP_CXX)
    endif()
    # GoogleTest is built with CMAKE_CXX_COMPILER, which may be newer than
    # nvcc's host compiler. Link with the C++ driver so the newer libstdc++ is
    # used; we do not use separable compilation, so no device link step is
    # needed.
    set_target_properties(run_cuda_tests PROPERTIES LINKER_LANGUAGE CXX)
    add_test(NAME CudaTests COMMAND run_cuda_tests)
endif()
