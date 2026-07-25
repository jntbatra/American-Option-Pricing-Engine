# American Options Pricing — CUDA & OpenMP

Reimplementation of *Cvetanoska & Stojanovski, "Using High Performance Computing and Monte Carlo Simulation for Pricing American Options"*, extended with true Quasi-Monte Carlo (QMC) backends sharing one math core:

- **Serial CPU** — reference implementation (LCG Pseudo-Random)
- **OpenMP** — multi-core parallel (LCG)
- **CUDA** — GPU parallel (LCG)
- **QMC OpenMP** — True low-discrepancy QMC (Sobol + Brownian Bridge)
- **QMC CUDA** — True low-discrepancy QMC on GPU

## Status

All five backends agree on the price to four decimal places. The pricer is
**not yet correct in absolute terms**: see [Known bias](#known-bias) below.

## Algorithm

American calls are priced as Bermudan options with `m` exercise points:

1. Forward-simulate `N` GBM paths (LCG + Moro inverse CND, or Sobol + Brownian
   bridge in the QMC backends).
2. Backward-induct: at each node, value = max(intrinsic, discounted continuation).
3. Average the time-zero values.

### Time grid

With `dt = T/(m+1)`, path value `S[i]` sits at time `i*dt` for `i = 0..m`. The
last grid point is at `T - dt`, not `T`: the `m` exercise opportunities are at
`dt, 2dt, ..., m*dt`, and the option still has `dt` of life left after the final
one, worth the European (Black-Scholes) value.

The whole recursion lives in one place, `src/core/backward_induction.hpp`, which
compiles for both host and device so the five backends cannot drift apart. It
returns a value already discounted to `t = 0`; callers average and do not
discount again.

### Known bias

The backward recursion is the naive per-path one from the original paper: each
path decides whether to exercise using **its own realised future**. That is
perfect foresight, so the estimator is biased high, and the bias grows with `m`:

```
m=1   price= 10.4518  bias=+0.0012
m=5   price= 13.3630  bias=+2.9124
m=9   price= 14.7195  bias=+4.2690
m=21  price= 16.3388  bias=+5.8882
```

The bias is measured against a hard reference: early exercise of an American
call on a non-dividend-paying stock is never optimal, so the correct price is
exactly the Black-Scholes European price (10.4506 for the benchmark case). At
`m=1` there is effectively one decision and the pricer reproduces it; beyond
that the foresight bias dominates.

Fixing this means replacing the recursion with **Longstaff-Schwartz** (regress
continuation value on in-the-money paths rather than peeking at the future).
`validate` reports the affected checks as `[XFAIL]` with that as the blocker,
and `CudaPricing.DISABLED_EqualsEuropeanForNonDividendCall` is the GPU-side
acceptance test to enable once it lands.

## Repository layout

```
src/
  core/       # shared math (BS formula, LCG, Moro, Sobol, Halton, BB, scrambling,
              #   backward_induction.hpp — the shared host/device recursion)
  cpu/        # serial pseudo-random reference
  openmp/     # OpenMP parallel (both standard & QMC)
  cuda/       # CUDA kernels and host launchers (both standard & QMC)
  benchmark/  # benchmark main()
tests/
  validate.cpp   # CPU validation suite, no GPU or GoogleTest needed
  test_cuda.cpp  # GoogleTest suite for the GPU backends
CMakeLists.txt
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

If CUDA is not installed, the GPU targets are skipped automatically and the CPU
targets still build. Pass `-DUSE_CUDA=OFF` to skip them deliberately.

> **CUDA arch:** defaults to `sm_80` (A100), with PTX emitted alongside so newer
> GPUs JIT. Override with `-DCMAKE_CUDA_ARCHITECTURES=75` (Turing), `86`
> (Ampere consumer), `89` (Ada), `90` (Hopper).

> **Host compiler:** nvcc pins its own supported GCC (CUDA 12.4 tops out at GCC
> 13). If your default `g++` is newer, configure with a matching one —
> `-DCMAKE_CXX_COMPILER=g++-13` — or linking `run_cuda_tests` fails on an
> undefined `__cxa_call_terminate`. CMake warns when it detects the mismatch.

## Test

```bash
ctest --test-dir build --output-on-failure
```

or run the suites directly:

```bash
./build/validate         # CPU: primitives, Sobol, cross-backend agreement
./build/run_cuda_tests   # GPU: kernel launch, CPU/GPU agreement, range guards
```

`validate` covers:

- Black-Scholes against the Hull textbook value
- Moro inverse CND at 0.975
- Sobol first points against the exact van der Corput values, `generate()` vs
  `point()` consistency, and per-dimension uniformity over 4096 points
- Serial vs OpenMP producing bit-identical prices (same per-path LCG stream)
- Serial vs QMC-OpenMP agreeing to within estimator difference
- American == European for a non-dividend call (currently `[XFAIL]`, see
  [Known bias](#known-bias))

## Benchmark

```bash
./build/american_serial
./build/american_omp
./build/american_qmc_omp
./build/american_cuda       # only if CUDA was built
./build/american_qmc_cuda   # only if CUDA was built
./build/cuda_benchmark      # both GPU backends in one run
```

Each prints a table over `N ∈ {10, 100, 1k, 10k, 100k, 200k, 300k, 500k, 1M}`
paths with timing and price.

> The reported GPU times currently include per-call setup (direction-number
> construction, seven `cudaMalloc`s, a `cudaMemcpyToSymbol`) because the host
> launcher does all of it on every invocation. They are not kernel times. Hoist
> the setup and time with `cudaEvent` before quoting any speedup.

## Notes

- The LCG CUDA kernel uses a per-thread stack array of 64 doubles, so `m ≤ 63`.
  The QMC kernel is capped at `m ≤ 21` by the Sobol direction-number table.
  Both launchers reject out-of-range `m` rather than corrupting the stack.
- `SobolGenerator::generate()` returns **point-major** data: coordinate `d` of
  point `n` is at `points[n*d_max + d]`. Getting this backwards silently
  shuffles coordinates between paths.
- LCG seeds are `(path_id + 1) * 1234567u`. This is reproducible but the
  resulting streams are *not* independent — a 2^32 LCG seeded by multiplication
  gives neighbouring paths correlated positions on the same lattice. Replacing
  it with a counter-based generator (cuRAND Philox) is pending.
