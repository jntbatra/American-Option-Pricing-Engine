# American Options Pricing — CUDA & OpenMP

Reimplementation of *Cvetanoska & Stojanovski, "Using High Performance Computing and Monte Carlo Simulation for Pricing American Options"*, extended with true Quasi-Monte Carlo (QMC) backends sharing one math core:

- **Serial CPU** — reference implementation (LCG Pseudo-Random)
- **OpenMP** — multi-core parallel (LCG)
- **CUDA** — GPU parallel (LCG)
- **QMC OpenMP** — True low-discrepancy QMC (Sobol + Brownian Bridge)
- **QMC CUDA** — True low-discrepancy QMC on GPU

## Status

Calls and puts, priced by Longstaff-Schwartz. All five backends agree, and the
prices are validated against two independent references:

| check | result |
| --- | --- |
| Call on non-dividend stock vs Black-Scholes (exact) | 10.4502 vs 10.4506 |
| American put vs binomial tree, same exercise dates | within 0.14% across four cases |
| GPU QMC vs host QMC | agree to 1e-6 |
| OpenMP at 1 thread vs 28 threads | agree to 2.8e-13 |

## Algorithm

Options are priced as Bermudans with `m` exercise points:

1. Forward-simulate `N` GBM paths (LCG + Moro inverse CND, or Sobol + Brownian
   bridge in the QMC backends).
2. Backward-induct with **Longstaff-Schwartz**: at each exercise date, regress
   the realised discounted future cashflow on functions of the current spot
   across the in-the-money paths, and exercise where the intrinsic value beats
   the *fitted* continuation.
3. Average the time-zero cashflows.

Two details in step 2 are what make it correct rather than merely plausible:

- The regression supplies the exercise **decision** only. Valuation always uses
  the realised cashflow, never the fitted value — substituting the fit back in
  reintroduces bias.
- Only in-the-money paths enter the regression. Out-of-the-money paths carry no
  decision, and including them degrades the fit exactly where the exercise
  boundary lives.

The estimator is low-biased: the fitted policy is suboptimal, and a suboptimal
exercise policy under-values. That is the safe direction, and it converges from
below as `N` and the basis grow.

### Time grid

With `dt = T/(m+1)`, path value `S[i]` sits at time `i*dt` for `i = 0..m`. The
last grid point is at `T - dt`, not `T`: the `m` exercise opportunities are at
`dt, 2dt, ..., m*dt`, and the option still has `dt` of life left after the final
one, worth the European (Black-Scholes) value.

Node `m` needs no regression: the value of holding there is the European value
over the remaining `dt`, which is an exact conditional expectation.

The backward pass lives in one place per side — `src/core/lsm.cpp` for the CPU
backends, `src/cuda/lsm_gpu.cu` for both GPU ones — and the option math they
share is in `src/core/hd_math.hpp`, compiled identically for host and device.

### Why not the paper's recursion

The original paper values each path with `max(intrinsic, discounted
continuation)` using **that path's own realised future**. That is perfect
foresight: every path exercises with hindsight, so the estimator is biased high
and the bias grows without bound in `m`. `validate` prints both side by side:

```
  call on non-dividend stock, exact price = 10.4506
    m          naive       bias        lsm       bias
    1        10.4518    +0.0012    10.4518    +0.0012
    5        13.3630    +2.9124    10.4426    -0.0080
    9        14.7195    +4.2690    10.4342    -0.0164
    13       15.4959    +5.0454    10.4525    +0.0019
    17       15.9839    +5.5333    10.4594    +0.0088
    21       16.3388    +5.8882    10.4573    +0.0067
```

At `m=1` there is effectively one decision and both methods reproduce the exact
price; beyond that the foresight bias takes over — +56% by `m=21`, while LSM
stays inside Monte Carlo noise. The naive recursion is kept in
`src/core/backward_induction.hpp` and reachable as
`price_american_call_serial_naive()` purely so this comparison can be made.

### Validating a put

The call test above cannot distinguish "LSM works" from "LSM never exercises",
because never exercising is the correct answer for a call on a non-dividend
stock. A put is the real test: early exercise binds, there is no closed form,
and `src/core/binomial.hpp` provides an independent CRR tree that shares no code
with the Monte Carlo path.

`binomial_bermudan()` restricts the tree to the same `m` exercise dates the
Monte Carlo grid uses, so both price the identical contract — comparing against
a fully American tree would be apples to oranges, since continuous exercise is
worth strictly more than `m` discrete dates.

## Repository layout

```
src/
  core/       # hd_math.hpp    — option math compiled for host AND device
              # lsm.{hpp,cpp}  — Longstaff-Schwartz regression + backward pass
              # paths.{hpp,cpp}— path simulation (shared by serial and OpenMP)
              # binomial.*     — CRR tree, independent oracle for the tests
              # sobol, halton, brownian_bridge, scramble, moro, black_scholes
              # backward_induction.hpp — the paper's naive recursion, kept for
              #   the bias comparison only
  cpu/        # serial entry point
  openmp/     # OpenMP parallel (both standard & QMC)
  cuda/       # path-generation kernels, lsm_gpu.cu (device backward pass),
              #   cuda_check.h (CUDA_TRY error macros)
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
- American == European for a non-dividend call, at m = 5, 10 and 20
- American put vs the binomial tree at four strikes/vols, plus a strictly
  positive early-exercise premium (which would be zero if LSM never exercised)
- OpenMP at one thread vs many, so a wrong reduction in the regression
  accumulation shows up as a thread-count-dependent price
- LCG vs QMC agreeing to within the difference between the two estimators

`run_cuda_tests` additionally checks GPU-vs-CPU agreement for both the LCG and
QMC backends, the put against the tree, and that out-of-range `m` is rejected.

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

- **Memory.** LSM regresses across paths at each exercise date, so the whole
  path set must be resident: `N × (m+1)` doubles, about 176 MB at `N = 1e6,
  m = 21`. The naive recursion could consume one path at a time; this cannot.
- **Precision.** The device LCG, Moro inverse CND and Sobol are all double
  precision, matching the host bit for bit in the integer parts. The earlier
  single-precision versions carried ~1e-7 of tail error — larger than the QMC
  convergence advantage they were supposed to demonstrate — and made CPU/GPU
  disagreement impossible to interpret. On a consumer GPU with 1/64-rate FP64
  this costs real throughput; a trustworthy comparison was judged worth more.
- The LCG CUDA path kernel is capped at `m ≤ 63`, the QMC one at `m ≤ 21` by the
  Sobol direction-number table. Both launchers reject out-of-range `m` rather
  than corrupting memory.
- The device regression reduces per block and finishes the sum on the host in
  block order. `atomicAdd` on doubles would make the price depend on block
  completion order, so two runs of the same pricer could disagree.
- `SobolGenerator::generate()` returns **point-major** data: coordinate `d` of
  point `n` is at `points[n*d_max + d]`. Getting this backwards silently
  shuffles coordinates between paths.
- LCG seeds are `(path_id + 1) * 1234567u`. This is reproducible but the
  resulting streams are *not* independent — a 2^32 LCG seeded by multiplication
  gives neighbouring paths correlated positions on the same lattice. Replacing
  it with a counter-based generator (cuRAND Philox) is pending.
