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
| Call on non-dividend stock vs Black-Scholes (exact) | error `-1.7e-5` at N=4×10⁶ (QMC) |
| American put vs binomial tree, same exercise dates | within 0.14% across four cases |
| GPU QMC vs host QMC | agree to 1e-6 |
| OpenMP at 1 thread vs 28 threads | agree to 2.8e-13 |

**[→ Full benchmark report with charts](docs/benchmark-results.html)** — a
self-contained HTML page, no build step and no network access needed; open it
straight from disk. Raw harness output is in [`docs/results/`](docs/results/).

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

Pass `put` as the first argument to price a put instead of a call. Each target
prints price, standard error, signed error against the exact reference where one
exists, and the median wall-clock of five runs after a warmup. GPU targets
additionally split setup from kernel time, measured with `cudaEvent` around
device work only.

### Results

Measured on an Intel i7-14700HX (20C/28T) and an RTX 5060 Laptop, CUDA 12.4.
Contract `S₀ = X = 100, T = 1, r = 5%, σ = 20%, m = 20`; exact Black-Scholes
price **10.450584**.

| Backend | Price @ 4×10⁶ | Error | Time | vs serial |
| --- | ---: | ---: | ---: | ---: |
| Serial (LCG) | 10.452417 | `+0.001834` | 3199.8 ms | 1.00× |
| OpenMP (LCG), 28 threads | 10.452417 | `+0.001834` | 596.7 ms | 5.36× |
| OpenMP QMC | 10.450566 | `−0.000017` | 1163.0 ms | 2.75× |
| CUDA (LCG), fp64 paths | 10.452417 | `+0.001834` | 160.6 ms | 19.9× |
| **CUDA (LCG), fp32 paths** | 10.452409 | `+0.001825` | **59.9 ms** | **53.4×** |
| CUDA QMC | 10.450566 | `−0.000017` | 178.9 ms | 17.9× |

Against the 28-thread OpenMP baseline rather than serial, the GPU is 10.0×.

Convergence, absolute error against the exact price:

| N | LCG | Sobol QMC |
| ---: | ---: | ---: |
| 1,000 | 2.4e-01 | 1.3e-02 |
| 10,000 | 9.2e-02 | 3.9e-03 |
| 100,000 | 1.2e-02 | 2.8e-04 |
| 1,000,000 | 7.6e-04 | 1.4e-05 |
| 4,000,000 | 1.8e-03 | 1.7e-05 |

QMC reaches three-decimal accuracy at 100,000 paths, where the pseudo-random
backends need 1,000,000 — about 8× less work for the same answer.

### Where the time goes, and what moved it

`./build/profile_gpu` splits device time into path generation and the LSM
backward pass. Two plausible hypotheses died on contact with the measurement:

- **Host round trips.** The backward pass synchronises 19 times to solve each
  3×3 regression on the CPU. That accounts for well under 1 ms of a 56 ms pass.
- **Uncoalesced reads.** Paths were stored point-major, putting consecutive
  threads 168 bytes apart at a fixed timestep. Switching the device layout to
  time-major changed runtime by under 2%. The layout is still correct and was
  kept, but it was not the bottleneck.

What dominated (≈63%) was FP64 arithmetic in path generation — one `exp` per
step plus two `log`s in Moro's tail branch, on a card that runs doubles at 1/64
the FP32 rate. Storing and generating paths in fp32 while keeping the regression
in double gives **2.7×** on the LCG backend, for a price change of 8×10⁻⁶
against a Monte Carlo standard error of 7.1×10⁻³ — a thousandth of the noise it
sits inside. Select it with `./build/american_cuda call fp32`.

Folding the accumulator's nine block reductions into one barrier (warp shuffles
need no `__syncthreads`) took a further 13% off the backward pass.

> **The GPU is still handicapped here.** CUDA 12.4 has no native sm_120 target,
> so kernels arrive by PTX JIT from sm_80. The default remains fp64 because it
> makes CPU/GPU agreement testable; fp32 is opt-in and its cost is measured
> above, not assumed.

## Notes

- **The standard-error column does not apply to QMC.** `sd/√N` assumes
  independent draws, and a Sobol sequence is deterministic and correlated by
  construction. At N=10⁶ the QMC backends report a standard error of 0.0143
  while their actual error is 0.000014 — a thousandfold overstatement. The QMC
  targets print this warning at run time. A real QMC error bar needs randomised
  QMC: average over several independent digital shifts and take the standard
  deviation of the means. Not yet implemented.
- **LSM is low-biased**, so it converges from below and a true confidence
  interval needs a paired upper bound (Andersen-Broadie dual). Not implemented.
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
