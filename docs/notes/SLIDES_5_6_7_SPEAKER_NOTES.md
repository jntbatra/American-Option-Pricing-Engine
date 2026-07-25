# Speaker Notes: Slides 5, 6, 7
## Monte Carlo CUDA — American Options Pricing

---

## Slide 5 — Repository Architecture (60-90 seconds)

"Let me walk you through how we've organized the codebase. At the core, we have a shared math library that contains everything: Black-Scholes pricing, the Moro inverse CND for normal variate generation, Sobol sequences, Brownian Bridge transforms, and our linear congruential generator. This shared foundation ensures that every backend—whether it's serial, OpenMP, or CUDA—is computing the same thing.

Below that, we branch into three parallel execution backends. First, the serial version runs on a single CPU core with straightforward sequential loops. It's our baseline—O(N times m) complexity with LCG random numbers. Then we have OpenMP, which distributes work across all your CPU cores with dynamic scheduling. And finally, CUDA on the GPU, where one thread handles one path, and thousands run simultaneously.

Notice we also have QMC variants for both OpenMP and CUDA. These replace the pseudo-random LCG with Sobol low-discrepancy sequences and Brownian Bridge, which converges faster with fewer paths.

At the bottom layer, we have a benchmark runner that sweeps from 10 to 1 million paths to measure performance, and a validation harness that confirms our Black-Scholes proxy and Moro inverse implementation are correct."

---

## Slide 6 — OpenMP Parallelization (60-90 seconds)

"OpenMP brings multi-core parallelism to the CPU. The core idea is simple: we use a parallel for pragma to distribute paths across all available threads. Each thread gets its own batch of paths to simulate—no shared state, no locks, just independent computation.

We use dynamic scheduling with a chunk size of 64. This keeps the load balanced across threads even if some paths take longer than others to compute.

For random numbers, each thread gets a unique, reproducible seed based on the path ID. The formula is straightforward: seed equals open-paren path plus one close-paren times 1234567u. This ensures statistically independent streams across threads.

When we're done, we use OpenMP's reduction clause to sum all the payoffs without any atomic operations—it's efficient and avoids contention.

Now, for the QMC variant, we replace the LCG with Sobol sequences. Each thread applies a different scrambling matrix to get its own low-discrepancy sub-sequence. We apply a Brownian Bridge transform before the GBM step, which distributes the quasi-random numbers more effectively across time. The result is faster convergence with the same thread count."

---

## Slide 7 — CUDA GPU Parallelization (60-90 seconds)

"CUDA lets us exploit massive parallelism on the GPU. The model is elegant: one GPU thread, one simulation path. Thousands of threads run simultaneously using SIMT—single instruction, multiple thread execution.

We launch 512 threads per block, and the block count scales dynamically: it's the ceiling of N divided by 512. So for a million paths, that's about 2,000 blocks, each with 512 threads all working in parallel.

Each thread does the full job: it simulates the Geometric Brownian Motion path over all m time steps using its own LCG stream—again seeded with that same formula, open-paren path plus one close-paren times 1234567u. Then it computes backward induction to determine the optimal exercise payoff.

The tricky part is reduction—aggregating millions of partial results efficiently. We use shared memory within each block and warp-level shuffle primitives to combine values without expensive atomics. Each block writes its partial sum to global memory, and the CPU does the final sum. This coalesced memory access pattern ensures the GPU memory bandwidth is used efficiently.

The result? For a million paths, we see roughly 33-times speedup over serial CPU."

---

## Timing Notes

- **Slide 5:** Approximately 85 seconds (detailed architecture overview)
- **Slide 6:** Approximately 70 seconds (OpenMP code walkthrough)  
- **Slide 7:** Approximately 75 seconds (CUDA kernel and reduction strategy)

**Total: ~230 seconds (~3.8 minutes) for the three slides together**

## Speaking Tips

1. **Slide 5:** Speak slowly through the architecture diagram. Pause after naming each backend to let the audience absorb the layout.
2. **Slide 6:** Emphasize the "no locks" and "dynamic scheduling" parts—these are why it scales.
3. **Slide 7:** When you reach "one GPU thread, one path," let that idea land before moving to parallelism numbers. The warp reduction is technical; consider gesturing to the code block on screen.

## Smooth Transitions

- **5 → 6:** "Let's zoom in on the OpenMP backend and see how we distribute work across CPU cores..."
- **6 → 7:** "OpenMP is powerful, but the GPU offers even more parallelism. Here's how CUDA handles it..."
