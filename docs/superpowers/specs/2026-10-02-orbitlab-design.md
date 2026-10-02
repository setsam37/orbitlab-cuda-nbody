# OrbitLab: CUDA N-Body Gravity Simulator

Design for review, 2026-10-02. Implementation has not started.

## Purpose and scope

Build a reproducible scientific-computing portfolio project in C++, CUDA, and Python, teaching the physics, numerical methods, testing, and performance analysis as we go. Run on a Colab GPU without requiring an FPGA or local GPU. Keep the repository usable independently of the notebook.

Deliver a CPU reference, basic and tiled CUDA acceleration kernels, velocity Verlet simulation, a deliberately simple forward Euler comparison, Google Test checks, reproducible experiments, benchmark CSV files and plots, a two-body animation, and a README explaining measured results and limitations. Start with two bodies, then deterministic small particle clouds and larger performance workloads.

Exclude collisions, merging, rendering engines, relativistic effects, Barnes-Hut trees, multi-GPU execution, and PyTorch bindings from this version. This is an all-pairs O(N^2) simulation, not a model intended for arbitrary astronomical accuracy.

## Physics and integration

Use dimensionless units with G=1. Bodies have positive mass, a 3D position, and a 3D velocity. For each body i, compute

    a_i = G * sum(j != i) m_j * (x_j - x_i)
          / (|x_j - x_i|^2 + epsilon^2)^(3/2).

Require finite epsilon > 0 and finite positive dt. Default epsilon=0.01. Keep epsilon fixed during a run and throughout convergence comparisons. Softening changes the physical model; it is not just a numerical workaround. Exact coincident bodies have finite potential and zero pair acceleration under this model; they do not collide or merge.

The production integrator is velocity Verlet:

1. Compute a(x_0) once at initialization.
2. x_new = x + v*dt + 0.5*a*dt^2.
3. Compute a_new from all new positions.
4. v_new = v + 0.5*(a + a_new)*dt.

Cache a_new for the next step, making a steady-state step require one new force evaluation. Use separate position and velocity update kernels so no body reads partially updated positions. Velocity Verlet is second order and symplectic for this position-dependent conservative model. That supports good long-term energy behavior with a suitable fixed time step; it does not guarantee an accurate orbit for an arbitrary dt or close encounter.

Implement forward Euler on the CPU as an educational comparison: x_new=x+v*dt and v_new=v+a(x)*dt, both from the old state. Plot its energy error beside Verlet on the same fixture and time grid. Do not assume every possible Euler run must exhibit monotonic drift. Euler is outside the main CPU/GPU performance comparison.

Diagnostics use the matching softened potential:

    E = sum_i 0.5*m_i*|v_i|^2
        - G*sum(i<j) m_i*m_j/sqrt(|x_j-x_i|^2+epsilon^2)
    P = sum_i m_i*v_i
    L = sum_i m_i*(x_i cross v_i).

Accumulate diagnostics in CPU double precision even for float trajectories. Record maximum energy deviation, final energy deviation, momentum error, and angular momentum error. Normalize against nonzero characteristic scales, not against an initial momentum of zero.

## Precision and reproducibility

Provide explicit float32 and float64 modes in both CPU and GPU implementations. Float32 is the primary speed study; float64 supports convergence and reference studies. Report the actual GPU and measured behavior rather than assume a universal double-precision penalty. Compare CPU and GPU of the same precision for speedup. A CPU double result is also the accuracy reference for float results.

Use the same seeded, serialized initial conditions for every backend. Generate random inputs once rather than rely on different language RNGs matching. Use ascending source-body order in each target's accumulation, without atomic force sums. Differences can still arise from arithmetic instructions and FMA. Do not compile with --use_fast_math in this version. Record compiler version and flags, GPU name, driver/runtime versions, seed, epsilon, dt, precision, N, and block size with results.

## Architecture and repository

Use CMake, C++17, and a pinned Google Test release. CPU-only builds and tests must work without CUDA; CUDA builds enable the GPU backends and GPU checks. Pin the dependency version during planning and document installation/build commands. CPU tests can run in GitHub Actions; GPU tests run in Colab or on a documented CUDA-capable machine.

Suggested layout:

    include/orbitlab/     state, configuration, backend interfaces
    src/                 CPU forces, integrators, diagnostics, CLI
    cuda/                basic forces, tiled forces, update kernels
    tests/               Google Test CPU and CUDA suites
    scripts/             experiments, plotting, animation
    notebooks/           thin Colab launcher
    results/             measured CSV, metadata, plots, verification log
    docs/                physics and short step-by-step lessons
    CMakeLists.txt
    README.md

Use a structure-of-arrays representation for mass, x/y/z, vx/vy/vz, and ax/ay/az. Keep the GPU simulation state resident in device memory. Allocate once per run, reuse buffers, and copy trajectories only at a configurable output interval. A shared backend interface selects cpu, cuda-basic, or cuda-tiled without changing the integrator equations.

One target body maps to one CUDA thread. The basic kernel reads source bodies from global memory. The tiled kernel cooperatively loads source position/mass chunks into shared memory, then reuses them across the block. Support block sizes 64, 128, and 256; default 128. Invalid target threads must still participate in every block barrier. Pad partial source tiles safely and never access beyond N. Explicitly skip self interactions.

Wrap every CUDA runtime call in an error-checking macro that reports the expression, error string, file, and line. Check cudaGetLastError after every launch and synchronize at test/output/timing boundaries to catch asynchronous errors. Cleanup must check errors without throwing from destructors. Runtime failures and nonfinite output cause a nonzero exit status.

Reject N=0, nonpositive masses, nonfinite state, unsupported block sizes, invalid precision/backend names, and invalid dt or epsilon. Handle file I/O errors explicitly. Never silently replace a failed GPU run with a CPU result.

## Numerical verification

Use Google Test for the C++ checks. Python analyzes experiment outputs and supplies supplementary convergence checks. Avoid bitwise equality requirements for computed GPU results.

Acceleration agreement is the main backend check, using identical positions before integration. For each body, compare the vector norm of the acceleration error against

    atol + rtol*|a_reference|.

For dimensionless test fixtures with masses and distances of order one, start with float32 atol=1e-5 and rtol=3e-4; float64 atol=1e-11 and rtol=1e-10. Require all values finite. Absolute tolerance handles cancellation and zero reference acceleration. These are scoped acceptance thresholds for the defined fixtures, not accuracy promises for any N or physical scale. Report maximum absolute error and maximum normalized error. If a test fails, investigate arithmetic and conditioning before changing a threshold; document any justified change.

Test N=1, N=2, N=17, N=63, N=64, N=65, N=127, N=128, N=129, N=257, and N=1003, across supported CUDA block sizes and both precisions. Fixtures include isolated motion, symmetric two-body forces, unequal masses with equal-and-opposite force checks, coincident particles, and seeded bounded particle clouds. Also test invalid input and simulated/observed error paths where practical.

Compare one full Verlet step and a short trajectory (10 steps, dt=0.001) against CPU double. For the position and velocity vector errors, use float32 atol=1e-5, rtol=1e-3; float64 atol=1e-10, rtol=1e-9 on these same scale-controlled fixtures. Long chaotic trajectories are not pointwise correctness tests.

Use a softened circular two-body fixture: equal masses m=1, initial positions (-0.5,0,0) and (0.5,0,0), equal/opposite tangential velocities of magnitude

    v = sqrt(0.5 / (1 + epsilon^2)^(3/2)).

This velocity follows the softened force law. Its circular angular frequency is omega=2*v and period T=2*pi/omega. The circular orbit has an analytic solution under this softened model; do not label it an unsoftened Kepler solution. For more general trajectories, use a high-resolution CPU double reference and demonstrate that further reference refinement changes the result negligibly.

Convergence study: run CPU double Verlet on the circular fixture over exactly one T with dt=T/128, T/256, and T/512. Measure a combined position/velocity error against the softened analytic orbit at the same final time. Target error ratios between 3.5 and 4.5 as dt halves, demonstrating second-order convergence before roundoff dominates. If those grids are outside the asymptotic regime, extend the refinement study and explain the evidence instead of declaring success from a visually closed orbit.

Invariant study: run the circular fixture for 10 periods with dt=T/512. For each backend in float32, initial acceptance budgets are max |E-E0|/|E0| <= 1e-3, |P-P0|/(sum_i m_i * v_characteristic) <= 1e-5, and |L-L0|/|L0| <= 1e-3. Use v_characteristic equal to the initial circular speed. Report float64 values with the same budgets, expecting tighter measured errors without promising them. For other fixtures with near-zero E0 or L0, use positive characteristic scales and document them.

Run Compute Sanitizer memcheck and synccheck on a small partial-tile fixture if available; record availability and results. Lack of a GPU must be reported as skipped GPU validation, never as passing it.

## Performance experiments

Answer four questions: how force time and full-step time scale with N; how GPU time compares with the same-precision single-threaded C++ CPU baseline; how tiling changes throughput; and how block size affects basic/tiled performance. No claim of outperforming optimized scientific libraries follows from this baseline.

Use N=128, 512, 1024, 2048, and 4096 for the primary float32 sweep. Add N=1003 as a performance/correctness boundary case. Sweep block sizes 64, 128, and 256 for both GPU kernels. Run a smaller float64 precision comparison at N=512 and 2048 with block size 128 to keep Colab cost reasonable.

Distinguish force-only timing from full steady-state Verlet step timing (position update, force evaluation, velocity update; initial acceleration excluded). For force-only experiments, repeat evaluation on frozen inputs with reusable output buffers. For steps, reset the fixture outside timed batches so backends start equivalently. Keep allocations, initial upload, trajectory output, and diagnostics outside device-only timings.

Use CUDA events in the same stream for GPU measurements and std::chrono::steady_clock for CPU measurements. After five warm-up iterations, measure five batches of 20 force calls or steps and report median per-call/per-step time plus batch minimum and maximum. Check event creation, recording, synchronization, elapsed-time queries, and destruction. Do not mix profiler-instrumented timing with normal event measurements.

For CPU use one warm-up and five batches of five calls/steps, with inputs reset outside timed step batches. Export per-batch measurements as well as summaries. If runtime limits require reducing repetitions, apply a documented consistent rule and preserve the actual counts in metadata.

Report force interactions/second as N*(N-1)/(force_time_seconds), excluding self interactions, and basic_time/tiled_time for tiling improvement. Report CPU/GPU speedup separately for force and full step. A tiled kernel need not win at every size.

Additionally report host-wall end-to-end time for a fixed short simulation, including initial H2D transfer and final D2H transfer, excluding compilation and allocation. Label it separately from resident-device time. Record hardware and configuration next to each measurement.

Attempt Nsight Compute CLI on one representative basic and tiled force run at N=4096, block size 128, in float32. Inspect available Memory Workload Analysis and occupancy information to distinguish requested loads, cache behavior, and actual DRAM traffic. Shared-memory reuse motivates fewer global load requests, but does not guarantee a proportional reduction in DRAM traffic because caches already reuse data. Colab may restrict performance counters; if unavailable, document the exact limitation and label memory-traffic explanations as hypotheses rather than measured findings. Do not require a legacy nvprof installation.

## Deliverables and teaching

The README leads with what the simulator does, a small animation, exact build/run instructions, verified results, a Markdown results table, benchmark plots, and limitations. The notebook only checks the environment, obtains/builds the repository, runs tests/experiments, and displays saved outputs. It must not duplicate the implementation with large embedded source cells. Keep deeper lesson text in docs and explain each milestone in conversation.

Deliver an orbit animation, a force/step-time-versus-N plot, block-size/tiling comparisons, an energy comparison plot for Euler/Verlet, a convergence plot, and invariant summaries. Produce standard Python plotting artifacts suitable for GitHub. Publish only real measurements and verified tests, with raw CSV, metadata, and logs alongside figures. No fabricated speedups or claimed GPU verification from CPU-only runs.

## Acceptance and next stage

This design is ready when its scope, numerical thresholds, integration choices, benchmark definitions, and repository structure are approved. The next artifact is a written implementation plan; implementation begins after that plan is reviewed and its execution method chosen, as required by the selected Superpowers workflow.

Project completion requires reproducible CPU and GPU checks (or an explicit external GPU blocker), documented convergence and invariants, actual benchmark artifacts, and a thin runnable Colab workflow. Performance improvements are findings to measure, not mandatory outcomes to manufacture.

## Primary references

- NVIDIA, Fast N-Body Simulation with CUDA: https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda
- NVIDIA, Floating Point and IEEE 754: https://docs.nvidia.com/cuda/archive/11.5.0/floating-point/index.html
- NVIDIA, Nsight Compute Profiling Guide: https://docs.nvidia.com/nsight-compute/ProfilingGuide/
