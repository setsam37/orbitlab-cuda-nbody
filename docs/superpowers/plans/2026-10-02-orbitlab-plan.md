# OrbitLab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Build and explain a verified gravitational N-body simulator with CPU and CUDA backends, scientific validation, and reproducible performance results.

**Architecture:** A templated C++ structure-of-arrays state feeds CPU and GPU force backends. CPU and resident-device velocity Verlet implementations share equations and initial conditions; Python consumes saved trajectories, diagnostics, and benchmarks. A thin Colab notebook builds and runs the repository.

**Tech Stack:** C++17, CMake 3.22 or newer, CUDA as supplied by Colab, Google Test v1.15.2, Python 3.10 or newer, NumPy and Matplotlib.

**Spec:** [approved design](../specs/2026-10-02-orbitlab-design.md). Read both documents before execution.

## Global constraints

- G=1; default epsilon=0.01; require finite epsilon > 0 and finite positive dt.
- Default CUDA block size 128; support 64, 128, and 256.
- Provide float32 and float64 on CPU and GPU; no --use_fast_math.
- Production velocity Verlet caches acceleration; forward Euler is CPU-only educational comparison.
- Check every CUDA runtime call, every launch, and asynchronous completion at validation/timing/output boundaries.
- CPU-only build and tests work without CUDA; report missing GPU validation as skipped.
- Use Google Test and deterministic serialized initial conditions; no bitwise GPU equality checks.
- A clean repository owns the implementation; the notebook does not embed copies of source code.
- Actual result artifacts are generated from executed runs; do not invent timings or profiler evidence.
- All paths below are relative to outputs/orbitlab/. Save spec and plan copies under docs/superpowers/ when initializing that repository.

## Review focus

1. Oversized N or steps must fail validation before allocation overflow; test in Task 1 and Task 7.
2. Reusing a simulator with changed input must reset cached acceleration; test in Tasks 3 and 5.
3. A nonzero duration not exactly divisible by dt must not silently change the final time; use explicit integer steps and test in Task 7.
4. An output path with spaces, missing parent, or unwritable destination must produce correct output or a clear error; test in Task 7.
5. Optional CUDA/profiler availability must not hide a failed test or replace a GPU result with CPU output; test in Tasks 4, 8, and 10.

## Interfaces and file map

- include/orbitlab/state.hpp: `template<class Real> struct State` with vectors mass,x,y,z,vx,vy,vz; `size()`; `validate_state(const State<Real>&)`.
- include/orbitlab/physics.hpp: `ForceConfig { double epsilon=0.01; }`; `Acceleration<Real>` with vectors x,y,z; `cpu_acceleration(const State<Real>&, ForceConfig) -> Acceleration<Real>`.
- include/orbitlab/fixtures.hpp: `circular_pair<Real>(double epsilon) -> State<Real>`; `cloud<Real>(size_t n, uint64_t seed) -> State<Real>`; `circular_period(double epsilon) -> double`.
- include/orbitlab/diagnostics.hpp: `Diagnostics { double energy; array<double,3> momentum, angular_momentum; }`; `diagnostics(const State<Real>&, ForceConfig) -> Diagnostics`.
- include/orbitlab/cpu_simulator.hpp: `enum class Integrator { Verlet, Euler }`; `CpuSimulator<Real>(State<Real>, ForceConfig, Integrator)`; `step(Real dt)`, `reset(State<Real>)`, `const State<Real>& state() const`.
- include/orbitlab/cuda_simulator.hpp: `enum class GpuKernel { Basic, Tiled }`; movable, noncopyable `GpuSimulator<Real>` with constructor `(State<Real>, ForceConfig, GpuKernel, int block_size)`; `step(Real dt)`, `reset(const State<Real>&)`, `download() -> State<Real>`, `acceleration_snapshot() -> Acceleration<Real>`, `force_only()`, `synchronize()`. Hide CUDA implementation through a private implementation object.
- include/orbitlab/comparison.hpp: `acceleration_matches(actual, reference, atol, rtol) -> Comparison` and `Comparison { bool passed; double max_absolute, max_normalized; }`, rejecting nonfinite values.
- src/: explicit float/double instantiations for fixtures, physics, diagnostics, integrators, CLI, and data I/O.
- cuda/: error/RAII support, resident simulator, basic and tiled force kernels, integration kernels.
- tests/: focused Google Test files for each milestone. scripts/: Python unittest checks and experiment/plot drivers.
- Results format: initial-state CSV `id,mass,x,y,z,vx,vy,vz`; trajectory CSV `step,time,id,x,y,z,vx,vy,vz`; diagnostics CSV `step,time,energy,px,py,pz,lx,ly,lz`. Decimal serialization uses max_digits10 for the underlying type. JSON metadata carries units, configuration, environment, and actual repetitions.

## Task 1: Buildable CPU state and deterministic fixtures

**Files:** CMakeLists.txt, .gitignore, include/orbitlab/{state,fixtures}.hpp, src/fixtures.cpp, tests/state_test.cpp, tests/fixtures_test.cpp, README.md.

**Produces:** State and fixture interfaces above. Google Test via FetchContent pinned to v1.15.2, optional preinstalled package mode for offline builds. `ORBITLAB_ENABLE_CUDA=OFF` default, tests ON by default.

- [x] Inspect local compiler/CMake/Git and Colab access; initialize a new local repository under outputs/orbitlab, preserving other projects. Read applicable AGENTS.md and the worktree skill before implementation. Copy approved design/plan into docs. A fresh standalone repository does not need a second checkout.
- [x] Add Google Test cases: empty state, inconsistent component lengths, zero/negative mass, NaN/Inf components, and checked count overflow must throw; a valid one-body state must pass. `EXPECT_THROW(validate_state(bad), std::invalid_argument);`
- [x] Add fixture tests: circular pair has masses 1, positions +/-0.5 on x, opposite y velocities with `v=sqrt(0.5/pow(1+eps*eps,1.5))`; repeated cloud seed gives identical serialized values; all cloud masses positive and positions/velocities bounded.
- [x] Configure a CPU-only test build and observe tests fail for missing/incorrect implementation, rather than missing dependencies. Implement validation, fixtures, and CMake.
- [x] Run `cmake -S . -B build-cpu -DORBITLAB_ENABLE_CUDA=OFF`, `cmake --build build-cpu --parallel`, and `ctest --test-dir build-cpu --output-on-failure`; require all fixture/state tests pass. Save evidence and commit `feat: add validated particle state and deterministic fixtures`.

**Teaching:** Explain structure-of-arrays, units, state versus acceleration, and why identical starting inputs matter.

## Task 2: CPU forces and conserved quantities

**Files:** include/orbitlab/{physics,diagnostics,comparison}.hpp, src/{physics,diagnostics,comparison}.cpp, tests/physics_test.cpp, tests/diagnostics_test.cpp.

**Consumes:** validated State. **Produces:** acceleration, diagnostics, and vector-norm Comparison interfaces.

- [x] Write tests: N=1 acceleration zero; two bodies separated by 1 have acceleration magnitude `m_other/pow(1+eps*eps,1.5)`; unequal masses satisfy `m1*a1 + m2*a2 == 0` within 1e-12 in double; coincident bodies have zero finite acceleration and finite softened energy. Invalid epsilon throws.
- [x] Write diagnostics tests with known kinetic energy, softened pair potential counted once, expected momentum and cross-product angular momentum. Comparison tests cover zero reference, cancellation, mismatched lengths, and nonfinite values. `EXPECT_FALSE(acceleration_matches(nan_result, reference, 1e-5, 3e-4).passed);`
- [x] Run filtered tests, observe a meaningful failure, implement ordered all-pairs summation (skip self), double diagnostic accumulation, and the norm-based tolerance formula.
- [x] Run `ctest --test-dir build-cpu --output-on-failure`; require all checks pass and commit `feat: implement softened gravity and invariant diagnostics`.

**Teaching:** Derive the force formula and matching potential; show why comparing acceleration alone misses momentum/energy bugs.

## Task 3: Velocity Verlet and Euler comparison

**Files:** include/orbitlab/cpu_simulator.hpp, src/cpu_simulator.cpp, tests/integrator_test.cpp, tests/convergence_test.cpp.

**Consumes:** cpu_acceleration and diagnostics. **Produces:** CpuSimulator with cached acceleration; reset recomputes it.

- [x] Add tests for one-body constant velocity, invalid dt, and reset to changed positions. Check one explicit Verlet step against independently calculated expected positions/velocities. For Euler, assert both updates use the old state.
- [x] Add softened circular-orbit convergence test with epsilon=0.01 and T/128, T/256, T/512 over one full T in double. Define combined error as `sqrt(sum_i(|dx_i|^2 + |dv_i|^2))` in dimensionless units against analytic rotations; assert successive ratios in [3.5,4.5].
- [x] Run tests red, implement velocity Verlet and forward Euler exactly as specified, then rerun. If convergence fails, investigate/reference the asymptotic regime rather than loosen limits without evidence.
- [x] Add 10-period, T/512 invariant tests: normalized max energy <=1e-3, momentum <=1e-5, angular momentum <=1e-3, float and double. Compare Euler/Verlet diagnostic data without assuming universally monotonic Euler drift.
- [x] Run full CPU suite and commit `feat: add verified Verlet and Euler integration`.

**Teaching:** Explain truncation error, stability, symplectic behavior, cached acceleration, and why halving dt should reduce second-order error about fourfold.

## Task 4: Checked CUDA forces and device ownership

**Files:** cuda/cuda_check.hpp, cuda/device_buffer.hpp, cuda/forces_basic.cu, cuda/gpu_simulator.cu, include/orbitlab/cuda_simulator.hpp, tests/cuda_force_test.cpp; modify CMakeLists.txt.

**Consumes:** State, ForceConfig, Comparison. **Produces:** GPU resident state, force_only, acceleration_snapshot, download, synchronize; integration methods are completed next.

- [x] Enable optional CUDA build with CMAKE_CUDA_ARCHITECTURES appropriate to the observed GPU. Find/report device count; explicitly skip device tests when absent. An enabled build on a machine without the toolkit fails with actionable instructions, not a false CPU substitute.
- [x] Write failing basic-force tests for N=1,2,17,63,64,65,127,128,129,257,1003, both precisions and block sizes 64/128/256. Include unequal-mass, coincident, and symmetric fixtures. Compare against CPU double using float atol=1e-5,rtol=3e-4 or double atol=1e-11,rtol=1e-10.
- [x] Add error-check helper test using a known CUDA error value (without causing unsafe device behavior) to verify non-success reports expression/file/line and fails the operation. Check download roundtrip, invalid block size, and move ownership.
- [x] Observe red tests, implement checked runtime calls and RAII buffers, and basic one-thread-per-target acceleration. Check launches and synchronization. Partial methods not used yet must fail explicitly, not return fabricated data.
- [x] Run Colab `ctest --test-dir build-gpu --output-on-failure` with the GPU enabled; save actual GPU/compile/test evidence. Run CPU suite again; commit `feat: add checked CUDA gravity backend`.

**Teaching:** Explain device allocation, host/device ownership, asynchronous errors, and why a launched kernel is not yet a verified result.

## Task 5: Resident GPU integration

**Files:** cuda/integrate.cu, cuda/gpu_simulator.cu, tests/cuda_integrator_test.cpp.

**Consumes:** resident force/state buffers. **Produces:** complete step/reset GPU interface. Separate position and velocity kernels and old/new acceleration buffers enforce ordering.

- [x] Write failing tests for constant velocity, changed-state reset, one Verlet step, and ten steps at dt=0.001. Compare body position and velocity vector norms against CPU double with float atol=1e-5,rtol=1e-3; double atol=1e-10,rtol=1e-9. Assert finite values.
- [x] Implement cached initialization and resident steps in a single stream; recompute acceleration on reset. Validate dt before launch and guard generated nonfinite outputs at checked boundaries.
- [x] Run GPU 10-period circular invariant checks with the Task 3 budgets, then full CPU/GPU tests. Preserve metadata/logs and commit `feat: integrate resident GPU particle trajectories`.

**Teaching:** Show why all positions must update before any new force reads, and why avoiding per-step transfers matters.

## Task 6: Shared-memory force tiling

**Files:** cuda/forces_tiled.cu, cuda/gpu_simulator.cu, tests/cuda_tiling_test.cpp.

**Consumes:** same force/simulator interface. **Produces:** Tiled backend selectable without changing physics or precision.

- [x] Reuse Task 4 force fixtures for Tiled; add Basic-versus-Tiled comparison and Task 5 integration/invariant checks. Ensure N=65,129,257,1003 exercise all supported block sizes.
- [x] Observe red tests, implement cooperative source position/mass loading, safe partial-tile padding, ordered accumulation, and unconditional barriers for every thread. Invalid target threads cannot return before a barrier.
- [x] Run full CPU/GPU suite. If available run `compute-sanitizer --tool memcheck ./build-gpu/orbitlab_cuda_tests --gtest_filter='*PartialTile*'` and corresponding synccheck; require zero reported errors. If unavailable record it explicitly.
- [x] Save evidence and commit `perf: add verified shared-memory gravity tiling`.

**Teaching:** Count source loads before/after tiling, distinguish cache reuse from DRAM traffic, and explain barrier correctness at the boundaries.

## Task 7: CLI, state files, and scientific experiments

**Files:** src/main.cpp, src/io.cpp, include/orbitlab/io.hpp, tests/io_test.cpp, scripts/run_science.py, scripts/test_science.py, scripts/plot_science.py.

**Produces:** `orbitlab simulate --backend cpu|cuda-basic|cuda-tiled --precision float|double --integrator verlet|euler --fixture circular|cloud --n N --seed SEED --epsilon E --dt DT --steps K --sample-every S --block-size B --output DIR`; defaults circular,N=2,seed=37,Verlet,float,epsilon=.01,dt=.001,steps=1000,S=10,B=128. Add `--initial-state FILE` overriding fixture generation. Reject Euler on GPU explicitly.

- [x] Write CLI/io tests for parse failures, oversized counts/product sizes, nonfinite numeric args, truncated CSV, duplicate/nonconsecutive IDs, roundtrip precision, steps=0 initialization output, and exact final `time=steps*dt` calculated from the index rather than cumulative additions.
- [x] Test paths with spaces and missing parents (create parents); unwritable destination must exit nonzero. CLI subprocess tests belong in Python unittest with temporary directories.
- [x] Observe failures, implement schema from the interface map, metadata JSON, sampled trajectory output including step zero and final step, and errors on invalid/missing inputs. All diagnostics refer to the same sampled state/time.
- [x] Implement `python scripts/run_science.py --exe build-gpu/orbitlab --output results/science`: circular convergence grids, Euler/Verlet energy comparison, CPU/basic/tiled invariant summaries, and short agreement runs. Save commands and raw data.
- [x] Use an additional small seeded N=17 double reference trajectory over duration .1 with dt=.1/4096 and .1/8192. Report refinement difference alongside coarse errors, not as a universal exact reference. Use Python tests with synthetic data to check convergence ratios, zero normalization, and rejected nonfinite rows.
- [x] Run CLI tests, scientific experiments, `python -m unittest discover -s scripts -p 'test_*.py'`, and full CTest. Generate static science plots with Matplotlib; commit `feat: add reproducible simulation and scientific experiments`.

**Teaching:** Interpret plots with error scales and units; distinguish numerical and model error from implementation failure.

## Task 8: Reproducible timing and optional profiling

**Files:** src/benchmark.cpp, include/orbitlab/benchmark.hpp, cuda/cuda_event.hpp, scripts/run_benchmarks.py, scripts/test_benchmarks.py, scripts/plot_benchmarks.py, scripts/profile.sh, tests/benchmark_test.cpp.

**Produces:** `orbitlab benchmark --backend BACKEND --precision PRECISION --n N --seed 37 --epsilon .01 --dt .001 --block-size B --mode force|step|end-to-end --output DIR`. Metadata and CSV preserve individual batches, actual counts, force/step distinction, transfer/allocation inclusion, and same-precision CPU ratios computed by Python.

- [x] Write tests for batch aggregation and normalized interaction rate: N=2,time=1 second means 2 directed interactions/second; speedup uses CPU/GPU same mode and precision only. Reject missing/zero/nonfinite timing and incompatible metadata. Absence of a requested GPU exits nonzero.
- [x] Implement checked CUDA events in the simulation stream, CPU steady_clock, GPU 5 warmups + 5 batches of 20, CPU 1 warmup + 5 batches of 5. Time frozen-input force calls and reset-outside-batch steady-state steps with initialized acceleration. Report median,min,max per call/step; no output/diagnostics/allocation/transfers inside device timing.
- [x] Add a 20-step end-to-end wall-time run including initial state upload/final download, excluding allocation and compilation. Record total and per-step times separately; use the same workload for CPU. Ensure results are synchronized before stopping the wall clock.
- [x] Verify correctness of each benchmark fixture outside timings before interpreting a speedup. Run float N=128,512,1003,1024,2048,4096 at blocks 64/128/256 for Basic/Tiled, CPU once per N; double N=512,2048 at block128 on all backends. Check predicted total runtime before launch; reductions in repetition must be applied consistently and logged.
- [x] Generate actual scaling, tiling, and block-size plots and a result table data file. Explain small-N overhead, uncertainty, and cases where tiling loses.
- [x] Attempt `ncu` Memory Workload Analysis and occupancy sections on Basic/Tiled N=4096,float,B=128 force runs, capturing tool/version/counter limitations. Keep profiling separate from event timings. Test wrapper handling of missing executable and permission denial; record these as unavailable, not success.
- [x] Run benchmark aggregation tests and full CTest; retain raw records and commit `perf: add reproducible gravity benchmarks and profiling workflow`.

## Task 9: Animation, README, and thin Colab notebook

**Files:** scripts/animate.py, scripts/test_animation.py, notebooks/orbitlab-colab.ipynb, requirements.txt, docs/{physics,numerics,cuda,results}.md, README.md.

- [x] Add lightweight Python tests for trajectory grouping: repeated IDs across different steps are valid; missing bodies, nonfinite coordinates, and mismatched time for a step are rejected. Implement `python scripts/animate.py --trajectory results/science/circular/trajectory.csv --output results/orbit.gif`, using Matplotlib/Pillow and fixed axis limits.
- [x] Save a two-body animation, standard scientific figures, and actual result table in the repository; link README to raw data and logs. Explain softened circular initialization, tolerance scope, Euler/Verlet, CPU baseline limits, precision choice, hardware, and timing boundaries.
- [x] Create notebook cells only for environment check, source acquisition, dependencies/build, tests, experiments, and display. Support upload of a source archive before a GitHub URL exists; switch to the verified repository URL when publishing is authorized. Never point at a guessed/nonexistent repo.
- [x] Run the notebook from a clean Colab work directory against saved repository sources, confirming it does not rely on files from the old matrix project. Validate notebook JSON and Python syntax locally; capture actual outputs separately rather than manufacture saved notebook execution.
- [x] Commit `docs: publish OrbitLab experiments and Colab workflow`.

**Teaching:** Walk through the final results and limits so the user can explain the project in an interview.

## Task 10: Independent review and final reproducibility check

**Files:** .github/workflows/cpu-tests.yml, verification-log.txt, results/environment.json, README.md as needed.

- [x] Add CPU CI configure/build/CTest on Ubuntu using Google Test; keep GPU tests separate and visibly skipped in CPU-only environments. Verify CI definition against actual build commands.
- [x] Execute clean CPU and GPU builds, all tests, one partial-tile sanitizer run if supported, and the science/benchmark scripts once after final code changes. Summarize evidence and genuine limitations. Avoid rerunning unchanged costly benchmarks without reason.
- [x] Use requesting-code-review skill for one independent whole-project review, with spec/plan and actual verification results. Resolve material issues and rerun affected checks. Review false CPU fallback, barrier bounds, precision/metadata consistency, and benchmark contamination.
- [x] Self-check spec coverage, repository source versus built source, README numbers versus raw data, and notebook cleanliness. Update verification log and checklist with actual outcomes, commit final reviewed changes, and package outputs/orbitlab.zip.
- [x] Present source/archive, teaching notes, scientific findings, and hardware limitations. Publishing to GitHub is a separately authorized action; this implementation plan does not expand repository visibility by itself.

## Execution handoff

Recommend native execution in this chat: the physics, kernels, diagnostics, and experiment schemas depend closely on each other, and explaining each milestone benefits from continuity. One independent review remains at the end. Subagent-driven execution is available if the user chooses fresh implementer/reviewer gates for each task.

This plan requires user review and an execution-method choice before implementation under the selected Superpowers workflow.
