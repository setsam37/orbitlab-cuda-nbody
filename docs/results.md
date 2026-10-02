# Lesson 8: interpreting actual results

Tesla T4 (compute capability7.5), CUDA13.0.88, g++13.3, CMake3.31.10, Colab Xeon CPU, Release. No fast-math flag was configured in the verified build. Raw batches, initial states, commands, environment/build provenance, science data, and profiler outputs are retained in `results/`.

## N=4096, float32, block128

| Mode | CPU ms | Basic GPU ms | Tiled GPU ms | Basic/Tiled | CPU/Tiled |
|---|---:|---:|---:|---:|---:|
| force | 88.5378 | 0.6651 | 0.6094 | 1.091x | 145.3x |
| step | 88.2064 | 0.7394 | 0.5684 | 1.301x | 155.2x |
| end-to-end | 91.5967 | 1.0555 | 0.7714 | 1.368x | 118.7x |

Values are per call/step, medians of five batch means. End-to-end divides a completed 20-step reset/upload/integration/download run by20. CPU is a checked single-thread reference, including finite/state validation. GPU events exclude host validation. Raw min/max ranges show uncertainty; these timings do not generalize to every Colab session.

## Numerical evidence

Halving dt reduced circular orbit error by 3.996958x and 3.999237x, consistent with second-order Verlet.

| Backend/precision | Max normalized energy error | Max normalized momentum | Max normalized angular momentum |
|---|---:|---:|---:|
| cpu-float | 1.01785e-05 | 0 | 5.09157e-06 |
| cpu-double | 5.66584e-09 | 0 | 2.9834e-15 |
| cuda-basic-float | 4.28176e-06 | 0 | 2.14458e-06 |
| cuda-basic-double | 5.66585e-09 | 0 | 1.14625e-14 |
| cuda-tiled-float | 4.28176e-06 | 0 | 2.14458e-06 |
| cuda-tiled-double | 5.66585e-09 | 0 | 1.14625e-14 |

Ten orbital periods at dt=T/512 were tested. Zero momentum in this symmetric fixture is expected and does not establish exact conservation for arbitrary clouds. The N=17 reference refinement difference was 0.000615065, compared with coarse combined error 10.2161; the coarse cloud run is substantially inaccurate. Short double CPU/GPU agreement passed, without claiming long chaotic trajectory equality.

## Performance questions

How does time scale with N? Direct summation performs N*(N-1) directed interactions; inspect `force-scaling.png` and raw throughput. Small N supplies few blocks, so launch/transfer overhead and underutilization can dominate.

What does tiling change? Source global loads are cooperatively reused within a block. At N=4096/float/B256 force-only, tiling lost: Basic/Tiled=0.874 (tiled median0.792 ms, range0.696-0.792 ms). This is a useful counterexample to assuming larger tiles are always faster. A larger block also means fewer blocks and more barriers/resources per block. Compare `block-size.png` before choosing256 by habit.

What does precision cost? At N=2048/B128, inspect the float/double force entries in `summary.csv` and `precision.png`; double improves numerical precision but substantially increases T4 time.

Why can GPU speedup differ between force, step and end-to-end? They include different work. End-to-end includes initial force and transfers amortized across20 steps. Never compare a CPU wall-time simulation to a GPU force-only number as if they were the same workload.

## Profiling interpretation

Nsight Compute MemoryWorkloadAnalysis/Occupancy completed on both kernels at N=4096/float/B128. Both reported100% theoretical occupancy but12.5% achieved occupancy (four active warps per SM); the grid contains just32 blocks. Basic reported98.84% L1/TEX hit rate, while Tiled reported2.16%. L2 reported99.70% and100.30%; the latter is a counter/replay measurement artifact, not a physically meaningful hit probability above100%. Both had zero spilling requests.

Inference: the basic kernel already gets substantial cache reuse. A lower L1 hit fraction in the tiled kernel does not mean its memory behavior is worse: it requests fewer global values and reuses shared values. These summary metrics alone do not quantify a reduction in DRAM bytes; retain/request explicit traffic counters before claiming that. Occupancy did not improve in this profile, so it cannot explain the tiling gain. Profiles are separate from event timings and profiler replay may alter cache state.

Read [NVIDIA’s CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) for timing, transfers, shared memory and occupancy, and [Google Test’s CMake quickstart](https://google.github.io/googletest/quickstart-cmake.html) for test discovery.

Exercise: explain the speedup table without using the word “faster” until you have named the measured workload, hardware, precision, and timing boundaries.
