# Independent review and decisions

A fresh reviewer inspected the whole implementation against the approved spec/plan. No critical numerical/CUDA correctness defect was found. The reviewer confirmed partial-tile bounds, unconditional barriers, resident Verlet ordering and matching softened initialization/potential.

Two important findings were fixed with observed failing/passing tests:

- Benchmark provenance assumed Release/no-fast-math and PATH tools. It now records adjacent CMakeCache configuration, generated flags, actual compiler paths, executable hash, and clearly scoped current-source hashes. Missing configuration is unknown. Tests cover Debug/custom compiler/fast-math and unavailable fields.
- The profiling CLI returned success on genuine profiler failure. It now retains both logs and exits nonzero on failed runs; unavailable counters/tools remain explicitly optional. A CLI-level mocked exit7 test failed before and passed after the fix.

Deferred minor findings (documented in README): benchmark metadata's general `steps` option differs from the actual20-step workload; use batch repetitions. Total wall time can be derived as per-call time multiplied by repetitions. Exporting explicit workload/total fields is a future refinement. Sweep command logs are finalized after success; an interrupted sweep can lose its aggregated log. Per-configuration CSVs already written remain available.

The checked CPU baseline includes validation inside timers, unlike GPU host checks outside event timing. README names this asymmetry rather than implying an optimized CPU comparison.

Local Windows filesystem integration tests encountered sandbox temporary-directory permission errors; all checks are run in Colab/Linux. Windows C++/CUDA execution and hosted GitHub CI execution are unverified. The CI definition is supplied and its CPU commands are verified locally in Colab.
