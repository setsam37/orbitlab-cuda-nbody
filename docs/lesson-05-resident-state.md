# Lesson 5: a complete GPU step

Velocity Verlet requires three ordered pieces of GPU work: update every position, evaluate acceleration at those positions, then update every velocity. Separate kernels in the same CUDA stream enforce this order without global barriers inside a kernel.

Two acceleration regions hold the old and new values. After the velocity update, swapping their pointers lets the next step reuse the new acceleration without copying it. Resetting a simulation must recompute acceleration for the reset positions.

We compare one step and ten-step trajectories against a CPU double reference. This catches wrong buffer offsets, stale cached acceleration, and update ordering. Longer simulations are checked through physical invariants rather than requiring chaotic particle paths to remain identical forever.
