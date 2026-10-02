# Lesson 4: moving gravity to CUDA

The CPU loop visits every target body in sequence. In the basic CUDA kernel, the index `blockIdx.x * blockDim.x + threadIdx.x` selects a target body. Each thread visits all source bodies, accumulating its own acceleration. No atomic additions are needed because threads write different targets.

Positions, masses, velocities, and two acceleration sets occupy one device allocation with separate contiguous regions. This is still a structure of arrays. Device state remains resident for a simulation; downloading it after every step would add transfer overhead.

CUDA calls can fail immediately, while kernel failures may appear only when work completes. We check the launch and synchronization, and report the call and source location. An animation or benchmark must never hide such a failure.

The first GPU tests compare acceleration at identical positions, not trajectories after a long chaotic run. Float and double have different tolerances, and an absolute tolerance covers forces near zero. Coincident particles also check the softened model.

The basic kernel may return early for an invalid target. In the future tiled kernel it must not: all threads in a block must participate in the shared-memory barriers, including those beyond N.
