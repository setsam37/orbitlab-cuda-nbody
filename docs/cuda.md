# CUDA design

Separate arrays store masses, positions, velocities, and two acceleration buffers in a resident device allocation. One target particle belongs to one thread. The basic kernel walks every source from global memory; the tiled kernel cooperatively loads source mass/position into shared memory.

Every block thread reaches both tile barriers, including threads beyond N. Stream ordering completes all position updates before new force evaluation and all new forces before velocity updates. No atomics are needed because each thread owns its output. Reset recomputes cached acceleration.

Read [CUDA ownership](lesson-04-cuda.md), [resident integration](lesson-05-resident-state.md), [tiling](lesson-06-tiling.md), and [measured results](results.md). GPU requests fail clearly when unsupported. CUDA runtime calls, event operations, launches, synchronization and RAII cleanup report errors.
