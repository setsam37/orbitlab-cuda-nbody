# Lesson 6: reusing source particles

For one source tile, threads cooperatively load masses and positions into shared memory. Every target thread in the block then reuses those source values, instead of issuing all of its own global loads.

Two barriers are needed: one after loading, and one after all threads finish reading the tile. The second prevents early threads from overwriting values that slower threads still need.

When N is not divisible by the block size, some threads have no target. Those threads still load source entries and reach both barriers. Out-of-range source loads are padded with zero, and the accumulation loop visits only valid source entries.

Tiling adds barriers and shared-memory work. It may lose at small N. We will measure it rather than assume it is faster, and distinguish requested global loads from actual DRAM traffic because caches can also reuse source values.
