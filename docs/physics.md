# Physical model

OrbitLab uses Newtonian attractive gravity with Plummer softening and G=1 in dimensionless units. The matching potential is `-m_i*m_j/sqrt(r²+epsilon²)`. Softening prevents a singularity at coincidence and changes the physical model; it is not a substitute for a suitable time step.

See [gravity](lesson-02-gravity.md), [integration](lesson-03-integration.md), and [numerical experiments](numerics.md) for the derivation and checks. Collisions, adaptive time steps, relativistic effects and tree approximations are outside this project.
