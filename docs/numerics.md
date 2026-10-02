# Lesson 7: proving a simulation is useful

A convincing orbit animation can hide an incorrect solver. Start with acceleration tests against a double CPU reference, then one and ten integration steps. Use vector tolerances that allow rounding while rejecting wrong forces.

Next, halve the time step. For a second-order method in its asymptotic regime, error behaves like C*dt², so halving dt should reduce error about fourfold. Our softened two-body fixture returns to its starting state after one period; compare the combined position/velocity norm after that period. The measured ratios are near four.

Energy, total momentum, and angular momentum check different properties. A force sign bug may break energy; asymmetric pair forces may break momentum; a directional bug can break angular momentum. Normalize energy by |E0|, momentum by sum(m*|v0|), and angular momentum by |L0| for the circular fixture. Synthetic zero-scale normalization tests cover stationary cases.

The N=17 cloud is more demanding. A double reference is refined from 4096 to 8192 steps over duration0.1. Report the difference alongside the error of a 128-step run. A small refinement difference relative to that coarse error supports that comparison; it does not prove an exact trajectory or every digit of the reference.

Exercise: if error changes from 0.008 to 0.002 when dt is halved, estimate the order. If CPU/GPU trajectories differ after a thousand chaotic steps but agree in acceleration and ten-step tests, what additional evidence would distinguish a bug from amplified rounding?
