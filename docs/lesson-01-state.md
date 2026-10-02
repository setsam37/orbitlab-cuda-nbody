# Lesson 1: describing particles

A simulation starts with data, before any physics. A body has mass, three position coordinates, and three velocity components. Acceleration will be calculated from the positions of all bodies.

For two bodies, `x = {-0.5, 0.5}` places them on opposite sides of the origin. `mass = {1, 1}` gives equal mass. Opposite tangential velocities make their total momentum zero.

Our `State<Real>` uses a separate vector for each component. This is called a structure of arrays. When adjacent GPU threads read adjacent bodies' x coordinates, their memory requests can access neighboring elements.

`Real` is a template parameter: `State<float>` uses 32-bit floating point; `State<double>` uses 64-bit. This is precision, not the integer bit width of the earlier ALU project.

Every component array must have the same length. Mass must be positive; every number must be finite. Rejecting invalid state at the boundary gives a useful error before it spreads through force calculations.

The two-body circular speed is derived from our softened gravity law, not copied from the unsoftened Kepler problem. Separation is one, each orbit radius is one half, and centripetal acceleration is v^2/(1/2). Equating this to softened gravitational acceleration gives

    v = sqrt(0.5 / (1 + epsilon^2)^(3/2)).

We will test the state and this initialization before implementing the force loop.
