# Lesson 3: advancing time

Velocity Verlet uses the acceleration before and after a position update:

    x_new = x + v*dt + 0.5*a_old*dt*dt
    a_new = gravity(x_new)
    v_new = v + 0.5*(a_old + a_new)*dt

The new acceleration becomes the old acceleration for the next step. After initialization, each step needs only one new all-pairs calculation.

Forward Euler uses old position, velocity, and acceleration for both updates. We keep it as a comparison to show why the integration method matters. Choosing a more accurate force calculation does not repair a poor time integration method.

A circular orbit has a known period under our softened law. After one period, the ideal state returns to its initial value. We compare the simulated final state to that reference at increasingly small steps. For a second-order method, the leading error is proportional to dt squared, so halving dt should reduce error by about four.

Energy need not stay exactly constant at every step. We measure its maximum deviation across ten periods, alongside momentum and angular momentum. Small errors in all three quantities give stronger evidence than an attractive animation alone.
