# Lesson 2: gravity and diagnostics

For body i, the displacement toward body j is `r = position[j] - position[i]`. Acceleration points along r and scales with j's mass. Multiplying acceleration by i's mass gives force. With unequal masses, accelerations differ but the two forces are equal and opposite.

The softened denominator is `(dot(r,r) + epsilon*epsilon)^(3/2)`. At a coincident position r is zero. Positive epsilon keeps the denominator nonzero and the acceleration vector is zero. This describes softened point particles; it does not simulate a collision.

The CPU implementation visits source bodies in ascending order for each target body. This is O(N^2): doubling N approximately quadruples the interaction count. CUDA will distribute targets among threads without changing the equation.

Diagnostics are independent checks. Energy combines kinetic energy and the matching softened potential; potential counts each unordered pair once. Momentum is the mass-weighted velocity sum. Angular momentum is the sum of `mass * (position cross velocity)`.

CPU and GPU calculations need not match bit for bit. We compare the norm of each acceleration error to `absolute_tolerance + relative_tolerance * norm(reference)`. The absolute term matters when symmetry makes the true acceleration close to zero.

Tests use isolated, unequal-mass, coincident, and hand-calculated fixtures before random inputs. A random comparison alone could let two implementations share the same mistake.
