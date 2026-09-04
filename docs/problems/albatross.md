# Albatross dynamic soaring

An albatross crosses oceans for days without flapping and without thermals. It
does it by repeatedly climbing into the wind and diving back downwind through
air that moves faster higher up. Each pass through the gradient takes a little
energy out of the wind, and over a closed cycle that can be exactly enough to
pay for the drag.

This is the flagship problem of the library. It is the only one that uses a
**statics chain**, a **path constraint**, **periodicity** and a **co-design
formulation** at once.

```python
from gemseo_tempo.problems.albatross import create_codesign_optimization

scenario, gemseo_scenario = create_codesign_optimization(size=20)
gemseo_scenario.execute(algo_name="SLSQP", max_iter=500)
print(gemseo_scenario.optimization_result.f_opt)  # about 8.9 m/s
```

## The bird

Six states place and orient a point mass: position `x` downwind, `y`
crosswind, height `h`, then airspeed `V`, climb angle `gamma` and heading
`psi`. The controls are the lift coefficient `C_L` and the bank angle `mu`.

```text
dx/dt = V cos(gamma) cos(psi) + W(h)
dy/dt = V cos(gamma) sin(psi)
dh/dt = V sin(gamma)
```

The wing comes from a three-model statics chain rather than a table of
constants, so the design variables reach the equations of motion through
quantities that mean something:

| model | in | out |
| --- | --- | --- |
| `WingGeometry` | area, span, Oswald efficiency | aspect ratio, induced factor `k` |
| `StructuralMass` | body mass, area, aspect ratio | wing mass, total mass |
| `FlightEnvelope` | mass, area, `C_L,max` | wing loading, stall speed |

The default bird is a wandering albatross to scale: 8.50 kg all told, 2.10 kg
of it wing, on 0.65 m² spanning 3.44 m. That is an aspect ratio of 18.2, a
best glide ratio of 19.7 and a stall speed of 11.8 m/s.

## Why the equations are not just a glide

The equations are written for the velocity relative to the **air**, while
Newton's law applies to the velocity relative to the **ground**. The two
differ by the wind, so changing height means moving into air travelling at a
different speed, and that enters as an acceleration:

```text
Wdot = (dW/dh) V sin(gamma)

dV/dt     = -D/m - g sin(gamma) - Wdot cos(gamma) cos(psi)
dgamma/dt = (L cos(mu) - m g cos(gamma)) / (m V) + Wdot sin(gamma) cos(psi) / V
dpsi/dt   = L sin(mu) / (m V cos(gamma)) + Wdot sin(psi) / (V cos(gamma))
```

Those three `Wdot` terms are the entire mechanism. Set the gradient to zero
and they vanish, leaving the textbook gliding equations, which can only lose.

The wind profile is logarithmic, `W(h) = w_ref ln(h/h0) / ln(h_ref/h0)`, with
the roughness length of open water. Its gradient is steepest near the surface,
which is why the bird flies low.

## Where the energy comes from

Differentiating the air-relative specific energy `e = h + V^2/(2g)` along the
equations leaves exactly two terms:

```text
de/dt = -D V / (m g)  -  (dW/dh) V^2 sin(gamma) cos(gamma) cos(psi) / g
```

Drag is always a loss. The gradient term's sign is the bird's to choose: with
the wind increasing upwards it is positive whenever `sin(gamma) cos(psi)` is
negative, which is to say **climb into the wind, descend with it**. The
manoeuvre falls out of the equations rather than being imposed.

That identity is asserted directly in the tests, on a finely integrated
trajectory, because a sign error in the wind terms produces a model that
appears to extract energy while the gain fails to match the work that made it.

## Two frames

`plot_energy_analysis` draws the cycle in both frames, which is the figure
worth reading. The **air-relative** energy is what the aerodynamics see: drag
acts on airspeed and the drag polar knows nothing about the ground. The
**inertial** energy, using ground speed, is what an observer on the water
sees. The gap between them is the wind the bird is being carried along with,
and it opens and closes as the bird climbs and dives.

The inertial energy swings several times harder than the air-relative one over
a cycle, since turning through the wind changes speed over the water far more
than speed through the air. The third panel is the audit: drag and gradient
work plotted against the measured rate, which must lie on top of their sum.

## Formulation A: how much can a cycle gain?

The wind and the bird are fixed. The bird must return to the height and
attitude it started at, but not to the same speed, so anything it extracts
shows up as airspeed. Sweeping the wind, over a nine-second cycle:

| `w_ref` (m/s) | drag work (m) | gradient work (m) | net (m) |
| --- | --- | --- | --- |
| 0 | −18.32 | 0.00 | −18.29 |
| 8 | −32.11 | +16.29 | −15.64 |
| 12 | −38.55 | +28.87 | −9.64 |
| 16 | −45.25 | +42.02 | −3.05 |
| 18 | −50.45 | +50.86 | **+0.45** |
| 22 | −59.26 | +66.85 | +7.58 |
| 26 | −67.58 | +82.43 | +14.90 |

With no wind the cycle is a pure glide and loses 18.3 m of specific energy,
all of it to drag. As the wind strengthens the gradient work grows faster than
the drag does, and the two cross at **`w_ref` = 17.7 m/s**: below that the
prescribed cycle cannot sustain itself, above it the bird gains.

## Formulation C: what is the weakest wind it could live in?

Now everything is free, and the cycle has to close completely, in speed as
well as height and attitude. The wind gradient must pay for the drag exactly,
and the objective is the wind itself.

| what is free | minimum `w_ref` |
| --- | --- |
| nothing (formulation A's fixed cycle) | 17.7 m/s |
| the cycle, its duration and its start | 10.25 m/s |
| all of that plus the wing | **8.89 m/s** |

Letting the optimizer choose the cycle rather than fly a prescribed one is
worth 42%. Adding the wing to the problem is worth a further **13.3%**, at
0.50 m² spanning 3.70 m, an aspect ratio of 27.4 flown over a 9.0 s cycle.

The 10.25 m/s figure is the one to compare against the literature, since it is
the real bird's wing: published estimates of the wind an albatross needs sit
in the same range.

Convergence is solid. Periodicity residuals come out at machine precision and
the collocation defects below 1e-6, with the answer stable across
discretization:

| segments | minimum `w_ref` | time |
| --- | --- | --- |
| 15 | 7.21 m/s | 12 s |
| 20 | 7.19 m/s | 12 s |
| 30 | 7.19 m/s | 23 s |

(Those are with the wider design bounds; see below.)

## Three things worth knowing

**A heading is periodic only modulo a turn.** The cycle sweeps the heading
through a full circle, so it comes back to the same direction of flight having
gained `2 pi`. Demanding plain equality of it forbids the very cycle the
problem is about, and the periodicity residual sits stubbornly at 6.283.
`PeriodicityConstraint` therefore takes an `offsets` mapping, and this problem
passes `{"psi": 2 pi}`.

**The guess is built, not picked.** A soaring cycle is not something an
optimizer stumbles onto. `soaring_cycle` chooses the shape the energy equation
asks for, sweeping the heading through a turn with the climb angle a quarter
cycle out of phase, then **inverts the two rotational equations of motion** for
the lift and bank that would fly it. That lands the guess with collocation
defects already near zero everywhere except the speed equation.

**The wing wants all the span it is allowed.** Within this model the wind a
cycle needs falls monotonically with aspect ratio, so the design bound is
active at the optimum whatever it is set to. A first attempt at a structural
penalty, wing mass growing as the square root of aspect ratio, did not change
that at all, because shrinking the area paid back almost exactly what
lengthening the span cost. The bending-driven law `m_wing ~ AR^1.5 sqrt(S)` is
steep enough to bite, but even a 7.6 kg wing on a 14 kg bird did not pull the
optimizer off its bound. The design bounds are therefore the observed range of
a real wandering albatross, so that an active bound at least means something.
What actually limits a real bird's span is not in this model.

## Reproducing the figures

```bash
python examples/albatross/run_albatross.py --output figures/albatross
```

This writes the 3D trajectory coloured by specific energy, the control
histories, the height-against-speed trade, and the two-frame energy analysis,
for both formulations.

## A note on finite-difference checks

The split assembly is verified with `check_jacobian`, and it needs
**fixed-step integration** to pass. Adaptive stepping changes its step
sequence discontinuously as the inputs move, and that noise swamps a 1e-7
probe. Passing `DiffraxIntegrator(dt0=0.02)` makes the whole map a smooth
function of its inputs, and all three split disciplines then agree with finite
differences to 1e-3.
