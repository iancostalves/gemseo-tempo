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
print(gemseo_scenario.optimization_result.f_opt)  # about 7.9 m/s
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

The wing comes from a four-model statics chain rather than a table of
constants, so the design variables reach the equations of motion through
quantities that mean something:

| model | in | out |
| --- | --- | --- |
| `AirfoilSection` | section | zero-lift drag `C_D0`, `C_L,max` |
| `WingGeometry` | area, span, Oswald efficiency | aspect ratio, induced factor `k` |
| `StructuralMass` | body mass, area, aspect ratio | wing mass, total mass |
| `FlightEnvelope` | mass, area, `C_L,max` | wing loading, stall speed |

The default bird is a wandering albatross to scale: 8.50 kg all told, 2.10 kg
of it wing, on 0.65 m² spanning 3.44 m. That is an aspect ratio of 18.2, a
best glide ratio of 19.7 and a stall speed of 11.8 m/s.

### The section decides what kind of bird this is

`AirfoilSection` is the model that matters most, and it was the last to be
added. A thin, clean section has little profile drag and glides beautifully
but stalls early; a thick, cambered one holds far more lift, which is what a
hard turn needs, and pays for it in drag every second of the flight. One
parameter walks between them:

| section | `C_D0` | `C_L,max` | best glide ratio |
| --- | --- | --- | --- |
| 0.0 (thin, clean) | 0.025 | 1.20 | 27.8 |
| 0.5 (nominal) | 0.033 | 1.50 | 24.2 |
| 1.0 (thick, cambered) | 0.041 | 1.80 | 21.7 |

`plot_polar` draws the drag polar and the glide ratio of any design, marking
both the best-glide lift coefficient and the stall limit. `plot_section_trade`
draws the table above as a curve.

Without that coupling the problem answers the wrong question, and for a while
it did. `C_L,max` was originally an independent constant that reached the
dynamics through nothing at all: the limit the optimizer actually met was a
hard-coded bound on the control. Measured that way the mission's sensitivity
to maximum lift came out **exactly zero**, which reads like a physical finding
and was an artefact of the wiring. The lift limit is now a path constraint
tied to the wing, and the optimized cycle sits on it for most of its length,
which makes it the most active constraint in the problem.

## The wind

The profile is logarithmic, which is what a boundary layer over open water
looks like:

```text
W(h) = w_ref ln(h / h0) / ln(h_ref / h0)
```

with `h0` the roughness length of the sea surface, 0.03 m, and `h_ref` the
10 m at which the reference wind is quoted. The logarithm is singular at `h0`
and the bird cannot fly into the water anyway, so the height is floored
smoothly at 0.5 m before the logarithm is taken, which keeps the profile and
its gradient finite without putting a kink in either.

The gradient is what the bird actually lives off, and it is the reason the
whole manoeuvre happens low down:

| height | wind at `w_ref` = 12 m/s | gradient |
| --- | --- | --- |
| 1 m | 5.8 m/s | 2.07 1/s |
| 5 m | 8.9 m/s | 0.41 1/s |
| 10 m | 12.0 m/s | 0.21 1/s |
| 30 m | 15.2 m/s | 0.07 1/s |

Thirty times weaker at 30 m than at 1 m. `plot_wind_profile` draws both
panels, and the gradient one needs a logarithmic axis to fit on the page.

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
| the cycle, its duration and its start | 10.18 m/s |
| all of that plus the wing | **7.87 m/s** |

Letting the optimizer choose the cycle rather than fly a prescribed one is
worth 42%. Adding the wing to the problem is worth a further **23%**, at
0.50 m² spanning 3.70 m and the thinnest section available: an aspect ratio of
27.4, a glide ratio of 27.8, and 9.8 kg of bird.

The 10.18 m/s figure is the one to compare against the literature, since it is
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

## What the optimizer is actually handed

Problem statements written by hand go stale. `describe_problem` reads them off
the transcribed problem instead, measuring the constraint sizes by evaluating
it once, so what follows is the problem that will be solved:

```python
from gemseo_tempo.scenario.description import describe_problem

print(describe_problem(scenario, formulation))
```

At 20 segments the three formulations come out as:

| | design variables | scalars | constraints |
| --- | --- | --- | --- |
| A, energy | control `C_L` and `mu`, state nodes | 162 | cycle (3), lift, load, height, defects |
| B, sizing | `wing_area`, `wing_span`, `section` | 3 | none: no cycle is flown |
| C, co-design | all of A plus `t_f`, `w_ref`, the wing | 170 | cycle (4), lift, load, height, defects |

The counts are worth seeing. A single `state.h` is one name covering a node
per segment, and a path constraint is one name covering a value per save
point, so a problem that reads as a handful of lines is a few hundred scalars
by the time the optimizer sees it. Note also that `state.h` has one fewer
component under A than under C: A pins its initial state, C frees it.

## Formulation B: size the bird, then fly it

Formulation C optimizes the wing and the cycle at once. That is not how the
work is usually organized: in practice the vehicle is settled first, against
figures of merit an engineer can reason about, and only then is a trajectory
flown on it. Formulation B is that route, in three steps.

**First, measure what the mission cares about.** Each vehicle-level figure of
merit is nudged and the mission **re-optimized**, so what is recorded is the
real effect rather than a linearization:

| figure of merit | moved through | baseline | elasticity |
| --- | --- | --- | --- |
| best glide ratio | `oswald` | 19.75 | **-1.117** |
| `C_L,max` | `section` | 1.50 | +0.531 |
| mass | `body_mass` | 8.50 | -0.005 |

Glide ratio dominates, and nearly proportionally: 1% more of it buys 0.94%
less wind. Mass barely registers. `C_L,max` is *positive*, which is the trade
made visible: reaching more of it means a thicker section, and for a mission
that only wants the weakest wind, the drag that costs outweighs the lift it
buys.

**Then choose a wing** against those elasticities as weights, with no
trajectory in sight, which is exactly what a designer does. **Then fly it.**

### Sequential and simultaneous agree here

| route | minimum `w_ref` |
| --- | --- |
| sequential: size, then fly | 7.87 m/s |
| simultaneous co-design | 7.87 m/s |

They pick the same bird. That is not the 2% margin the sprint expected, and it
is the more useful result: **co-design is worth exactly the misalignment
between the figure of merit and the mission**, and when the merit is measured
against the mission rather than assumed, there is none left to recover.

What it costs to get that merit wrong is worth quantifying, since a designer
has every reason to weight mass more heavily than this mission does:

| weight on mass | aspect ratio | glide ratio | mass | minimum `w_ref` | penalty |
| --- | --- | --- | --- | --- | --- |
| 0.07 (measured) | 27.4 | 24.2 | 9.80 kg | 8.89 m/s | 0% |
| 1.0 | 26.4 | 23.8 | 9.62 kg | 9.00 m/s | +1.3% |
| 2.0 | 13.2 | 16.8 | 7.54 kg | 12.22 m/s | +37% |
| 4.0 | 7.8 | 13.0 | 7.06 kg | 15.40 m/s | +73% |

Sequential design is not worse than co-design because it is sequential. It is
worse when it is sized for the wrong thing.

## What the bird is flying for

Minimum wind is a strange thing to want. A real albatross is going somewhere,
so formulation A takes a `mission`:

| mission | closes on | maximizes | result at `w_ref` = 22 m/s |
| --- | --- | --- | --- |
| `energy` | `h, gamma, psi` | final specific energy | +7.58 m banked, 229 m covered |
| `distance` | `h, V, gamma, psi` | ground made good | 334 m in 9 s, exactly neutral |
| `time` | `h, V, gamma, psi` | minus the duration | 300 m in 8.36 s, exactly neutral |

`energy` leaves speed open, so the surplus is banked as airspeed. The other
two close in speed as well, making the cycle exactly energy-neutral, and spend
the surplus on progress instead: 334 m against 229 m, a 46% gain.

### And this is where `C_L,max` earns its keep

The section trade looked one-sided above, with the thinnest section winning on
minimum wind. Under a mission that rewards progress it is not one-sided at
all:

| section | glide ratio | `C_L,max` | min wind | distance in 9 s | time for 300 m |
| --- | --- | --- | --- | --- | --- |
| 0.00 | 27.8 | 1.20 | **7.88** | 342.9 | 8.248 |
| 0.25 | 25.8 | 1.35 | 8.30 | 351.5 | 8.085 |
| 0.50 | 24.2 | 1.50 | 8.70 | **355.2** | **7.983** |
| 0.75 | 22.9 | 1.65 | 9.09 | 354.4 | **7.983** |
| 1.00 | 21.7 | 1.80 | 9.47 | 351.4 | 8.068 |

Minimum wind wants the thinnest section available and pins the bound. Both
cross-country missions have an **interior optimum** near section 0.5 to 0.75,
which is a maximum lift coefficient of 1.50 to 1.65: about where a real
albatross sits.

The mechanism is visible in the cycle. More `C_L,max` turns tighter, and the
turn radius falls from 32 m at section 0 to 23 m at section 1. Minimum wind
does not care, because nothing in it penalizes a wide cycle and the low-lift
bird simply flies faster, 21.3 m/s against 18.1, and draws a bigger loop. A
mission that has to *get somewhere* does care, because a wide turn spends
distance on turning.

So the cruise-against-manoeuvre trade is real, and it stays invisible until
the mission is one that values manoeuvring. Optimizing for the weakest wind
gives a glider; optimizing for cross-country speed gives something much closer
to the bird.

## The shape of the cycle

The heading can come back to where it started in two ways, and the `offsets`
mapping on `PeriodicityConstraint` is the only thing that distinguishes them:
a **loop** gains a full turn, a **weave** gains nothing.

| topology | mission | result |
| --- | --- | --- |
| loop | `energy` | +7.24 m banked, 230 m covered |
| loop | `distance` | 332 m, neutral |
| weave | `energy` | **-9.80 m**: it loses |
| weave | `distance` | 312 m, neutral |

A weave has no heading reversal to exploit, and asked to maximize energy it
converges to a cycle that loses nearly 10 m. Forced to close in speed as well
it does find a neutral cycle, covering 94% of what the loop manages, so what
fails in the first case is the search rather than the topology.

## Three things that had to be got right

**A heading is periodic only modulo a turn.** A loop comes back to the same
direction of flight having gained a full turn. Demanding plain equality
forbids the very cycle the problem is about, and the residual sits stubbornly
at 6.283. `PeriodicityConstraint` therefore takes an `offsets` mapping.

**The guess is built, not picked.** A soaring cycle is not something an
optimizer stumbles onto. `soaring_cycle` chooses the shape the energy equation
asks for, then **inverts the two rotational equations of motion** for the lift
and bank that would fly it. That lands the guess with collocation defects
already near zero everywhere except the speed equation.

**Finite-difference checks need fixed-step integration.** The split assembly is
verified with `check_jacobian`, and adaptive stepping changes its step sequence
discontinuously as the inputs move, which swamps a 1e-7 probe. Passing
`DiffraxIntegrator(dt0=0.01)` makes the whole map smooth, and all three split
disciplines then agree to 1e-3.

## Still open

**The wing still wants all the span it is allowed.** Aspect ratio raises the
glide ratio, the glide ratio dominates every mission measured here, and
nothing in the model bites hard enough to stop it: a structural law growing as
the square root of aspect ratio changed nothing, and the bending law
`AR^1.5 sqrt(S)` still leaves the optimizer on its bound at 7.6 kg of wing on
a 14 kg bird. Only the section trade produces an interior optimum, and only
for the cross-country missions. The area and span bounds are therefore the
observed range of a real wandering albatross, so that an active bound at least
means something. What actually limits a real bird's span is not in this model.

## Reproducing everything here

```bash
python examples/albatross/run_albatross.py --output figures/albatross
```
