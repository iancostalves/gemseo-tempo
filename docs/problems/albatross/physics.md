# Physics

```{variable-context} gemseo_tempo.problems.albatross.model:REGISTRY
```

## Kinematics

```{eval-rst}
.. plot:: _ext/gallery/albatross_kinematics.py
   :include-source: false
```

One loop, leg by leg. The manoeuvre is three-dimensional and a side-on
wave hides the half of it that matters: the bird **climbs while headed into
the wind**, turns through the top of the gradient, **descends headed with
it**, and turns back at the bottom. The four legs are named as
{cite}`hong2023dynamic` names them — windward climb and leeward descent,
either side of the upwind-to-downwind turn — and
{cite}`koessler2019dynamic` makes the case that the turn is not merely the
join between the two gradient crossings but carries energy exchange of its
own, which is the reason to draw it as a leg rather than a corner.

Six states place and orient a point mass, and two controls fly it. Both
tables are generated from the registry the model itself reads, so they
cannot drift from the code:

```{variables} gemseo_tempo.problems.albatross.model:REGISTRY
:filter: tags=state
:columns: name, full_name, unit, bounds
```

```{variables} gemseo_tempo.problems.albatross.model:REGISTRY
:filter: tags=control
:columns: name, full_name, unit, bounds
```

Note what the bound on {var}`cl <symbol>` is not: a number written in this
page or in the dynamics. It is the stall limit the statics chain computes
for the wing being designed, which is what stops a design buying glide
ratio without paying for the manoeuvre.

## Dynamics

```{math}
:label: albatross-kinematics
\begin{split}
\dot{x} &= V \cos\gamma \cos\psi + W(h) \\
\dot{y} &= V \cos\gamma \sin\psi \\
\dot{h} &= V \sin\gamma
\end{split}
```

The wind term in {math}`\dot x` is the velocity triangle,
{math}`\mathbf{v}_K = \mathbf{v}_W + \mathbf{v}_A`: the bird's velocity
over the ground is the wind plus its velocity through the air. It is the
same construction the boat closes from the other side, where the unknown is
the speed rather than the wind — see {doc}`../sail/physics`.

The equations are written for the velocity relative to the **air**, while
Newton's law applies to the velocity relative to the **ground**. The two
differ by the wind, so changing height means moving into air travelling at a
different speed, and that enters as an acceleration:

```{math}
:label: albatross-dynamics
\begin{split}
\dot{W} &= \frac{dW}{dh} V \sin\gamma \\
\dot{V} &= -\frac{D}{m} - g \sin\gamma - \dot{W} \cos\gamma \cos\psi \\
\dot{\gamma} &= \frac{L \cos\mu - m g \cos\gamma}{m V}
   + \frac{\dot{W} \sin\gamma \cos\psi}{V} \\
\dot{\psi} &= \frac{L \sin\mu}{m V \cos\gamma}
   + \frac{\dot{W} \sin\psi}{V \cos\gamma}
\end{split}
```

with the lift and drag of the wing the statics chain sized,
{math}`L = \tfrac12 \rho V^2 S C_L` and
{math}`D = \tfrac12 \rho V^2 S C_D`, on the parabolic polar
{math}`C_D = C_{D0} + k C_L^2`, and the load factor
{math}`n = L / (m g)` that the path constraint limits.

Those three `Wdot` terms are the entire mechanism. Set the gradient to zero
and they vanish, leaving the textbook gliding equations, which can only lose.

**Where the energy comes from.** Differentiating the air-relative specific
energy `e = h + V^2/(2g)` along the equations leaves exactly two terms:

```{math}
:label: albatross-energy
\frac{de}{dt} = -\frac{D V}{m g}
   - \frac{1}{g} \frac{dW}{dh} V^2 \sin\gamma \cos\gamma \cos\psi
```

Drag is always a loss. The gradient term's sign is the bird's to choose: with
the wind increasing upwards it is positive whenever `sin(gamma) cos(psi)` is
negative, which is to say **climb into the wind, descend with it**. The
manoeuvre falls out of the equations rather than being imposed. That identity
is asserted directly in the tests, on a finely integrated trajectory, because
a sign error in the wind terms produces a model that appears to extract
energy while the gain fails to match the work that made it.

Which energy, though, is a question with a literature behind it, and it is
worth saying which side this model takes. Kinetic energy can be written
against the **inertial** speed, as an observer on the water measures it, or
against the **airspeed**, as the aerodynamics feel it; {cite}`koessler2019dynamic`
sets out how easily the two are confused and what goes wrong when the wrong
one is differentiated, and {cite}`hong2023dynamic` adopts the airspeed form
"because the energy concept related to airspeed shows a clear connection to
the wind profile". {eq}`albatross-energy` is that form, for the same reason:
written this way the wind gradient appears explicitly as the term the bird
lives off, rather than hiding inside a frame change.

`plot_energy_analysis` draws the cycle in **both** frames, which is the
figure worth reading. The air-relative energy is what the aerodynamics see:
drag acts on airspeed and the drag polar knows nothing about the ground. The
inertial energy, using ground speed, is what an observer on the water sees.
The gap between them is the wind the bird is being carried along with, and it
opens and closes as the bird climbs and dives. The inertial energy swings
several times harder than the air-relative one over a cycle, since turning
through the wind changes speed over the water far more than speed through
the air — which is exactly the discrepancy that makes the choice of frame
worth being explicit about. The third panel is the audit: drag and gradient
work plotted against the measured rate, which must lie on top of their sum.

## The wing

The wing comes from a five-model statics chain rather than a table of
constants, so the design variables reach the equations of motion through
quantities that mean something:

| model | in | out |
| --- | --- | --- |
| `AirfoilSection` | section | zero-lift drag `C_D0`, `C_L,max` |
| `WingGeometry` | area, span, Oswald efficiency | aspect ratio, induced factor `k` |
| `FiniteWing` | plan, aspect ratio, section properties | chord, Reynolds, `C_D0`, `C_L,max` |
| `StructuralMass` | body mass, area, aspect ratio | wing mass, total mass |
| `FlightEnvelope` | mass, area, `C_L,max` | wing loading, stall speed |

The default bird is a wandering albatross to scale: 8.50 kg all told, 2.10 kg
of it wing, on 0.65 m² spanning 3.44 m. That is an aspect ratio of 18.2, a
best glide ratio of 19.7 and a stall speed of 11.8 m/s.

**The section decides what kind of bird this is.** `AirfoilSection` is the
model that matters most, and it was the last to be added. A thin, clean
section has little profile drag and glides beautifully but stalls early; a
thick, cambered one holds far more lift, which is what a hard turn needs, and
pays for it in drag every second of the flight. One parameter walks between
them:

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

**Aspect ratio reaches the maximum lift too.** Aspect ratio is not only an
induced-drag story, and modelling it as one is what let the optimizer ask
for a sailplane. `FiniteWing` adds the other half, in two layers that pull
opposite ways.

*The lift-curve slope.* A finite wing flies in its own downwash, so it needs
a larger angle than the section to reach a given lift coefficient:
`a = a0 / (1 + a0 / (pi e AR))`. A longer wing suffers less of that and holds a
little more lift. Across the aspect ratios here it is worth about 4%: real,
and far too small to carry the argument.

*The Reynolds number, which is what bites.* The mean chord is `S / b`, so at
fixed area a longer wing is a narrower one, and a narrower wing flies slower
in Reynolds terms. Across the design box the chord runs 0.135 m to 0.360 m and
the Reynolds number 1.8e5 to 4.9e5, which is exactly where a wing starts
losing maximum lift and gaining profile drag:

| Reynolds | `C_D0` | `C_L,max` |
| --- | --- | --- |
| 1.5e5 | ×1.23 | ×0.80 |
| 1.8e5 (high-AR corner) | ×1.13 | ×0.88 |
| 2.5e5 (nominal) | ×1.00 | ×1.00 |
| 3.5e5 | ×0.87 | ×1.12 |

`C_D0` scales as `Re^-0.4`, between the laminar `-0.5` and turbulent `-0.2`,
and `C_L,max` saturates as `Re / (Re + Re_half)`. Both are calibrated choices
rather than derivations, and both are normalized at the nominal bird so it
keeps its section properties exactly. This is the standard reason small flyers
cannot use sailplane aspect ratios.

The effect on the answer is the point. Through two earlier passes **every**
wing variable sat on a bound, because nothing charged aspect ratio for
anything but structure, and a wing mass law was not enough to stop it even at
7.6 kg of wing on a 14 kg bird. With the chord penalty in place the wing area
settles at 0.628 m2, between its limits, and the aspect ratio drops from 27.4
to 21.8.

## The wind

The profile is logarithmic, which is what a boundary layer over open water
looks like, and is the model the dynamic-soaring literature is built on
{cite}`hong2023dynamic`:

```{math}
:label: albatross-wind
W(h) = w_{ref} \frac{\ln(h / h_0)}{\ln(h_{ref} / h_0)}
```

with `h0` the roughness length of the sea surface, 0.03 m, and `h_ref` the
10 m at which the reference wind is quoted. It is a simplification worth
naming: a real marine boundary layer is only logarithmic when it is
neutrally stable, and {cite}`hong2023dynamic` shows the profile shifting
between convective and stable over a diurnal cycle, with the minimum wind a
cycle needs moving accordingly. This model holds the neutral case, so
{var}`roughness <symbol>` carrying an uncertainty is standing in for that
whole effect rather than for a poorly measured seabed.

The logarithm is singular at `h0`
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
panels, and the gradient one needs a logarithmic axis to fit on the page. The
roughness length `h0` and the reference wind `w_ref` are exactly the two
quantities that carry uncertainty in this problem; see
{ref}`UQ strategies comparison <uq-strategies-comparison>`.

## An example simulation

No optimizer, no constraints: `soaring_cycle` builds a closed cycle directly
from the shape the energy equation asks for, then **inverts the two
rotational equations of motion** for the lift and bank that would fly it.
Nothing here is solved; it is a construction, evaluated once.

```{eval-rst}
.. plot:: _ext/gallery/albatross_example_simulation.py
   :include-source: false
```

The guessed cycle already comes close to closing, which is what makes it a
usable starting point for the optimizations below rather than a stumbling
block: see {ref}`The shape of the cycle <the-shape-of-the-cycle>` and
{ref}`Three things that had to be got right <three-things-that-had-to-be-got-right>`.
