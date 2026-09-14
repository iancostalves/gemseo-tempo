# Physics

```{variable-context} gemseo_tempo.problems.sail.model:REGISTRY
```

A boat sails to a mark through a steady, spatially varying wind. It is a
point over the water, steered on a heading, moving at whatever speed its
polar allows at the wind angle that heading produces.

## The points of sail

```{eval-rst}
.. plot:: _ext/gallery/sail_points_of_sail.py
   :include-source: false
```

The vocabulary first, because the rest of the problem is written in it.
Adapted from the points-of-sail diagram in {cite}`ciccarella2024sailboat`,
which sets the same terms out for the same kind of problem.

The angle between the heading and the wind is the **true wind angle**, and
it names where the boat is sailing: close-hauled, on a beam reach, on a
broad reach, or running. Below some angle the boat cannot advance at all,
and that wedge is the **no-go zone**. Which side the wind enters on is the
**tack** the boat is on, port or starboard. Crossing the wind from one tack
to the other is a **tack** going upwind and a **gybe** going downwind — and
a tack passes through the no-go zone, where the boat is briefly not
propelled at all and carries through on its own momentum. That is not a
detail: it is why {doc}`the upwind leg <optimization>` is the harder
problem, and why one smooth heading over a whole leg represents it badly.

The convention is worth reading twice, because the model and the sailor
disagree about direction. The wind field gives the velocity the air is
travelling *towards*; wind is named for where it comes *from*, which is the
opposite bearing, and the true wind angle is measured against that:

| TWA | meaning |
| --- | --- |
| 0 deg | head to wind, which no boat can sail |
| 90 deg | beam reach, across the wind |
| 180 deg | dead downwind, running |

The sign of the true wind angle says which tack the boat is on. The polar
is symmetric in it, because the two tacks are equally fast.

## Kinematics

Two states and one control. The boat is a point, and the water is static,
so its velocity over the ground is also its velocity through the water:

```{math}
:label: sail-kinematics
\dot{x} = V(\theta_W, W) \cos\psi, \qquad
\dot{y} = V(\theta_W, W) \sin\psi
```

with {var}`psi <symbol>` the heading, {math}`\theta_W` the true wind angle
and {math}`W` the true wind speed at the boat's position.

```{variables} gemseo_tempo.problems.sail.model:REGISTRY
:filter: tags=state,control,output
:columns: name, full_name, unit, description
```

## The velocity triangle

```{eval-rst}
.. plot:: _ext/gallery/sail_wind_triangle.py
   :include-source: false
```

A sail does not feel the true wind. It feels the **apparent** wind, which
is the true wind minus the boat's own velocity — and with the water static,
the boat's velocity over the ground is also its velocity through the water,
so there is only one velocity to subtract:

```{math}
:label: sail-triangle
\mathbf{v}_A = \mathbf{v}_W - \mathbf{v}_B, \qquad
\mathbf{v}_B = V (\cos\psi, \sin\psi)
```

Adapted from the wind triangle of {cite}`hong2023dynamic`, which draws the
same construction for an aircraft, where it is written
{math}`\mathbf{v}_K = \mathbf{v}_W + \mathbf{v}_A`: inertial velocity is
wind plus airspeed. The albatross on {doc}`../albatross/physics` closes that
triangle from the aircraft's side and this problem from the boat's. Sailing
and dynamic soaring are the same construction read in different directions,
which is a better reason to keep both problems in one library than that
they are both interesting.

## Where the polar comes from

```{eval-rst}
.. plot:: _ext/gallery/sail_coefficients.py
   :include-source: false
```

The sail is a wing when it is close to the wind and a parachute when it is
running, and one angle carries that story: the angle of attack left over
once the sail has been trimmed to the apparent wind. On a flat-plate
surrogate, lift peaks where the sail sits at forty-five degrees to the
apparent wind and drag grows first with the lift and then with the plate
held across the flow:

```{math}
:label: sail-coefficients
C_L = C_{L,max} \sin 2\alpha, \qquad
C_D = C_{D0} + k C_L^2 + \tfrac12 \left(1 - \cos 2\alpha\right)
```

Lift acts across the apparent wind and drag along it, so what is left to
push the boat forward is

```{math}
:label: sail-drive
F = \tfrac12 \rho_{air} S \left|\mathbf{v}_A\right|^2
    \left(C_L \sin\theta_A - C_D \cos\theta_A\right)
```

which is the right-hand panel above. It is negative close to the wind — the
sail cannot beat its own drag along the heading — and where it crosses zero
is a no-go zone that nobody had to declare. The dip towards dead downwind,
which is what makes gybing pay, is the sail stalling at ninety degrees of
attack.

The two accounts do not agree exactly, and it is worth saying which way.
The surrogate crosses zero at **18.4 degrees**, the same at every wind speed
— as it should be, since the cone is a property of the polar and not of the
day — while the table declares **30**. The table is the more conservative of
the two, which is the right direction for the disagreement to run: a table
claiming the boat points *higher* than a flat plate could would be claiming
something the physics does not support. The test suite asserts that
direction rather than the agreement.

```{admonition} This is an explanation, not the model
:class: note

The dynamics do **not** evaluate {eq}`sail-drive`. The trajectory reads a
fitted polar, `V = shape(twa) * saturation(tws)`, and these curves are
where that shape comes from.

Solving the force balance inside the right-hand side was tried, and works:
the speed is where {eq}`sail-drive` meets the hull's resistance, found by
six Newton steps, matching a bracketing solver to a micrometre per second
and producing a no-go cone of 18.5 degrees that nobody had to declare. It
costs about six times as much per evaluation, and this problem's
uncertainty studies integrate the trajectory once per sampled wind, so the
robust design study went from minutes to hours. For a problem whose subject
is the *trajectory*, a table that reproduces the same polar is the better
trade — and the reasoning is kept here rather than deleted, because "we
could have computed this and chose not to" is a design decision, not an
omission. The code is still in
{mod}`~gemseo_tempo.problems.sail.aero`, and draws these figures.
```

## What the smoothness costs

The polar is symmetric between tacks, so the obvious way to write it is to
interpolate against the angle and fold the sign away with `abs`. That gives
every value correctly and one derivative wrongly: the folded curve meets its
own mirror image at dead downwind, where the table still has slope, so the
model has a **corner at exactly 180 degrees** that the physics does not —
and it sits in the middle of where a downwind leg spends its time. It is
invisible in every value the model reports.

The table is therefore interpolated against the **cosine** of the wind
angle, which is even and stationary at both ends, so symmetry between tacks
is automatic and the curve is smooth through dead downwind and head to wind
alike. Every tabulated point is reproduced exactly; only the shape between
them changes, and only near the ends. The measurable consequence is that
`d(speed)/d(TWA)` at 180 degrees is zero instead of `0.0027` per degree, and
the best gybing angle moves from 151.8 to 153.9 degrees.

The same trap caught the aerodynamic surrogate above, in a different
disguise: trimming the sail at any fraction of the apparent wind angle other
than a half makes {eq}`sail-coefficients` non-periodic in that angle, and
the mismatch lands exactly where the angle wraps from `+180` to `-180`. At
0.72 it put a corner there of 0.043 m/s per degree, sixteen times the one
the table was rewritten to remove. The check that sees it takes **one-sided**
slopes approaching 180 degrees: a symmetric difference across a symmetric
corner reports zero and looks perfect.

This is the same kind of error as the lift limit that turned out to be a
hardcoded bound in the albatross: a modelling artefact producing numbers
that look like physics. Neither shows up in a value; both show up in a
derivative.

## Wind fields

Three deterministic fields, selected by `wind_name`:

```{eval-rst}
.. plot:: _ext/gallery/sail_wind_field.py
   :include-source: false
```

- `uniform`, blowing east at 10 m/s everywhere (drawn above).
- `sheared`, the same direction with speed growing with `y`.
- `gyre`, built from a streamfunction `sin(pi x / L) sin(pi y / L)`, so it
  circulates and has no divergence. Its cells are half an extent across, and
  the origin is a stagnation point between four of them rather than the centre
  of one.

Each field is a pair of `SpatialField` interpolants, one per velocity
component, since a `SpatialField` is scalar-valued. Two parameters act on
whatever the field returns — {var}`wind_scale <symbol>` on its strength and
{var}`wind_veer <symbol>` on its direction — because the shape of the breeze
over a course is a feature of the course and is known, while how hard it is
blowing on the day is not. They are the uncertainties
{doc}`uncertainty` propagates, and the choice of which to make uncertain
follows {cite}`ciccarella2024sailboat`, whose model likewise holds the wind
speed fixed and lets the direction move, on the grounds that moderate
fluctuations in direction matter more than those in strength.
