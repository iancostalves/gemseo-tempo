# Sailing to a mark

```{variable-context} gemseo_tempo.problems.sail.model:REGISTRY
```

Minimum-time sailing to a mark through a steady, spatially varying wind. The
boat is a point moving over the water at whatever speed its polar allows at
the wind angle it is being steered on:

```text
dx/dt = V(TWA, TWS) cos(psi)
dy/dt = V(TWA, TWS) sin(psi)
```

with `psi` the heading and `V` the boat speed read off the polar.

```python
from gemseo_tempo.problems.sail import create_optimization

scenario, gemseo_scenario = create_optimization(leg="downwind", size=25)
gemseo_scenario.execute(algo_name="SLSQP", max_iter=300)
print(gemseo_scenario.optimization_result.f_opt)  # about 160 s
```

This is the first problem in the library with a genuine **path constraint**,
and the first whose interesting answer is qualitative: the optimizer is never
told to gybe, and gybes anyway.

## Angles

The angle convention is the one sailors use, and it is worth reading twice.
The wind field gives the velocity the air is travelling *towards*. Wind is
named for where it comes *from*, which is the opposite direction. The true
wind angle is the angle between the heading and that "from" direction, so:

| TWA | meaning |
| --- | --- |
| 0 deg | head to wind, which no boat can sail |
| 90 deg | beam reach, across the wind |
| 180 deg | dead downwind, running |

The two states, the control and the quantities the trajectory reports:

```{variables} gemseo_tempo.problems.sail.model:REGISTRY
:filter: tags=state,control,output
:columns: name, full_name, unit, description
```

The sign of the TWA says which tack the boat is on. The polar is symmetric in
it, because the two tacks are equally fast.

## The polar

The polar is authored in the repository from an analytic shape, so the problem
carries no measured polar data from any real boat. Speed is separable,

```text
V = f(|TWA|) * g(TWS)
```

with `f` a smooth table over the true wind angle and `g` a saturating factor
so that speed does not grow without bound in strong wind.

Three details matter. The table is **dense through the no-go transition**: a
C2 spline through a sparse table undershoots to `-0.12` there, and negative
boat speed is something an optimizer will exploit cheerfully. Even densely
sampled it dips slightly below zero, so the result is **clamped** with a
smooth maximum against zero, which floors it at `+0.0008` while leaving the
rest of the curve untouched.

The third is about the other end of the table, and it was a defect for
several sprints before a Jacobian check found it. The polar is symmetric
between tacks, so the obvious way to write it is to interpolate against the
angle and fold the sign away with `abs`. That gives every value correctly and
one derivative wrongly: the folded curve meets its own mirror image at dead
downwind, where the table still has slope, so the model has a **corner at
exactly 180 degrees** that the physics does not. It sits in the middle of a
downwind leg, which is where this problem spends its time, and it is
invisible in every value the model reports.

The table is therefore interpolated against the **cosine** of the wind angle
rather than against the angle. The cosine is even and stationary at both ends,
so symmetry between tacks is automatic and the curve is smooth through dead
downwind and head to wind alike. Every tabulated point is reproduced exactly;
only the shape between them changes, and only near the ends. The measurable
consequence is that `d(speed)/d(TWA)` at 180 degrees is now zero instead of
`0.0027` per degree, and the best gybing angle moves from 151.8 to 153.9
degrees.

This is the same kind of error as the lift limit that turned out to be a
hardcoded bound in the albatross: a modelling artefact that produces numbers
which look like physics. Neither shows up in a value; both show up in a
derivative.

## Why gybing pays

Sailing straight at a mark that lies dead downwind is slow, because a boat
running dead downwind is slow. Heading up onto a broad reach costs distance
but buys enough speed to more than pay for it. The quantity that decides this
is the **speed made good**, the part of the boat speed that actually shortens
the distance to the mark:

```text
VMG = V(TWA) * |cos(TWA)|
```

For this polar, in 10 m/s of wind:

| sailing | TWA | boat speed | speed made good |
| --- | --- | --- | --- |
| dead downwind | 180 deg | 4.50 m/s | 4.50 m/s |
| best gybing angle | 153.9 deg | 5.62 m/s | 5.05 m/s |

So gybing is worth **12.2%** in principle, and over the 800 m leg that is
158.4 s against 177.8 s sailed straight. `best_vmg_angle` computes the angle,
and it is what the default starting point is built from.

## What the optimizer finds

Both direct transcriptions find the gybe, from a starting guess that only
knows the best VMG angle and not the trajectory:

| transcription | nodes | time | above the ideal 158.4 s | widest point |
| --- | --- | --- | --- | --- |
| collocation | 25 | 161.71 s | 2.09% | 102.1 m |
| collocation | 40 | 162.07 s | 2.32% | 86.6 m |
| single shooting | 25 | 158.81 s | 0.26% | 189.4 m |
| single shooting | 40 | 158.68 s | 0.18% | 192.1 m |

Against the 177.78 s straight run, shooting is a **10.7% gain** landing within
a quarter of a percent of the theoretical VMG optimum. The heading history
comes back symmetric about the bearing to the mark, near `+26` and `-26`
degrees, which is the best VMG angle of 153.9 degrees expressed as a heading;
sailing half the leg on each of those tacks puts the widest point at about
`400 * tan(26.1 deg)`, or 196 m, which is what shooting finds.

Collocation does **not** find it, and got worse when the polar was smoothed:
it stops around 102 m across and 2% above the ideal, and adding nodes does not
help. That is a local optimum rather than a discretization error, and the
smoothing is what made it reachable. The old polar had a corner at dead
downwind with slope `0.0027` per degree on either side, which pushed a
trajectory off the wind axis; without it the region near dead downwind is flat
to first order, the gybe angle is less sharply determined, and collocation
settles for a narrower one. The corner was making the problem easier by being
wrong. Shooting, which integrates the dynamics rather than treating the states
as design variables, is unaffected.

## The path constraint

The channel constraint limits how far the boat may stray from the direct line
to the mark. It is the constraint worth having, because straying wide is
precisely what gybing buys speed with, so it trades against the objective
rather than fencing off a region the boat had no reason to enter:

| half-width | time | furthest across |
| --- | --- | --- |
| unconstrained | 158.81 s | 189.4 m |
| 250 m | 158.81 s | 189.4 m |
| 150 m | 160.24 s | 150.0 m |
| 80 m | 161.86 s | 80.0 m |

A limit the free optimum already respects leaves it untouched; tighter limits
bind exactly, and cost time monotonically.

Those numbers are the **best of several starting points**, and they have to be.
Every entry here is a different local optimum depending on where the solve
begins, and no single strategy wins at every width. Warm-starting from the
unconstrained optimum is what gets 150 m to bind at 150.0 m rather than
stalling at 98.2 m and a second slower; chaining that same design onward to
80 m stalls instead at 42.0 m and 162.79 s, where a cold start from the
polar-derived guess binds properly at 161.86 s. The guess sails a narrow gybe,
so it is a good start for a narrow channel and a poor one for a wide one.

This is not conditioning. Scaling the constraint by 1, by its own half-width
and by the leg length gives byte-identical answers, and 800 and 2000 iterations
give identical designs. It is the shape of the problem: the channel-constrained
downwind leg has several local optima and the solver finds the one nearest
where it started.

The no-go cone gets **no** path constraint by default. The polar already gives
a boat pointed inside the cone essentially no speed, which is the physical
reason it cannot go there, and a hard constraint on top of that forbids the
brief crossing that a tack is made of. It is available as `enforce_no_go` for
the cases where the explicit constraint is wanted.

## The adjoint this problem needs

This is the only problem in the library that does not use the library's
default adjoint, and the only one whose right-hand side reads an interpolated
[`SpatialField`][gemseo_tempo.dynamics.fields.SpatialField]. Interpolation
gathers from the grid; its transpose scatters back into it; and diffrax's
`RecursiveCheckpoint` adjoint gets that wrong once the reverse pass is itself
batched. Measured at the downwind optimum, against the sensitivity of the
along-track arrival point to the wind strength that central differences give
as `1241.15` at two step sizes:

| how the derivative is taken | RecursiveCheckpoint | Direct |
| --- | --- | --- |
| one cotangent, `jax.grad` | exact | exact |
| `jax.jacrev`, several outputs at once | +0.61% | exact |
| a `vmap` over that | -8.95% | exact |

Nothing raises and no value is wrong; only derivatives are. The middle row is
the one that matters in practice, because a GEMSEO discipline Jacobian *is* a
`jacrev` over every output at once. So `create_scenario` builds the integrator
with the `Direct` adjoint, and **refuses** one handed to it that carries the
default, since passing an integrator is the ordinary way to fix a step size
and doing so used to lose the correction without a word. Build one with
`sail_integrator()`. The Direct adjoint unrolls the solve rather than
checkpointing it and costs about a third more (23.0 s against 17.0 s on the
downwind leg).

Worth recording how this was found, because the first diagnosis was wrong. The
two adjoints also disagreed at the starting guess, and that disagreement was
not the adjoint: it was the polar's corner at dead downwind, which the guess
sails straight through, with the two adjoints landing on opposite sides of it.
Only once the polar was smooth did the guess agree under both and the real
effect show up alone at the optimum. From a single number, a one-sided
derivative at a kink and a wrong derivative look exactly the same.

## Two things that had to be got right

**The heading bounds are a single turn** centred on the bearing to the mark.
The heading enters the dynamics only through its sine and cosine, so any wider
bound re-admits every trajectory again, shifted by a multiple of `2*pi`. Given
two turns the optimizer drifts through a wrap instead of gybing, and the
downwind result degrades from a 10% gain to 3%.

**The starting point is read off the polar** rather than picked by hand, by
`default_guess`: the best VMG angle gives the two headings, one per tack, and
the guess sweeps from one to the other, while the made-good speed gives the
final time as `distance / VMG`. A poor guess is not harmless here. Started
38% slow in the final time, single shooting stalls on its first line search
and returns the guess unchanged; started from the polar it converges to within
0.4% of collocation.

## The upwind leg

The `upwind` leg puts the mark dead into the wind, where the boat cannot sail
at it at all, so `straight_run_time` is infinite and beating is the only way
there. It is the harder problem, and it is included as an honest hard case
rather than a solved one: with a single smooth heading over the whole leg,
neither transcription converges reliably.

| transcription | nodes | time | above the ideal 248.1 s |
| --- | --- | --- | --- |
| collocation | 25 | 300.13 s | 21.0% |
| collocation | 40 | 258.05 s | 4.0% |
| single shooting | 25 | 256.37 s | 3.3% |
| single shooting | 40 | 343.67 s | 38.5% |

The difficulty is real rather than numerical. A tack has to cross head to
wind, where the polar says the boat stops, and one smooth heading over the
whole leg makes that crossing slow. A piecewise-linear heading helps shooting
(273 s at 40 nodes) but not collocation. Representing a tack properly needs
either a phase structure that lets the heading jump, or a control fine enough
to turn sharply without oscillating elsewhere.

## Wind fields

Three deterministic fields, selected by `wind_name`:

- `uniform`, blowing east at 10 m/s everywhere.
- `sheared`, the same direction with speed growing with `y`.
- `gyre`, built from a streamfunction `sin(pi x / L) sin(pi y / L)`, so it
  circulates and has no divergence. Its cells are half an extent across, and
  the origin is a stagnation point between four of them rather than the centre
  of one.

Each field is a pair of `SpatialField` interpolants, one per velocity
component, since a `SpatialField` is scalar-valued.
