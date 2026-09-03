# Sailing to a mark

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

Two details matter. The table is **dense through the no-go transition**: a C2
spline through a sparse table undershoots to `-0.12` there, and negative boat
speed is something an optimizer will exploit cheerfully. Even densely sampled
it dips slightly below zero, so the result is **clamped** with a smooth
maximum against zero, which floors it at `+0.0008` while leaving the rest of
the curve untouched.

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
| best gybing angle | 151.8 deg | 5.71 m/s | 5.03 m/s |

So gybing is worth **11.7%** in principle, and over the 800 m leg that is
159.1 s against 177.8 s sailed straight. `best_vmg_angle` computes the angle,
and it is what the default starting point is built from.

## What the optimizer finds

Both direct transcriptions find the gybe, from a starting guess that only
knows the best VMG angle and not the trajectory:

| transcription | nodes | time | above the ideal |
| --- | --- | --- | --- |
| collocation | 25 | 160.27 s | 0.74% |
| single shooting | 25 | 159.40 s | 0.19% |
| single shooting | 40 | 159.30 s | 0.13% |

Against the 177.78 s straight run, that is a **10% gain**, and it lands within
a fraction of a percent of the theoretical VMG optimum. The heading history
comes back symmetric about the bearing to the mark, near `+29` and `-29`
degrees, which is exactly the best VMG angle expressed as a heading.

## The path constraint

The channel constraint limits how far the boat may stray from the direct line
to the mark. It is the constraint worth having, because straying wide is
precisely what gybing buys speed with, so it trades against the objective
rather than fencing off a region the boat had no reason to enter:

| half-width | time | furthest across |
| --- | --- | --- |
| unconstrained | 160.31 s | 180.9 m |
| 250 m | 160.31 s | 180.9 m |
| 150 m | 161.31 s | 149.3 m |
| 80 m | 163.23 s | 80.0 m |

A limit the free optimum already respects leaves it untouched; tighter limits
bind exactly, and cost time monotonically.

The no-go cone gets **no** path constraint by default. The polar already gives
a boat pointed inside the cone essentially no speed, which is the physical
reason it cannot go there, and a hard constraint on top of that forbids the
brief crossing that a tack is made of. It is available as `enforce_no_go` for
the cases where the explicit constraint is wanted.

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
