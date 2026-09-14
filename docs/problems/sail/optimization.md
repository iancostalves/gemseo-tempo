# Optimization

```{variable-context} gemseo_tempo.problems.sail.model:REGISTRY
```

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

## The gybe nobody asked for

```{eval-rst}
.. plot:: _ext/gallery/sail_main_result.py
   :include-source: false
```

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

```{eval-rst}
.. plot:: _ext/gallery/sail_channel_trade.py
   :include-source: false
```

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

## What compiling costs

One discipline evaluation, compiled against eager, at 25 nodes and 75
variables: 1.39 s to compile, 0.7 ms compiled, 460.8 ms eager, a 628x
speedup with the crossover at 3 evaluations. Compiling pays sooner here
than on the brachistochrone, because a single eager evaluation of a
wind-field lookup is far more expensive to begin with; see
{doc}`../../guides/benchmarking` for how the measurement avoids timing a
discipline's own input cache instead.

## Transcription benchmarks

```{image} ../../_data/sail/transcription_study.png
:alt: Transcription benchmark on the sail
```

Reused from `benchmarks/baselines/sail.json` rather than re-run here, since
it was blessed with this same polar and nothing since has changed the
physics it measures. `python benchmarks/bless.py` regenerates it when
something does.

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

The difficulty is real rather than numerical, and it is a known feature of
the problem rather than a peculiarity of this implementation. Beating upwind
means alternating between port and starboard tack at the best upwind angle,
and a tack **crosses the no-go zone**, where the boat is briefly not
propelled at all and carries through on its own momentum
{cite}`ciccarella2024sailboat`. A single smooth heading over the whole leg
has to make that crossing slowly, so it pays the no-go zone's price at every
tack instead of passing through it.

A piecewise-linear heading helps shooting (273 s at 40 nodes) but not
collocation. Representing a tack properly needs either a phase structure
that lets the heading jump, or a control fine enough to turn sharply without
oscillating elsewhere — and either way the count of tacks becomes part of
the answer, which is what makes the upwind problem a different shape from
the downwind one rather than a harder instance of it. The stochastic
formulation in {cite}`ciccarella2024sailboat` makes that explicit by
charging a fixed cost per tack and optimizing over the switching times,
which is the structure this library would need to add.

## Discussion

Findings about this library, surfaced by this problem, that do not belong
under Physics, Optimization or Optimization under uncertainty.

### The adjoint this problem needs

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

### Two things that had to be got right

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
