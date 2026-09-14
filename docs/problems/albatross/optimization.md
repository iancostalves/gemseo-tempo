# Optimization

```{variable-context} gemseo_tempo.problems.albatross.model:REGISTRY
```

## What is the weakest wind it could live in?

Now everything is free, and the cycle has to close completely, in speed as
well as height and attitude. The wind gradient must pay for the drag exactly,
and the objective is the wind itself.

| what is free | minimum `w_ref` |
| --- | --- |
| nothing (a fixed, prescribed cycle) | 17.7 m/s |
| the cycle, its duration and its start | 10.18 m/s |
| all of that plus the wing | **8.65 m/s** |

Letting the optimizer choose the cycle rather than fly a prescribed one is
worth 43%. Adding the wing to the problem is worth a further **15%**, at
0.628 m2 spanning 3.70 m and the thinnest section available: an aspect ratio
of 21.8, a glide ratio of 24.3, and 9.1 kg of bird. The area is the first wing
variable in three passes to settle between its bounds rather than on one.

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

(Those are with wider design bounds than the default; see
{doc}`../../guides/scenario` for how bounds are declared.)

## How much can a fixed cycle gain?

Before letting the optimizer choose the cycle, it is worth asking what a
*prescribed* one can already do. The wind and the bird are fixed. The bird
must return to the height and attitude it started at, but not to the same
speed, so anything it extracts shows up as airspeed. Sweeping the wind, over
a nine-second cycle:

| `w_ref` (m/s) | drag work (m) | gradient work (m) | net (m) |
| --- | --- | --- | --- |
| 0 | −18.16 | 0.00 | −18.13 |
| 8 | −31.56 | +15.60 | −15.67 |
| 12 | −38.04 | +28.20 | −9.78 |
| 16 | −44.58 | +41.27 | −3.26 |
| 18 | −50.69 | +51.13 | **+0.13** |
| 22 | −58.32 | +65.93 | +7.26 |
| 26 | −68.04 | +81.87 | +14.44 |

With no wind the cycle is a pure glide and loses 18.1 m of specific energy,
all of it to drag. As the wind strengthens the gradient work grows faster than
the drag does, and the two cross at **`w_ref` = 17.9 m/s**: below that the
prescribed cycle cannot sustain itself, above it the bird gains. This is the
17.7 m/s "nothing free" row above, and the gap between it and 10.18 m/s is
exactly what letting the optimizer choose the cycle buys.

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

**And this is where `C_L,max` earns its keep.** The section trade looked
one-sided above, with the thinnest section winning on minimum wind. Under a
mission that rewards progress it is not one-sided at all:

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

(the-shape-of-the-cycle)=
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

(formulation-b-what-did-you-size-it-for)=
## Formulation B: what did you size it for?

Formulation C optimizes the wing and the cycle at once. That is not how the
work is usually organized: the vehicle is settled first, against figures of
merit an engineer can reason about, and only then is a trajectory flown on it.
Formulation B is that route, and it comes in two versions, because the merit
you pick is an assumption about the mission whether you write it down or not.

**B-cruise** sizes for the glide ratio alone. This is what conventional
practice does when nobody states the assumption: the best-gliding wing is
taken to be the best wing.

**B-dynamics** sizes for gliding *and* turning, using a proxy computable from
the wing alone so it never flies the cycle: the load factor the wing can pull
at a reference airspeed, `n = 0.5 rho V^2 S C_L,max / (m g)`, which is what
sets the minimum turn radius. The two terms carry equal weight, which is a
modelling choice and is stated as one.

Each is then flown on **both** missions, against simultaneous co-design:

| sized for | area | AR | glide | `n_max` | min wind | vs C | crossing | vs C |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cruise | 0.500 | 27.4 | 26.0 | 1.38 | 8.90 | +2.9% | 8.48 s | +6.5% |
| dynamics | 0.900 | 9.7 | 14.2 | 5.84 | 14.00 | +61.8% | 11.13 s | +39.8% |
| best blend | 0.900 | 15.2 | 21.8 | 3.47 | 9.15 | +5.8% | 8.26 s | +3.7% |
| **co-design** | **0.628** | 21.8 | 24.3 | 2.02 | **8.65** | ref | **7.96 s** | ref |

Three things fall out of that table.

**Cruise sizing is nearly right for the wrong reason.** Minimum wind is a
cruise-like objective, so a wing sized for gliding is only 2.9% off it. Ask
the same bird to *get somewhere* and the gap more than doubles. The assumption
that cruise is the goal is invisible until the mission stops being cruise.

**Sizing hard for the manoeuvre is far worse than sizing for cruise.** The
equal-weighted proxy produces a stubby, thick-sectioned bird that turns
beautifully and glides at 14. Turning matters, but not that much, and a proxy
has no way of knowing how much.

**No weighting reaches the co-design wing at all.** This is the part that
makes the argument structural rather than a matter of tuning. The vehicle
merit is close to linear in its normalized quantities over a box, so its
optimum sits at a **vertex** and jumps between vertices as the weights move:
the area goes 0.500, 0.500, 0.900, 0.900 as the turn weight is swept, never
stopping in between. The mission's optimum is **interior**, at 0.628. There is
no weight that lands there, so no amount of hindsight would let the sequential
route reproduce the co-design answer. This is exactly the setting the
{ref}`Multiobjective optimization under uncertainty <multiobjective-optimization-under-uncertainty>`
subsection below asks the next question of: does the forecast error the two
wings face change this ranking?

## What compiling costs

| size | variables | compile | compiled | eager | speedup | crossover |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 71 | 0.27 s | 0.9 ms | 55.0 ms | 58x | 5 |
| 16 | 135 | 0.27 s | 1.1 ms | 55.6 ms | 52x | 5 |

Compiling pays within five evaluations here too; see
{doc}`../../guides/benchmarking` for the general shape of this measurement
and {doc}`../brachistochrone/index` for the same table on the smallest problem.

## Optimization benchmarks

```{image} ../../_data/albatross/optimizer_study.png
:alt: Optimizer benchmark on the albatross
```

Reused from `benchmarks/baselines/albatross.json`. The albatross offers only
collocation, so it carries no transcription study: a study that varies one
thing needs at least two levels of it, and this problem does not have a
second. See {doc}`../../guides/benchmarking` for what that comparison measures
and how the baseline is kept current.

## Weaker wind or a faster crossing

The two things a soaring bird might want cannot both be had, and mNBI draws
the trade directly rather than scalarizing it. Over a 300 m crossing:

| `w_ref` needed | crossing time | cross-country speed |
| --- | --- | --- |
| 10.97 m/s | 15.56 s | 19.3 m/s |
| 12.06 m/s | 11.91 s | 25.2 m/s |
| 16.45 m/s | 9.41 s | 31.9 m/s |
| 23.06 m/s | 7.67 s | 39.1 m/s |
| 31.14 m/s | 6.45 s | 46.5 m/s |
| 40.00 m/s | 5.50 s | 54.6 m/s |

The low end is worth a second look: **10.97 m/s to cross 300 m per cycle,
against 8.65 m/s merely to sustain one.** Covering ground costs more than
staying airborne, which is the difference between a bird that survives a calm
and a bird that gets anywhere. `plot_pareto` draws the curve; anything marked
on it has to be measured in the same two objectives, since a minimum wind
taken without a distance requirement is a different quantity and will appear
to dominate the front.

## The time budget

The sprint asks for formulation C under five minutes on one core with
fixed-step integration. It is not close:

| integration | minimum `w_ref` | time |
| --- | --- | --- |
| adaptive, rtol 1e-8 | 8.6332 m/s | 6.6 s |
| fixed step, 1/200 | 8.6332 m/s | 3.7 s |

Identical answers, and fixed stepping is the faster of the two, because an
adaptive controller spends steps chasing a tolerance the optimizer never asked
for. Eighty times inside the budget.

## Discussion

Findings about this library, surfaced by the albatross, that do not belong
under the physics, the optimization or the uncertainty.

### Are these optima real?

Every number above comes from one inverse-dynamics guess, which is a strong
start and therefore a suspicious one. GEMSEO's `MultiStart` over a Latin
hypercube is the check:

| start | minimum `w_ref` |
| --- | --- |
| seeded, inverse dynamics | 8.6486 m/s |
| four random starts | 8.6400 m/s |

Random search does slightly better, by 0.10%, so the seeded optimum is
marginally local. The margin is small enough that the basin is clearly broad
rather than the guess being wrong, and it is reported rather than tidied away.

(three-things-that-had-to-be-got-right)=
### Three things that had to be got right

**A heading is periodic only modulo a turn.** A loop comes back to the same
direction of flight having gained a full turn. Demanding plain equality
forbids the very cycle the problem is about, and the residual sits stubbornly
at 6.283. `PeriodicityConstraint` therefore takes an `offsets` mapping.

**The guess is built, not picked.** A soaring cycle is not something an
optimizer stumbles onto. `soaring_cycle` chooses the shape the energy equation
asks for, then inverts the two rotational equations of motion for the lift
and bank that would fly it, which is exactly the construction the example
simulation under Physics draws. That lands the guess with collocation
defects already near zero everywhere except the speed equation.

**Finite-difference checks need fixed-step integration.** The split assembly is
verified with `check_jacobian`, and adaptive stepping changes its step sequence
discontinuously as the inputs move, which swamps a 1e-7 probe. Passing
`DiffraxIntegrator(dt0=0.01)` makes the whole map smooth, and all three split
disciplines then agree to 1e-3.

### Closing the cycle with an inner solve

The guess satisfies every equation of motion except the one for speed, and
speed is also the state whose periodicity is hardest to close. That suggests
a nested formulation: solve for the initial speed **inside** each evaluation,
so the outer problem loses a design variable and an equality.

The mechanism works. The residual `r(V0) = V(t_f; V0) - V0` is solved by
Newton, and the derivative comes from the implicit function theorem rather
than from differentiating through the iterations, which keeps the gradient
exact and its cost independent of how many steps were taken. Against a central
difference the two agree to five significant figures.

What stops it being usable as it stands is physical rather than numerical:

| start | lowest height | physical | converged |
| --- | --- | --- | --- |
| 28 m/s | −14.9 m | no | no |
| 32 m/s | −12.0 m | no | no |
| 34 m/s | −6.8 m | no | yes |
| 36 m/s | +2.0 m | yes | yes |
| 40 m/s | +8.3 m | yes | yes |

Below about 35 m/s the prescribed cycle flies the bird under the water. The
wind profile's smooth floor keeps the arithmetic finite, so what comes back is
a number computed from a trajectory that does not exist, and the integration
underneath the root finder fails outright rather than returning a poor value.
Inside an optimizer that is a discipline which did not evaluate.

So the inner solve is conditionally robust, and the condition is now measured
rather than discovered: `robustness_map` reports where a start can be taken
from. A formulation built on this must bracket the root or start from the
inverse-dynamics speed, which is already in the right region. That formulation
is the remaining step, and it is not attempted here on the strength of a solve
that fails outside a region this narrow.

### Still open

**The span still wants everything it is allowed**, even though the area no
longer does. Aspect ratio is bought two ways here, by lengthening the span and
by shrinking the area, and only the second now carries a penalty: a smaller
area means a smaller chord and a worse Reynolds number, which is what pulled
the area off its bound. A longer span at fixed area does the same thing to the
chord, so the two are not independent, and what actually limits a real bird's
span, structural and otherwise, is still not in this model.

**The wing mass law barely matters.** Growing as the square root of aspect
ratio it changed nothing at all, and the bending law `AR^1.5 sqrt(S)` moves
the answer far less than the Reynolds penalty does. The interesting physics
here turned out to be aerodynamic rather than structural, which was not the
expectation going in.

## Reproducing everything here

```bash
python examples/albatross/run_albatross.py --output figures/albatross
```

Every number on this page is measured at 16 collocation segments unless the
table says otherwise, with the model as of the third pass of Sprint 9.
