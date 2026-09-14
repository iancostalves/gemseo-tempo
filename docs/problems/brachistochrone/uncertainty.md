# Optimization under uncertainty

```{variable-context} gemseo_tempo.problems.brachistochrone.model:REGISTRY
```

Three of this problem's constants are declared uncertain, and they are
declared where the variable is rather than in an analysis:

```{variables} gemseo_tempo.problems.brachistochrone.model:REGISTRY
:filter: tags=constant
:columns: name, full_name, distribution
```

Each is uncertain for a different reason, which is why this short list is
worth having:

- {var}`drag_coefficient <symbol>` is uncertain about the *shape*. A
  circular cylinder's drag coefficient genuinely runs about 1.0 to 1.3 with
  Reynolds number and surface finish, and since the quadratic drag group
  follows from it by {eq}`brachistochrone-drag`, that is the honest place
  to put the spread.
- {var}`mu <symbol>` is uncertain about the *contact*: a sliding friction
  coefficient is not a property anyone measures to three figures.
- {var}`g <symbol>` is in the list precisely because it is **not**
  uncertain in any useful sense. It is known to three decimal places, and a
  screening that cannot say so is not doing its job.

That last one is the point of including it: the brachistochrone is where
the uncertainty machinery is checked against an answer somebody already
knows, in the same spirit as checking the optimizer against the cycloid.

## What a screening is actually ranking

Frozen at the drag variant's optimum, Morris ranks
{var}`mu <symbol>` first, {var}`drag_coefficient <symbol>` a long way
behind, and {var}`g <symbol>` last; Sobol confirms the same ordering, with
total indices of 0.92, 0.07 and 0.01. The cheap screening anticipates the
expensive decomposition, which is the whole argument for doing it first.

The ordering was the other way round one sprint ago, and that is the more
useful thing on this page. Drag led while `k/m` was sampled over 0.03 to
0.07 — bounds with nothing behind them. Deriving it from a cylinder's drag
coefficient by {eq}`brachistochrone-drag` left a quantity that is still
genuinely uncertain, over 1.0 to 1.3, but much less so; meanwhile friction
picked up the centripetal term in {eq}`brachistochrone-normal` and grew.
Both moved for physical reasons, in opposite directions, and the ranking
followed.

So a sensitivity ranking is a statement about the *declared distributions*
as much as about the model. It is exactly as trustworthy as the reasons
behind the bounds, which is why this problem's bounds now have reasons.

```{note}
A near-miss worth recording, because the library has made this mistake
before. Dropping the friction term for variants where `mu` is zero looked
like a harmless optimization — it keeps the control-rate machinery, and so
the indirect transcription, available where friction is definitionally
absent. But the drag variant also has `mu = 0` nominally, and it is a
variant the uncertainty studies run on: the model stopped reading a
declared-uncertain parameter, and its sensitivity to friction came back as
*exactly zero*.

That is the albatross's hardcoded lift limit again — a number that reads
like a physical finding and is an artefact of the wiring. Friction is now
dropped for the ideal variant alone, where it is zero by definition rather
than by accident, and
`tests/problems/test_brachistochrone.py::test_every_uncertain_constant_reaches_the_dynamics`
perturbs every declared uncertainty on every variant a study uses and
insists the trajectory moves.
```

There is no dedicated uncertainty study on this page. The problem is here
to be the smallest complete example, and propagating a forecast error
through it would measure the machinery rather than teach anything about
beads. The studies that do the work are on the two problems where a wrong
forecast costs something:

- {doc}`../sail/uncertainty` — six estimators scored against a large
  sample, and what a safety margin costs against carrying the probability
  in the problem.
- {doc}`../albatross/uncertainty` — screening against variance
  decomposition, and a co-design wing read under an uncertain wind.

See {doc}`../../guides/uncertainty` for the API those studies use, and for
why {func}`~gemseo_tempo.uncertainty.batch.freeze_design` is an explicit
step rather than something implied.
