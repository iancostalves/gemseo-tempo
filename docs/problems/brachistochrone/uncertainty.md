# Uncertainty

```{variable-context} gemseo_tempo.problems.brachistochrone.model:REGISTRY
```

A wire, once built, meets friction and drag nobody measured exactly. Three
of this problem's constants are declared uncertain, and they are declared
where the variable is rather than in an analysis:

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

Their means make a bead with a little of each variant. The wire's shape
does not change with them, so the bead is integrated along the wire rather
than in time, by {eq}`eq-brach-wire`: the end of the wire is then pure
geometry, and only the descent time is uncertain. Optimized this way for
the ideal bead, the wire is the cycloid again, to a part in a million.

## Uncertainty propagation

:::{admonition} Takeaway: uncertainty propagation
:class: tip

- **Sampling:** vectorized with `vmap`, never looped, and on a wire, not a
  steering schedule, so that only the time is uncertain.
- **Estimator:** first-order Taylor, two model calls, since the descent time
  is nearly linear in the three constants.
- **Differentiation:** finite differences, batched into one call: with
  three uncertain inputs they lose nothing and are the quicker.
:::

Optimized for the mean constants and frozen, the wire takes
{docvalue}`brachistochrone/propagation.json:reference.mean:.4f` s
on average, give or take
{docvalue}`brachistochrone/propagation.json:reference.standard_deviation:.4f` s.
Each way of estimating that, against 20 000 draws, where AD marks exact
derivatives by reverse-mode automatic differentiation and FD central
differences, every perturbed point in the same vectorized call, at the
step closest to the exact derivative at the mean:

| method | error, mean | error, deviation | model calls | time |
| --- | --- | --- | --- | --- |
| sampling, 256 draws | {docvalue}`brachistochrone/propagation.json:estimates.sampling_vmapped.mean_error:+.0e` σ | {docvalue}`brachistochrone/propagation.json:estimates.sampling_vmapped.deviation_error:+.0e` | {docvalue}`brachistochrone/propagation.json:estimates.sampling_vmapped.model_calls:d` | {docvalue}`brachistochrone/propagation.json:estimates.sampling_vmapped.seconds:.3f` s, looped {docvalue}`brachistochrone/propagation.json:estimates.sampling_looped.seconds:.2f` s |
| Taylor, 1st order, AD | {docvalue}`brachistochrone/propagation.json:estimates.taylor1_ad.mean_error:+.0e` σ | {docvalue}`brachistochrone/propagation.json:estimates.taylor1_ad.deviation_error:+.0e` | {docvalue}`brachistochrone/propagation.json:estimates.taylor1_ad.model_calls:d` | {docvalue}`brachistochrone/propagation.json:estimates.taylor1_ad.seconds:.3f` s |
| Taylor, 1st order, FD | {docvalue}`brachistochrone/propagation.json:estimates.taylor1_fd.mean_error:+.0e` σ | {docvalue}`brachistochrone/propagation.json:estimates.taylor1_fd.deviation_error:+.0e` | {docvalue}`brachistochrone/propagation.json:estimates.taylor1_fd.model_calls:d` | {docvalue}`brachistochrone/propagation.json:estimates.taylor1_fd.seconds:.3f` s |
| Taylor, 2nd order, AD | {docvalue}`brachistochrone/propagation.json:estimates.taylor2_ad.mean_error:+.0e` σ | {docvalue}`brachistochrone/propagation.json:estimates.taylor2_ad.deviation_error:+.0e` | {docvalue}`brachistochrone/propagation.json:estimates.taylor2_ad.model_calls:d` | {docvalue}`brachistochrone/propagation.json:estimates.taylor2_ad.seconds:.3f` s |
| Taylor, 2nd order, FD | {docvalue}`brachistochrone/propagation.json:estimates.taylor2_fd.mean_error:+.0e` σ | {docvalue}`brachistochrone/propagation.json:estimates.taylor2_fd.deviation_error:+.0e` | {docvalue}`brachistochrone/propagation.json:estimates.taylor2_fd.model_calls:d` | {docvalue}`brachistochrone/propagation.json:estimates.taylor2_fd.seconds:.3f` s |
| chaos, 12 points | {docvalue}`brachistochrone/propagation.json:estimates.chaos.mean_error:+.0e` σ | {docvalue}`brachistochrone/propagation.json:estimates.chaos.deviation_error:+.0e` | {docvalue}`brachistochrone/propagation.json:estimates.chaos.model_calls:d` | {docvalue}`brachistochrone/propagation.json:estimates.chaos.seconds:.3f` s |
| chaos + AD gradients | {docvalue}`brachistochrone/propagation.json:estimates.chaos_ad.mean_error:+.0e` σ | {docvalue}`brachistochrone/propagation.json:estimates.chaos_ad.deviation_error:+.0e` | {docvalue}`brachistochrone/propagation.json:estimates.chaos_ad.model_calls:d` | {docvalue}`brachistochrone/propagation.json:estimates.chaos_ad.seconds:.3f` s |
| chaos + FD gradients | {docvalue}`brachistochrone/propagation.json:estimates.chaos_fd.mean_error:+.0e` σ | {docvalue}`brachistochrone/propagation.json:estimates.chaos_fd.deviation_error:+.0e` | {docvalue}`brachistochrone/propagation.json:estimates.chaos_fd.model_calls:d` | {docvalue}`brachistochrone/propagation.json:estimates.chaos_fd.seconds:.3f` s |

Over these ranges the descent time is nearly linear in the three
constants, so every method lands within a few hundredths of a deviation of
the reference, and first-order Taylor, one evaluation and one gradient, is
enough. What separates the methods is cost, in two ways. Vectorizing the
sample with `vmap` runs the same 256 draws in one dispatch, in
{docvalue}`brachistochrone/propagation.json:estimates.sampling_vmapped.seconds:.2f` s against
{docvalue}`brachistochrone/propagation.json:estimates.sampling_looped.seconds:.1f` s in a loop. And with only three uncertain inputs, finite
differences cost six evaluations per gradient and reproduce the AD numbers
exactly; on a model this cheap they are even quicker than a reverse pass,
which costs a few evaluations. Where AD earns its keep is a gradient with
respect to many variables, which is what the next section needs.

### What a screening is actually ranking

Frozen at the drag variant's optimum, Morris ranks
{var}`mu <symbol>` first, {var}`drag_coefficient <symbol>` a long way
behind, and {var}`g <symbol>` last; Sobol confirms the same ordering, with
total indices of 0.92, 0.07 and 0.01. The cheap screening anticipates the
expensive decomposition, which is the whole argument for doing it first.

The ordering used to be the other way round, and that is the more useful
thing in this section. Drag led while `k/m` was sampled over 0.03 to
0.07, bounds with nothing behind them. Deriving it from a cylinder's drag
coefficient by {eq}`brachistochrone-drag` left a quantity that is still
genuinely uncertain, over 1.0 to 1.3, but much less so; meanwhile friction
picked up the turning term in {eq}`brachistochrone-normal` and grew.
Both moved for physical reasons, in opposite directions, and the ranking
followed.

So a sensitivity ranking is a statement about the *declared distributions*
as much as about the model. It is exactly as trustworthy as the reasons
behind the bounds, which is why this problem's bounds now have reasons.

```{note}
A near-miss worth recording, because the library has made this mistake
before. Dropping the friction term for variants where `mu` is zero looked
like a harmless optimization: it keeps the control-rate machinery, and so
the indirect transcription, available where friction is definitionally
absent. But the drag variant also has `mu = 0` nominally, and it is a
variant the uncertainty studies run on: the model stopped reading a
declared-uncertain parameter, and its sensitivity to friction came back as
*exactly zero*.

That is the albatross's hardcoded lift limit again, a number that reads
like a physical finding and is an artefact of the wiring. Friction is now
dropped for the ideal variant alone, where it is zero by definition rather
than by accident, and
`tests/problems/test_brachistochrone.py::test_every_uncertain_constant_reaches_the_dynamics`
perturbs every declared uncertainty on every variant a study uses and
insists the trajectory moves.
```

## Optimization under uncertainty

:::{admonition} Takeaway: optimization under uncertainty
:class: tip

- **Formulation:** the wire designed for the mean constants; a robust
  objective finds the same wire.
- **Sampling:** one fixed sample of {docvalue}`brachistochrone/robust.json:design_sample:d` draws,
  the same at every iteration, checked afterwards on
  {docvalue}`brachistochrone/robust.json:check_sample:d` fresh ones.
- **Differentiation:** automatic, in the loop, once the design is large:
  an AD gradient costs about the same whatever its size. At twenty
  variables, finite differences batched into one call compile in half the
  time and win a single solve.
:::

Three wires, each flown on
{docvalue}`brachistochrone/robust.json:check_sample:d` fresh draws:

| wire | mean | deviation | 95th percentile |
| --- | --- | --- | --- |
| the cycloid, for the ideal bead | {docvalue}`brachistochrone/robust.json:designs.ideal.mean:.5f` s | {docvalue}`brachistochrone/robust.json:designs.ideal.std:.5f` s | {docvalue}`brachistochrone/robust.json:designs.ideal.p95:.5f` s |
| designed for the mean constants | {docvalue}`brachistochrone/robust.json:designs.nominal.mean:.5f` s | {docvalue}`brachistochrone/robust.json:designs.nominal.std:.5f` s | {docvalue}`brachistochrone/robust.json:designs.nominal.p95:.5f` s |
| minimizing mean + 2σ over {docvalue}`brachistochrone/robust.json:design_sample:d` draws | {docvalue}`brachistochrone/robust.json:designs.robust.mean:.5f` s | {docvalue}`brachistochrone/robust.json:designs.robust.std:.5f` s | {docvalue}`brachistochrone/robust.json:designs.robust.p95:.5f` s |

The three agree to a few parts in ten thousand: the uncertainty moves the
descent time, not the best wire, and even the cycloid, designed for no
friction at all, gives almost nothing away. That is a property of this
problem, and it leaves the method to compare. The robust problem, solved
by SLSQP from the same start with each kind of gradient:

| gradient | iterations | compilation | optimization |
| --- | --- | --- | --- |
| AD, reverse mode | {docvalue}`brachistochrone/robust.json:designs.robust.calls:d` | {docvalue}`brachistochrone/robust.json:designs.robust.compile_seconds:.1f` s | {docvalue}`brachistochrone/robust.json:designs.robust.seconds:.1f` s |
| FD, batched | {docvalue}`brachistochrone/robust.json:designs.robust_fd.calls:d` | {docvalue}`brachistochrone/robust.json:designs.robust_fd.compile_seconds:.1f` s | {docvalue}`brachistochrone/robust.json:designs.robust_fd.seconds:.1f` s |

Both reach the same wire in the same number of iterations. Each iteration
differentiates the mean and deviation of
{docvalue}`brachistochrone/robust.json:design_sample:d` descents with respect to all
{docvalue}`brachistochrone/robust.json:gradient_cost.1.n_design:d` design variables. By AD that is one
reverse pass through the whole batch, through the checkpointing adjoint,
the fastest exact route measured; by finite differences, every perturbed
wire flown in the same vectorized call, at the step closest to the exact
gradient. The loop runs about twice as fast with AD, but AD compiles for
twice as long, and over one solve of twenty iterations, compilation
included, the batched differences come out ahead. What turns it is the
size of the design: an AD gradient costs about the same however many
variables there are, while the batch of differences grows with every one,
to {docvalue}`brachistochrone/robust.json:gradient_cost.3.fd_seconds:.2f` s against
{docvalue}`brachistochrone/robust.json:gradient_cost.3.ad_seconds:.2f` s at
{docvalue}`brachistochrone/robust.json:gradient_cost.3.n_design:d` variables:

```{eval-rst}
.. plot:: _ext/gallery/brachistochrone_gradient_cost.py
   :include-source: false
```

See {doc}`../../guides/uncertainty` for the API these studies use, and for
why {func}`~gemseo_tempo.uncertainty.batch.freeze_design` is an explicit
step rather than something implied.
