# Uncertainty

```{variable-context} gemseo_tempo.problems.albatross.model:REGISTRY
```

## Uncertainty propagation

:::{admonition} Takeaway: uncertainty propagation
:class: tip

- **Transcription:** single shooting, so that the frozen cycle responds to
  the wind, started from a forty-segment collocation optimum. From the
  default guess, or from twenty segments, shooting does not converge.
- **Sampling:** vectorized with `vmap`.
- **Estimator:** first-order Taylor, two model calls.
- **Differentiation:** finite differences, batched into one call: with two
  uncertain inputs they serve as well as exact derivatives, and are the
  quicker.
:::

The wind at 10 m is forecast at 22 m/s and known to 10%, and the roughness
of the sea, which sets how fast the wind grows with height, is log-normal
about 3 cm. The cycle that banks the most energy at the forecast, the energy
mission on {doc}`optimization`, is flown here by single shooting on forty
intervals, started from the collocation answer, so that a frozen cycle
responds to the wind. It ends
with an energy height of {docvalue}`albatross/propagation.json:objective:.1f*-1` m at the
forecast and {docvalue}`albatross/propagation.json:reference.mean:.1f` ±
{docvalue}`albatross/propagation.json:reference.standard_deviation:.1f` m over the uncertainty.
Each way of estimating that, against 20 000 draws, where AD marks exact
derivatives by reverse-mode automatic differentiation and FD central
differences, every perturbed point in the same vectorized call, at the
step closest to the exact derivative at the mean:

| method | error, mean | error, deviation | model calls | time |
| --- | --- | --- | --- | --- |
| sampling, 256 draws | {docvalue}`albatross/propagation.json:estimates.sampling_vmapped.mean_error:+.0e` σ | {docvalue}`albatross/propagation.json:estimates.sampling_vmapped.deviation_error:+.0e` | {docvalue}`albatross/propagation.json:estimates.sampling_vmapped.model_calls:d` | {docvalue}`albatross/propagation.json:estimates.sampling_vmapped.seconds:.3f` s, looped {docvalue}`albatross/propagation.json:estimates.sampling_looped.seconds:.2f` s |
| Taylor, 1st order, AD | {docvalue}`albatross/propagation.json:estimates.taylor1_ad.mean_error:+.0e` σ | {docvalue}`albatross/propagation.json:estimates.taylor1_ad.deviation_error:+.0e` | {docvalue}`albatross/propagation.json:estimates.taylor1_ad.model_calls:d` | {docvalue}`albatross/propagation.json:estimates.taylor1_ad.seconds:.3f` s |
| Taylor, 1st order, FD | {docvalue}`albatross/propagation.json:estimates.taylor1_fd.mean_error:+.0e` σ | {docvalue}`albatross/propagation.json:estimates.taylor1_fd.deviation_error:+.0e` | {docvalue}`albatross/propagation.json:estimates.taylor1_fd.model_calls:d` | {docvalue}`albatross/propagation.json:estimates.taylor1_fd.seconds:.3f` s |
| Taylor, 2nd order, AD | {docvalue}`albatross/propagation.json:estimates.taylor2_ad.mean_error:+.0e` σ | {docvalue}`albatross/propagation.json:estimates.taylor2_ad.deviation_error:+.0e` | {docvalue}`albatross/propagation.json:estimates.taylor2_ad.model_calls:d` | {docvalue}`albatross/propagation.json:estimates.taylor2_ad.seconds:.3f` s |
| Taylor, 2nd order, FD | {docvalue}`albatross/propagation.json:estimates.taylor2_fd.mean_error:+.0e` σ | {docvalue}`albatross/propagation.json:estimates.taylor2_fd.deviation_error:+.0e` | {docvalue}`albatross/propagation.json:estimates.taylor2_fd.model_calls:d` | {docvalue}`albatross/propagation.json:estimates.taylor2_fd.seconds:.3f` s |
| chaos, 12 points | {docvalue}`albatross/propagation.json:estimates.chaos.mean_error:+.0e` σ | {docvalue}`albatross/propagation.json:estimates.chaos.deviation_error:+.0e` | {docvalue}`albatross/propagation.json:estimates.chaos.model_calls:d` | {docvalue}`albatross/propagation.json:estimates.chaos.seconds:.3f` s |
| chaos + AD gradients | {docvalue}`albatross/propagation.json:estimates.chaos_ad.mean_error:+.0e` σ | {docvalue}`albatross/propagation.json:estimates.chaos_ad.deviation_error:+.0e` | {docvalue}`albatross/propagation.json:estimates.chaos_ad.model_calls:d` | {docvalue}`albatross/propagation.json:estimates.chaos_ad.seconds:.3f` s |
| chaos + FD gradients | {docvalue}`albatross/propagation.json:estimates.chaos_fd.mean_error:+.0e` σ | {docvalue}`albatross/propagation.json:estimates.chaos_fd.deviation_error:+.0e` | {docvalue}`albatross/propagation.json:estimates.chaos_fd.model_calls:d` | {docvalue}`albatross/propagation.json:estimates.chaos_fd.seconds:.3f` s |

The energy banked is nearly linear in the wind and the roughness over their
ranges, so every method lands within a few hundredths of a deviation of the
reference, and first-order Taylor, one evaluation and one gradient, is as
good as any. Vectorizing pays most again: the 256 draws take
{docvalue}`albatross/propagation.json:estimates.sampling_vmapped.seconds:.1f` s in one `vmap`
dispatch against {docvalue}`albatross/propagation.json:estimates.sampling_looped.seconds:.1f` s
in a loop. With two uncertain inputs, finite differences need four
evaluations per gradient and are cheaper than a reverse pass through nine
seconds of flight: exact derivatives with respect to two inputs are not what
this problem is short of. The gradient that matters is the one with respect
to the cycle's controls, which the next section needs.

## Optimization under uncertainty

:::{admonition} Takeaway: optimization under uncertainty
:class: tip

- **Formulation:** the mean less twice the deviation of the energy,
  started from the deterministic cycle.
- **Sampling:** one fixed sample of
  {docvalue}`albatross/robust.json:design_sample:d` draws, checked on
  {docvalue}`albatross/robust.json:check_sample:d` fresh ones.
- **Differentiation:** automatic, with the Jacobian taken group by group:
  the constraints on the forecast flight in forward mode, the statistic
  through the sample in reverse mode, both through the Direct adjoint. One
  gradient costs a seventh of what finite differences do, even batched
  into one call.
:::

The robust cycle maximizes the mean energy less twice its deviation over
{docvalue}`albatross/robust.json:design_sample:d` draws of the wind and the sea, starting
from the deterministic one. Both, flown on
{docvalue}`albatross/robust.json:check_sample:d` fresh draws:

| cycle | at the forecast | mean | deviation | 5th percentile |
| --- | --- | --- | --- | --- |
| deterministic | {docvalue}`albatross/robust.json:designs.deterministic.nominal:.2f` m | {docvalue}`albatross/robust.json:designs.deterministic.mean:.2f` m | {docvalue}`albatross/robust.json:designs.deterministic.std:.2f` m | {docvalue}`albatross/robust.json:designs.deterministic.p05:.2f` m |
| robust, mean − 2σ | {docvalue}`albatross/robust.json:designs.robust.nominal:.2f` m | {docvalue}`albatross/robust.json:designs.robust.mean:.2f` m | {docvalue}`albatross/robust.json:designs.robust.std:.2f` m | {docvalue}`albatross/robust.json:designs.robust.p05:.2f` m |

The robust cycle gives up most of a metre at the forecast and a quarter of
one on average, and in exchange its spread falls by a third and its worst
cycle in twenty banks almost two metres more.

Here the gradient is the bottleneck. Every iteration differentiates the
mean and deviation of {docvalue}`albatross/robust.json:design_sample:d` nine-second flights
with respect to all {docvalue}`albatross/robust.json:gradient_cost.n_design:d` controls:
one gradient takes {docvalue}`albatross/robust.json:gradient_cost.ad_seconds:.1f` s by AD,
one reverse pass through the whole batch, against
{docvalue}`albatross/robust.json:gradient_cost.fd_seconds:.1f` s by finite differences, even
with every perturbed cycle flown in the same vectorized call. With AD the
robust solve ran its {docvalue}`albatross/robust.json:designs.robust.calls:d` iterations, its
budget, in {docvalue}`albatross/robust.json:designs.robust.seconds:.0f*0.016666666666666666` min.
How the Jacobian is split matters as much:
{class}`~gemseo_tempo.uncertainty.robust.GroupedJAXDiscipline`
differentiates the cycle's three hundred constraint rows on the forecast
flight alone, in forward mode, eighty-two tangents rather than three
hundred cotangents, and only the statistic through the sample, in reverse
mode. Taken together, every constraint row would be back-propagated through
all {docvalue}`albatross/robust.json:design_sample:d` flights. One integrator serves both
groups, and the Direct adjoint is the better one for the pair: it lets the
constraints go forward, at a quarter of what reverse mode costs them, and
costs the statistic half as much again as the checkpointing adjoint would.
