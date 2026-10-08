# Uncertainty

```{variable-context} gemseo_tempo.problems.sail.model:REGISTRY
```

## Uncertainty propagation

:::{admonition} Takeaway: uncertainty propagation
:class: tip

- **Sampling:** vectorized with `vmap`: the largest saving of all, a
  hundredfold on the same draws.
- **Estimator:** first-order Taylor, two model calls, for the spread;
  gradient-enhanced chaos when the mean is wanted to a thousandth of a
  deviation.
- **Differentiation:** finite differences, batched into one call: with two
  uncertain inputs they are quicker than the reverse passes, and give the
  same answer.
:::

The wind on the day is not the forecast one: its strength
{var}`wind_scale <symbol>` is known to 12% and its direction
{var}`wind_veer <symbol>` to 10°, both normally distributed. The downwind
leg's optimal headings, sailed as planned for their
{docvalue}`sail/propagation.json:objective:.1f` s without correction, then end short of or
past the mark by
{docvalue}`sail/propagation.json:reference.standard_deviation:.0f` m along the course, one
deviation either way. Single shooting is used here, so that a frozen heading
schedule responds to the wind. Each way of estimating that spread, against
20 000 draws, where AD marks exact derivatives by reverse-mode automatic
differentiation and FD central
differences, every perturbed point in the same vectorized call, at the
step closest to the exact derivative at the mean:

| method | error, mean | error, deviation | model calls | time |
| --- | --- | --- | --- | --- |
| sampling, 256 draws | {docvalue}`sail/propagation.json:estimates.sampling_vmapped.mean_error:+.0e` σ | {docvalue}`sail/propagation.json:estimates.sampling_vmapped.deviation_error:+.0e` | {docvalue}`sail/propagation.json:estimates.sampling_vmapped.model_calls:d` | {docvalue}`sail/propagation.json:estimates.sampling_vmapped.seconds:.3f` s, looped {docvalue}`sail/propagation.json:estimates.sampling_looped.seconds:.2f` s |
| Taylor, 1st order, AD | {docvalue}`sail/propagation.json:estimates.taylor1_ad.mean_error:+.0e` σ | {docvalue}`sail/propagation.json:estimates.taylor1_ad.deviation_error:+.0e` | {docvalue}`sail/propagation.json:estimates.taylor1_ad.model_calls:d` | {docvalue}`sail/propagation.json:estimates.taylor1_ad.seconds:.3f` s |
| Taylor, 1st order, FD | {docvalue}`sail/propagation.json:estimates.taylor1_fd.mean_error:+.0e` σ | {docvalue}`sail/propagation.json:estimates.taylor1_fd.deviation_error:+.0e` | {docvalue}`sail/propagation.json:estimates.taylor1_fd.model_calls:d` | {docvalue}`sail/propagation.json:estimates.taylor1_fd.seconds:.3f` s |
| Taylor, 2nd order, AD | {docvalue}`sail/propagation.json:estimates.taylor2_ad.mean_error:+.0e` σ | {docvalue}`sail/propagation.json:estimates.taylor2_ad.deviation_error:+.0e` | {docvalue}`sail/propagation.json:estimates.taylor2_ad.model_calls:d` | {docvalue}`sail/propagation.json:estimates.taylor2_ad.seconds:.3f` s |
| Taylor, 2nd order, FD | {docvalue}`sail/propagation.json:estimates.taylor2_fd.mean_error:+.0e` σ | {docvalue}`sail/propagation.json:estimates.taylor2_fd.deviation_error:+.0e` | {docvalue}`sail/propagation.json:estimates.taylor2_fd.model_calls:d` | {docvalue}`sail/propagation.json:estimates.taylor2_fd.seconds:.3f` s |
| chaos, 12 points | {docvalue}`sail/propagation.json:estimates.chaos.mean_error:+.0e` σ | {docvalue}`sail/propagation.json:estimates.chaos.deviation_error:+.0e` | {docvalue}`sail/propagation.json:estimates.chaos.model_calls:d` | {docvalue}`sail/propagation.json:estimates.chaos.seconds:.3f` s |
| chaos + AD gradients | {docvalue}`sail/propagation.json:estimates.chaos_ad.mean_error:+.0e` σ | {docvalue}`sail/propagation.json:estimates.chaos_ad.deviation_error:+.0e` | {docvalue}`sail/propagation.json:estimates.chaos_ad.model_calls:d` | {docvalue}`sail/propagation.json:estimates.chaos_ad.seconds:.3f` s |
| chaos + FD gradients | {docvalue}`sail/propagation.json:estimates.chaos_fd.mean_error:+.0e` σ | {docvalue}`sail/propagation.json:estimates.chaos_fd.deviation_error:+.0e` | {docvalue}`sail/propagation.json:estimates.chaos_fd.model_calls:d` | {docvalue}`sail/propagation.json:estimates.chaos_fd.seconds:.3f` s |

The miss is close to linear in the two inputs, so first-order Taylor,
one evaluation and one gradient, already matches 20 000 draws to a hundredth
of a deviation; second order adds a little curvature to the mean and nothing
useful. Gradient-enhanced chaos is the most accurate of all, and it needs the
gradient at each of its twelve points: twelve reverse passes by AD, or 48
more evaluations by FD, for the same answer, and with two inputs the
differences, all in one vectorized call, are the quicker. The largest
saving is
vectorization: the 256 draws take
{docvalue}`sail/propagation.json:estimates.sampling_vmapped.seconds:.2f` s in
one `vmap` dispatch, against
{docvalue}`sail/propagation.json:estimates.sampling_looped.seconds:.0f` s in a loop.

## Optimization under uncertainty

:::{admonition} Takeaway: optimization under uncertainty
:class: tip

- **Formulation:** a chance constraint, P(inside) ≥ 95%, solved once,
  rather than a safety margin found by trial and error.
- **Sampling:** one fixed sample of {docvalue}`sail/margin.json:design_sample:d`
  draws, the probability smoothed so that it has a gradient, and the design
  checked on 4000 fresh ones.
- **Differentiation:** automatic, in the loop: one gradient of the
  probability with respect to every heading costs a reverse pass through the
  batch, a quarter of what finite differences cost even batched into one
  call.
:::

With the race area narrowed to
{docvalue}`sail/margin.json:half_width:.0f` m either side, the downwind leg's optimum
runs along its edge, and in the wind that actually blows it stays inside
only {docvalue}`sail/margin.json:designs.0.reliability:.0%` of the time. There are two
ways to fix that. A safety margin tightens the channel by a fixed amount and
solves as before. A chance constraint asks for the probability itself,
P(inside) ≥ 95% over {docvalue}`sail/margin.json:design_sample:d` fixed draws, smoothed
so that it has a gradient. Each design, checked on 4000 fresh draws:

| design | time | P(inside) | furthest across, as planned |
| --- | --- | --- | --- |
| deterministic | {docvalue}`sail/margin.json:designs.0.objective:.2f` s | {docvalue}`sail/margin.json:designs.0.reliability:.1%` | {docvalue}`sail/margin.json:designs.0.across:.0f` m |
| margin 25 m | {docvalue}`sail/margin.json:designs.1.objective:.2f` s | {docvalue}`sail/margin.json:designs.1.reliability:.1%` | {docvalue}`sail/margin.json:designs.1.across:.0f` m |
| margin 50 m | {docvalue}`sail/margin.json:designs.2.objective:.2f` s | {docvalue}`sail/margin.json:designs.2.reliability:.1%` | {docvalue}`sail/margin.json:designs.2.across:.0f` m |
| margin 75 m | {docvalue}`sail/margin.json:designs.3.objective:.2f` s | {docvalue}`sail/margin.json:designs.3.reliability:.1%` | {docvalue}`sail/margin.json:designs.3.across:.0f` m |
| P(inside) ≥ 95% | {docvalue}`sail/margin.json:designs.4.objective:.2f` s | {docvalue}`sail/margin.json:designs.4.reliability:.1%` | {docvalue}`sail/margin.json:designs.4.across:.0f` m |

A margin buys reliability with time, but which margin is right can only be
found by trying them: 25 m is not enough and 75 m is more than asked for.
The chance constraint reaches the reliability asked for in one solve, and
lands on the same design as the 50 m margin, which only hindsight could
have picked.

Its gradient is a derivative of a probability, with respect to all
{docvalue}`sail/margin.json:gradient_cost.n_design:d` design variables, through
{docvalue}`sail/margin.json:design_sample:d` sailed legs. By AD that is one reverse
pass through the batch, {docvalue}`sail/margin.json:gradient_cost.ad_seconds:.1f` s,
against {docvalue}`sail/margin.json:gradient_cost.fd_seconds:.1f` s by finite differences
even with every perturbed design sailed in the same vectorized call. The
trajectory's own rows, the channel at every save point, are taken
separately in forward mode, which the differentiation study on
{doc}`optimization` found the faster on the sail. The solve took {docvalue}`sail/margin.json:designs.4.calls:d` iterations,
{docvalue}`sail/margin.json:designs.4.seconds:.0f` s after
{docvalue}`sail/margin.json:designs.4.compile_seconds:.0f` s of compilation.

See {doc}`../../guides/uncertainty` for how `robust_discipline` builds the
chance constraint this design was found under.
