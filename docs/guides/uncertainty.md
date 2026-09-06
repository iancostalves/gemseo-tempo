# Uncertainty

A trajectory optimized for one set of parameters is a plan made for a forecast.
This guide is about what happens when the forecast is wrong: which parameters
matter, how far the answer moves, how cheaply that can be established, and how
to design against it rather than pad against it.

The thread running through all of it is that these disciplines are not black
boxes. They are jitted JAX programs, so a whole sample of parameters goes
through the dynamics in **one dispatch** instead of one at a time, and every
derivative with respect to those parameters comes out of the same pass. Methods
that the literature describes as conditionally affordable are unconditionally
affordable here, and the interesting question stops being which method you can
pay for and becomes which method is right.

## Declaring an uncertainty

An uncertain variable is declared where the variable is, in the problem's
registry, by giving it a distribution:

```python
Variable(
    name="wind_scale",
    full_name="Wind strength factor",
    distribution={"name": "OTNormalDistribution", "mu": 1.0, "sigma": 0.12},
    tags=frozenset({"constant"}),
)
```

`make_parameter_space` turns those declarations into a GEMSEO `ParameterSpace`,
and **refuses** a variable that has none. That refusal is the point: a default
would be a guess wearing a distribution's clothes, and sampling it would produce
numbers that look like uncertainty quantification without being it.

```python
from gemseo_tempo.uncertainty import make_parameter_space, uncertain_names
from gemseo_tempo.problems.sail.model import REGISTRY

names = uncertain_names(REGISTRY)          # ['wind_scale', 'wind_veer']
space = make_parameter_space(REGISTRY, names)
```

A parameter can only be made uncertain if the dynamics take it as a parameter.
The sail's wind used to be closed over at build time, which is why it grew
`wind_scale` and `wind_veer`: the shape of the breeze over a course is a feature
of the course and is known, while how hard it is blowing on the day is not.

## Freezing the design first

Propagating uncertainty through a design means through **a** design, and which
one matters more than it looks:

```python
from gemseo_tempo.uncertainty import freeze_design, nominal_values

gemseo_scenario.execute(algo_name="SLSQP", max_iter=300)
freeze_design(scenario, gemseo_scenario, nominal_values(REGISTRY, names))
```

Frozen at the starting guess instead, the analysis measures the sensitivity of a
trajectory nobody would fly, and the Sobol indices that come out are not merely
different but ill-formed, with total indices below first-order ones. That is how
the mistake was found, and it is why `freeze_design` exists as an explicit step
rather than something implied.

## A whole sample at once

```python
from gemseo_tempo.uncertainty import batch_evaluator, measure_speedup

evaluate = batch_evaluator(scenario, names, ["constraint.final.x"])
values = evaluate(sample)          # (n, 1), one dispatch
```

`measure_speedup` runs the same evaluations through the ordinary looped path and
reports the ratio, so the claim is measured rather than asserted. On 512
brachistochrone draws it is **56x**, with the two paths agreeing to 3e-15.

The saving is not the arithmetic, which is identical, but everything around it:
one dispatch instead of five hundred. So it is largest where the per-evaluation
work is smallest, and on the sail, whose right-hand side interpolates a wind
field, it is nearer 9x on the same measurement. Both are real; neither
generalizes to the other.

Two things about that measurement were wrong before they were right, and both
are worth knowing:

- A jitted function specializes on the **shape** of its argument. Warming up on
  two rows and timing five hundred measures a recompilation and reports it as
  the cost of vectorizing. The first measurement said 0.5x for this reason.
- Under adaptive stepping the batched and looped paths agree only to about 1e-5
  on the sail, because a vmapped adaptive controller steps the whole batch in
  lockstep. With a fixed step they agree to 1e-12. If exact agreement matters,
  fix the step.

## Screening before measuring

A Sobol budget spent on a variable that turns out not to matter is a budget
wasted, so rank first and decompose second:

```python
from gemseo_tempo.uncertainty.sensitivity import screen, sobol_indices

ranking = screen(scenario, REGISTRY, names, "constraint.final.y").ranking
result = sobol_indices(scenario, REGISTRY, names, "constraint.final.y")
```

On the brachistochrone, Morris ranks drag above friction above gravity for 80
evaluations. Sobol then confirms exactly that ordering, 0.65 and 0.34 and 0.02,
for 20480 evaluations that take 434 ms because they go through together. The
cheap ranking was right, for 0.4% of the cost. Gravity is in the list on purpose:
it is known to three decimal places, and a screening ought to be able to say so.

## Six ways to the same number

`gemseo_tempo.uncertainty.estimators` estimates the mean and standard deviation
of one quantity several ways and scores them against a large sample. On the sail
downwind leg, with the along-track arrival error as the quantity and the wind
strength and veer as the uncertainties, against 50 000 samples taking 49 s:

| estimator | model calls | seconds | mean error | std error |
| --- | --- | --- | --- | --- |
| sampling, vmapped | 512 | 0.53 | +0.9% | -2.1% |
| sampling, looped | 512 | 22.57 | +0.9% | -2.1% |
| Taylor, first order | 2 | 0.29 | +2.0% | +1.0% |
| Taylor, second order | 6 | | -3.6% | +1.5% |
| chaos, 32 points | 32 | 0.07 | +0.5% | -0.3% |
| chaos + gradients, 32 | 64 | 0.68 | +0.6% | -0.3% |
| chaos, 12 points | 12 | 0.06 | -2.5% | +0.1% |
| chaos + gradients, 12 | 24 | 0.43 | -1.5% | -0.4% |

Errors are relative to the reference **spread**, the mean included. A converged
design misses its target by little on average, so the reference mean is near
zero, and dividing an error by it turns a small absolute error into an enormous
relative one that says nothing about the estimator.

What the table says:

- The two sampling rows are one estimator on identical points, differing only in
  dispatch. The 43x between them is the whole claim of the batch evaluator,
  measured on a real analysis rather than a synthetic loop.
- **First-order Taylor gets the spread to 1.0% for two model calls**, beating
  512 samples. This is the CSMA paper's remark holding exactly: the cost is
  about twice a deterministic evaluation when the derivatives are exact, and
  here they are, by construction.
- Second order is *worse* than first on the mean here. The curvature correction
  overshoots. It is reported rather than dressed up.
- Chaos on 12 points costs 0.06 s and gets the spread to 0.1%. Gradient
  enhancement earns its keep where points are scarce, halving the error of the
  mean at 12 points, and adds nothing at 32.

A warning drawn from getting this wrong first. With a subtly incorrect Jacobian,
first-order Taylor came out 17% wrong on the spread, and that looked exactly like
the expected failure of a first-order method on a nonlinear response. It was not.
See {doc}`../problems/sail` for the adjoint fault behind it. An estimator that
depends on derivatives is only as good as the derivatives, and a plausible-looking
failure is the most expensive kind.

## Designing against a probability

`gemseo_tempo.uncertainty.robust` turns a statistic over a sample into a
constraint an ordinary optimizer can be handed:

```python
from gemseo_tempo.uncertainty.robust import statistics_discipline
from gemseo_tempo.scenario.formulations import OptimalControl

discipline = statistics_discipline(
    scenario, names, sample, "constraint.channel",
    reduction=jnp.max, target=0.95,
)
formulation = OptimalControl(
    scenario,
    extra_disciplines=[discipline],
    extra_constraints=[("constraint.channel.shortfall", "ineq")],
)
```

Because the discipline is a JAX discipline like every other one here, the
optimizer receives the **exact derivative of a probability**, not a finite
difference of a noisy estimate of one.

Two choices decide whether this works at all, and both are deliberate:

- **The sample is fixed once** and reused at every iteration. With a fresh
  sample per iteration the statistic is noisy and the optimizer chases the
  noise. The price is that the design is optimal for that sample, so its
  reliability has to be checked on an independent one, which `assess` does.
- **The probability is smoothed.** An indicator has no useful gradient, so a
  logistic in the constraint value stands in for it. The width is a modelling
  choice, reported rather than buried: too wide and the constraint is not the
  one you asked for, too narrow and the gradient vanishes again. On the sail it
  agrees with the counted fraction to 3e-4.
