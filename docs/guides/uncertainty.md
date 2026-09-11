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

Everything here is reproducible from one script:

```bash
python examples/uncertainty/run_uncertainty.py --analysis speedup screen
python examples/uncertainty/run_uncertainty.py --analysis all
```

The examples are run by hand rather than executed when the documentation is
built. A gallery that ran them would add tens of minutes of solver time to
every push and every tag, and would fail the docs build whenever a solver
had an off day, which the numbers on this page show is a real possibility.
The measured figures are quoted here instead, and the ones that matter are
checked against the code by `tests/test_docs.py`.

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

`gemseo_tempo.uncertainty.estimators` estimates the mean and standard
deviation of one quantity several ways and scores them against a large
sample: sampling (vmapped and looped), first- and second-order Taylor, and
polynomial chaos with and without gradient enhancement. See the UQ
strategies comparison on {doc}`../problems/sail` for what that comparison
looks like on a real analysis, and the warning that follows it about what
an estimator that depends on derivatives depends on.

## Designing against a probability

`gemseo_tempo.uncertainty.robust` turns a statistic over a sample into a
constraint an ordinary optimizer can be handed:

```python
from gemseo_tempo.uncertainty.robust import robust_discipline
from gemseo_tempo.scenario.formulations import OptimalControl

discipline = robust_discipline(
    scenario, names, sample, "constraint.channel",
    reduction=jnp.max, target=0.95,
)
formulation = OptimalControl(
    scenario,
    disciplines=[discipline],          # replaces, does not extend
    extra_constraints=[("constraint.channel.shortfall", "ineq")],
)
```

Because the discipline is a JAX discipline like every other one here, the
optimizer receives the **exact derivative of a probability**, not a finite
difference of a noisy estimate of one.

Note that `robust_discipline` computes the trajectory *and* the statistics in
one program, and that `disciplines` replaces the scenario's own rather than
adding to them. Chaining a second discipline alongside the first is the
obvious thing to try and it fails silently: a JAX discipline filters its
Jacobian to the inputs and outputs GEMSEO declares to it, those declarations
do not reach a discipline inside a chain, and an empty filter returns zeros
rather than raising. Measured here, the objective and both terminal
constraints came back with gradient exactly **zero** while the added
constraint alone had one, and SLSQP stopped on its first iteration reporting
a singular LSQ matrix. Nothing said the derivatives had gone.

The cost is honest: every evaluation integrates the trajectory once per draw,
so a 256-draw sample makes each iteration 257 integrations. That is what
designing against a probability costs, and vectorizing is what makes it one
dispatch rather than 257.

Two further choices decide whether this works at all, and both are
deliberate:

- **The sample is fixed once** and reused at every iteration. With a fresh
  sample per iteration the statistic is noisy and the optimizer chases the
  noise. The price is that the design is optimal for that sample, so its
  reliability has to be checked on an independent one, which `assess` does.
- **The probability is smoothed.** An indicator has no useful gradient, so a
  logistic in the constraint value stands in for it. The width is a modelling
  choice, reported rather than buried: too wide and the constraint is not the
  one you asked for, too narrow and the gradient vanishes again. At the width
  used on {doc}`../problems/sail`'s margin design it agrees with the counted
  fraction to about a point.

## What a safety margin costs

The argument: a deterministic design that satisfies a constraint at one set
of parameter values can be a coin flip under an uncertain one, a margin is
the conventional repair and it is not monotone in what it buys, and putting
the probability directly in the problem can dominate the margin outright,
both faster and more reliable at once. See the main result on
{doc}`../problems/sail` for the numbers this plays out with.
