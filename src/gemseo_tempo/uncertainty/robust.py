# Copyright 2026 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Designing against a constraint that has to hold under uncertainty.

A deterministic design satisfies its constraints at one set of parameter
values and says nothing about any other. The conventional repair is a
safety margin: tighten the constraint by a fixed amount, solve as before,
and hope. The margin is chosen from experience rather than from the
uncertainty, so the design it produces is either over-conservative, buying
reliability with performance nobody asked to spend, or under-conservative,
and **it has no way of telling which**. That is the argument of the
satellite paper this follows, where accounting for the uncertainty instead
of padding against it recovered two thirds of the performance the margin
was costing.

The alternative here is to put the probability into the problem: minimize
the objective subject to the constraint holding with probability at least
``alpha``. What makes that affordable is the same thing that makes
everything else in this package affordable. The inner sample goes through
the dynamics in **one dispatch**, and the whole statistic is differentiable
with respect to the design, so the outer optimizer gets exact gradients of
a probability rather than finite differences of a noisy estimate.

Two details decide whether this works at all.

**The sample is fixed once and reused at every iteration.** Common random
numbers: with a fresh sample per iteration the statistic is a noisy
function of the design and the optimizer chases the noise rather than the
optimum. With one fixed sample it is a deterministic, smooth, and
differentiable function, and the estimate it converges to is the estimate
that sample supports. The price is that the design is optimal for *that*
sample, so the reliability it claims must be checked on an independent one,
which :func:`assess` does.

**The probability is smoothed.** An indicator function has zero gradient
almost everywhere and no gradient at all where it matters, so counting
failures gives the optimizer nothing to work with. A logistic in the
constraint value, sharp enough to be close to the indicator and smooth
enough to differentiate, gives the same estimate and a usable gradient. The
smoothing width is reported alongside the result rather than buried, since
it is a modelling choice: too wide and the constraint is not the one asked
for, too narrow and the gradient vanishes again.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy

from gemseo_tempo.uncertainty.batch import CONSTANT_PREFIX
from gemseo_tempo.uncertainty.batch import _quantity

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

MEAN = "mean"
"""The output name suffix of an estimated mean."""

DEVIATION = "std"
"""The output name suffix of an estimated standard deviation."""

MARGIN = "margin"
"""The output name suffix of a mean plus a multiple of a deviation."""

RELIABILITY = "reliability"
"""The output name suffix of an estimated probability of satisfaction."""

SHORTFALL = "shortfall"
"""The output name suffix of a reliability target minus the reliability.

The reliability constraint in the form an optimizer wants it, negative
when the target is met.
"""


def statistics_function(
    scenario,
    uncertain: Sequence[str],
    sample,
    quantity: str,
    reduction: Callable | None = None,
    factor: float = 2.0,
    smoothing: float = 1.0,
    target: float = 0.95,
    prefix: str = CONSTANT_PREFIX,
) -> Callable:
    """Build the statistics of one quantity as a function of the design.

    The returned function takes the design inputs the scenario already
    knows by name and returns five numbers: the mean of the quantity over
    the sample, its standard deviation, the margin ``mean + factor * std``,
    and the smoothed probability that the quantity is not positive. It is
    an ordinary JAX function, so it composes, jits and differentiates like
    any other.

    Parameters
    ----------
    scenario
        The dynamic scenario.
    uncertain
        The raw names of the uncertain inputs, in the order the sample
        columns are given in.
    sample
        The ``(n, len(uncertain))`` sample, fixed for the whole
        optimization.
    quantity
        The output whose statistics are wanted; a constraint, so that
        "not positive" means "satisfied".
    reduction
        How to reduce the quantity to a scalar when it is a path
        constraint with one value per node; the worst violation along the
        trajectory is the usual choice.
    factor
        The multiple of the standard deviation in the margin.
    smoothing
        The width of the logistic used for the probability, in the units
        of the quantity. It has to be small against the spread of the
        quantity and large against nothing in particular; the result
        carries it so that a reader can see what was assumed.
    target
        The reliability the shortfall is measured against, so that the
        reliability requirement reaches an optimizer in the form it wants:
        one number, negative when the requirement is met.
    prefix
        The grammar prefix the uncertain names carry as inputs.

    Returns
    -------
    Callable
        ``design -> dict`` of the five statistics.
    """
    problem = scenario.problem
    defaults = dict(problem.input_defaults)
    names = [prefix + name for name in uncertain]
    sample = jnp.asarray(numpy.asarray(sample, dtype=float))
    reductions = {quantity: reduction} if reduction is not None else None

    def one(design: dict, row):
        """Evaluate the quantity for one draw at a given design."""
        inputs = dict(defaults)
        inputs.update(design)
        for index, name in enumerate(names):
            inputs[name] = jnp.reshape(row[index], jnp.shape(defaults[name]))
        return _quantity(problem.jax_function(inputs), quantity, reductions)

    mapped = jax.vmap(one, in_axes=(None, 0))

    def statistics(design: dict) -> dict:
        """Return the statistics of the quantity over the fixed sample."""
        values = mapped(design, sample)
        mean = jnp.mean(values)
        deviation = jnp.std(values)
        # The logistic stands in for the indicator of "not positive".
        # Sharp enough to be nearly it, smooth enough to differentiate.
        reliability = jnp.mean(jax.nn.sigmoid(-values / smoothing))
        return {
            MEAN: mean,
            DEVIATION: deviation,
            MARGIN: mean + factor * deviation,
            RELIABILITY: reliability,
            SHORTFALL: target - reliability,
        }

    return statistics


def statistics_discipline(
    scenario,
    uncertain: Sequence[str],
    sample,
    quantity: str,
    design_names: Sequence[str] | None = None,
    reduction: Callable | None = None,
    factor: float = 2.0,
    smoothing: float = 1.0,
    target: float = 0.95,
    name: str = "uncertainty",
    prefix: str = CONSTANT_PREFIX,
):
    """Wrap the statistics as a GEMSEO discipline over the design variables.

    This is what turns a probability into a constraint an ordinary
    optimizer can be handed. Chained after the scenario's own discipline it
    sees the same design variables, and being a JAX discipline like every
    other one here its Jacobian is exact: the optimizer gets the derivative
    of a probability, not a finite difference of a noisy estimate of one.

    Parameters
    ----------
    scenario
        The dynamic scenario.
    uncertain
        The raw names of the uncertain inputs.
    sample
        The fixed sample the statistics are taken over.
    quantity
        The constraint output, satisfied when not positive.
    design_names
        The inputs to expose; the scenario's design variables when
        omitted.
    reduction
        How to reduce a path constraint to a scalar.
    factor
        The multiple of the standard deviation in the margin.
    smoothing
        The width of the logistic standing in for the indicator.
    target
        The reliability the shortfall output is measured against.
    name
        The discipline name.
    prefix
        The grammar prefix of the uncertain inputs.

    Returns
    -------
    JAXDiscipline
        The discipline, one output per statistic, each named
        ``"<quantity>.<statistic>"``.
    """
    from gemseo_jax.jax_discipline import JAXDiscipline

    statistics = statistics_function(
        scenario,
        uncertain,
        sample,
        quantity,
        reduction=reduction,
        factor=factor,
        smoothing=smoothing,
        target=target,
        prefix=prefix,
    )
    defaults = scenario.problem.input_defaults
    names = list(design_names or scenario.problem.design_variables)

    def function(inputs: dict) -> dict:
        """Return the statistics, named as the discipline declares them."""
        values = statistics({key: inputs[key] for key in names})
        return {
            f"{quantity}.{statistic}": jnp.atleast_1d(value)
            for statistic, value in values.items()
        }

    return JAXDiscipline(
        function=function,
        input_names=names,
        output_names=[
            f"{quantity}.{statistic}"
            for statistic in (MEAN, DEVIATION, MARGIN, RELIABILITY, SHORTFALL)
        ],
        default_inputs={key: defaults[key] for key in names},
        differentiation_method=JAXDiscipline.DifferentiationMethod.REVERSE,
        name=name,
    )


@dataclass
class Assessment:
    """What a design actually achieves, measured on a fresh sample."""

    name: str
    """The design."""

    objective: float
    """The objective it attained."""

    reliability: float
    """The fraction of an independent sample in which the constraint held.

    Counted, not smoothed. The smoothed probability is what the optimizer
    saw; this is what the design gets.
    """

    worst: float
    """The largest constraint value seen, positive when it was violated."""

    quantile: float
    """The 95th percentile of the constraint value."""

    n_samples: int
    """The size of the assessment sample."""

    extra: dict = field(default_factory=dict)
    """Anything the caller wants carried alongside."""

    @property
    def feasible_fraction(self) -> float:
        """Return the reliability, named as the paper names it.

        Returns
        -------
        float
            The fraction of draws in which the constraint held.
        """
        return self.reliability


def assess(
    scenario,
    uncertain: Sequence[str],
    sample,
    quantity: str,
    objective: float,
    name: str,
    reduction: Callable | None = None,
    prefix: str = CONSTANT_PREFIX,
) -> Assessment:
    """Measure what a frozen design achieves on a sample it did not see.

    The design must already be frozen into the scenario's input defaults;
    see :func:`~gemseo_tempo.uncertainty.batch.freeze_design`. The sample
    should be independent of the one the design was optimized against,
    which is the whole point: a design optimized on a sample is optimal for
    that sample, and the reliability it claims there is not evidence.

    Parameters
    ----------
    scenario
        The scenario, frozen at the design to assess.
    uncertain
        The raw names of the uncertain inputs.
    sample
        The assessment sample.
    quantity
        The constraint output, satisfied when not positive.
    objective
        The objective the design attained, carried into the result.
    name
        The name of the design.
    reduction
        How to reduce a path constraint to a scalar.
    prefix
        The grammar prefix.

    Returns
    -------
    Assessment
        What the design achieves.
    """
    statistics = statistics_function(
        scenario, uncertain, sample, quantity, reduction=reduction, prefix=prefix
    )
    problem = scenario.problem
    defaults = dict(problem.input_defaults)
    names = [prefix + one for one in uncertain]
    reductions = {quantity: reduction} if reduction is not None else None

    def value(row):
        """Evaluate the constraint for one draw."""
        inputs = dict(defaults)
        for index, key in enumerate(names):
            inputs[key] = jnp.reshape(row[index], jnp.shape(defaults[key]))
        return _quantity(problem.jax_function(inputs), quantity, reductions)

    values = numpy.asarray(
        jax.jit(jax.vmap(value))(jnp.asarray(numpy.asarray(sample, dtype=float)))
    )
    smoothed = statistics({})
    return Assessment(
        name=name,
        objective=float(objective),
        reliability=float(numpy.mean(values <= 0.0)),
        worst=float(numpy.max(values)),
        quantile=float(numpy.percentile(values, 95.0)),
        n_samples=int(len(values)),
        extra={"smoothed_reliability": float(smoothed[RELIABILITY])},
    )


def format_assessments(assessments: Sequence[Assessment]) -> str:
    """Render assessments as a table.

    Parameters
    ----------
    assessments
        The assessments, in the order to show them.

    Returns
    -------
    str
        One row per design.
    """
    lines = [
        "| design | objective | reliability | 95th pct | worst |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {assessment.name} | {assessment.objective:.2f}"
        f" | {assessment.reliability:.1%} | {assessment.quantile:+.1f}"
        f" | {assessment.worst:+.1f} |"
        for assessment in assessments
    )
    return "\n".join(lines)


__all__ = [
    "DEVIATION",
    "MARGIN",
    "MEAN",
    "RELIABILITY",
    "SHORTFALL",
    "Assessment",
    "assess",
    "format_assessments",
    "statistics_discipline",
    "statistics_function",
]
