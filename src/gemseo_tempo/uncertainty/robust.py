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
from gemseo_jax.jax_discipline import JAXDiscipline

from gemseo_tempo.dynamics.integrators import AdjointMethod
from gemseo_tempo.transcriptions.collocation import HermiteSimpson
from gemseo_tempo.uncertainty.batch import CONSTANT_PREFIX
from gemseo_tempo.uncertainty.batch import _quantity

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
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


def _batched_differences(subset, step: float):
    """Return the Jacobian of ``subset`` by forward differences, vectorized.

    Parameters
    ----------
    subset
        ``(differentiated, fixed) -> {output: value}``.
    step
        The step, relative to each input.

    Returns
    -------
    Callable
        ``(differentiated, fixed) -> {output: {input: block}}``, every
        perturbed point evaluated in one ``vmap``.
    """
    from jax.flatten_util import ravel_pytree

    def jacobian(differentiated, fixed):
        flat, unravel = ravel_pytree(differentiated)
        outputs = subset(differentiated, fixed)
        names = list(outputs)

        def flat_outputs(x):
            values = subset(unravel(x), fixed)
            return jnp.concatenate([jnp.ravel(values[name]) for name in names])

        steps = step * jnp.maximum(1.0, jnp.abs(flat))
        points = jnp.vstack([flat, flat + jnp.diag(steps)])
        values = jax.vmap(flat_outputs)(points)
        matrix = ((values[1:] - values[0]) / steps[:, None]).T
        blocks, row = {}, 0
        for name in names:
            count = jnp.size(outputs[name])
            column, blocks[name] = 0, {}
            # The order ravel_pytree flattens a dictionary in.
            for key in sorted(differentiated):
                width = jnp.size(differentiated[key])
                blocks[name][key] = matrix[row : row + count, column : column + width]
                column += width
            row += count
        return blocks

    return jacobian


class GroupedJAXDiscipline(JAXDiscipline):
    """A JAX discipline whose Jacobian is taken one group of outputs at a time.

    Reverse mode costs a pass per output row, and each pass runs back
    through everything the outputs are computed from together. A robust
    discipline returns the transcription's constraints, computed on the
    nominal trajectory, beside statistics computed over a whole sample;
    differentiated as one, every constraint row is back-propagated through
    every draw although none depends on them, and an iteration costs as
    many passes through the sample as there are constraint rows. Taken
    group by group, each compiled transform keeps only what its outputs
    need, so the constraints are differentiated on the one trajectory and
    the handful of statistics through the sample.

    Each group also takes its own mode. ``"auto"`` chooses forward mode when
    the group has fewer inputs than outputs and the integration allows it,
    reverse mode otherwise, which is what makes it cheap: a path constraint
    checked at every save point is many rows against few controls, and a
    statistic is one row against all of them. ``"fd"`` takes forward
    differences instead, every perturbed point in one vectorized call,
    which is the best finite differences can do and the fair comparison.
    """

    MODES = ("auto", "forward", "reverse", "fd")
    """The ways a group can be differentiated."""

    def __init__(
        self,
        *args,
        groups: Sequence[Sequence[str]],
        modes: Sequence[str] | None = None,
        forward_allowed: bool = True,
        fd_step: float = 1e-7,
        **kwargs,
    ) -> None:
        """Store the output groups and how each is differentiated.

        Parameters
        ----------
        *args
            The positional arguments of
            :class:`~gemseo_jax.jax_discipline.JAXDiscipline`.
        groups
            The output names, partitioned into the groups differentiated
            together.
        modes
            The mode of each group, one of :attr:`MODES`; ``"auto"`` for
            every group when omitted.
        forward_allowed
            Whether forward mode can differentiate the function at all, which
            an integration through a reverse-only adjoint forbids.
        fd_step
            The finite-difference step, relative to each input.
        **kwargs
            The keyword arguments of the base discipline.

        Raises
        ------
        ValueError
            If a mode is unknown, or forward mode is asked for where it is
            not allowed.
        """
        self._groups = [tuple(group) for group in groups]
        modes = tuple(modes or ("auto",) * len(self._groups))
        unknown = sorted(set(modes) - set(self.MODES))
        if unknown or len(modes) != len(self._groups):
            msg = f"Expected one of {list(self.MODES)} per group, got {list(modes)}."
            raise ValueError(msg)
        if "forward" in modes and not forward_allowed:
            msg = "Forward mode is not allowed through this integration."
            raise ValueError(msg)
        self._modes = modes
        self._forward_allowed = forward_allowed
        self._fd_step = fd_step
        self._compiled_groups: dict[tuple, Callable] = {}
        super().__init__(*args, **kwargs)

    def _mode(self, mode: str, outputs: tuple, inputs: tuple, data: dict) -> str:
        """Resolve ``"auto"`` into the cheaper exact mode for a group's shape."""
        if mode != "auto":
            return mode
        n_inputs = sum(numpy.size(data[name]) for name in inputs)
        n_outputs = sum(numpy.size(self.io.data[name]) for name in outputs)
        return (
            "forward" if self._forward_allowed and n_inputs < n_outputs else "reverse"
        )

    def _group_jacobian(
        self, outputs: tuple, inputs: tuple, data: dict, mode: str = "reverse"
    ) -> dict:
        """Differentiate one group of outputs.

        Parameters
        ----------
        outputs
            The outputs of the group.
        inputs
            The inputs to differentiate with respect to.
        data
            The full input data.
        mode
            ``"forward"``, ``"reverse"`` or ``"fd"``.

        Returns
        -------
        dict
            ``{output: {input: array}}``.
        """
        key = (outputs, inputs, mode)
        compiled = self._compiled_groups.get(key)
        if compiled is None:

            def subset(differentiated, fixed):
                values = self.jax_out_func({**fixed, **differentiated})
                return {name: values[name] for name in outputs}

            if mode == "fd":
                transform = _batched_differences(subset, self._fd_step)
            else:
                transform = (
                    jax.jacfwd(subset) if mode == "forward" else jax.jacrev(subset)
                )
            compiled = jax.jit(transform)
            self._compiled_groups[key] = compiled
        return compiled({name: data[name] for name in inputs}, data)

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        """Assemble the Jacobian group by group."""
        data = {name: self.io.data[name] for name in self.input_grammar.names}
        inputs = tuple(self._differentiated_input_names)
        wanted = set(self._differentiated_output_names)
        jacobian = {}
        for group, mode in zip(self._groups, self._modes, strict=True):
            outputs = tuple(name for name in group if name in wanted)
            if outputs:
                mode = self._mode(mode, outputs, inputs, data)
                jacobian.update(self._group_jacobian(outputs, inputs, data, mode))
        sizes = {name: numpy.asarray(value).size for name, value in data.items()}
        self.jac = {
            output: {
                name: (
                    numpy.atleast_2d(numpy.asarray(block)).reshape((-1, 1))
                    if sizes[name] == 1
                    else numpy.atleast_2d(numpy.asarray(block))
                )
                for name, block in inner.items()
            }
            for output, inner in jacobian.items()
        }
        self._has_jacobian = True


def robust_discipline(
    scenario,
    uncertain: Sequence[str],
    sample,
    quantity: str,
    reduction: Callable | None = None,
    factor: float = 2.0,
    smoothing: float = 1.0,
    target: float = 0.95,
    name: str = "robust_scenario",
    prefix: str = CONSTANT_PREFIX,
    modes: Sequence[str] = ("auto", "auto"),
    fd_step: float = 1e-7,
):
    """Wrap the transcription and the statistics as **one** JAX discipline.

    This is what turns a probability into a constraint an ordinary
    optimizer can be handed. Its Jacobian is exact, so the optimizer gets
    the derivative of a probability rather than a finite difference of a
    noisy estimate of one.

    One discipline, not two, and that is not tidiness. Chaining a second
    discipline alongside the scenario's own leaves the first one
    undifferentiated: a JAX discipline filters its Jacobian down to the
    inputs and outputs GEMSEO declares to it, those declarations do not
    reach a discipline inside a chain, and an empty filter returns zeros
    rather than raising. Measured on the sailing problem, the objective and
    both terminal constraints came back with gradient exactly zero while
    the added constraint alone had a gradient, and SLSQP stopped on its
    first iteration reporting a singular LSQ matrix. Nothing anywhere said
    the derivatives had gone. Computing both in the same jitted program
    avoids the seam entirely, and these problems are one jitted program
    each to begin with.

    The cost is honest and unavoidable: every evaluation of the objective
    now integrates the trajectory once per draw in the sample, so a
    256-draw sample makes each iteration 257 integrations. That is what
    designing against a probability costs, and vectorizing is what makes it
    one dispatch rather than 257. Its derivative costs a few reverse passes
    through the sample, one per statistic, because the Jacobian is taken
    group by group; see :class:`GroupedJAXDiscipline`.

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
    modes
        How the transcription's outputs and the statistics are each
        differentiated; see :class:`GroupedJAXDiscipline`. Forward mode is
        allowed when the transcription does not integrate, or integrates
        through the Direct adjoint.
    fd_step
        The finite-difference step of a group differentiated by ``"fd"``.

    Returns
    -------
    JAXDiscipline
        One discipline producing every transcription output and, in
        addition, ``"<quantity>.<statistic>"`` for each statistic.
    """
    problem = scenario.problem
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
    design = list(problem.design_variables)
    statistic_names = (MEAN, DEVIATION, MARGIN, RELIABILITY, SHORTFALL)

    def function(inputs: dict) -> dict:
        """Return the transcription outputs and the statistics together."""
        outputs = dict(problem.jax_function(inputs))
        values = statistics({key: inputs[key] for key in design})
        for statistic in statistic_names:
            outputs[f"{quantity}.{statistic}"] = jnp.atleast_1d(values[statistic])
        return outputs

    statistic_outputs = [f"{quantity}.{statistic}" for statistic in statistic_names]
    method = JAXDiscipline.DifferentiationMethod(scenario.differentiation_method)
    integrator = scenario.definition["integrator"]
    forward_allowed = (
        isinstance(scenario.transcription, HermiteSimpson)
        or getattr(integrator, "adjoint_method", None) == AdjointMethod.DIRECT
    )
    return GroupedJAXDiscipline(
        function=function,
        input_names=list(problem.input_defaults),
        output_names=[*problem.output_names, *statistic_outputs],
        default_inputs=problem.input_defaults,
        differentiation_method=method,
        name=name,
        groups=(problem.output_names, statistic_outputs),
        modes=modes,
        forward_allowed=forward_allowed,
        fd_step=fd_step,
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
    "robust_discipline",
    "statistics_function",
]
