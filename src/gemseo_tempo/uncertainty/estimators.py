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
"""Estimating the same statistic several ways, and what each one costs.

Every estimator here answers one question, the mean and standard deviation
of a quantity at a fixed design under uncertain parameters, and they are
scored against each other on cost and on error against a large sample.

The comparison is worth making because the methods differ by orders of
magnitude in cost and by very little in what they claim. The CSMA paper
sets out the taxonomy and, more usefully, the condition under which each is
affordable: first-order propagation costs about two evaluations *when the
derivatives with respect to the uncertain variables are exact*, and by
finite differences it costs one per uncertain variable instead. The same
holds for gradient-enhanced chaos, which wants those derivatives in its
training basis. Here they come out of the same pass by automatic
differentiation, so the condition holds by construction and the question
becomes which method is accurate rather than which is affordable.

The estimators take a :class:`Propagation` rather than a scenario, so the
same code runs on an analytic function whose answer is known. That is not
tidiness: the first surprising indices this library produced came from a
correct estimator on a wrongly frozen design, and telling those apart needs
a reference problem.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import numpy
import openturns

from gemseo_tempo.uncertainty.batch import batch_evaluator
from gemseo_tempo.uncertainty.batch import batch_jacobian
from gemseo_tempo.uncertainty.spaces import make_parameter_space

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from gemseo.algos.parameter_space import ParameterSpace


@dataclass
class Propagation:
    """A scalar quantity, its uncertain inputs, and how to evaluate them.

    Everything an estimator needs, and nothing about where it came from.
    """

    evaluate: Callable
    """``(n, d) -> (n,)``, the quantity over a whole sample at once."""

    space: ParameterSpace
    """The uncertain inputs, in the order the sample columns are given in."""

    names: tuple[str, ...]
    """The uncertain variable names, in that same order."""

    jacobian: Callable | None = None
    """``(n, d) -> (n, d)``, the derivatives, when they are available."""

    @property
    def dimension(self) -> int:
        """Return the number of uncertain variables.

        Returns
        -------
        int
            The dimension.
        """
        return len(self.names)

    @property
    def distribution(self) -> openturns.Distribution:
        """Return the joint distribution, with independent margins.

        Returns
        -------
        openturns.Distribution
            The distribution.
        """
        return openturns.JointDistribution(
            [self.space.distributions[name].distribution for name in self.names]
        )

    @property
    def mean_point(self) -> numpy.ndarray:
        """Return the mean of every input, as a one-row sample.

        Returns
        -------
        numpy.ndarray
            The point a Taylor expansion is taken about.
        """
        return numpy.array(
            [[float(self.space.distributions[name].mean[0]) for name in self.names]]
        )

    @property
    def standard_deviations(self) -> numpy.ndarray:
        """Return the standard deviation of every input.

        Returns
        -------
        numpy.ndarray
            The deviations, in input order.
        """
        return numpy.array(
            [
                float(self.space.distributions[name].standard_deviation[0])
                for name in self.names
            ]
        )


def from_scenario(
    scenario,
    registry,
    names: Sequence[str],
    quantity: str,
    differentiable: bool = True,
) -> Propagation:
    """Build a propagation from a frozen dynamic scenario.

    Parameters
    ----------
    scenario
        The scenario, frozen at the design the uncertainty is propagated
        through; see :func:`~gemseo_tempo.uncertainty.batch.freeze_design`.
    registry
        The variable metadata carrying the distributions.
    names
        The uncertain variable names.
    quantity
        The output to propagate to.
    differentiable
        Whether to attach the derivatives, which the Taylor and
        gradient-enhanced routes need.

    Returns
    -------
    Propagation
        The propagation.
    """
    names = tuple(names)
    evaluate = batch_evaluator(scenario, names, [quantity])
    derivatives = batch_jacobian(scenario, names, [quantity])

    def values(sample):
        """Evaluate the quantity over a sample."""
        return numpy.asarray(evaluate(sample)).ravel()

    def gradients(sample):
        """Differentiate the quantity over a sample."""
        return numpy.asarray(derivatives(sample))[:, 0, :]

    return Propagation(
        evaluate=values,
        space=make_parameter_space(registry, names),
        names=names,
        jacobian=gradients if differentiable else None,
    )


def _warm_up(propagation: Propagation, sample) -> None:
    """Compile before the clock starts, on the shape that will be timed.

    Without this the seconds column compares compilation against
    arithmetic, and reports a two-call estimator as the slowest of the
    six. The shape matters as much as the call: a jitted function
    specializes on it, so warming up on one row and timing five hundred
    measures a recompilation.

    Parameters
    ----------
    propagation
        The propagation.
    sample
        The sample the estimator is about to be timed on.
    """
    propagation.evaluate(sample)
    if propagation.jacobian is not None:
        propagation.jacobian(sample)


@dataclass
class Estimate:
    """What one estimator said, and what it cost to say it."""

    name: str
    """The estimator."""

    mean: float
    """The estimated mean of the quantity."""

    standard_deviation: float
    """The estimated standard deviation."""

    n_evaluations: int
    """Evaluations of the quantity."""

    n_gradients: int = 0
    """Evaluations of its gradient, each one reverse pass."""

    seconds: float = 0.0
    """Wall-clock time, excluding compilation."""

    extra: dict = field(default_factory=dict)
    """Anything the estimator wants carried alongside."""

    @property
    def cost(self) -> int:
        """Return the total number of model calls.

        A gradient counts as one call rather than as one per uncertain
        variable, which is the whole point: by automatic differentiation it
        is one reverse pass, and by finite differences it would not be.

        Returns
        -------
        int
            The calls.
        """
        return self.n_evaluations + self.n_gradients


def monte_carlo(
    propagation: Propagation,
    n_samples: int = 2048,
    seed: int = 0,
    evaluate: Callable | None = None,
    name: str = "sampling",
) -> Estimate:
    """Estimate the statistics by sampling.

    Parameters
    ----------
    propagation
        The propagation.
    n_samples
        The sample size.
    seed
        The sampling seed.
    evaluate
        An evaluator to use instead of the propagation's own. This is what
        makes the looped and vectorized arms of the comparison the same
        estimator on the same points, differing only in how they dispatch.
    name
        The name to record.

    Returns
    -------
    Estimate
        The mean, the standard deviation, and the cost.
    """
    openturns.RandomGenerator.SetSeed(seed)
    sample = numpy.asarray(propagation.distribution.getSample(n_samples))
    call = evaluate or propagation.evaluate
    call(sample)
    started = time.perf_counter()
    values = numpy.asarray(call(sample)).ravel()
    seconds = time.perf_counter() - started
    return Estimate(
        name=name,
        mean=float(values.mean()),
        standard_deviation=float(values.std(ddof=1)),
        n_evaluations=n_samples,
        seconds=seconds,
    )


def taylor(propagation: Propagation, second_order: bool = False) -> Estimate:
    """Estimate the statistics by expanding about the mean.

    First order takes the mean of the quantity to be its value at the mean
    of the inputs, and its variance to be ``sum (df/du_i * sigma_i)^2``. It
    is the cheapest thing that is not a guess, and it is exactly as good as
    the response is straight. Note what it cannot do: the mean it reports
    has no contribution from the spread at all, so a response with any
    curvature has its mean estimated by a method that cannot see curvature.

    Second order adds ``0.5 tr(H C)`` to the mean and ``0.5 tr(H C H C)``
    to the variance, which is where the curvature enters.

    Parameters
    ----------
    propagation
        The propagation, which must carry its derivatives.
    second_order
        Whether to add the curvature terms.

    Returns
    -------
    Estimate
        The mean, the standard deviation, and the cost.

    Raises
    ------
    ValueError
        If the propagation carries no derivatives.
    """
    if propagation.jacobian is None:
        msg = (
            "A Taylor expansion needs the derivatives; build the"
            " propagation with differentiable=True."
        )
        raise ValueError(msg)

    point = propagation.mean_point
    deviations = propagation.standard_deviations
    _warm_up(propagation, point)
    offsets = _hessian_points(point, deviations) if second_order else None
    if offsets is not None:
        # A second shape, so a second compilation, and it belongs outside
        # the clock like the first one.
        propagation.jacobian(offsets)

    started = time.perf_counter()
    value = float(numpy.asarray(propagation.evaluate(point)).ravel()[0])
    gradient = numpy.asarray(propagation.jacobian(point))[0]
    variance = float(numpy.sum((gradient * deviations) ** 2))
    mean = value
    n_gradients = 1

    if offsets is not None:
        hessian = _hessian(propagation, offsets, deviations)
        product = hessian @ numpy.diag(deviations**2)
        mean = value + 0.5 * float(numpy.trace(product))
        variance += 0.5 * float(numpy.trace(product @ product))
        n_gradients += 2 * propagation.dimension

    seconds = time.perf_counter() - started
    return Estimate(
        name="taylor 2" if second_order else "taylor 1",
        mean=mean,
        standard_deviation=float(numpy.sqrt(max(variance, 0.0))),
        n_evaluations=1,
        n_gradients=n_gradients,
        seconds=seconds,
    )


def _hessian_points(point, deviations) -> numpy.ndarray:
    """Return the points at which the gradient is differenced.

    Parameters
    ----------
    point
        The one-row sample to expand about.
    deviations
        The standard deviations, which set the step of each variable.

    Returns
    -------
    numpy.ndarray
        Two rows per variable, stepped up and down in turn.
    """
    steps = _hessian_steps(deviations)
    points = []
    for index in range(len(steps)):
        for sign in (1.0, -1.0):
            shifted = point.copy()
            shifted[0, index] += sign * steps[index]
            points.append(shifted[0])
    return numpy.asarray(points)


def _hessian_steps(deviations) -> numpy.ndarray:
    """Return the differencing step of each variable.

    Parameters
    ----------
    deviations
        The standard deviations.

    Returns
    -------
    numpy.ndarray
        The steps, scaled to each variable's own spread.
    """
    return 1e-4 * numpy.where(deviations > 0.0, deviations, 1.0)


def _hessian(propagation: Propagation, points, deviations) -> numpy.ndarray:
    """Return the Hessian of the quantity, by differencing its gradient.

    Differencing rather than differentiating twice, deliberately. Nesting
    the derivatives needs an adjoint that supports forward over reverse,
    which the library's default does not: it is a ``custom_vjp`` and
    refuses forward mode outright. Differencing an exact gradient is
    accurate to the square of the step where differencing a value is
    accurate to the step, and the second-order correction is a small term
    added to a first-order one, so an error in it is an error in a
    correction. Both routes cost one gradient per uncertain variable, and
    by automatic differentiation each of those is a reverse pass.

    Parameters
    ----------
    propagation
        The propagation.
    points
        The stepped points, from :func:`_hessian_points`.
    deviations
        The standard deviations, which set the step.

    Returns
    -------
    numpy.ndarray
        The Hessian, symmetrized.
    """
    steps = _hessian_steps(deviations)
    dimension = len(steps)
    gradients = numpy.asarray(propagation.jacobian(points))
    hessian = numpy.empty((dimension, dimension))
    for index in range(dimension):
        hessian[index] = (gradients[2 * index] - gradients[2 * index + 1]) / (
            2.0 * steps[index]
        )
    return 0.5 * (hessian + hessian.T)


def _basis(propagation: Propagation, degree: int) -> list:
    """Return the orthonormal polynomial basis up to a total degree.

    Parameters
    ----------
    propagation
        The propagation, whose margins choose the polynomial family.
    degree
        The total degree.

    Returns
    -------
    list
        The basis functions, the constant one first.
    """
    distribution = propagation.distribution
    factory = openturns.OrthogonalProductPolynomialFactory(
        [
            openturns.StandardDistributionPolynomialFactory(
                distribution.getMarginal(index)
            )
            for index in range(propagation.dimension)
        ]
    )
    size = factory.getEnumerateFunction().getStrataCumulatedCardinal(degree)
    return [factory.build(index) for index in range(size)]


def polynomial_chaos(
    propagation: Propagation,
    n_training: int = 64,
    degree: int = 4,
    seed: int = 0,
    gradient_enhanced: bool = False,
) -> Estimate:
    """Estimate the statistics by fitting an orthonormal polynomial expansion.

    The basis is orthonormal for the input distribution, so the statistics
    are read straight off the coefficients: the mean is the constant term
    and the variance is the sum of the squares of the others. Nothing is
    sampled from the fitted expansion, which is why the estimate costs
    nothing once the expansion exists.

    ``gradient_enhanced`` adds one regression row per uncertain variable
    per training point, matching the expansion's derivative to the model's.
    Those rows are free here, since a reverse pass gives the whole
    gradient, and they are worth having exactly when training points are
    scarce: the fit sees ``d + 1`` numbers per point instead of one. The
    two blocks are scaled to comparable magnitude before being stacked, so
    the derivative rows inform the fit rather than dominate it.

    Parameters
    ----------
    propagation
        The propagation; the derivatives are needed only when
        ``gradient_enhanced`` is set.
    n_training
        The number of training points.
    degree
        The total degree of the expansion.
    seed
        The sampling seed.
    gradient_enhanced
        Whether to put the derivatives into the regression.

    Returns
    -------
    Estimate
        The mean, the standard deviation, and the cost.

    Raises
    ------
    ValueError
        If derivatives are asked for and the propagation has none.
    """
    if gradient_enhanced and propagation.jacobian is None:
        msg = (
            "Gradient-enhanced chaos needs the derivatives; build the"
            " propagation with differentiable=True."
        )
        raise ValueError(msg)

    openturns.RandomGenerator.SetSeed(seed)
    inputs = numpy.asarray(propagation.distribution.getSample(n_training))
    functions = _basis(propagation, degree)

    _warm_up(propagation, inputs)
    started = time.perf_counter()
    values = numpy.asarray(propagation.evaluate(inputs)).ravel()
    design = numpy.array(
        [[float(function(row)[0]) for function in functions] for row in inputs]
    )
    right_hand_side = [values]
    blocks = [design]
    n_gradients = 0

    if gradient_enhanced:
        derivatives = numpy.asarray(propagation.jacobian(inputs))
        n_gradients = n_training
        slopes = numpy.array(
            [
                [
                    numpy.asarray(function.gradient(row)).ravel()
                    for function in functions
                ]
                for row in inputs
            ]
        )
        spread = float(numpy.std(values))
        derivative_spread = float(numpy.std(derivatives))
        scale = spread / derivative_spread if derivative_spread > 0.0 else 1.0
        for index in range(propagation.dimension):
            blocks.append(slopes[:, :, index] * scale)
            right_hand_side.append(derivatives[:, index] * scale)

    coefficients, *_ = numpy.linalg.lstsq(
        numpy.vstack(blocks), numpy.concatenate(right_hand_side), rcond=None
    )
    seconds = time.perf_counter() - started
    return Estimate(
        name="pce + grad" if gradient_enhanced else "pce",
        mean=float(coefficients[0]),
        standard_deviation=float(numpy.sqrt(numpy.sum(coefficients[1:] ** 2))),
        n_evaluations=n_training,
        n_gradients=n_gradients,
        seconds=seconds,
        extra={"n_terms": len(functions)},
    )


@dataclass
class Comparison:
    """Several estimates of one statistic, scored against a reference."""

    reference: Estimate
    """The large-sample estimate the others are scored against."""

    estimates: tuple[Estimate, ...]
    """The estimates, in the order they were made."""

    def relative_error(self, estimate: Estimate) -> tuple[float, float]:
        """Return the error of one estimate against the reference.

        Both errors are relative to the reference *standard deviation*,
        the mean included. A mean near zero is the normal case here, since
        a converged design misses its target by little on average, and
        dividing an error by that mean turns a small absolute error into
        an enormous relative one that says nothing about the estimator.
        The spread is the scale the question is asked on.

        Parameters
        ----------
        estimate
            The estimate.

        Returns
        -------
        tuple
            The error of the mean and of the standard deviation.
        """
        scale = self.reference.standard_deviation
        return (
            (estimate.mean - self.reference.mean) / scale,
            (estimate.standard_deviation - self.reference.standard_deviation) / scale,
        )

    def table(self) -> str:
        """Render the comparison as a table.

        Returns
        -------
        str
            One row per estimate, with its cost and its errors.
        """

        def row(estimate: Estimate, errors: str) -> str:
            """Render one estimate as a table row."""
            return (
                f"| {estimate.name} | {estimate.cost}"
                f" | {estimate.seconds:.2f} | {estimate.mean:.4g}"
                f" | {estimate.standard_deviation:.4g} |{errors}"
            )

        lines = [
            "| estimator | calls | s | mean | std | mean err | std err |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            row(self.reference, " | |"),
        ]
        for estimate in self.estimates:
            mean_error, deviation_error = self.relative_error(estimate)
            lines.append(
                row(estimate, f" {mean_error:+.1%} | {deviation_error:+.1%} |")
            )
        return "\n".join(lines)


__all__ = [
    "Comparison",
    "Estimate",
    "Propagation",
    "from_scenario",
    "monte_carlo",
    "polynomial_chaos",
    "taylor",
]
