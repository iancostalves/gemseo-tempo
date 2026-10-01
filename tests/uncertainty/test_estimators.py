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
"""Six ways to the same statistic, checked where the answer is known.

Every estimator is exercised on a function whose mean and variance can be
written down, because that is the only way to tell an estimator that is
wrong from a problem that is surprising. The library learned that the
expensive way once already, when a correct Sobol pipeline was blamed for
indices that came from freezing the wrong design.
"""

from __future__ import annotations

import numpy
import pytest
from gemseo.algos.parameter_space import ParameterSpace

from gemseo_tempo.uncertainty.estimators import Comparison
from gemseo_tempo.uncertainty.estimators import Estimate
from gemseo_tempo.uncertainty.estimators import Propagation
from gemseo_tempo.uncertainty.estimators import finite_differences
from gemseo_tempo.uncertainty.estimators import monte_carlo
from gemseo_tempo.uncertainty.estimators import polynomial_chaos
from gemseo_tempo.uncertainty.estimators import taylor


def _standard_normals(count: int) -> ParameterSpace:
    """Return a space of independent standard normals named u0, u1, ...

    Parameters
    ----------
    count
        The dimension.

    Returns
    -------
    ParameterSpace
        The space.
    """
    space = ParameterSpace()
    for index in range(count):
        space.add_random_variable(
            f"u{index}", "OTNormalDistribution", mu=0.0, sigma=1.0
        )
    return space


def _propagation(function, gradient, count: int = 2) -> Propagation:
    """Wrap an analytic function as a propagation over standard normals.

    Parameters
    ----------
    function
        ``(n, d) -> (n,)``.
    gradient
        ``(n, d) -> (n, d)``, or ``None`` for a propagation without
        derivatives.
    count
        The dimension.

    Returns
    -------
    Propagation
        The propagation.
    """
    space = _standard_normals(count)
    return Propagation(
        evaluate=function,
        space=space,
        names=tuple(f"u{index}" for index in range(count)),
        jacobian=gradient,
    )


# ------------------------------------------------------------ the linear case

LINEAR_SLOPES = numpy.array([3.0, -4.0])
"""A plane whose mean is 2 and whose variance is 9 + 16 = 25."""


def _linear(sample):
    return 2.0 + sample @ LINEAR_SLOPES


def _linear_gradient(sample):
    return numpy.tile(LINEAR_SLOPES, (len(sample), 1))


def test_first_order_is_exact_on_a_plane():
    """Where the response is straight, one gradient is the whole answer."""
    estimate = taylor(_propagation(_linear, _linear_gradient))
    assert estimate.mean == pytest.approx(2.0, abs=1e-12)
    assert estimate.standard_deviation == pytest.approx(5.0, abs=1e-12)
    # One evaluation and one reverse pass, whatever the dimension.
    assert estimate.n_evaluations == 1
    assert estimate.n_gradients == 1
    assert estimate.cost == 2


def test_sampling_agrees_with_the_plane_it_cannot_see():
    estimate = monte_carlo(_propagation(_linear, None), n_samples=20000)
    assert estimate.mean == pytest.approx(2.0, abs=0.1)
    assert estimate.standard_deviation == pytest.approx(5.0, rel=0.02)
    assert estimate.n_evaluations == 20000
    assert estimate.seconds > 0.0


# --------------------------------------------------------- the quadratic case


def _quadratic(sample):
    return sample[:, 0] ** 2 - 2.0 * sample[:, 1] ** 2


def _quadratic_gradient(sample):
    return numpy.stack([2.0 * sample[:, 0], -4.0 * sample[:, 1]], axis=1)


def test_first_order_cannot_see_the_mean_of_a_curved_response():
    """The failure that matters, and the reason second order exists.

    For this function the mean is ``1 - 2 = -1`` and the variance is
    ``2 + 8 = 10``. Expanded to first order about the mean of the inputs,
    every gradient is zero: the estimate says the quantity is exactly zero
    with no spread at all. It is not a poor estimate, it is no estimate,
    and nothing about it announces that.
    """
    propagation = _propagation(_quadratic, _quadratic_gradient)
    first = taylor(propagation)
    assert first.mean == pytest.approx(0.0, abs=1e-12)
    assert first.standard_deviation == pytest.approx(0.0, abs=1e-12)

    second = taylor(propagation, second_order=True)
    assert second.mean == pytest.approx(-1.0, rel=1e-6)
    assert second.standard_deviation == pytest.approx(numpy.sqrt(10.0), rel=1e-6)
    # Second order costs two more gradients per uncertain variable.
    assert second.n_gradients == first.n_gradients + 2 * propagation.dimension


def test_chaos_recovers_the_quadratic_from_a_handful_of_points():
    """A degree-2 expansion represents this exactly, so the fit is exact."""
    estimate = polynomial_chaos(
        _propagation(_quadratic, _quadratic_gradient), n_training=40, degree=2
    )
    assert estimate.mean == pytest.approx(-1.0, abs=1e-8)
    assert estimate.standard_deviation == pytest.approx(numpy.sqrt(10.0), rel=1e-8)
    assert estimate.extra["n_terms"] == 6


def test_the_derivatives_earn_their_place_when_points_are_scarce():
    """Fewer points than basis terms: the gradients make the fit possible.

    A degree-3 expansion in two variables has ten terms, and six training
    points cannot determine them. Six points *with* their gradients carry
    eighteen numbers, and do.
    """
    propagation = _propagation(_quadratic, _quadratic_gradient)
    plain = polynomial_chaos(propagation, n_training=6, degree=3)
    enhanced = polynomial_chaos(
        propagation, n_training=6, degree=3, gradient_enhanced=True
    )
    truth = numpy.sqrt(10.0)
    assert abs(enhanced.standard_deviation - truth) < abs(
        plain.standard_deviation - truth
    )
    assert enhanced.standard_deviation == pytest.approx(truth, rel=1e-6)
    # And they are counted, since the comparison is about cost as well.
    assert plain.n_gradients == 0
    assert enhanced.n_gradients == 6


# ------------------------------------------------------------------- Ishigami


def _ishigami(sample):
    return (
        numpy.sin(sample[:, 0])
        + 7.0 * numpy.sin(sample[:, 1]) ** 2
        + 0.1 * sample[:, 2] ** 4 * numpy.sin(sample[:, 0])
    )


def _uniform_propagation() -> Propagation:
    space = ParameterSpace()
    for index in range(3):
        space.add_random_variable(
            f"u{index}",
            "OTUniformDistribution",
            minimum=-numpy.pi,
            maximum=numpy.pi,
        )
    return Propagation(
        evaluate=_ishigami, space=space, names=("u0", "u1", "u2"), jacobian=None
    )


def test_chaos_beats_sampling_on_ishigami_for_a_tenth_of_the_cost():
    """The claim the comparison exists to make, on a known function.

    Ishigami has mean 3.5 and variance 13.845. A degree-8 expansion trained
    on 400 points is closer than 4000 random draws, and every draw after
    the fit is free.
    """
    propagation = _uniform_propagation()
    truth_mean, truth_deviation = 3.5, numpy.sqrt(13.8445)
    chaos = polynomial_chaos(propagation, n_training=400, degree=8)
    sampling = monte_carlo(propagation, n_samples=4000)
    assert chaos.mean == pytest.approx(truth_mean, abs=0.05)
    assert chaos.standard_deviation == pytest.approx(truth_deviation, rel=0.02)
    assert chaos.cost * 10 <= sampling.cost
    assert abs(chaos.standard_deviation - truth_deviation) < abs(
        sampling.standard_deviation - truth_deviation
    )


# --------------------------------------------------------- refusals and shape


def test_a_taylor_expansion_without_derivatives_is_refused():
    with pytest.raises(ValueError, match="needs the derivatives"):
        taylor(_propagation(_linear, None))


def test_gradient_enhanced_chaos_without_derivatives_is_refused():
    with pytest.raises(ValueError, match="needs the derivatives"):
        polynomial_chaos(_propagation(_linear, None), gradient_enhanced=True)


def test_the_comparison_scores_against_the_reference_spread():
    """Not against the reference mean, which is often near zero here."""
    reference = Estimate(
        name="reference", mean=0.0, standard_deviation=10.0, n_evaluations=1000
    )
    estimate = Estimate(
        name="cheap", mean=1.0, standard_deviation=8.0, n_evaluations=2, n_gradients=1
    )
    comparison = Comparison(reference=reference, estimates=(estimate,))
    mean_error, deviation_error = comparison.relative_error(estimate)
    assert mean_error == pytest.approx(0.1)
    assert deviation_error == pytest.approx(-0.2)
    table = comparison.table()
    assert "cheap" in table
    assert "+10.0%" in table
    assert "-20.0%" in table


def test_the_propagation_reports_its_own_moments():
    propagation = _propagation(_linear, _linear_gradient)
    assert propagation.dimension == 2
    assert propagation.mean_point.tolist() == [[0.0, 0.0]]
    assert propagation.standard_deviations.tolist() == [1.0, 1.0]
    assert propagation.distribution.getDimension() == 2


def test_finite_differences_reproduce_the_exact_gradient_of_a_smooth_response():
    """On a quadratic, central differences are exact up to rounding."""
    exact = _propagation(_quadratic, _quadratic_gradient)
    differenced = finite_differences(exact)
    points = numpy.array([[0.3, -1.2], [2.0, 0.5]])
    assert numpy.allclose(
        differenced.jacobian(points), exact.jacobian(points), atol=1e-6
    )
    assert taylor(differenced).standard_deviation == pytest.approx(
        taylor(exact).standard_deviation, rel=1e-6
    )
