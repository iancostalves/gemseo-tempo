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
"""Screening first, then decomposing the variance."""

from __future__ import annotations

import logging

import numpy
import openturns
import pytest

from gemseo_tempo.dynamics.integrators import DiffraxIntegrator
from gemseo_tempo.problems.brachistochrone import create_optimization
from gemseo_tempo.problems.brachistochrone.model import REGISTRY
from gemseo_tempo.transcriptions import SingleShooting
from gemseo_tempo.uncertainty import freeze_design
from gemseo_tempo.uncertainty import nominal_values
from gemseo_tempo.uncertainty import uncertain_names
from gemseo_tempo.uncertainty.sensitivity import format_indices
from gemseo_tempo.uncertainty.sensitivity import screen
from gemseo_tempo.uncertainty.sensitivity import sobol_indices

logging.disable(logging.CRITICAL)

QUANTITY = "constraint.final.y"
"""How far the bead misses in depth."""


@pytest.fixture(scope="module")
def frozen():
    """Return a shooting brachistochrone frozen at its nominal optimum."""
    names = uncertain_names(REGISTRY)
    scenario, gemseo_scenario = create_optimization(
        transcription=SingleShooting(),
        size=20,
        variant="drag",
        integrator=DiffraxIntegrator(dt0=0.005, n_save=21),
    )
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=200)
    freeze_design(scenario, gemseo_scenario, nominal_values(REGISTRY, names))
    return scenario, names


def test_screening_ranks_the_uncertainties_cheaply(frozen):
    """Morris finds the ordering for a fraction of a Sobol budget.

    Drag dominates, friction follows, and gravity does almost nothing,
    which is the point of including a variable known to three decimal
    places: a screening should be able to say so.
    """
    scenario, names = frozen
    result = screen(scenario, REGISTRY, names, QUANTITY, n_trajectories=20)
    assert result.n_evaluations == 20 * (len(names) + 1)
    assert result.ranking[0] == "k_over_m"
    assert result.ranking[-1] == "g"
    largest = max(result.mean_effect)
    gravity = result.mean_effect[result.names.index("g")]
    assert gravity < 0.2 * largest


def test_screening_names_what_is_not_worth_measuring(frozen):
    scenario, names = frozen
    result = screen(scenario, REGISTRY, names, QUANTITY, n_trajectories=12)
    assert "g" in result.negligible(threshold=0.2)
    assert "k_over_m" not in result.negligible(threshold=0.2)


def test_a_screening_that_finds_nothing_calls_everything_negligible():
    """A flat response has no ranking to give, and says so."""
    from gemseo_tempo.uncertainty.sensitivity import Screening

    flat = Screening(
        names=("a", "b"),
        mean_effect=(0.0, 0.0),
        effect_spread=(0.0, 0.0),
        n_evaluations=4,
    )
    assert flat.negligible() == ["a", "b"]


def test_sobol_indices_are_well_formed(frozen):
    """Total at least first order, and the shares add to about one.

    Frozen at the starting guess rather than the optimum these come out
    ill-conditioned, with totals below first-order indices, which is how
    the mistake was found.
    """
    scenario, names = frozen
    result = sobol_indices(scenario, REGISTRY, names, QUANTITY, n_samples=1024)
    assert result.n_evaluations == 1024 * (len(names) + 2)
    assert result.seconds > 0.0
    for first, total in zip(result.first_order, result.total, strict=True):
        assert total > first - 0.05
    assert 0.85 < sum(result.first_order) < 1.15


def test_sobol_agrees_with_the_screening(frozen):
    """The cheap ranking and the expensive decomposition tell one story."""
    scenario, names = frozen
    ranked = screen(scenario, REGISTRY, names, QUANTITY, n_trajectories=20).ranking
    result = sobol_indices(scenario, REGISTRY, names, QUANTITY, n_samples=1024)
    order = numpy.argsort(result.total)[::-1]
    assert [result.names[index] for index in order] == ranked


def test_the_indices_render_as_a_table(frozen):
    scenario, names = frozen
    result = sobol_indices(scenario, REGISTRY, names, QUANTITY, n_samples=256)
    table = format_indices(result)
    assert "interaction" in table
    for name in names:
        assert f"`{name}`" in table


def test_the_estimator_matches_ishigami():
    """The pipeline is checked where the answer is known.

    Ishigami has analytic Sobol indices, and it is the reference that says
    whether an unexpected result on a real problem is the problem or the
    estimator. It was: the first surprising brachistochrone indices came
    from freezing the wrong design, not from a broken pipeline.
    """
    openturns.RandomGenerator.SetSeed(0)
    distribution = openturns.JointDistribution(
        [openturns.Uniform(-numpy.pi, numpy.pi)] * 3
    )
    size = 4096
    design = numpy.asarray(
        openturns.SobolIndicesExperiment(distribution, size).generate()
    )
    values = (
        numpy.sin(design[:, 0])
        + 7.0 * numpy.sin(design[:, 1]) ** 2
        + 0.1 * design[:, 2] ** 4 * numpy.sin(design[:, 0])
    ).reshape(-1, 1)
    algorithm = openturns.SaltelliSensitivityAlgorithm(
        openturns.Sample(design), openturns.Sample(values), size
    )
    first = numpy.asarray(algorithm.getFirstOrderIndices())
    total = numpy.asarray(algorithm.getTotalOrderIndices())
    assert first == pytest.approx([0.3139, 0.4424, 0.0], abs=0.05)
    assert total == pytest.approx([0.5576, 0.4424, 0.2437], abs=0.05)
