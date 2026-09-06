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
"""Statistics of a constraint as a differentiable function of the design."""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy
import pytest

from gemseo_tempo.dynamics.integrators import DiffraxIntegrator
from gemseo_tempo.problems.brachistochrone import create_optimization
from gemseo_tempo.problems.brachistochrone.model import REGISTRY
from gemseo_tempo.scenario.formulations import OptimalControl
from gemseo_tempo.transcriptions import SingleShooting
from gemseo_tempo.uncertainty import batch_evaluator
from gemseo_tempo.uncertainty import freeze_design
from gemseo_tempo.uncertainty import make_parameter_space
from gemseo_tempo.uncertainty import nominal_values
from gemseo_tempo.uncertainty import uncertain_names
from gemseo_tempo.uncertainty.robust import DEVIATION
from gemseo_tempo.uncertainty.robust import MARGIN
from gemseo_tempo.uncertainty.robust import MEAN
from gemseo_tempo.uncertainty.robust import RELIABILITY
from gemseo_tempo.uncertainty.robust import Assessment
from gemseo_tempo.uncertainty.robust import assess
from gemseo_tempo.uncertainty.robust import format_assessments
from gemseo_tempo.uncertainty.robust import robust_discipline
from gemseo_tempo.uncertainty.robust import statistics_function

logging.disable(logging.CRITICAL)

QUANTITY = "constraint.final.y"
"""A scalar output, treated here as a constraint satisfied when negative."""


@pytest.fixture(scope="module")
def frozen():
    """Return a brachistochrone frozen at its optimum, plus two samples."""
    names = uncertain_names(REGISTRY)
    scenario, gemseo_scenario = create_optimization(
        transcription=SingleShooting(),
        size=20,
        variant="drag",
        integrator=DiffraxIntegrator(dt0=0.005, n_save=21),
    )
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=200)
    freeze_design(scenario, gemseo_scenario, nominal_values(REGISTRY, names))
    space = make_parameter_space(REGISTRY, names)
    design = numpy.asarray(space.compute_samples(128))
    check = numpy.asarray(space.compute_samples(512))
    return scenario, names, design, check


def test_the_statistics_are_the_sample_statistics(frozen):
    """The same numbers a plain batch evaluation gives, and nothing else."""
    scenario, names, sample, _ = frozen
    statistics = jax.jit(statistics_function(scenario, names, sample, QUANTITY))({})
    values = numpy.asarray(batch_evaluator(scenario, names, [QUANTITY])(sample)).ravel()
    assert float(statistics[MEAN]) == pytest.approx(values.mean(), rel=1e-10)
    assert float(statistics[DEVIATION]) == pytest.approx(values.std(), rel=1e-10)
    assert float(statistics[MARGIN]) == pytest.approx(
        values.mean() + 2.0 * values.std(), rel=1e-10
    )


def test_the_smoothed_probability_is_the_probability(frozen):
    """The logistic exists for its gradient, not to change the answer.

    Too wide and the constraint is not the one that was asked for, so the
    agreement with the counted fraction is what says the width is small
    enough. Here it agrees to a fraction of a percentage point.
    """
    scenario, names, sample, _ = frozen
    values = numpy.asarray(batch_evaluator(scenario, names, [QUANTITY])(sample)).ravel()
    counted = float(numpy.mean(values <= 0.0))
    spread = float(values.std())
    smoothed = float(
        jax.jit(
            statistics_function(
                scenario, names, sample, QUANTITY, smoothing=spread / 50.0
            )
        )({})[RELIABILITY]
    )
    assert smoothed == pytest.approx(counted, abs=0.02)


def test_a_wider_smoothing_pulls_the_probability_towards_a_half(frozen):
    """Which is the failure mode the width has to be chosen against."""
    scenario, names, sample, _ = frozen
    blurred = float(
        jax.jit(statistics_function(scenario, names, sample, QUANTITY, smoothing=1e6))(
            {}
        )[RELIABILITY]
    )
    assert blurred == pytest.approx(0.5, abs=1e-3)


def test_the_statistics_respond_to_the_design(frozen):
    """A statistic nothing can move is a statistic nothing can optimize."""
    scenario, names, sample, _ = frozen
    statistics = statistics_function(scenario, names, sample, QUANTITY)
    design_names = [
        name
        for name in scenario.problem.input_defaults
        if name.startswith("control.") or name == "t_f"
    ]
    design = {
        name: jnp.asarray(scenario.problem.input_defaults[name])
        for name in design_names
    }
    gradient = jax.jit(jax.grad(lambda values: statistics(values)[MEAN]))(design)
    stacked = numpy.concatenate([numpy.ravel(value) for value in gradient.values()])
    assert numpy.isfinite(stacked).all()
    assert numpy.abs(stacked).max() > 0.0

    # And it is the derivative of the statistic, not of one draw: stepping
    # the design moves the mean by what the gradient says it will. The
    # design entry is a vector, so stepping all of it by the same amount
    # moves the mean by the *sum* of its partials.
    step = 1e-6
    key = design_names[0]
    moved = dict(design)
    moved[key] = design[key] + step
    difference = (
        float(statistics(moved)[MEAN]) - float(statistics(design)[MEAN])
    ) / step
    assert difference == pytest.approx(
        float(numpy.sum(numpy.ravel(gradient[key]))), rel=1e-3
    )


def test_a_path_constraint_needs_a_reduction(frozen):
    """One value per node is not a quantity of interest on its own."""
    scenario, names, sample, _ = frozen
    with pytest.raises(ValueError, match="not a quantity of interest"):
        statistics_function(scenario, names, sample, "trajectory.y")({})


def test_the_reduction_says_which_scalar_the_path_becomes(frozen):
    scenario, names, sample, _ = frozen
    statistics = jax.jit(
        statistics_function(scenario, names, sample, "trajectory.y", reduction=jnp.max)
    )({})
    assert numpy.isfinite(float(statistics[MEAN]))


def test_the_assessment_counts_rather_than_smooths(frozen):
    """What the design gets, on a sample it was not optimized against."""
    scenario, names, _, check = frozen
    values = numpy.asarray(batch_evaluator(scenario, names, [QUANTITY])(check)).ravel()
    assessment = assess(scenario, names, check, QUANTITY, objective=1.23, name="frozen")
    assert assessment.n_samples == 512
    assert assessment.objective == pytest.approx(1.23)
    assert assessment.reliability == pytest.approx(numpy.mean(values <= 0.0))
    assert assessment.feasible_fraction == assessment.reliability
    assert assessment.worst == pytest.approx(values.max())
    assert assessment.quantile == pytest.approx(numpy.percentile(values, 95.0))
    assert "smoothed_reliability" in assessment.extra


def test_the_assessments_render_as_a_table():
    table = format_assessments(
        [
            Assessment(
                name="deterministic",
                objective=161.41,
                reliability=0.8575,
                worst=77.2,
                quantile=17.0,
                n_samples=4000,
            )
        ]
    )
    assert "deterministic" in table
    assert "85.8%" in table
    assert "161.41" in table


# ------------------------------------------ the discipline and its gradients


def test_the_robust_discipline_keeps_every_gradient(frozen):
    """The test whose absence let a dead Jacobian through.

    Adding a statistic to a scenario by chaining a second discipline
    alongside its own leaves the **first** one undifferentiated. A JAX
    discipline filters its Jacobian down to the inputs and outputs GEMSEO
    declares to it, those declarations do not reach a discipline inside a
    chain, and an empty filter returns zeros rather than raising. The
    objective and every transcription constraint come back with gradient
    exactly zero, the only surviving row is the added one, and the
    optimizer stops on its first iteration with a singular subproblem.

    So the statistics are computed in the same discipline, and this
    asserts what that is for: every gradient is still there.
    """
    scenario, names, sample, _ = frozen
    discipline = robust_discipline(
        scenario, names, sample[:16], QUANTITY, smoothing=0.05
    )
    formulation = OptimalControl(
        scenario,
        disciplines=[discipline],
        extra_constraints=[(f"{QUANTITY}.shortfall", "ineq")],
    )
    gemseo_scenario = formulation.create_gemseo_scenario()
    problem = gemseo_scenario.formulation.optimization_problem
    point = problem.design_space.get_current_value()

    assert numpy.linalg.norm(problem.objective.jac(point)) > 0.0
    names_seen = set()
    for constraint in problem.constraints:
        names_seen.add(constraint.name)
        gradient = numpy.atleast_2d(constraint.jac(point))
        assert numpy.isfinite(gradient).all()
        assert numpy.linalg.norm(gradient) > 0.0, constraint.name
    # The reliability constraint is wired, alongside the transcription's own.
    assert f"{QUANTITY}.shortfall" in names_seen
    assert len(names_seen) > 1


def test_the_robust_discipline_reports_the_statistics_as_outputs(frozen):
    scenario, names, sample, _ = frozen
    discipline = robust_discipline(scenario, names, sample[:16], QUANTITY)
    outputs = set(discipline.io.output_grammar)
    for statistic in (MEAN, DEVIATION, MARGIN, RELIABILITY, "shortfall"):
        assert f"{QUANTITY}.{statistic}" in outputs
    # And it still produces everything the transcription did.
    assert set(scenario.problem.output_names) <= outputs
