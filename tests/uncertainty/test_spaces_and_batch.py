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
"""Declared distributions, and evaluating a whole sample at once."""

from __future__ import annotations

import logging

import numpy
import pytest

from gemseo_tempo.dynamics.integrators import DiffraxIntegrator
from gemseo_tempo.metadata.registry import VariableRegistry
from gemseo_tempo.metadata.variable import Variable
from gemseo_tempo.problems.brachistochrone import create_optimization
from gemseo_tempo.problems.brachistochrone.model import REGISTRY
from gemseo_tempo.transcriptions import SingleShooting
from gemseo_tempo.uncertainty import batch_evaluator
from gemseo_tempo.uncertainty import batch_jacobian
from gemseo_tempo.uncertainty import freeze_design
from gemseo_tempo.uncertainty import looped_evaluator
from gemseo_tempo.uncertainty import make_parameter_space
from gemseo_tempo.uncertainty import measure_speedup
from gemseo_tempo.uncertainty import nominal_values
from gemseo_tempo.uncertainty import uncertain_names

logging.disable(logging.CRITICAL)

QUANTITIES = ("constraint.final.x", "constraint.final.y")
"""The miss in each direction, which is what an uncertain bead ends up with."""


def _scenario(size: int = 20):
    """Build a shooting brachistochrone frozen at its nominal optimum.

    Shooting rather than collocation: under collocation the states are
    design variables, so changing the physics with the design frozen only
    breaks the defects. Under shooting the trajectory is integrated, so it
    genuinely responds.

    Parameters
    ----------
    size
        The discretization.

    Returns
    -------
    DynamicScenario
        The frozen scenario.
    """
    names = uncertain_names(REGISTRY)
    scenario, gemseo_scenario = create_optimization(
        transcription=SingleShooting(),
        size=size,
        variant="drag",
        integrator=DiffraxIntegrator(dt0=0.005, n_save=21),
    )
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=200)
    return freeze_design(scenario, gemseo_scenario, nominal_values(REGISTRY, names))


@pytest.fixture(scope="module")
def frozen():
    """Return the frozen scenario, its uncertain names and a sample."""
    names = uncertain_names(REGISTRY)
    sample = numpy.asarray(make_parameter_space(REGISTRY, names).compute_samples(512))
    return _scenario(), names, sample


# ------------------------------------------------------------- the spaces


def test_the_registry_is_what_declares_an_uncertainty():
    """Distributions live with the variable, not in the analysis."""
    names = uncertain_names(REGISTRY)
    assert set(names) == {"g", "mu", "k_over_m"}
    space = make_parameter_space(REGISTRY, names)
    assert space.uncertain_variables == list(names)


def test_a_variable_without_a_distribution_cannot_be_made_uncertain():
    """Refused rather than defaulted.

    A default would be a guess wearing a distribution's clothes, and
    sampling it would produce numbers that look like uncertainty
    quantification without being it.
    """
    registry = VariableRegistry([Variable(name="x", full_name="Something")])
    with pytest.raises(ValueError, match="declares no distribution"):
        make_parameter_space(registry, ["x"])


def test_a_distribution_must_name_itself():
    registry = VariableRegistry(
        [Variable(name="x", full_name="Something", distribution={"mu": 1.0})]
    )
    with pytest.raises(ValueError, match="has no 'name'"):
        make_parameter_space(registry, ["x"])


def test_an_unknown_variable_is_reported():
    with pytest.raises(KeyError, match="not in the registry"):
        make_parameter_space(REGISTRY, ["not_a_variable"])


def test_the_nominal_values_are_the_means():
    values = nominal_values(REGISTRY, ["g", "mu", "k_over_m"])
    assert values["g"] == pytest.approx(9.81, abs=1e-6)
    # Uniform on [0.02, 0.08] and [0.03, 0.07].
    assert values["mu"] == pytest.approx(0.05, abs=1e-9)
    assert values["k_over_m"] == pytest.approx(0.05, abs=1e-9)


def test_a_prefix_names_the_variables_as_the_discipline_knows_them():
    space = make_parameter_space(REGISTRY, ["mu"], prefix="constant.")
    assert space.uncertain_variables == ["constant.mu"]


# -------------------------------------------------------------- the batch


def test_the_batched_and_looped_paths_agree_exactly(frozen):
    """Not approximately: it is the same function, so any gap is a bug."""
    scenario, names, sample = frozen
    batched = numpy.asarray(batch_evaluator(scenario, names, QUANTITIES)(sample))
    looped = looped_evaluator(scenario, names, QUANTITIES)(sample)
    assert batched.shape == (len(sample), len(QUANTITIES))
    assert numpy.max(numpy.abs(batched - looped)) < 1e-12


def test_the_whole_sample_goes_through_far_faster(frozen):
    """The sprint asks for ten times on 512 draws; it is much more.

    The saving is not the arithmetic, which is identical, but everything
    around it: one dispatch instead of five hundred.
    """
    scenario, names, sample = frozen
    result = measure_speedup(scenario, names, QUANTITIES, sample)
    assert result.n_samples == 512
    assert result.max_difference < 1e-12
    assert result.speedup > 10.0


def test_the_derivatives_come_out_with_the_sample(frozen):
    """What makes Taylor and gradient-enhanced chaos cheap here.

    The CSMA paper calls first-order propagation affordable only when these
    derivatives are exact, and by finite differences their cost multiplies
    by the number of uncertain variables. By automatic differentiation they
    come out of the same pass.
    """
    scenario, names, sample = frozen
    small = sample[:16]
    jacobian = numpy.asarray(batch_jacobian(scenario, names, QUANTITIES)(small))
    assert jacobian.shape == (16, len(QUANTITIES), len(names))

    evaluate = batch_evaluator(scenario, names, QUANTITIES)
    step = 1e-6
    point = small[:1]
    for index in range(len(names)):
        up = point.copy()
        down = point.copy()
        up[0, index] += step
        down[0, index] -= step
        difference = (numpy.asarray(evaluate(up)) - numpy.asarray(evaluate(down))) / (
            2.0 * step
        )
        assert jacobian[0, :, index] == pytest.approx(difference[0], rel=1e-4, abs=1e-8)


def test_freezing_uses_the_optimum_rather_than_the_guess(frozen):
    """Which design the uncertainty is propagated through matters.

    Frozen at the starting guess, the analysis measures the sensitivity of
    a trajectory nobody would fly, and the Sobol indices that come out of it
    are not merely different but ill-conditioned.
    """
    scenario, names, sample = frozen
    values = numpy.asarray(batch_evaluator(scenario, names, QUANTITIES)(sample))
    # A converged design misses by a little under uncertainty, not a lot.
    assert numpy.abs(values).max() < 0.5
    # And it genuinely responds: a frozen design under varying physics does
    # not give the same answer twice.
    assert values[:, 0].std() > 1e-4
