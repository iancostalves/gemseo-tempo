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
"""Reporting the problem a formulation hands the optimizer."""

from __future__ import annotations

import logging

import numpy
import pytest
from tests.scenario.test_formulations import make_min_energy_scenario

from gemseo_tempo.scenario import OptimalControl
from gemseo_tempo.scenario.description import describe_problem

logging.disable(logging.CRITICAL)


def test_the_description_matches_the_formulation():
    """The design variables are the formulation's, with its bounds."""
    scenario = make_min_energy_scenario()
    formulation = OptimalControl(scenario, bounds={"control.u": (-3.0, 3.0)})
    described = describe_problem(scenario, formulation)

    assert [variable.name for variable in described.design_variables] == ["control.u"]
    assert described.design_variables[0].lower_bound == -3.0
    assert described.design_variables[0].upper_bound == 3.0
    assert described.objective == "energy"
    assert described.maximize is False


def test_constraint_sizes_are_measured_not_guessed():
    """Reported sizes match what evaluating the problem actually returns."""
    scenario = make_min_energy_scenario()
    described = describe_problem(scenario)
    outputs = scenario.problem.jax_function(scenario.problem.input_defaults)
    for constraint in described.constraints:
        actual = numpy.atleast_1d(numpy.asarray(outputs[constraint.name])).size
        assert constraint.size == actual
    assert described.constraint_size == sum(
        constraint.size for constraint in described.constraints
    )


def test_without_a_formulation_every_variable_is_reported():
    """The transcription's own variables and bounds are used."""
    scenario = make_min_energy_scenario()
    described = describe_problem(scenario)
    assert set(scenario.problem.design_variables) == {
        variable.name for variable in described.design_variables
    }
    assert described.design_size > 0


def test_maximizing_is_reported():
    scenario = make_min_energy_scenario()
    formulation = OptimalControl(scenario, maximize=True)
    assert describe_problem(scenario, formulation).maximize is True


def test_unbounded_variables_read_as_infinite():
    scenario = make_min_energy_scenario()
    described = describe_problem(scenario, OptimalControl(scenario))
    variable = described.design_variables[0]
    assert numpy.isinf(variable.lower_bound)
    assert variable.bounds_text == "[-inf, inf]"


def test_the_markdown_carries_every_row():
    """The rendered table has a line per variable and per constraint."""
    scenario = make_min_energy_scenario()
    described = describe_problem(scenario, OptimalControl(scenario))
    markdown = described.to_markdown()
    assert "minimize" in markdown
    for variable in described.design_variables:
        assert f"`{variable.name}`" in markdown
    for constraint in described.constraints:
        assert f"`{constraint.name}`" in markdown


def test_the_text_summary_reads_as_a_problem_statement():
    scenario = make_min_energy_scenario()
    text = str(describe_problem(scenario, OptimalControl(scenario)))
    assert "design variables:" in text
    assert "constraints:" in text
    assert "control.u" in text


def test_bounds_text_prints_finite_sides():
    scenario = make_min_energy_scenario()
    described = describe_problem(
        scenario, OptimalControl(scenario, bounds={"control.u": (-2.0, 2.5)})
    )
    assert described.design_variables[0].bounds_text == "[-2, 2.5]"


def test_a_constant_is_sized_from_its_default():
    """A design constant is not a transcription variable, so it is measured."""
    from gemseo_tempo.problems.albatross import create_codesign_scenario
    from gemseo_tempo.scenario import ControlAndDesign

    scenario = create_codesign_scenario(size=6)
    formulation = ControlAndDesign(scenario, ["w_ref"])
    described = describe_problem(scenario, formulation)
    constants = [
        variable
        for variable in described.design_variables
        if variable.name == "constant.w_ref"
    ]
    assert len(constants) == 1
    assert constants[0].size == 1
    with pytest.raises(StopIteration):
        next(
            variable
            for variable in described.design_variables
            if variable.name == "constant.wing_area"
        )
