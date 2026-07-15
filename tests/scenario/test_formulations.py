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
"""Formulation builders and end-to-end optimizations."""

from __future__ import annotations

import jax.numpy as jnp
import numpy
import pytest

from gemseo_tempo.dynamics import DiffraxIntegrator
from gemseo_tempo.dynamics import PiecewiseLinearControl
from gemseo_tempo.dynamics import StateSpace
from gemseo_tempo.metadata import Variable
from gemseo_tempo.metadata import VariableRegistry
from gemseo_tempo.outputs import Lagrange
from gemseo_tempo.outputs import Mayer
from gemseo_tempo.scenario import ControlAndDesign
from gemseo_tempo.scenario import DynamicScenario
from gemseo_tempo.scenario import OptimalControl
from gemseo_tempo.scenario import SystemDesign
from gemseo_tempo.statics import JAXModel
from gemseo_tempo.statics import StaticsChain
from gemseo_tempo.transcriptions import BoundarySpec

KNOTS = jnp.linspace(0.0, 1.0, 5)


def make_gain_statics():
    gain = JAXModel(
        function=lambda inputs: {"c": 2.0 * inputs["k"]},
        input_names=["k"],
        output_names=["c"],
        default_inputs={"k": 0.5},
        name="gain",
    )
    return StaticsChain([gain], ["k"])


def make_min_energy_scenario(registry=None):
    """Minimum-energy transfer: min integral u^2 with x(1) = 1, x' = u."""
    space = StateSpace(
        states=["x"],
        controls=["u"],
        rhs=lambda t, s, u, theta: {"x": u["u"]},
    )
    return DynamicScenario(
        state_space=space,
        integrator=DiffraxIntegrator(rtol=1e-12, atol=1e-12, n_save=11),
        controls=(
            PiecewiseLinearControl("u", KNOTS, default_params=0.2 * jnp.ones(5)),
        ),
        functionals=(Lagrange("energy", lambda t, s, u, theta: u["u"] ** 2),),
        boundary=BoundarySpec(s0={"x": 0.0}, fix_final={"x": True}, s1={"x": 1.0}),
        registry=registry,
        t0=0.0,
        t1=1.0,
    )


def make_design_scenario():
    """Tune k (through c = 2k) so that x(1) = exp(-2k) hits 0.5."""
    space = StateSpace(
        states=["x"],
        parameters=["c"],
        rhs=lambda t, s, u, theta: {"x": -theta["c"] * s["x"]},
    )
    return DynamicScenario(
        state_space=space,
        integrator=DiffraxIntegrator(rtol=1e-12, atol=1e-12, n_save=11),
        controls=(),
        functionals=(Mayer("miss", lambda t1, s1, theta: (s1["x"] - 0.5) ** 2),),
        boundary=BoundarySpec(s0={"x": 1.0}),
        statics=make_gain_statics(),
        t0=0.0,
        t1=1.0,
    )


def test_design_space_bounds_from_registry():
    registry = VariableRegistry(
        [
            Variable(name="u", full_name="Control", lower_bound=-2.0, upper_bound=2.0),
        ]
    )
    formulation = OptimalControl(make_min_energy_scenario(registry))
    design_space = formulation.create_design_space()
    assert design_space.variable_names == ["control.u"]
    assert numpy.all(design_space.get_lower_bounds() == -2.0)
    assert numpy.all(design_space.get_upper_bounds() == 2.0)


def test_design_space_explicit_bounds_win():
    registry = VariableRegistry(
        [
            Variable(name="u", full_name="Control", lower_bound=-2.0, upper_bound=2.0),
        ]
    )
    formulation = OptimalControl(
        make_min_energy_scenario(registry),
        bounds={"control.u": (-5.0, 5.0)},
    )
    design_space = formulation.create_design_space()
    assert numpy.all(design_space.get_lower_bounds() == -5.0)


def test_design_space_unbounded_without_registry():
    formulation = OptimalControl(make_min_energy_scenario())
    design_space = formulation.create_design_space()
    assert numpy.all(numpy.isinf(design_space.get_lower_bounds()))


def test_system_design_unknown_constant_raises():
    with pytest.raises(ValueError, match="not inputs of the problem"):
        SystemDesign(make_design_scenario(), ["not_a_constant"])


def test_control_and_design_union():
    scenario = make_design_scenario()
    formulation = ControlAndDesign(scenario, ["k"], bounds={"constant.k": (0.0, 2.0)})
    assert formulation._design_variable_names() == ["constant.k"]
    energy = make_min_energy_scenario()
    # A scenario with controls: union of control and design variables.
    formulation = OptimalControl(energy)
    assert formulation._design_variable_names() == ["control.u"]


def test_minimum_energy_optimum_recovered():
    """SLSQP recovers the analytic optimum u* = 1, J* = 1 to 1e-8."""
    formulation = OptimalControl(
        make_min_energy_scenario(), bounds={"control.u": (-3.0, 3.0)}
    )
    gemseo_scenario = formulation.create_gemseo_scenario()
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=100)
    optimum = gemseo_scenario.optimization_result
    assert optimum.f_opt == pytest.approx(1.0, abs=1e-8)
    u_opt = optimum.x_opt_as_dict["control.u"]
    assert numpy.max(numpy.abs(u_opt - 1.0)) < 1e-6


def test_system_design_optimum():
    """SLSQP finds k* = ln(2) / 2 so that exp(-2k) = 0.5."""
    formulation = SystemDesign(
        make_design_scenario(), ["k"], bounds={"constant.k": (0.01, 2.0)}
    )
    gemseo_scenario = formulation.create_gemseo_scenario()
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=50)
    optimum = gemseo_scenario.optimization_result
    assert optimum.f_opt == pytest.approx(0.0, abs=1e-10)
    k_opt = float(optimum.x_opt_as_dict["constant.k"][0])
    assert k_opt == pytest.approx(numpy.log(2.0) / 2.0, abs=1e-5)
