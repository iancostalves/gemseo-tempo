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
"""Tests of the Hermite-Simpson collocation transcription."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy
import pytest

from gemseo_tempo.dynamics import DiffraxIntegrator
from gemseo_tempo.dynamics import PiecewiseLinearControl
from gemseo_tempo.dynamics import StateSpace
from gemseo_tempo.dynamics import Trajectory
from gemseo_tempo.outputs import ControlGroupConstraint
from gemseo_tempo.outputs import FinalRate
from gemseo_tempo.outputs import Lagrange
from gemseo_tempo.outputs import Mayer
from gemseo_tempo.outputs import PathConstraint
from gemseo_tempo.outputs import PeriodicityConstraint
from gemseo_tempo.outputs import TerminalConstraint
from gemseo_tempo.outputs import TimeIntegral
from gemseo_tempo.scenario import DynamicScenario
from gemseo_tempo.scenario import OptimalControl
from gemseo_tempo.statics import JAXModel
from gemseo_tempo.statics import StaticsChain
from gemseo_tempo.transcriptions import BoundarySpec
from gemseo_tempo.transcriptions import HermiteSimpson
from gemseo_tempo.transcriptions import SingleShooting

KNOTS = jnp.linspace(0.0, 1.0, 5)


def min_energy_space():
    """x' = u, minimize integral u^2 with x(0)=0, x(1)=1. u*=1, J*=1."""
    return StateSpace(
        states=["x"],
        controls=["u"],
        rhs=lambda t, s, u, theta: {"x": u["u"]},
    )


def make_min_energy(transcription, **kwargs):
    return DynamicScenario(
        state_space=min_energy_space(),
        integrator=DiffraxIntegrator(rtol=1e-11, atol=1e-11, n_save=11),
        controls=(
            PiecewiseLinearControl("u", KNOTS, default_params=0.2 * jnp.ones(5)),
        ),
        functionals=(Lagrange("energy", lambda t, s, u, theta: u["u"] ** 2),),
        boundary=BoundarySpec(s0={"x": 0.0}, fix_final={"x": True}, s1={"x": 1.0}),
        transcription=transcription,
        **kwargs,
    )


def solve(scenario, **bounds):
    formulation = OptimalControl(scenario, bounds=bounds or None)
    gemseo_scenario = formulation.create_gemseo_scenario()
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=200)
    return gemseo_scenario


# ------------------------------------------------------------- structure


def test_collocation_is_monolithic_only():
    scenario = make_min_energy(HermiteSimpson(num_segments=10))
    assert len(scenario.disciplines) == 1
    with pytest.raises(ValueError, match="does not support the split"):
        make_min_energy(HermiteSimpson(num_segments=10), assembly="split")


def test_design_variables_and_defects():
    scenario = make_min_energy(HermiteSimpson(num_segments=8))
    problem = scenario.problem
    # x has fixed initial and final: 9 nodes minus 2 -> 7 free state nodes.
    assert problem.design_variables["state.x"].size == 7
    # controls sampled at all 9 nodes.
    assert problem.design_variables["control.u"].size == 9
    assert ("constraint.defect.x", "eq") in problem.constraint_names
    outputs = problem.jax_function(problem.input_defaults)
    assert outputs["constraint.defect.x"].shape == (8,)


def test_defect_ref_scaling():
    problem = make_min_energy(
        HermiteSimpson(num_segments=6, defect_ref={"x": 4.0})
    ).problem
    scaled = problem.jax_function(problem.input_defaults)["constraint.defect.x"]
    reference = make_min_energy(HermiteSimpson(num_segments=6)).problem
    unscaled = reference.jax_function(reference.input_defaults)["constraint.defect.x"]
    assert jnp.max(jnp.abs(scaled * 4.0 - unscaled)) < 1e-12
    # A single float applies to every state.
    flat = make_min_energy(HermiteSimpson(num_segments=6, defect_ref=2.0)).problem
    flat_defect = flat.jax_function(flat.input_defaults)["constraint.defect.x"]
    assert jnp.max(jnp.abs(flat_defect * 2.0 - unscaled)) < 1e-12


# --------------------------------------------------------- optimization


def test_matches_shooting_optimum():
    """Collocation and shooting reach the same J* = 1 to 1e-6."""
    shooting = solve(make_min_energy(SingleShooting()), **{"control.u": (-3.0, 3.0)})
    collocation = solve(
        make_min_energy(HermiteSimpson(num_segments=6)),
        **{"control.u": (-3.0, 3.0)},
    )
    j_shooting = float(shooting.optimization_result.f_opt)
    j_colloc = float(collocation.optimization_result.f_opt)
    assert j_shooting == pytest.approx(1.0, abs=1e-6)
    assert j_colloc == pytest.approx(1.0, abs=1e-6)
    assert j_colloc == pytest.approx(j_shooting, abs=1e-6)


def test_defect_residuals_small_at_optimum():
    scenario = make_min_energy(HermiteSimpson(num_segments=6))
    gemseo_scenario = solve(scenario, **{"control.u": (-3.0, 3.0)})
    x_opt = gemseo_scenario.design_space.get_current_value(as_dict=True)
    outputs = scenario.problem.jax_function(
        {k: jnp.asarray(v) for k, v in x_opt.items()}
    )
    assert jnp.max(jnp.abs(outputs["constraint.defect.x"])) < 1e-9


def test_order_of_accuracy():
    """Hermite-Simpson state error decays at (at least) third order.

    Order of accuracy is a property of the scheme, so the defects are
    solved directly (a square feasibility system, one defect per free
    node) rather than through an optimizer, which would mix solver
    tolerances into the measurement.
    """
    a, t1 = 0.7, 2.0
    exact = math.exp(-a * t1)

    def node_error(n_seg):
        problem = HermiteSimpson(num_segments=n_seg).transcribe(
            None,
            StateSpace(
                states=["x"],
                parameters=["a"],
                rhs=lambda t, s, u, theta: {"x": -theta["a"] * s["x"]},
            ),
            None,  # collocation solves no ODE
            (),
            (Mayer("obj", lambda t_f, s_f, theta: s_f["x"]),),
            (),
            BoundarySpec(s0={"x": 1.0}),
            constants={"a": a},
            t0=0.0,
            t1=t1,
        )
        defaults = problem.input_defaults

        def residual(free_nodes):
            outputs = problem.jax_function({**defaults, "state.x": free_nodes})
            return outputs["constraint.defect.x"]

        # The decay dynamics are linear, so the defects are linear in the
        # node states and one Newton step solves them exactly.
        start = defaults["state.x"]
        jacobian = jax.jacobian(residual)(start)
        solved = start - jnp.linalg.solve(jacobian, residual(start))
        assert jnp.max(jnp.abs(residual(solved))) < 1e-12
        # Node 0 is fixed, so the last free node is the state at t1.
        return abs(float(solved[-1]) - exact)

    coarse = node_error(4)
    fine = node_error(8)
    order = math.log2(coarse / fine)
    assert order >= 3.0


def test_warm_start_reduces_iterations():
    """Seeding the node values from a trajectory cuts the SLSQP work."""
    # Cold start from the default linear ramp and the control defaults.
    cold_scenario = make_min_energy(HermiteSimpson(num_segments=6))
    cold = solve(cold_scenario, **{"control.u": (-3.0, 3.0)})
    cold_calls = len(cold.formulation.optimization_problem.database)

    # Warm start from a trajectory close to the known optimum x(t) = t,
    # u = 1 (as a cheap shooting run would produce).
    ts = jnp.linspace(0.0, 1.0, 41)
    trajectory = Trajectory(ts=ts, states={"x": ts}, controls={"u": jnp.ones_like(ts)})
    warm_scenario = make_min_energy(HermiteSimpson(num_segments=6))
    warm_scenario.problem.set_initial_guess(trajectory)
    warm = solve(warm_scenario, **{"control.u": (-3.0, 3.0)})
    warm_calls = len(warm.formulation.optimization_problem.database)

    assert float(warm.optimization_result.f_opt) == pytest.approx(1.0, abs=1e-6)
    assert warm_calls < cold_calls


# ------------------------------------------------------------ behaviors


def test_free_final_time():
    """Node times scale with the free final time."""
    space = StateSpace(
        states=["x"],
        controls=["u"],
        rhs=lambda t, s, u, theta: {"x": u["u"]},
    )
    problem = HermiteSimpson(num_segments=6).transcribe(
        None,
        space,
        DiffraxIntegrator(dt0=0.1),
        (PiecewiseLinearControl("u", jnp.linspace(0.0, 1.0, 7)),),
        (Mayer("time", lambda t1, s1, theta: t1),),
        (),
        BoundarySpec(s0={"x": 0.0}, free_final_time=True, t_f=2.0),
    )
    assert "t_f" in problem.design_variables
    outputs = problem.jax_function(problem.input_defaults)
    assert float(outputs["times"][-1]) == pytest.approx(2.0)
    inputs = dict(problem.input_defaults)
    inputs["t_f"] = jnp.asarray([3.5])
    outputs = problem.jax_function(inputs)
    assert float(outputs["times"][-1]) == pytest.approx(3.5)


def test_path_constraint_at_nodes_and_midpoints():
    space = StateSpace(
        states=["x"],
        controls=["u"],
        rhs=lambda t, s, u, theta: {"x": u["u"]},
    )
    problem = HermiteSimpson(num_segments=4).transcribe(
        None,
        space,
        DiffraxIntegrator(dt0=0.1),
        (PiecewiseLinearControl("u", KNOTS),),
        (Mayer("goal", lambda t1, s1, theta: s1["x"]),),
        (PathConstraint("u_max", lambda t, s, u, theta: u["u"] - 1.0),),
        BoundarySpec(s0={"x": 0.0}),
    )
    outputs = problem.jax_function(problem.input_defaults)
    # 5 nodes + 4 midpoints = 9 evaluation points.
    assert outputs["constraint.u_max"].shape == (9,)


def test_control_group_and_periodicity():
    space = StateSpace(
        states=["x"],
        controls=["u1", "u2"],
        rhs=lambda t, s, u, theta: {"x": u["u1"] + u["u2"]},
    )
    problem = HermiteSimpson(num_segments=5).transcribe(
        None,
        space,
        DiffraxIntegrator(dt0=0.1),
        (
            PiecewiseLinearControl("u1", jnp.linspace(0.0, 1.0, 6)),
            PiecewiseLinearControl("u2", jnp.linspace(0.0, 1.0, 6)),
        ),
        (Mayer("goal", lambda t1, s1, theta: s1["x"]),),
        (
            ControlGroupConstraint("shares", ["u1", "u2"]),
            PeriodicityConstraint("cycle", ["x"]),
        ),
        BoundarySpec(s0={"x": 0.0}, fix_initial={"x": False}),
    )
    outputs = problem.jax_function(problem.input_defaults)
    assert outputs["constraint.shares"].shape == (6,)
    assert numpy.allclose(numpy.asarray(outputs["constraint.cycle"]), 0.0)


def test_set_initial_guess_unsupported_on_shooting():
    problem = make_min_energy(SingleShooting()).problem
    with pytest.raises(NotImplementedError, match="does not support"):
        problem.set_initial_guess(Trajectory(ts=jnp.zeros(3)))


def test_guess_setter_skips_names_absent_from_the_trajectory():
    problem = make_min_energy(HermiteSimpson(num_segments=6)).problem
    before_state = problem.input_defaults["state.x"]
    before_control = problem.input_defaults["control.u"]
    # A trajectory carrying neither "x" nor "u" leaves the defaults alone.
    problem.set_initial_guess(Trajectory(ts=jnp.linspace(0.0, 1.0, 5)))
    assert jnp.array_equal(problem.input_defaults["state.x"], before_state)
    assert jnp.array_equal(problem.input_defaults["control.u"], before_control)


def test_statics_chain_outputs_and_functional_variants():
    """A statics-fed parameter, declared outputs, and every functional kind."""
    gain = JAXModel(
        function=lambda inputs: {"c": 2.0 * inputs["k"]},
        input_names=["k"],
        output_names=["c"],
        default_inputs={"k": 0.5},
        name="gain",
    )
    space = StateSpace(
        states=["x"],
        controls=["u"],
        parameters=["c"],
        outputs=["y"],
        rhs=lambda t, s, u, theta: {"x": -theta["c"] * s["x"] + u["u"]},
        output=lambda t, s, u, theta: {"y": s["x"] ** 2},
    )
    problem = HermiteSimpson(num_segments=5).transcribe(
        StaticsChain([gain], ["k"]),
        space,
        None,
        (PiecewiseLinearControl("u", jnp.linspace(0.0, 1.0, 6)),),
        (
            TimeIntegral("cumulative.y", "y"),
            FinalRate("final_rate.y", "y"),
            Mayer("terminal", lambda t_f, s_f, theta: s_f["x"]),
        ),
        (
            TerminalConstraint(
                "arrival", lambda t_f, s_f, theta: s_f["x"] - 0.1, kind="ineq"
            ),
        ),
        BoundarySpec(s0={"x": 1.0}),
        constants={"k": 0.5},
        t0=0.0,
        t1=1.0,
    )
    outputs = problem.jax_function(problem.input_defaults)
    assert outputs["trajectory.y"].shape == (6,)
    # c = 2k is produced by the statics chain, so it is not a raw input.
    assert "constant.c" not in problem.input_defaults
    assert "constant.k" in problem.input_defaults
    for name in ("cumulative.y", "final_rate.y", "terminal", "constraint.arrival"):
        assert outputs[name].shape == (1,)


def test_validation_errors():
    space = StateSpace(
        states=["x"],
        controls=["u"],
        rhs=lambda t, s, u, theta: {"x": u["u"]},
    )
    transcription = HermiteSimpson(num_segments=4)
    control = PiecewiseLinearControl("u", KNOTS)
    goal = (Mayer("goal", lambda t_f, s_f, theta: s_f["x"]),)
    boundary = BoundarySpec(s0={"x": 0.0})

    def transcribe(**overrides):
        kwargs = {
            "statics": None,
            "state_space": space,
            "integrator": None,
            "controls": (control,),
            "functionals": goal,
            "constraints": (),
            "boundary": boundary,
        }
        kwargs.update(overrides)
        return transcription.transcribe(**kwargs)

    with pytest.raises(ValueError, match="do not match the state-space"):
        transcribe(controls=(PiecewiseLinearControl("v", KNOTS),))
    with pytest.raises(ValueError, match="At least one functional"):
        transcribe(functionals=())
    with pytest.raises(ValueError, match="requires t0 = 0"):
        transcribe(
            boundary=BoundarySpec(s0={"x": 0.0}, free_final_time=True, t_f=1.0),
            t0=1.0,
        )
    with pytest.raises(ValueError, match="neither a state nor an output"):
        transcribe(functionals=(FinalRate("rate", "unknown"),))
    with pytest.raises(ValueError, match="is not an output"):
        transcribe(functionals=(TimeIntegral("cum", "unknown"),))
    with pytest.raises(ValueError, match="'g' is neither an output"):
        transcribe(
            state_space=StateSpace(
                states=["x"],
                controls=["u"],
                parameters=["g"],
                rhs=lambda t, s, u, theta: {"x": theta["g"] * u["u"]},
            )
        )
