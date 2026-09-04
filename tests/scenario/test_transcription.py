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
"""Tests of the transcription machinery beyond assembly equivalence."""

from __future__ import annotations

import jax.numpy as jnp
import numpy
import pytest

from gemseo_tempo.dynamics import DiffraxIntegrator
from gemseo_tempo.dynamics import PiecewiseLinearControl
from gemseo_tempo.dynamics import QuasiStaticIntegrator
from gemseo_tempo.dynamics import StateSpace
from gemseo_tempo.outputs import ControlGroupConstraint
from gemseo_tempo.outputs import Lagrange
from gemseo_tempo.outputs import Mayer
from gemseo_tempo.outputs import PathConstraint
from gemseo_tempo.outputs import PeriodicityConstraint
from gemseo_tempo.outputs import TimeIntegral
from gemseo_tempo.outputs.constraints import AbstractConstraint
from gemseo_tempo.transcriptions import AbstractTranscription
from gemseo_tempo.transcriptions import BoundarySpec
from gemseo_tempo.transcriptions import SingleShooting

KNOTS = jnp.linspace(0.0, 2.0, 5)


def make_integrator(**kwargs):
    kwargs.setdefault("rtol", 1e-12)
    kwargs.setdefault("atol", 1e-12)
    kwargs.setdefault("n_save", 21)
    return DiffraxIntegrator(**kwargs)


def transcribe(**overrides):
    space = overrides.pop(
        "state_space",
        StateSpace(
            states=["x"],
            controls=["u"],
            rhs=lambda t, s, u, theta: {"x": u["u"]},
        ),
    )
    defaults = {
        "statics": None,
        "state_space": space,
        "integrator": make_integrator(),
        "controls": (PiecewiseLinearControl("u", KNOTS),),
        "functionals": (Mayer("goal", lambda t1, s1, theta: s1["x"]),),
        "constraints": (),
        "boundary": BoundarySpec(s0={"x": 0.0}),
        "t0": 0.0,
        "t1": 2.0,
    }
    defaults.update(overrides)
    return SingleShooting().transcribe(**defaults)


# ------------------------------------------------------------- validation


def test_abstract_transcription_raises():
    with pytest.raises(NotImplementedError):
        AbstractTranscription().transcribe(None, None, None, (), (), (), None)


def test_control_mismatch_raises():
    with pytest.raises(ValueError, match="do not match the state-space"):
        transcribe(controls=(PiecewiseLinearControl("v", KNOTS),))


def test_no_functional_raises():
    with pytest.raises(ValueError, match="At least one functional"):
        transcribe(functionals=())


def test_missing_parameter_source_raises():
    space = StateSpace(
        states=["x"],
        controls=["u"],
        parameters=["gain"],
        rhs=lambda t, s, u, theta: {"x": theta["gain"] * u["u"]},
    )
    with pytest.raises(ValueError, match="'gain' is neither an output"):
        transcribe(state_space=space)


def test_free_final_time_requires_zero_t0():
    boundary = BoundarySpec(s0={"x": 0.0}, free_final_time=True, t_f=1.0)
    with pytest.raises(ValueError, match="requires t0 = 0"):
        transcribe(boundary=boundary, t0=1.0)


def test_final_rate_unknown_series_raises():
    from gemseo_tempo.outputs import FinalRate

    with pytest.raises(ValueError, match="neither a state nor an output"):
        transcribe(
            functionals=(FinalRate("rate", "unknown"),),
        )


def test_time_integral_unknown_output_raises():
    with pytest.raises(ValueError, match="is not an output"):
        transcribe(functionals=(TimeIntegral("cum", "unknown"),))


def test_boundary_spec_validation():
    with pytest.raises(ValueError, match="no target"):
        BoundarySpec(s0={"x": 0.0}, fix_final={"x": True})
    with pytest.raises(ValueError, match="requires an initial guess"):
        BoundarySpec(s0={"x": 0.0}, free_final_time=True)
    spec = BoundarySpec(s0={"x": 0.0}, fix_final={"x": False})
    assert spec.is_initial_fixed("x")
    assert not spec.is_final_fixed("x")


def test_constraint_kind_validation():
    with pytest.raises(ValueError, match="Unknown constraint kind"):
        AbstractConstraint("bad", "leq")
    with pytest.raises(ValueError, match="Unknown aggregation"):
        PathConstraint("bad", lambda t, s, u, theta: 0.0, aggregate="max")


# ------------------------------------------------------------- behaviors


def test_discount_rate():
    """Lagrange with discount: integral of exp(-rho t) over [0, 2]."""
    problem = transcribe(
        functionals=(Lagrange("cost", lambda t, s, u, theta: 1.0, rho=0.5),),
    )
    outputs = problem.jax_function(problem.input_defaults)
    exact = (1.0 - numpy.exp(-1.0)) / 0.5
    # Scalar functionals are exposed as size-1 arrays (GEMSEO convention).
    assert float(outputs["cost"][0]) == pytest.approx(exact, abs=1e-10)


def test_control_group_constraint_matches_noads():
    """sum(controls) - 1 at every node, as in the NOADS scenario."""
    space = StateSpace(
        states=["x"],
        controls=["u1", "u2"],
        rhs=lambda t, s, u, theta: {"x": u["u1"] + u["u2"]},
    )
    params_1 = jnp.array([0.2, 0.4, 0.6, 0.8, 1.0])
    params_2 = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    problem = transcribe(
        state_space=space,
        controls=(
            PiecewiseLinearControl("u1", KNOTS, default_params=params_1),
            PiecewiseLinearControl("u2", KNOTS, default_params=params_2),
        ),
        constraints=(ControlGroupConstraint("shares", ["u1", "u2"]),),
    )
    outputs = problem.jax_function(problem.input_defaults)
    times = outputs["times"]
    expected = (
        jnp.interp(times, KNOTS, params_1) + jnp.interp(times, KNOTS, params_2) - 1.0
    )
    constraint = outputs["constraint.shares"]
    assert constraint.shape == times.shape
    assert jnp.max(jnp.abs(constraint - expected)) < 1e-12
    assert ("constraint.shares", "ineq") in problem.constraint_names


def test_path_constraint_ks_aggregation():
    """The KS envelope upper-bounds the max and converges to it."""
    problem = transcribe(
        constraints=(
            PathConstraint(
                "u_max",
                lambda t, s, u, theta: u["u"] - 0.5,
                aggregate="ks",
                ks_rho=200.0,
            ),
        ),
    )
    inputs = dict(problem.input_defaults)
    inputs["control.u"] = jnp.array([0.0, 1.0, 0.2, 0.1, 0.0])
    outputs = problem.jax_function(inputs)
    value = float(outputs["constraint.u_max"][0])
    assert value >= 0.5  # upper bound of max(u - 0.5) = 0.5
    assert value == pytest.approx(0.5, abs=0.05)


def test_free_final_time_and_free_initial():
    """t_f and initial.x enter the design space; times scale with t_f."""
    boundary = BoundarySpec(
        s0={"x": 0.3},
        fix_initial={"x": False},
        free_final_time=True,
        t_f=2.0,
    )
    problem = transcribe(
        boundary=boundary,
        t1=1.0,
        constraints=(PeriodicityConstraint("cycle", ["x"]),),
    )
    assert set(problem.design_variables) == {"control.u", "initial.x", "t_f"}
    outputs = problem.jax_function(problem.input_defaults)
    assert float(outputs["times"][-1]) == pytest.approx(2.0)
    # With u = 0 the state stays at its (free) initial value: periodic.
    assert numpy.allclose(numpy.asarray(outputs["constraint.cycle"]), 0.0)
    inputs = dict(problem.input_defaults)
    inputs["t_f"] = jnp.asarray([3.0])
    outputs = problem.jax_function(inputs)
    assert float(outputs["times"][-1]) == pytest.approx(3.0)


def test_some_initial_states_free_and_others_fixed():
    """Shooting must cope with a problem that frees only part of the state.

    A free initial value is a length-one design variable and a fixed one is
    a bare scalar. Mixing them used to make the state vector impossible to
    stack, so any problem that pinned where it started but let the rest of
    the state float failed before it ever integrated.
    """
    space = StateSpace(
        states=["x", "y"],
        controls=["u"],
        rhs=lambda t, s, u, theta: {"x": u["u"], "y": -s["y"]},
    )
    boundary = BoundarySpec(
        s0={"x": 0.0, "y": 1.0},
        fix_initial={"x": True, "y": False},
    )
    problem = transcribe(
        state_space=space,
        boundary=boundary,
        constraints=(PeriodicityConstraint("cycle", ["y"]),),
    )
    assert set(problem.design_variables) == {"control.u", "initial.y"}
    outputs = problem.jax_function(problem.input_defaults)
    # y decays from its free initial value, so the cycle does not close.
    assert numpy.asarray(outputs["constraint.cycle"]).shape == (1,)
    assert float(outputs["constraint.cycle"][0]) < 0.0


def test_periodicity_offsets_close_a_winding_state():
    """A state that winds returns to itself having gained the offset."""
    boundary = BoundarySpec(s0={"x": 0.0}, fix_initial={"x": False})
    problem = transcribe(
        boundary=boundary,
        constraints=(PeriodicityConstraint("cycle", ["x"], offsets={"x": 2.0}),),
    )
    inputs = dict(problem.input_defaults)
    # x' = u with u = 1 over [0, 2] gains exactly 2, which the offset expects.
    inputs["control.u"] = jnp.ones(5)
    outputs = problem.jax_function(inputs)
    assert float(outputs["constraint.cycle"][0]) == pytest.approx(0.0, abs=1e-9)


def test_fix_final_terminal_equality():
    boundary = BoundarySpec(s0={"x": 0.0}, fix_final={"x": True}, s1={"x": 1.5})
    problem = transcribe(boundary=boundary)
    assert ("constraint.final.x", "eq") in problem.constraint_names
    inputs = dict(problem.input_defaults)
    inputs["control.u"] = jnp.ones(5)  # x(2) = 2 with u = 1
    outputs = problem.jax_function(inputs)
    assert float(outputs["constraint.final.x"][0]) == pytest.approx(0.5, abs=1e-9)


# ------------------------------------------------- quasi-static quadrature


def test_time_integral_augmentation_ode():
    """TimeIntegral through the ODE integrator uses a quadrature state."""
    space = StateSpace(
        states=["x"],
        controls=["u"],
        outputs=["y"],
        rhs=lambda t, s, u, theta: {"x": u["u"]},
        output=lambda t, s, u, theta: {"y": s["x"]},
    )
    # x(t) = t (u = 1), so integral of y = x over [0, 2] is 2.
    problem = transcribe(
        state_space=space,
        controls=(PiecewiseLinearControl("u", KNOTS, default_params=jnp.ones(5)),),
        functionals=(TimeIntegral("cumulative.y", "y"),),
        boundary=BoundarySpec(s0={"x": 0.0}),
    )
    outputs = problem.jax_function(problem.input_defaults)
    assert float(outputs["cumulative.y"][0]) == pytest.approx(2.0, abs=1e-8)


def test_quasi_static_time_integral_trapezoid():
    """Under quasi-static execution, integrals fall back to trapezoid."""
    space = StateSpace(
        states=[],
        controls=["u"],
        outputs=["y"],
        output=lambda t, s, u, theta: {"y": u["u"] ** 2},
    )
    params = jnp.array([0.0, 1.0, 0.0, 1.0, 0.0])
    problem = transcribe(
        state_space=space,
        integrator=QuasiStaticIntegrator(time_step=0.1),
        controls=(PiecewiseLinearControl("u", KNOTS, default_params=params),),
        functionals=(
            TimeIntegral("cumulative.y", "y"),
            Lagrange("cost", lambda t, s, u, theta: u["u"] ** 2),
        ),
        boundary=BoundarySpec(s0={}),
    )
    outputs = problem.jax_function(problem.input_defaults)
    times = outputs["times"]
    series = jnp.interp(times, KNOTS, params) ** 2
    exact = jnp.trapezoid(series, x=times)
    assert float(outputs["cumulative.y"][0]) == pytest.approx(float(exact), abs=1e-12)
    assert float(outputs["cost"][0]) == pytest.approx(float(exact), abs=1e-12)
