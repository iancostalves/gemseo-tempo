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
"""Tests of the indirect single-shooting transcription."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy
import pytest

from gemseo_tempo.dynamics import DiffraxIntegrator
from gemseo_tempo.dynamics import PiecewiseLinearControl
from gemseo_tempo.dynamics import StateSpace
from gemseo_tempo.outputs import FinalRate
from gemseo_tempo.outputs import Lagrange
from gemseo_tempo.outputs import Mayer
from gemseo_tempo.outputs import PathConstraint
from gemseo_tempo.outputs import TimeIntegral
from gemseo_tempo.scenario import DynamicScenario
from gemseo_tempo.scenario import OptimalControl
from gemseo_tempo.statics import JAXModel
from gemseo_tempo.statics import StaticsChain
from gemseo_tempo.transcriptions import BoundarySpec
from gemseo_tempo.transcriptions import IndirectShooting


def integrator():
    return DiffraxIntegrator(rtol=1e-12, atol=1e-12, n_save=21)


def scalar_space():
    """x' = u, used with a terminal miss cost (free final state)."""
    return StateSpace(
        states=["x"], controls=["u"], rhs=lambda t, s, u, theta: {"x": u["u"]}
    )


def double_integrator_space():
    """x' = v, v' = u: a genuinely 2x2 dynamics Jacobian."""
    return StateSpace(
        states=["x", "v"],
        controls=["u"],
        rhs=lambda t, s, u, theta: {"x": s["v"], "v": u["u"]},
    )


def make_scalar(transcription=None, **kwargs):
    """min int u^2 dt + 0.5 (x(1) - 1)^2 with x(0) = 0; optimum u = 1/3."""
    return DynamicScenario(
        state_space=scalar_space(),
        integrator=integrator(),
        controls=(),
        functionals=(
            Lagrange("energy", lambda t, s, u, theta: u["u"] ** 2),
            Mayer("miss", lambda t1, s1, theta: 0.5 * (s1["x"] - 1.0) ** 2),
        ),
        boundary=BoundarySpec(s0={"x": 0.0}),
        transcription=transcription or IndirectShooting(),
        t0=0.0,
        t1=1.0,
        **kwargs,
    )


def make_double_integrator(transcription=None):
    """min int u^2 dt with x(0)=v(0)=0, x(1)=1, v(1)=0; u* = 6 - 12t."""
    return DynamicScenario(
        state_space=double_integrator_space(),
        integrator=integrator(),
        controls=(),
        functionals=(Lagrange("energy", lambda t, s, u, theta: u["u"] ** 2),),
        boundary=BoundarySpec(
            s0={"x": 0.0, "v": 0.0},
            fix_final={"x": True, "v": True},
            s1={"x": 1.0, "v": 0.0},
        ),
        transcription=transcription or IndirectShooting(),
        t0=0.0,
        t1=1.0,
    )


def solve(scenario, **bounds):
    formulation = OptimalControl(scenario, bounds=bounds or None)
    gemseo_scenario = formulation.create_gemseo_scenario()
    gemseo_scenario.execute(algo_name="L-BFGS-B", max_iter=200)
    return gemseo_scenario


def outputs_at(scenario, gemseo_scenario):
    """Evaluate the transcription at the converged design point."""
    values = gemseo_scenario.design_space.get_current_value(as_dict=True)
    return scenario.problem.jax_function(
        {
            **scenario.problem.input_defaults,
            **{k: jnp.asarray(v) for k, v in values.items()},
        }
    )


# --------------------------------------------------------------- analytics


def test_transversality_matches_analytic_solution():
    """Free final state: lam(1) = dphi/dx pins u = 1/(2 + T) = 1/3."""
    scenario = make_scalar()
    gemseo_scenario = solve(scenario, **{"costate.x": (-5.0, 5.0)})
    result = outputs_at(scenario, gemseo_scenario)
    assert float(result["trajectory.costate.x"][0]) == pytest.approx(
        -2.0 / 3.0, abs=1e-7
    )
    assert float(result["trajectory.u"][0]) == pytest.approx(1.0 / 3.0, abs=1e-7)
    assert float(result["trajectory.x"][-1]) == pytest.approx(1.0 / 3.0, abs=1e-7)
    assert float(result["residual_norm"][0]) < 1e-9
    # J = 1/9 + 0.5 * (2/3)^2 = 1/3.
    assert float(result["ocp_cost"][0]) == pytest.approx(1.0 / 3.0, abs=1e-7)


def test_double_integrator_minimum_energy():
    """Fixed final states on both components: u*(t) = 6 - 12t, J* = 12."""
    scenario = make_double_integrator()
    gemseo_scenario = solve(
        scenario, **{"costate.x": (-50.0, 50.0), "costate.v": (-50.0, 50.0)}
    )
    result = outputs_at(scenario, gemseo_scenario)
    times = numpy.asarray(result["times"])
    assert float(result["residual_norm"][0]) < 1e-9
    assert (
        numpy.max(
            numpy.abs(numpy.asarray(result["trajectory.u"]) - (6.0 - 12.0 * times))
        )
        < 1e-7
    )
    assert (
        numpy.max(
            numpy.abs(
                numpy.asarray(result["trajectory.x"])
                - (3.0 * times**2 - 2.0 * times**3)
            )
        )
        < 1e-7
    )
    assert float(result["ocp_cost"][0]) == pytest.approx(12.0, abs=1e-7)
    # Constant costate on x, linear on v.
    assert float(result["trajectory.costate.x"][0]) == pytest.approx(-24.0, abs=1e-6)
    assert float(result["trajectory.costate.v"][0]) == pytest.approx(-12.0, abs=1e-6)


# ------------------------------------------------------------------ AD mode


def test_forward_and_reverse_modes_agree():
    """The Jacobian form of the costate equation runs in either mode."""
    reverse = make_double_integrator(IndirectShooting(differentiation_method="reverse"))
    forward = make_double_integrator(IndirectShooting(differentiation_method="forward"))
    guess = {
        "costate.x": jnp.asarray([-20.0]),
        "costate.v": jnp.asarray([-10.0]),
    }
    reverse_out = reverse.problem.jax_function(
        {**reverse.problem.input_defaults, **guess}
    )
    forward_out = forward.problem.jax_function(
        {**forward.problem.input_defaults, **guess}
    )
    for name in ("trajectory.x", "trajectory.v", "trajectory.costate.v", "residual"):
        assert (
            numpy.max(
                numpy.abs(
                    numpy.asarray(reverse_out[name]) - numpy.asarray(forward_out[name])
                )
            )
            < 1e-10
        ), name


def test_unknown_differentiation_method_raises():
    with pytest.raises(ValueError, match="Unknown differentiation method"):
        IndirectShooting(differentiation_method="sideways")


# ------------------------------------------------------------- optimal control


def test_analytic_and_newton_optimal_control_agree():
    """u* = -lam/2 by hand equals the inner stationarity solve."""
    analytic = make_scalar(
        IndirectShooting(u_star=lambda t, s, lam, theta: {"u": -0.5 * lam["x"]})
    )
    newton = make_scalar(IndirectShooting())
    guess = {"costate.x": jnp.asarray([-0.4])}
    analytic_out = analytic.problem.jax_function(
        {**analytic.problem.input_defaults, **guess}
    )
    newton_out = newton.problem.jax_function({**newton.problem.input_defaults, **guess})
    for name in ("trajectory.u", "trajectory.x", "residual"):
        assert (
            numpy.max(
                numpy.abs(
                    numpy.asarray(analytic_out[name]) - numpy.asarray(newton_out[name])
                )
            )
            < 1e-9
        ), name


def test_control_bounds_are_smoothly_saturated():
    """The unconstrained optimum 1/3 is pulled back inside [-0.1, 0.1]."""
    scenario = make_scalar(
        IndirectShooting(
            control_bounds={"u": (-0.1, 0.1)}, smooth_bounds_sharpness=50.0
        )
    )
    result = scenario.problem.jax_function(
        {
            **scenario.problem.input_defaults,
            "costate.x": jnp.asarray([-2.0 / 3.0]),
        }
    )
    controls = numpy.asarray(result["trajectory.u"])
    assert numpy.all(controls <= 0.1 + 1e-2)
    assert numpy.all(controls >= -0.1 - 1e-2)


# ------------------------------------------------------------------ gradients


def test_residual_gradient_vs_finite_differences():
    scenario = make_scalar()
    defaults = scenario.problem.input_defaults

    def residual_norm(costate):
        outputs = scenario.problem.jax_function(
            {
                **defaults,
                "costate.x": jnp.asarray([costate]),
            }
        )
        return outputs["residual_norm"][0]

    gradient = float(jax.grad(residual_norm)(-0.4))
    step = 1e-6
    reference = (residual_norm(-0.4 + step) - residual_norm(-0.4 - step)) / (2 * step)
    assert gradient == pytest.approx(float(reference), rel=1e-5)


# ----------------------------------------------------------------- multistart


def test_multistart_recovers_the_optimum():
    """Costate guesses are sensitive; MultiStart needs no extra machinery."""
    scenario = make_scalar()
    formulation = OptimalControl(scenario, bounds={"costate.x": (-5.0, 5.0)})
    gemseo_scenario = formulation.create_gemseo_scenario()
    gemseo_scenario.execute(
        algo_name="MultiStart",
        n_start=3,
        opt_algo_name="L-BFGS-B",
        doe_algo_name="LHS",
        max_iter=200,
    )
    result = outputs_at(scenario, gemseo_scenario)
    assert float(result["residual_norm"][0]) < 1e-9
    assert float(result["trajectory.u"][0]) == pytest.approx(1.0 / 3.0, abs=1e-6)


# --------------------------------------------------------------- scope limits


def test_free_final_time_adds_the_hamiltonian_residual():
    problem = IndirectShooting().transcribe(
        None,
        scalar_space(),
        integrator(),
        (),
        (
            Lagrange("energy", lambda t, s, u, theta: u["u"] ** 2),
            Mayer("miss", lambda t1, s1, theta: 0.5 * (s1["x"] - 1.0) ** 2),
        ),
        (),
        BoundarySpec(s0={"x": 0.0}, free_final_time=True, t_f=1.0),
    )
    assert set(problem.design_variables) == {"costate.x", "t_f"}
    result = problem.jax_function(problem.input_defaults)
    # One residual per state, plus the H(t1) condition.
    assert result["residual"].shape == (2,)
    assert float(result["times"][-1]) == pytest.approx(1.0)


def test_constraints_are_rejected():
    with pytest.raises(ValueError, match="does not support explicit constraints"):
        make_scalar(
            IndirectShooting(),
            constraints=(PathConstraint("u_max", lambda t, s, u, theta: u["u"] - 1.0),),
        )


def test_final_rate_is_rejected():
    space = StateSpace(
        states=["x"],
        controls=["u"],
        outputs=["y"],
        rhs=lambda t, s, u, theta: {"x": u["u"]},
        output=lambda t, s, u, theta: {"y": s["x"]},
    )
    with pytest.raises(ValueError, match="not part of a Bolza"):
        IndirectShooting().transcribe(
            None,
            space,
            integrator(),
            (),
            (FinalRate("rate", "y"),),
            (),
            BoundarySpec(s0={"x": 0.0}),
        )


def test_free_initial_state_is_rejected():
    with pytest.raises(ValueError, match="fix_initial=False"):
        IndirectShooting().transcribe(
            None,
            scalar_space(),
            integrator(),
            (),
            (),
            (),
            BoundarySpec(s0={"x": 0.0}, fix_initial={"x": False}),
        )


def test_control_mismatch_is_rejected():
    with pytest.raises(ValueError, match="do not match the state-space"):
        IndirectShooting().transcribe(
            None,
            scalar_space(),
            integrator(),
            (PiecewiseLinearControl("v", jnp.linspace(0.0, 1.0, 5)),),
            (),
            (),
            BoundarySpec(s0={"x": 0.0}),
        )


def test_free_final_time_requires_zero_t0():
    with pytest.raises(ValueError, match="requires t0 = 0"):
        IndirectShooting().transcribe(
            None,
            scalar_space(),
            integrator(),
            (),
            (),
            (),
            BoundarySpec(s0={"x": 0.0}, free_final_time=True, t_f=1.0),
            t0=1.0,
        )


def test_split_assembly_is_unsupported():
    with pytest.raises(ValueError, match="does not support the split"):
        make_scalar(IndirectShooting(), assembly="split")


def test_statics_chain_time_integral_and_constants():
    """A statics-fed parameter, a direct constant, and a TimeIntegral cost."""
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
        parameters=["c", "g"],
        outputs=["y"],
        rhs=lambda t, s, u, theta: {"x": -theta["c"] * s["x"] + theta["g"] * u["u"]},
        output=lambda t, s, u, theta: {"y": s["x"] ** 2},
    )
    problem = IndirectShooting().transcribe(
        StaticsChain([gain], ["k"]),
        space,
        integrator(),
        (),
        (
            TimeIntegral("cumulative.y", "y"),
            Lagrange("energy", lambda t, s, u, theta: u["u"] ** 2),
            Mayer("miss", lambda t1, s1, theta: 0.5 * s1["x"] ** 2),
        ),
        (),
        BoundarySpec(s0={"x": 1.0}),
        constants={"k": 0.5, "g": 1.0},
        t0=0.0,
        t1=1.0,
    )
    # c comes from the statics chain, g straight from the constants.
    assert "constant.k" in problem.input_defaults
    assert "constant.g" in problem.input_defaults
    assert "constant.c" not in problem.input_defaults
    result = problem.jax_function(problem.input_defaults)
    assert result["residual"].shape == (1,)
    assert numpy.isfinite(float(result["ocp_cost"][0]))
    assert result["trajectory.y"].shape == result["times"].shape


def test_state_space_without_controls():
    """Pure drift: the Hamiltonian has no control to minimize."""
    space = StateSpace(
        states=["x"],
        parameters=["a"],
        rhs=lambda t, s, u, theta: {"x": -theta["a"] * s["x"]},
    )
    problem = IndirectShooting().transcribe(
        None,
        space,
        integrator(),
        (),
        (Mayer("miss", lambda t1, s1, theta: 0.5 * (s1["x"] - 0.5) ** 2),),
        (),
        BoundarySpec(s0={"x": 1.0}),
        constants={"a": 0.7},
        t0=0.0,
        t1=1.0,
    )
    result = problem.jax_function(problem.input_defaults)
    # Transversality alone: lam(1) = x(1) - 0.5 with x(1) = exp(-0.7).
    expected = numpy.exp(-0.7) - 0.5
    assert float(result["residual"][0]) == pytest.approx(-expected, abs=1e-8)


def test_matching_control_parameterizations_are_accepted():
    """Passing controls is allowed as long as the names line up."""
    problem = IndirectShooting().transcribe(
        None,
        scalar_space(),
        integrator(),
        (PiecewiseLinearControl("u", jnp.linspace(0.0, 1.0, 5)),),
        (Lagrange("energy", lambda t, s, u, theta: u["u"] ** 2),),
        (),
        BoundarySpec(s0={"x": 0.0}, fix_final={"x": True}, s1={"x": 1.0}),
    )
    assert set(problem.design_variables) == {"costate.x"}
