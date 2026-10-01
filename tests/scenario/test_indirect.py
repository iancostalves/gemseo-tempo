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
from gemseo_tempo.outputs import PeriodicityConstraint
from gemseo_tempo.outputs import TerminalConstraint
from gemseo_tempo.outputs import TimeIntegral
from gemseo_tempo.scenario import ControlAndDesign
from gemseo_tempo.scenario import DynamicScenario
from gemseo_tempo.scenario import OptimalControl
from gemseo_tempo.statics import JAXModel
from gemseo_tempo.statics import StaticsChain
from gemseo_tempo.transcriptions import BoundarySpec
from gemseo_tempo.transcriptions import HermiteSimpson
from gemseo_tempo.transcriptions import IndirectShooting
from gemseo_tempo.transcriptions import indirect_from_direct


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


def test_inequality_terminal_constraints_are_rejected():
    with pytest.raises(ValueError, match="does not support the constraint 'cap'"):
        make_scalar(
            IndirectShooting(),
            constraints=(
                TerminalConstraint(
                    "cap", lambda t1, s1, theta: s1["x"] - 1.0, kind="ineq"
                ),
            ),
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


# ------------------------------------------------- beyond the textbook case


def test_a_terminal_equality_is_enforced_through_its_multiplier():
    """The double integrator again, its targets now terminal constraints.

    A multiplier per equality replaces fixing the final states, and the
    answer is the same: u*(t) = 6 - 12t, so lam_v(0) = -12, lam_x = -24.
    """
    scenario = DynamicScenario(
        state_space=double_integrator_space(),
        integrator=integrator(),
        controls=(),
        functionals=(Lagrange("energy", lambda t, s, u, theta: u["u"] ** 2),),
        constraints=(
            TerminalConstraint("at_one", lambda t1, s1, theta: s1["x"] - 1.0),
            TerminalConstraint("at_rest", lambda t1, s1, theta: s1["v"]),
        ),
        boundary=BoundarySpec(s0={"x": 0.0, "v": 0.0}),
        transcription=IndirectShooting(),
        t0=0.0,
        t1=1.0,
    )
    result = outputs_at(scenario, solve(scenario))
    assert float(result["residual_norm"][0]) < 1e-10
    assert float(result["trajectory.costate.v"][0]) == pytest.approx(-12.0, abs=1e-3)
    assert float(result["trajectory.costate.x"][0]) == pytest.approx(-24.0, abs=1e-4)


def decaying_space():
    """x' = -x + u."""
    return StateSpace(
        states=["x"],
        controls=["u"],
        rhs=lambda t, s, u, theta: {"x": -s["x"] + u["u"]},
    )


def test_a_periodic_cycle_finds_its_own_starting_state():
    """min int (x - 1)^2 + u^2 over a cycle of x' = -x + u.

    The cycle is free to start anywhere it returns to, and the optimum is
    the steady state x = u = 1/2 with lam = -1 throughout: the initial
    transversality condition and the periodicity multiplier pin it.
    """
    scenario = DynamicScenario(
        state_space=decaying_space(),
        integrator=integrator(),
        controls=(),
        functionals=(
            Lagrange(
                "tracking", lambda t, s, u, theta: (s["x"] - 1.0) ** 2 + u["u"] ** 2
            ),
        ),
        constraints=(PeriodicityConstraint("cycle", ["x"]),),
        boundary=BoundarySpec(s0={"x": 0.0}, fix_initial={"x": False}),
        transcription=IndirectShooting(),
        t0=0.0,
        t1=1.0,
    )
    assert set(scenario.problem.design_variables) == {
        "costate.x",
        "initial.x",
        "multiplier.cycle",
    }
    result = outputs_at(scenario, solve(scenario))
    assert float(result["residual_norm"][0]) < 1e-10
    assert float(result["trajectory.x"][0]) == pytest.approx(0.5, abs=1e-4)
    assert float(result["trajectory.x"][-1]) == pytest.approx(0.5, abs=1e-4)
    assert float(result["trajectory.costate.x"][0]) == pytest.approx(-1.0, abs=1e-4)


def drifting_space(parameter="a"):
    """x' = a + u, with the drift a a parameter."""
    return StateSpace(
        states=["x"],
        controls=["u"],
        parameters=[parameter],
        rhs=lambda t, s, u, theta: {"x": theta[parameter] + u["u"]},
    )


DRIFT_COST = (
    Lagrange("energy", lambda t, s, u, theta: u["u"] ** 2),
    Mayer(
        "miss",
        lambda t1, s1, theta: 0.5 * (s1["x"] - 1.0) ** 2 + 0.5 * theta["a"] ** 2,
    ),
)
"""min int u^2 + 0.5 (x(1) - 1)^2 + 0.5 a^2 over u and the drift a.

With u constant, the cost is u^2 + 0.5 (a + u - 1)^2 + 0.5 a^2, whose
minimum is a = 0.4, u = 0.2.
"""


def test_a_design_constant_is_solved_for_with_the_trajectory():
    """dPhi/da + int dH/da dt = 0 is one more residual, and one more unknown."""
    scenario = DynamicScenario(
        state_space=drifting_space(),
        integrator=integrator(),
        controls=(),
        functionals=DRIFT_COST,
        boundary=BoundarySpec(s0={"x": 0.0}),
        transcription=IndirectShooting(design_parameters=("a",)),
        constants={"a": 0.0},
        t0=0.0,
        t1=1.0,
    )
    result = outputs_at(scenario, solve(scenario))
    assert float(result["residual_norm"][0]) < 1e-10
    assert float(result["trajectory.u"][0]) == pytest.approx(0.2, abs=1e-4)
    assert float(result["trajectory.x"][-1]) == pytest.approx(0.6, abs=1e-4)


def test_a_design_constant_feeding_the_statics_is_solved_for_too():
    """The drift is twice a constant b, computed by a statics chain.

    The stationarity is taken back through the chain, so b = a / 2 = 0.2.
    """
    doubling = JAXModel(
        function=lambda inputs: {"a": 2.0 * inputs["b"]},
        input_names=["b"],
        output_names=["a"],
        default_inputs={"b": 0.0},
        name="doubling",
    )
    scenario = DynamicScenario(
        state_space=drifting_space(),
        integrator=integrator(),
        controls=(),
        functionals=DRIFT_COST,
        boundary=BoundarySpec(s0={"x": 0.0}),
        transcription=IndirectShooting(design_parameters=("b",)),
        statics=StaticsChain([doubling], ["b"]),
        constants={"b": 0.0},
        t0=0.0,
        t1=1.0,
    )
    gemseo_scenario = solve(scenario)
    result = outputs_at(scenario, gemseo_scenario)
    assert float(result["residual_norm"][0]) < 1e-10
    design = gemseo_scenario.design_space.get_current_value(as_dict=True)
    assert float(design["constant.b"][0]) == pytest.approx(0.2, abs=1e-4)


def test_design_parameters_must_be_constants():
    with pytest.raises(ValueError, match=r"Design parameters \['wing'\]"):
        make_scalar(IndirectShooting(design_parameters=("wing",)))


def test_a_path_inequality_is_penalized_into_place():
    """Capping u at 0.25 where the free optimum is 1/3: the cap binds."""
    scenario = make_scalar(
        IndirectShooting(path_penalty=1e4, path_smoothing=1e-3),
        constraints=(PathConstraint("cap", lambda t, s, u, theta: u["u"] - 0.25),),
    )
    result = outputs_at(scenario, solve(scenario))
    assert float(result["residual_norm"][0]) < 1e-10
    assert float(numpy.max(result["trajectory.u"])) == pytest.approx(0.25, abs=5e-3)


def heading_space():
    """x' = sin(u): the rate is largest at u = pi / 2."""
    return StateSpace(
        states=["x"],
        controls=["u"],
        rhs=lambda t, s, u, theta: {"x": jnp.sin(u["u"])},
    )


def make_heading(transcription):
    """Reach x = 1 as fast as possible: the answer is t_f = 1 at u = pi / 2."""
    return DynamicScenario(
        state_space=heading_space(),
        integrator=integrator(),
        controls=(),
        functionals=(Mayer("time", lambda t1, s1, theta: t1),),
        boundary=BoundarySpec(
            s0={"x": 0.0},
            fix_final={"x": True},
            s1={"x": 1.0},
            free_final_time=True,
            t_f=1.5,
        ),
        transcription=transcription,
        t0=0.0,
        t1=1.0,
    )


def test_a_grid_finds_the_minimizer_newton_alone_cannot():
    """H = 1 + lam sin(u) is flat at u = 0, where Newton alone starts.

    The grid starts the polish near pi / 2, where the minimizer is. The
    costate guess must not be zero: there H vanishes for every control and
    nothing tells the grid, or the optimizer, which way to go.
    """
    scenario = make_heading(
        IndirectShooting(
            control_grid=33,
            control_bounds={"u": (0.0, jnp.pi)},
            costate_guess={"x": -0.5},
        )
    )
    result = outputs_at(
        scenario, solve(scenario, **{"costate.x": (-5.0, 0.0), "t_f": (0.5, 3.0)})
    )
    assert float(result["residual_norm"][0]) < 1e-10
    assert float(result["times"][-1]) == pytest.approx(1.0, abs=1e-5)
    assert float(result["trajectory.u"][0]) == pytest.approx(jnp.pi / 2, abs=1e-4)


def test_the_grid_needs_bounds_on_every_control():
    with pytest.raises(ValueError, match=r"needs bounds on controls \['u'\]"):
        make_heading(IndirectShooting(control_grid=9))


# ------------------------------------------------ from a direct solution


def test_a_direct_solution_seeds_the_costates_it_implies():
    """Solve the double integrator by collocation, then read lam off it.

    The seed is already the answer to a few percent, so the shooting that
    starts from it has almost nothing left to do.
    """
    direct = DynamicScenario(
        state_space=double_integrator_space(),
        integrator=integrator(),
        controls=(PiecewiseLinearControl("u", jnp.linspace(0.0, 1.0, 21)),),
        functionals=(Lagrange("energy", lambda t, s, u, theta: u["u"] ** 2),),
        boundary=BoundarySpec(
            s0={"x": 0.0, "v": 0.0},
            fix_final={"x": True, "v": True},
            s1={"x": 1.0, "v": 0.0},
        ),
        transcription=HermiteSimpson(num_segments=20),
        t0=0.0,
        t1=1.0,
    )
    gemseo_scenario = OptimalControl(direct).create_gemseo_scenario()
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=200)
    indirect = indirect_from_direct(
        direct, gemseo_scenario, control_bounds={"u": (-20.0, 20.0)}
    )
    seed = indirect.problem.input_defaults
    assert float(seed["costate.v"][0]) == pytest.approx(-12.0, rel=0.05)
    assert float(seed["costate.x"][0]) == pytest.approx(-24.0, rel=0.05)
    result = outputs_at(indirect, solve(indirect))
    assert float(result["residual_norm"][0]) < 1e-10
    assert float(result["trajectory.u"][0]) == pytest.approx(6.0, abs=1e-3)


def test_a_periodic_direct_solution_seeds_its_multiplier_and_start():
    """The steady cycle x = u = 1/2 again, reached from collocation."""
    direct = DynamicScenario(
        state_space=decaying_space(),
        integrator=integrator(),
        controls=(PiecewiseLinearControl("u", jnp.linspace(0.0, 1.0, 11)),),
        functionals=(
            Lagrange(
                "tracking", lambda t, s, u, theta: (s["x"] - 1.0) ** 2 + u["u"] ** 2
            ),
        ),
        constraints=(PeriodicityConstraint("cycle", ["x"]),),
        boundary=BoundarySpec(s0={"x": 0.0}, fix_initial={"x": False}),
        transcription=HermiteSimpson(num_segments=10),
        t0=0.0,
        t1=1.0,
    )
    gemseo_scenario = OptimalControl(direct).create_gemseo_scenario()
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=200)
    indirect = indirect_from_direct(
        direct, gemseo_scenario, control_bounds={"u": (-5.0, 5.0)}
    )
    seed = indirect.problem.input_defaults
    assert float(seed["initial.x"][0]) == pytest.approx(0.5, abs=1e-3)
    assert float(seed["costate.x"][0]) == pytest.approx(-1.0, abs=1e-2)
    assert float(seed["multiplier.cycle"][0]) == pytest.approx(-1.0, abs=1e-2)


def test_the_same_problem_can_be_built_again_under_another_transcription():
    indirect = make_double_integrator()
    direct = indirect.retranscribe(
        HermiteSimpson(num_segments=4),
        controls=(PiecewiseLinearControl("u", jnp.linspace(0.0, 1.0, 5)),),
    )
    assert isinstance(direct.transcription, HermiteSimpson)
    assert "control.u" in direct.problem.design_variables
    assert direct.definition["functionals"] == indirect.definition["functionals"]


def test_multiple_shooting_finds_the_same_answer_in_segments():
    """Four segments joined by continuity: the double integrator again."""
    scenario = make_double_integrator(IndirectShooting(segments=4))
    names = set(scenario.problem.design_variables)
    assert {"segment.x", "segment.v", "costate.segment.x"} <= names
    assert scenario.problem.design_variables["segment.x"].size == 3
    result = outputs_at(scenario, solve(scenario))
    assert float(result["residual_norm"][0]) < 1e-8
    # The stitched trajectory keeps each shared node once.
    assert result["times"].shape == (4 * 20 + 1,)
    assert float(result["trajectory.costate.v"][0]) == pytest.approx(-12.0, abs=1e-3)


def test_an_active_path_constraint_shapes_the_seeded_costates():
    """min int u^2 - u with u capped at 0.25, where the free optimum is 1/2.

    Transversality says lam = 0 while stationarity, 2u - 1 + lam = 0, says
    lam = 1/2 wherever the cap binds; the cap's multiplier, 1/2, is what
    reconciles them, and the warm start solves for it rather than blending
    the two into a wrong compromise.
    """
    direct = DynamicScenario(
        state_space=scalar_space(),
        integrator=integrator(),
        controls=(PiecewiseLinearControl("u", jnp.linspace(0.0, 1.0, 11)),),
        functionals=(Lagrange("gain", lambda t, s, u, theta: u["u"] ** 2 - u["u"]),),
        constraints=(PathConstraint("cap", lambda t, s, u, theta: u["u"] - 0.25),),
        boundary=BoundarySpec(s0={"x": 0.0}),
        transcription=HermiteSimpson(num_segments=10),
        t0=0.0,
        t1=1.0,
    )
    gemseo_scenario = OptimalControl(direct).create_gemseo_scenario()
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=200)
    indirect = indirect_from_direct(
        direct, gemseo_scenario, control_bounds={"u": (-1.0, 1.0)}, segments=2
    )
    seed = indirect.problem.input_defaults
    assert float(seed["costate.x"][0]) == pytest.approx(0.0, abs=1e-6)
    assert float(seed["costate.segment.x"][0]) == pytest.approx(0.0, abs=1e-6)
    assert float(seed["segment.x"][0]) == pytest.approx(0.125, abs=1e-6)


def test_a_design_objective_sets_the_scale_of_the_seeded_costates():
    """min a over x' = a - u^2 with x(0) = 0, x(1) = 1: u = 0, a = 1.

    Stationarity, -2 lam u = 0, holds for any lam at u = 0, and no final
    state is free, so only the design stationarity 1 + int lam dt = 0 can
    say that lam = -1. Without it the seed would be zero.
    """
    space = StateSpace(
        states=["x"],
        controls=["u"],
        parameters=["a"],
        rhs=lambda t, s, u, theta: {"x": theta["a"] - u["u"] ** 2},
    )
    direct = DynamicScenario(
        state_space=space,
        integrator=integrator(),
        controls=(PiecewiseLinearControl("u", jnp.linspace(0.0, 1.0, 11)),),
        functionals=(Mayer("drift", lambda t1, s1, theta: theta["a"]),),
        boundary=BoundarySpec(s0={"x": 0.0}, fix_final={"x": True}, s1={"x": 1.0}),
        transcription=HermiteSimpson(num_segments=10),
        constants={"a": 2.0},
        t0=0.0,
        t1=1.0,
    )
    gemseo_scenario = ControlAndDesign(
        direct, ["a"], bounds={"constant.a": (0.0, 5.0), "control.u": (-1.0, 1.0)}
    ).create_gemseo_scenario()
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=200)
    indirect = indirect_from_direct(direct, gemseo_scenario)
    seed = indirect.problem.input_defaults
    assert float(seed["constant.a"][0]) == pytest.approx(1.0, abs=1e-6)
    assert float(seed["costate.x"][0]) == pytest.approx(-1.0, abs=1e-6)


def test_a_direct_solution_reaches_the_dynamics_through_its_statics():
    """x' = a + u with a = 2 b from a statics chain, b = 0.2: lam = -1.2.

    The warm start evaluates the chain at the direct problem's constants to
    find the drift the dynamics saw. Reaching x(1) = 1 at least effort takes
    u = 1 - a = 0.6 throughout, and stationarity, 2 u + lam = 0, gives the
    costate. It is also handed the integrator the indirect problem is to use.
    """
    doubling = JAXModel(
        function=lambda inputs: {"a": 2.0 * inputs["b"]},
        input_names=["b"],
        output_names=["a"],
        default_inputs={"b": 0.0},
        name="doubling",
    )
    direct = DynamicScenario(
        state_space=drifting_space(),
        integrator=integrator(),
        controls=(PiecewiseLinearControl("u", jnp.linspace(0.0, 1.0, 11)),),
        functionals=(Lagrange("energy", lambda t, s, u, theta: u["u"] ** 2),),
        boundary=BoundarySpec(s0={"x": 0.0}, fix_final={"x": True}, s1={"x": 1.0}),
        transcription=HermiteSimpson(num_segments=10),
        statics=StaticsChain([doubling], ["b"]),
        constants={"b": 0.2},
        t0=0.0,
        t1=1.0,
    )
    gemseo_scenario = OptimalControl(direct).create_gemseo_scenario()
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=200)
    chosen = DiffraxIntegrator(rtol=1e-10, atol=1e-10, n_save=11)
    indirect = indirect_from_direct(
        direct,
        gemseo_scenario,
        integrator=chosen,
        control_bounds={"u": (-1.0, 1.0)},
    )
    assert indirect.definition["integrator"] is chosen
    seed = indirect.problem.input_defaults
    assert float(seed["costate.x"][0]) == pytest.approx(-1.2, abs=1e-3)


def test_a_free_final_time_seeds_the_hamiltonian_condition():
    """The brachistochrone, from a coarse collocation, nearly converged."""
    from gemseo_tempo.problems.brachistochrone import create_optimization

    direct, gemseo_scenario = create_optimization(variant="ideal", size=10)
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=200)
    indirect = indirect_from_direct(direct, gemseo_scenario, control_grid=16)
    outputs = indirect.problem.jax_function(indirect.problem.input_defaults)
    assert float(numpy.max(numpy.abs(outputs["residual"]))) < 0.05


def test_an_active_path_constraint_and_a_design_objective_seed_together():
    """min a over x' = a - u^2 with u >= 0.1 binding: a = 1.01, lam = -1.

    The floor binds everywhere, so its multiplier, 0.2, enters the design
    stationarity too, alongside the costate it shares the rows with.
    """
    space = StateSpace(
        states=["x"],
        controls=["u"],
        parameters=["a"],
        rhs=lambda t, s, u, theta: {"x": theta["a"] - u["u"] ** 2},
    )
    direct = DynamicScenario(
        state_space=space,
        integrator=integrator(),
        controls=(PiecewiseLinearControl("u", jnp.linspace(0.0, 1.0, 11)),),
        functionals=(Mayer("drift", lambda t1, s1, theta: theta["a"]),),
        constraints=(PathConstraint("floor", lambda t, s, u, theta: 0.1 - u["u"]),),
        boundary=BoundarySpec(s0={"x": 0.0}, fix_final={"x": True}, s1={"x": 1.0}),
        transcription=HermiteSimpson(num_segments=10),
        constants={"a": 2.0},
        t0=0.0,
        t1=1.0,
    )
    gemseo_scenario = ControlAndDesign(
        direct, ["a"], bounds={"constant.a": (0.0, 5.0), "control.u": (-1.0, 1.0)}
    ).create_gemseo_scenario()
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=200)
    indirect = indirect_from_direct(direct, gemseo_scenario)
    seed = indirect.problem.input_defaults
    assert float(seed["constant.a"][0]) == pytest.approx(1.01, abs=1e-4)
    assert float(seed["costate.x"][0]) == pytest.approx(-1.0, abs=1e-3)


def test_a_control_outside_the_design_space_takes_its_registry_bounds():
    from gemseo.algos.design_space import DesignSpace

    from gemseo_tempo.metadata import Variable
    from gemseo_tempo.metadata import VariableRegistry
    from gemseo_tempo.transcriptions.indirect import _control_bounds

    registry = VariableRegistry(
        [
            Variable(name="u", full_name="u", lower_bound=-2.0, upper_bound=3.0),
            Variable(name="w", full_name="w", lower_bound=0.0),
        ]
    )
    bounds = _control_bounds(DesignSpace(), registry, ["u", "w", "z"])
    assert bounds == {"u": (-2.0, 3.0)}
