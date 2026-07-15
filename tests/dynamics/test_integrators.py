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
"""Tests of gemseo_tempo.dynamics.integrators."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from gemseo_tempo.dynamics import AdjointMethod
from gemseo_tempo.dynamics import DelayedControl
from gemseo_tempo.dynamics import DiffraxIntegrator
from gemseo_tempo.dynamics import PiecewiseLinearControl
from gemseo_tempo.dynamics import QuasiStaticIntegrator
from gemseo_tempo.dynamics import StateSpace
from gemseo_tempo.dynamics import check_adjoint_compatibility

KNOTS = jnp.linspace(0.0, 5.0, 6)


def make_decay():
    return StateSpace(
        states=["x"],
        parameters=["a"],
        outputs=["energy"],
        rhs=lambda t, s, u, theta: {"x": -theta["a"] * s["x"]},
        output=lambda t, s, u, theta: {"energy": s["x"] ** 2},
    )


def make_forced():
    """x' = -x + u, an algebraic output stage for quasi-static tests."""
    return StateSpace(
        states=[],
        controls=["u"],
        parameters=["gain"],
        outputs=["y"],
        output=lambda t, s, u, theta: {"y": theta["gain"] * u["u"] + 0.1 * t},
    )


# ---------------------------------------------------------------- adjoints


@pytest.mark.parametrize("adjoint", [AdjointMethod.RECURSIVE, AdjointMethod.BACKSOLVE])
def test_forward_incompatible_adjoints(adjoint):
    with pytest.raises(ValueError, match="does not support 'forward'"):
        check_adjoint_compatibility(adjoint, "forward")


def test_compatibility_matrix():
    """Direct supports both modes; Recursive/Backsolve support reverse only."""
    # Every adjoint supports reverse.
    for adjoint in AdjointMethod:
        check_adjoint_compatibility(adjoint, "reverse")
    # Only Direct supports forward.
    check_adjoint_compatibility(AdjointMethod.DIRECT, "forward")
    # String adjoints accepted too.
    check_adjoint_compatibility("Direct", "forward")


def test_unknown_differentiation_method_raises():
    with pytest.raises(ValueError, match="Unknown differentiation method"):
        check_adjoint_compatibility(AdjointMethod.DIRECT, "sideways")


def test_integrator_check_differentiation_method():
    DiffraxIntegrator(dt0=0.1, adjoint="Direct").check_differentiation_method("forward")
    with pytest.raises(ValueError, match="does not support 'forward'"):
        DiffraxIntegrator(dt0=0.1).check_differentiation_method("forward")
    QuasiStaticIntegrator().check_differentiation_method("reverse")


def test_diffrax_integrator_requires_stepping():
    with pytest.raises(ValueError, match="Give either dt0"):
        DiffraxIntegrator()


# ---------------------------------------------------- diffrax integration


def test_exponential_decay_analytic():
    integrator = DiffraxIntegrator(rtol=1e-12, atol=1e-12, n_save=11)
    trajectory = integrator.integrate(
        make_decay(), {"x": 1.0}, (), {}, {"a": 1.3}, 0.0, 2.0
    )
    exact = jnp.exp(-1.3 * trajectory.ts)
    assert jnp.max(jnp.abs(trajectory.states["x"] - exact)) < 1e-8
    assert jnp.max(jnp.abs(trajectory.outputs["energy"] - exact**2)) < 1e-8


def test_one_compile_per_integrator_instance():
    traces = []

    def rhs(t, s, u, theta):
        traces.append(1)
        return {"x": -theta["a"] * s["x"]}

    space = StateSpace(states=["x"], parameters=["a"], rhs=rhs)
    integrator = DiffraxIntegrator(rtol=1e-8, atol=1e-8, n_save=5)
    integrator.integrate(space, {"x": 1.0}, (), {}, {"a": 1.0}, 0.0, 1.0)
    n_traces = len(traces)
    integrator.integrate(space, {"x": 2.0}, (), {}, {"a": 0.5}, 0.0, 2.0)
    assert len(traces) == n_traces  # second call reuses the compiled program


def test_delay_step_response_order_1():
    """Order-1 delay to a unit step: 1 - exp(-t / tau)."""
    tau = 0.8
    step = PiecewiseLinearControl("u", KNOTS, default_params=jnp.ones(6))
    delayed = DelayedControl(step, time_constant=tau, order=1, initial_value=0.0)
    space = StateSpace(
        states=["x"],
        controls=["u"],
        rhs=lambda t, s, u, theta: {"x": 0.0 * s["x"]},
    )
    integrator = DiffraxIntegrator(rtol=1e-10, atol=1e-10, n_save=26)
    trajectory = integrator.integrate(
        space, {"x": 0.0}, (delayed,), {"u": jnp.ones(6)}, {}, 0.0, 5.0
    )
    exact = 1.0 - jnp.exp(-trajectory.ts / tau)
    assert jnp.max(jnp.abs(trajectory.controls["u"] - exact)) < 1e-6


def test_delay_step_response_order_3():
    """Order-3 delay to a unit step: the Erlang(3) response."""
    tau = 1.5
    order = 3
    rate = order / tau
    step = PiecewiseLinearControl("u", KNOTS, default_params=jnp.ones(6))
    delayed = DelayedControl(step, time_constant=tau, order=order, initial_value=0.0)
    space = StateSpace(
        states=["x"],
        controls=["u"],
        rhs=lambda t, s, u, theta: {"x": 0.0 * s["x"]},
    )
    integrator = DiffraxIntegrator(rtol=1e-10, atol=1e-10, n_save=26)
    trajectory = integrator.integrate(
        space, {"x": 0.0}, (delayed,), {"u": jnp.ones(6)}, {}, 0.0, 5.0
    )
    kt = rate * trajectory.ts
    exact = 1.0 - jnp.exp(-kt) * (1.0 + kt + kt**2 / 2.0)
    assert jnp.max(jnp.abs(trajectory.controls["u"] - exact)) < 1e-6


def central_difference(function, x, step=1e-6):
    return (function(x + step) - function(x - step)) / (2 * step)


def test_diffrax_gradient_vs_finite_differences():
    space = make_decay()
    integrator = DiffraxIntegrator(rtol=1e-12, atol=1e-12, n_save=5)

    def final_state(a):
        trajectory = integrator.integrate(space, {"x": 1.0}, (), {}, {"a": a}, 0.0, 2.0)
        return trajectory.states["x"][-1]

    gradient = jax.grad(final_state)(1.3)
    reference = central_difference(final_state, 1.3)
    assert gradient == pytest.approx(reference, rel=1e-5)
    # Analytic check: d/da exp(-2a) = -2 exp(-2a).
    assert gradient == pytest.approx(-2.0 * math.exp(-2.6), rel=1e-8)


def test_diffrax_gradient_wrt_control_params():
    space = StateSpace(
        states=["x"],
        controls=["u"],
        rhs=lambda t, s, u, theta: {"x": u["u"]},
    )
    control = PiecewiseLinearControl("u", KNOTS)
    # Fixed stepping: adaptive controllers mis-estimate the error at the
    # control kinks, which makes the FD reference noisy; with fixed steps
    # final_state is exactly linear in the params.
    integrator = DiffraxIntegrator(dt0=0.05, n_save=6)

    def final_state(params):
        trajectory = integrator.integrate(
            space, {"x": 0.0}, (control,), {"u": params}, {}, 0.0, 5.0
        )
        return trajectory.states["x"][-1]

    params = jnp.array([0.0, 1.0, 0.5, 0.2, 0.8, 1.0])
    gradient = jax.grad(final_state)(params)
    # final_state is linear in the params, so a large FD step is exact and
    # keeps the solver noise (~atol) negligible against the step.
    for index in range(6):
        basis = jnp.zeros(6).at[index].set(1.0)
        reference = central_difference(
            lambda h, b=basis, p=params: final_state(p + h * b), 0.0, step=1e-3
        )
        assert gradient[index] == pytest.approx(reference, rel=1e-5, abs=1e-8)
    # Analytic cross-check: the integral of the interpolant gives the
    # trapezoid weights on the unit-spaced knots.
    assert jnp.allclose(gradient, jnp.array([0.5, 1.0, 1.0, 1.0, 1.0, 0.5]), atol=1e-7)


# ------------------------------------------------- quasi-static execution


def test_quasi_static_rejects_states():
    integrator = QuasiStaticIntegrator()
    with pytest.raises(ValueError, match="algebraic state spaces"):
        integrator.integrate(make_decay(), {"x": 1.0}, (), {}, {"a": 1.0}, 0.0, 5.0)


def test_quasi_static_broadcast_and_interpolation():
    space = make_forced()
    control = PiecewiseLinearControl("u", KNOTS)
    integrator = QuasiStaticIntegrator(time_step=0.5)
    params = jnp.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    trajectory = integrator.integrate(
        space, {}, (control,), {"u": params}, {"gain": 2.0}, 0.0, 5.0
    )
    assert trajectory.ts.shape == (11,)
    expected_u = jnp.interp(trajectory.ts, KNOTS, params)
    assert jnp.max(jnp.abs(trajectory.controls["u"] - expected_u)) < 1e-12
    expected_y = 2.0 * expected_u + 0.1 * trajectory.ts
    assert jnp.max(jnp.abs(trajectory.outputs["y"] - expected_y)) < 1e-12


def test_quasi_static_delay_matches_analytic():
    tau = 1.0
    step = PiecewiseLinearControl("u", KNOTS, default_params=jnp.ones(6))
    delayed = DelayedControl(step, time_constant=tau, order=1, initial_value=0.0)
    integrator = QuasiStaticIntegrator(time_step=0.25)
    trajectory = integrator.integrate(
        make_forced(), {}, (delayed,), {"u": jnp.ones(6)}, {"gain": 1.0}, 0.0, 5.0
    )
    exact = 1.0 - jnp.exp(-trajectory.ts / tau)
    assert jnp.max(jnp.abs(trajectory.controls["u"] - exact)) < 1e-4


def test_quasi_static_compiles_once():
    traces = []

    def output(t, s, u, theta):
        traces.append(1)
        return {"y": theta["gain"] * u["u"]}

    space = StateSpace(
        states=[], controls=["u"], parameters=["gain"], outputs=["y"], output=output
    )
    control = PiecewiseLinearControl("u", KNOTS)
    integrator = QuasiStaticIntegrator(time_step=1.0)
    integrator.integrate(space, {}, (control,), {"u": KNOTS}, {"gain": 1.0}, 0.0, 5.0)
    n_traces = len(traces)
    integrator.integrate(space, {}, (control,), {"u": KNOTS}, {"gain": 2.0}, 0.0, 5.0)
    assert len(traces) == n_traces


def test_quasi_static_gradient_vs_finite_differences():
    space = make_forced()
    control = PiecewiseLinearControl("u", KNOTS)
    integrator = QuasiStaticIntegrator(time_step=0.5)

    def objective(gain):
        trajectory = integrator.integrate(
            space, {}, (control,), {"u": KNOTS}, {"gain": gain}, 0.0, 5.0
        )
        return jnp.sum(trajectory.outputs["y"])

    gradient = jax.grad(objective)(2.0)
    reference = central_difference(objective, 2.0)
    assert gradient == pytest.approx(reference, rel=1e-5)


def test_quasi_static_gradient_through_delay():
    step = PiecewiseLinearControl("u", KNOTS, default_params=jnp.ones(6))
    delayed = DelayedControl(step, time_constant=1.0, order=1, initial_value=0.0)
    integrator = QuasiStaticIntegrator(time_step=0.5)
    space = make_forced()

    def objective(level):
        trajectory = integrator.integrate(
            space,
            {},
            (delayed,),
            {"u": level * jnp.ones(6)},
            {"gain": 1.0},
            0.0,
            5.0,
        )
        return jnp.sum(trajectory.outputs["y"])

    gradient = jax.grad(objective)(1.0)
    reference = central_difference(objective, 1.0)
    assert gradient == pytest.approx(reference, rel=1e-5)
