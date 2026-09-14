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
"""Tests of gemseo_tempo.dynamics.state_space and .time_scaling."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from gemseo_tempo.dynamics import DiffraxIntegrator
from gemseo_tempo.dynamics import StateSpace
from gemseo_tempo.dynamics import normalize_time


def decay_rhs(t, s, u, theta):
    return {"x": -theta["a"] * s["x"]}


def decay_output(t, s, u, theta):
    return {"energy": s["x"] ** 2}


@pytest.fixture
def decay():
    return StateSpace(
        states=["x"],
        parameters=["a"],
        outputs=["energy"],
        rhs=decay_rhs,
        output=decay_output,
    )


def test_flatten_unflatten_round_trip(decay):
    flat = decay.flatten({"x": 3.0})
    assert flat.shape == (1,)
    assert decay.unflatten(flat) == {"x": 3.0}


def test_flatten_empty():
    space = StateSpace(states=[])
    assert space.flatten({}).shape == (0,)
    assert space.unflatten(jnp.zeros(0)) == {}


def test_rhs_not_implemented():
    space = StateSpace(states=["x"])
    with pytest.raises(NotImplementedError, match="Pass rhs="):
        space.rhs(0.0, {"x": 1.0}, {}, {})


def test_output_defaults_to_empty():
    space = StateSpace(states=["x"])
    assert space.output(0.0, {"x": 1.0}, {}, {}) == {}


def test_rhs_and_output_dispatch(decay):
    rates = decay.rhs(0.0, {"x": 2.0}, {}, {"a": 3.0})
    assert rates == {"x": -6.0}
    assert decay.output(0.0, {"x": 2.0}, {}, {"a": 3.0}) == {"energy": 4.0}


def test_normalize_time_metadata(decay):
    scaled = normalize_time(decay)
    assert scaled.parameters == ("a", "t_f")
    assert scaled.states == decay.states
    assert scaled.outputs == decay.outputs


def test_normalize_time_rejects_existing_t_f():
    space = StateSpace(states=["x"], parameters=["t_f"])
    with pytest.raises(ValueError, match="already has a 't_f'"):
        normalize_time(space)


def test_normalized_time_equals_unnormalized(decay):
    """Fixed t_f: integrating on tau in [0, 1] equals integrating on t."""
    t_final = 2.5
    integrator = DiffraxIntegrator(rtol=1e-10, atol=1e-10, n_save=11)
    physical = integrator.integrate(decay, {"x": 1.0}, (), {}, {"a": 0.7}, 0.0, t_final)
    scaled_space = normalize_time(decay)
    normalized = integrator.integrate(
        scaled_space, {"x": 1.0}, (), {}, {"a": 0.7, "t_f": t_final}, 0.0, 1.0
    )
    assert jnp.max(jnp.abs(physical.ts - normalized.ts * t_final)) < 1e-12
    error = jnp.max(jnp.abs(physical.states["x"] - normalized.states["x"]))
    assert error < 1e-9
    out_error = jnp.max(
        jnp.abs(physical.outputs["energy"] - normalized.outputs["energy"])
    )
    assert out_error < 1e-9


# ------------------------------------------------- controls with a rate


def ramp_rhs(t, s, u, theta):
    """Integrate the control's own time-derivative, so it can be read back."""
    return {"x": u["gamma_rate"]}


def rate_space(name: str = "gamma"):
    """Return a model whose rate of change is the control's rate."""
    return StateSpace(
        states=["x"],
        controls=[name],
        rhs=lambda t, s, u, theta: {"x": u[f"{name}_rate"]},
        control_rates=[name],
    )


def test_a_rate_can_only_be_asked_for_a_control():
    with pytest.raises(ValueError, match="not among the controls"):
        StateSpace(states=["x"], controls=["gamma"], control_rates=["theta"])


def test_the_rate_key_is_the_control_name_and_a_suffix():
    space = rate_space()
    assert space.rate_name("gamma") == "gamma_rate"
    assert space.control_rates == ("gamma",)


def test_the_integrator_hands_the_model_the_control_slope():
    """A control ramping at a known rate must arrive at that rate.

    The model integrates whatever rate it is given, so the state at the
    end is the rate times the span: a plumbing error shows up as the wrong
    slope rather than as a crash.
    """
    from gemseo_tempo.dynamics import PiecewiseLinearControl

    slope = 0.75
    times = jnp.linspace(0.0, 2.0, 5)
    control = PiecewiseLinearControl("gamma", times, default_params=slope * times)
    integrator = DiffraxIntegrator(rtol=1e-10, atol=1e-10, n_save=9)
    trajectory = integrator.integrate(
        rate_space(),
        {"x": 0.0},
        (control,),
        {"gamma": slope * times},
        {},
        0.0,
        2.0,
    )
    assert float(trajectory.states["x"][-1]) == pytest.approx(slope * 2.0, rel=1e-6)


def test_normalizing_time_rescales_the_rate_to_physical_time():
    """The factor that would otherwise be silently wrong.

    Controls are parameterized on tau, so the rate computed for a
    normalized model is per unit tau; the model underneath is written on
    physical time. Integrating the same ramp both ways has to give the
    same answer, and does so only if that conversion happens.
    """
    from gemseo_tempo.dynamics import PiecewiseLinearControl

    t_final = 3.0
    slope = 0.4
    integrator = DiffraxIntegrator(rtol=1e-10, atol=1e-10, n_save=9)

    physical_times = jnp.linspace(0.0, t_final, 5)
    physical = integrator.integrate(
        rate_space(),
        {"x": 0.0},
        (PiecewiseLinearControl("gamma", physical_times),),
        {"gamma": slope * physical_times},
        {},
        0.0,
        t_final,
    )

    tau = jnp.linspace(0.0, 1.0, 5)
    normalized = integrator.integrate(
        normalize_time(rate_space()),
        {"x": 0.0},
        (PiecewiseLinearControl("gamma", tau),),
        # The same physical ramp, expressed on tau.
        {"gamma": slope * tau * t_final},
        {"t_f": t_final},
        0.0,
        1.0,
    )
    assert float(normalized.states["x"][-1]) == pytest.approx(
        float(physical.states["x"][-1]), rel=1e-6
    )


def test_a_delayed_control_cannot_be_asked_for_its_rate():
    """The one control whose rate is not a derivative of a function of time.

    A delayed control is carried as extra state and advanced by a stage
    equation, so there is no closure over ``t`` to differentiate. Saying so
    is better than handing back the derivative of something else.
    """
    from gemseo_tempo.dynamics import PiecewiseLinearControl
    from gemseo_tempo.dynamics.controls import DelayedControl

    integrator = DiffraxIntegrator(rtol=1e-8, atol=1e-8, n_save=5)
    inner = PiecewiseLinearControl("u", jnp.linspace(0.0, 1.0, 3))
    delayed = DelayedControl(inner, time_constant=0.5, order=2)
    with pytest.raises(TypeError, match="delayed control"):
        integrator.integrate(
            rate_space("u"),
            {"x": 0.0},
            (delayed,),
            {"u": delayed.default_params},
            {},
            0.0,
            1.0,
        )
