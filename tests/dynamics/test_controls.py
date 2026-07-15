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
"""Tests of gemseo_tempo.dynamics.controls and .interpolation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from gemseo_tempo.dynamics import AbstractControl
from gemseo_tempo.dynamics import CubicControl
from gemseo_tempo.dynamics import DelayedControl
from gemseo_tempo.dynamics import ParametricControl
from gemseo_tempo.dynamics import PiecewiseLinearControl
from gemseo_tempo.dynamics.interpolation import CubicSpline
from gemseo_tempo.dynamics.interpolation import interpolate_data

TIMES = jnp.array([0.0, 1.0, 2.0, 3.0])
VALUES = jnp.array([0.0, 1.0, 0.0, 2.0])


def test_abstract_control_call_raises():
    control = AbstractControl("u", jnp.zeros(2))
    assert control.n_params == 2
    with pytest.raises(NotImplementedError):
        control(0.0, jnp.zeros(2))


def test_piecewise_linear_control():
    control = PiecewiseLinearControl("u", TIMES)
    assert control.n_params == 4
    assert jnp.all(control.default_params == 0.0)
    assert control(0.5, VALUES) == 0.5
    assert control(2.5, VALUES) == 1.0
    # Constant beyond the boundary nodes.
    assert control(-1.0, VALUES) == 0.0
    assert control(4.0, VALUES) == 2.0


def test_cubic_control_passes_through_nodes():
    control = CubicControl("u", TIMES, default_params=VALUES)
    values = jax.vmap(lambda t: control(t, VALUES))(TIMES)
    assert jnp.max(jnp.abs(values - VALUES)) < 1e-12


def test_cubic_control_is_differentiable():
    control = CubicControl("u", TIMES)
    gradient = jax.grad(lambda p: control(1.5, p))(VALUES)
    assert gradient.shape == (4,)
    assert jnp.all(jnp.isfinite(gradient))


def test_parametric_control():
    control = ParametricControl(
        "u", lambda t, p: p[0] / (1 + jnp.exp(-p[1] * t)), [1.0, 2.0]
    )
    assert control(0.0, jnp.array([1.0, 2.0])) == 0.5
    assert control.n_params == 2


def test_delayed_control_exposes_inner():
    inner = PiecewiseLinearControl("u", TIMES, default_params=VALUES)
    delayed = DelayedControl(inner, time_constant=1.0, order=3)
    assert delayed.name == "u"
    assert delayed.n_params == 4
    assert delayed(0.5, VALUES) == inner(0.5, VALUES)


def test_delayed_control_initial_states():
    inner = PiecewiseLinearControl("u", TIMES, default_params=VALUES)
    delayed = DelayedControl(inner, time_constant=1.0, order=2)
    steady = delayed.initial_states(1.0, VALUES)
    assert steady.shape == (2,)
    assert jnp.all(steady == inner(1.0, VALUES))
    forced = DelayedControl(inner, time_constant=1.0, order=2, initial_value=0.3)
    assert jnp.all(forced.initial_states(1.0, VALUES) == 0.3)


def test_delayed_control_stage_rhs():
    inner = PiecewiseLinearControl("u", TIMES, default_params=VALUES)
    delayed = DelayedControl(inner, time_constant=2.0, order=2)
    # Each stage relaxes toward its driver at rate order / time_constant.
    stages = jnp.array([0.5, 0.25])
    rates = delayed.stage_rhs(0.5, stages, VALUES)  # input = 0.5
    assert rates == pytest.approx([0.0, 0.25])


def test_interpolate_data_linear_and_cubic():
    x = jnp.linspace(0.0, 3.0, 7)
    linear = interpolate_data(x, TIMES, VALUES)
    assert linear[1] == 0.5
    cubic = interpolate_data(TIMES, TIMES, VALUES, cubic=True)
    assert jnp.max(jnp.abs(cubic - VALUES)) < 1e-12


def test_cubic_spline_reproduces_scipy():
    """The not-a-knot spline matches scipy.interpolate.CubicSpline."""
    from scipy.interpolate import CubicSpline as ScipyCubicSpline

    x = jnp.array([0.0, 0.7, 1.5, 2.2, 3.0])
    y = jnp.array([1.0, -0.5, 0.3, 2.0, 0.0])
    ours = CubicSpline(x, y)
    theirs = ScipyCubicSpline(x, y)  # not-a-knot by default
    queries = jnp.linspace(0.0, 3.0, 50)
    assert jnp.max(jnp.abs(ours(queries) - theirs(queries))) < 1e-10


def test_cubic_spline_natural_endpoints():
    x = jnp.array([0.0, 1.0, 2.0, 3.0])
    y = jnp.array([0.0, 1.0, 4.0, 9.0])
    spline = CubicSpline(x, y, endpoints="natural")
    # Natural spline: zero second derivative at both ends.
    second = jax.grad(jax.grad(lambda t: spline(t)))
    assert abs(second(0.0)) < 1e-10
    assert abs(second(3.0)) < 1e-10


def test_cubic_spline_validation():
    with pytest.raises(ValueError, match="at least 4 points"):
        CubicSpline(jnp.array([0.0, 1.0, 2.0]), jnp.array([0.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="Unknown endpoint condition"):
        CubicSpline(TIMES, VALUES, endpoints="clamped")
