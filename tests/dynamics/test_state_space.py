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
