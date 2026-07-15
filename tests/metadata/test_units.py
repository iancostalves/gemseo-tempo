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
"""Tests of gemseo_tempo.metadata.units."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pint
import pytest

from gemseo_tempo.metadata import base_unit
from gemseo_tempo.metadata import convert
from gemseo_tempo.metadata import from_base
from gemseo_tempo.metadata import make_unit_shell
from gemseo_tempo.metadata import to_base


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (3.6, "km/h", 1.0),
        (1.0, "ft", 0.3048),
        (180.0, "deg", math.pi),
        (1.0, "kWh/kg", 3.6e6),
        (25.0, "degC", 298.15),
    ],
)
def test_to_base(value, unit, expected):
    assert to_base(value, unit) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("unit", ["km/h", "ft", "deg", "kWh/kg", "degC"])
def test_from_base_round_trip(unit):
    assert from_base(to_base(123.456, unit), unit) == pytest.approx(123.456, rel=1e-12)


def test_base_unit():
    assert base_unit("km/h") == "meter / second"
    assert base_unit("kWh/kg") == "meter ** 2 / second ** 2"


def test_convert():
    assert convert(100.0, "km/h", "mph") == pytest.approx(62.137119224, rel=1e-9)
    # Offset units convert through the affine map, not the pure factor.
    assert convert(212.0, "degF", "degC") == pytest.approx(100.0, rel=1e-12)


def test_convert_incommensurable():
    with pytest.raises(pint.DimensionalityError):
        convert(1.0, "m", "s")


def test_to_base_jit_safety():
    """to_base compiles once and matches eager output to 1e-12."""
    traces = []

    @jax.jit
    def speed_in_base(x):
        traces.append(1)
        return to_base(x, "km/h")

    values = jnp.array([3.6, 36.0, 360.0])
    first = speed_in_base(values)
    second = speed_in_base(values + 1.0)
    assert len(traces) == 1  # one compile for both calls
    eager = to_base(values, "km/h")
    assert jnp.max(jnp.abs(first - eager)) < 1e-12
    assert jnp.max(jnp.abs(second - to_base(values + 1.0, "km/h"))) < 1e-12


def test_make_unit_shell():
    def double_speed(inputs):
        # Operates in base units (m/s); untouched keys pass through.
        return {"v_out": 2.0 * inputs["v_in"], "label": inputs["label"]}

    shell = make_unit_shell({"v_in": "km/h"}, {"v_out": "km/h"}, double_speed)
    outputs = shell({"v_in": 10.0, "label": 7})
    assert outputs["v_out"] == pytest.approx(20.0, rel=1e-12)
    assert outputs["label"] == 7


def test_make_unit_shell_jit_safety():
    def kinetic_energy(inputs):
        return {"e": 0.5 * inputs["m"] * inputs["v"] ** 2}

    shell = make_unit_shell({"v": "km/h", "m": "t"}, {"e": "kWh"}, kinetic_energy)
    jitted = jax.jit(shell)
    inputs = {"v": jnp.asarray(36.0), "m": jnp.asarray(2.0)}
    eager = shell(inputs)["e"]
    compiled = jitted(inputs)["e"]
    # 2 t at 36 km/h (= 10 m/s): 0.5 * 2000 * 100 J = 1e5 J.
    assert eager == pytest.approx(1e5 / 3.6e6, rel=1e-12)
    assert jnp.abs(compiled - eager) < 1e-12
