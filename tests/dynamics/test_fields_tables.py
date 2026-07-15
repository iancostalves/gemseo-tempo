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
"""Tests of gemseo_tempo.dynamics.fields and .tables."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from gemseo_tempo.dynamics import SpatialField
from gemseo_tempo.dynamics import smooth_abs
from gemseo_tempo.dynamics import smooth_clip
from gemseo_tempo.dynamics import smooth_max
from gemseo_tempo.dynamics import smooth_min
from gemseo_tempo.dynamics import smooth_table

X_AXIS = jnp.linspace(0.0, 4.0, 9)
Y_AXIS = jnp.linspace(-1.0, 1.0, 11)


def linear_values():
    grid_x, grid_y = jnp.meshgrid(X_AXIS, Y_AXIS, indexing="ij")
    return 2.0 * grid_x - 3.0 * grid_y + 1.0


# ------------------------------------------------------------------ fields


def test_spatial_field_validation():
    with pytest.raises(ValueError, match="Unknown interpolation method"):
        SpatialField([X_AXIS, Y_AXIS], linear_values(), method="quintic")
    with pytest.raises(ValueError, match="2D and 3D grids only"):
        SpatialField([X_AXIS], jnp.zeros(9))
    with pytest.raises(ValueError, match="does not match the axis"):
        SpatialField([X_AXIS, Y_AXIS], jnp.zeros((9, 5)))
    with pytest.raises(ValueError, match="uniformly spaced"):
        SpatialField(
            [jnp.array([0.0, 1.0, 3.0, 4.0, 5.0]), Y_AXIS],
            jnp.zeros((5, 11)),
        )


@pytest.mark.parametrize("method", ["linear", "cubic"])
def test_spatial_field_reproduces_linear_function(method):
    """Both interpolants are exact on a linear field (interior points)."""
    field = SpatialField([X_AXIS, Y_AXIS], linear_values(), method=method)
    points = [(1.3, 0.21), (2.05, -0.47), (3.1, 0.68)]
    for x, y in points:
        assert field(x, y) == pytest.approx(2.0 * x - 3.0 * y + 1.0, abs=1e-10)


def test_spatial_field_matches_grid_values():
    values = jnp.sin(X_AXIS)[:, None] * jnp.cos(Y_AXIS)[None, :]
    for method in ("linear", "cubic"):
        field = SpatialField([X_AXIS, Y_AXIS], values, method=method)
        # Interior nodes are reproduced exactly by both interpolants.
        assert field(X_AXIS[3], Y_AXIS[4]) == pytest.approx(values[3, 4], abs=1e-12)


def test_spatial_field_3d():
    z_axis = jnp.linspace(0.0, 1.0, 6)
    grid = (
        2.0 * X_AXIS[:, None, None]
        - 3.0 * Y_AXIS[None, :, None]
        + 0.5 * z_axis[None, None, :]
    )
    for method in ("linear", "cubic"):
        field = SpatialField([X_AXIS, Y_AXIS, z_axis], grid, method=method)
        assert field(1.7, 0.33, 0.52) == pytest.approx(
            2.0 * 1.7 - 3.0 * 0.33 + 0.5 * 0.52, abs=1e-10
        )


@pytest.mark.parametrize("method", ["linear", "cubic"])
def test_spatial_field_gradient(method):
    """Gradients match central finite differences (and the exact slope)."""
    field = SpatialField([X_AXIS, Y_AXIS], linear_values(), method=method)
    gradient = jax.grad(lambda x, y: field(x, y), argnums=(0, 1))(1.3, 0.21)
    assert gradient[0] == pytest.approx(2.0, abs=1e-8)
    assert gradient[1] == pytest.approx(-3.0, abs=1e-8)


def test_spatial_field_cubic_gradient_smooth_data():
    values = jnp.sin(X_AXIS)[:, None] * jnp.cos(Y_AXIS)[None, :]
    field = SpatialField([X_AXIS, Y_AXIS], values, method="cubic")

    def scalar(x):
        return field(x, 0.13)

    gradient = jax.grad(scalar)(1.9)
    step = 1e-6
    reference = (scalar(1.9 + step) - scalar(1.9 - step)) / (2 * step)
    assert gradient == pytest.approx(reference, rel=1e-5)


def test_spatial_field_clamps_outside():
    field = SpatialField([X_AXIS, Y_AXIS], linear_values(), method="linear")
    inside = field(0.0, 1.0)
    assert field(-5.0, 5.0) == pytest.approx(inside)


# ------------------------------------------------------------------ tables


def test_smooth_table_passes_through_points():
    x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = jnp.array([0.0, 0.8, 1.0, 1.0, 0.2])
    table = smooth_table(x, y)
    assert jnp.max(jnp.abs(table(x) - y)) < 1e-12


def test_smooth_table_is_c2():
    """The second derivative is continuous across an interior knot."""
    x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = jnp.array([0.0, 0.8, 1.0, 1.0, 0.2])
    table = smooth_table(x, y)
    second = jax.grad(jax.grad(lambda t: table(t)))
    assert second(2.0 - 1e-7) == pytest.approx(second(2.0 + 1e-7), abs=1e-4)


def test_smooth_saturations():
    assert smooth_abs(3.0, eps=0.0) == 3.0
    assert smooth_max(5.0, 0.0, eps=0.0) == 5.0
    assert smooth_max(-5.0, 0.0, eps=0.0) == 0.0
    assert smooth_min(5.0, 1.0, eps=0.0) == 1.0
    assert smooth_clip(5.0, 0.0, 1.0, eps=0.0) == 1.0
    assert smooth_clip(-5.0, 0.0, 1.0, eps=0.0) == 0.0
    assert smooth_clip(0.5, 0.0, 1.0, eps=0.0) == 0.5


def test_smooth_saturations_are_differentiable_at_kinks():
    gradient = jax.grad(lambda x: smooth_max(x, 0.0))(0.0)
    assert gradient == pytest.approx(0.5)  # symmetric smoothing at the kink
    gradient = jax.grad(lambda x: smooth_abs(x))(0.0)
    assert gradient == 0.0
