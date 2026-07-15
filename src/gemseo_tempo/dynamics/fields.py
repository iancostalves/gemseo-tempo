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
"""Gridded spatial fields, jit- and grad-safe (e.g. wind fields)."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

if TYPE_CHECKING:
    from collections.abc import Sequence

    from jax import Array


def _catmull_rom_weights(fraction):
    """Return the four Catmull-Rom weights for a unit-interval fraction.

    Parameters
    ----------
    fraction
        The position inside the interval, in [0, 1].

    Returns
    -------
    tuple
        The weights of the stencil points at offsets (-1, 0, 1, 2).
    """
    t = fraction
    return (
        0.5 * (-(t**3) + 2 * t**2 - t),
        0.5 * (3 * t**3 - 5 * t**2 + 2),
        0.5 * (-3 * t**3 + 4 * t**2 + t),
        0.5 * (t**3 - t**2),
    )


class SpatialField:
    """A field on a uniform 2D or 3D grid, interpolated smoothly.

    Linear interpolation goes through
    :func:`jax.scipy.ndimage.map_coordinates` (order 1). Cubic
    interpolation is a Catmull-Rom tensor product over the 4-point stencil
    per axis: C1, passes through the grid values, and evaluates as pure
    gathers, so it is jit- and grad-safe. Queries outside the grid are
    clamped to the boundary.
    """

    def __init__(
        self,
        axes: Sequence,
        values,
        method: str = "linear",
    ) -> None:
        """Store the grid and precompute the index mapping.

        Parameters
        ----------
        axes
            One uniformly spaced, strictly increasing 1D array of grid
            coordinates per dimension (2 or 3 of them).
        values
            The field values on the grid, shaped by the axis lengths.
        method
            The interpolation method, ``"linear"`` or ``"cubic"``.

        Raises
        ------
        ValueError
            If the method is unknown, the dimension is not 2 or 3, an axis
            is not uniformly spaced, or the value shape does not match the
            axes.
        """
        if method not in ("linear", "cubic"):
            raise ValueError(f"Unknown interpolation method {method!r}.")
        self.method = method
        self.axes = tuple(jnp.asarray(axis, dtype=float) for axis in axes)
        self.values = jnp.asarray(values, dtype=float)
        if len(self.axes) not in (2, 3):
            raise ValueError("SpatialField supports 2D and 3D grids only.")
        if self.values.shape != tuple(axis.shape[0] for axis in self.axes):
            raise ValueError(
                f"Value shape {self.values.shape} does not match the axis"
                f" lengths {tuple(axis.shape[0] for axis in self.axes)}."
            )
        self._origins = tuple(float(axis[0]) for axis in self.axes)
        self._spacings = []
        for axis in self.axes:
            steps = jnp.diff(axis)
            if not bool(jnp.allclose(steps, steps[0])):
                raise ValueError("SpatialField requires uniformly spaced axes.")
            self._spacings.append(float(steps[0]))

    def __call__(self, *coordinates) -> Array:
        """Interpolate the field at a point.

        Parameters
        ----------
        *coordinates
            One coordinate per grid dimension.

        Returns
        -------
        Array
            The interpolated value.
        """
        indices = [
            (jnp.asarray(coordinate) - origin) / spacing
            for coordinate, origin, spacing in zip(
                coordinates, self._origins, self._spacings, strict=True
            )
        ]
        if self.method == "linear":
            return map_coordinates(self.values, indices, order=1, mode="nearest")
        return self._cubic(indices)

    def _cubic(self, indices: list) -> Array:
        """Evaluate the Catmull-Rom tensor product at fractional indices.

        Parameters
        ----------
        indices
            The fractional grid indices, one per dimension.

        Returns
        -------
        Array
            The interpolated value.
        """
        n_dims = len(indices)
        sizes = self.values.shape
        bases = []
        weights = []
        for index, size in zip(indices, sizes, strict=True):
            # Clamp so the 4-point stencil stays inside the grid.
            clamped = jnp.clip(index, 0.0, size - 1.0)
            base = jnp.clip(jnp.floor(clamped).astype(int), 1, size - 3)
            bases.append(base)
            weights.append(_catmull_rom_weights(clamped - base))
        result = jnp.zeros_like(jnp.asarray(indices[0], dtype=float))
        for offsets in itertools.product((-1, 0, 1, 2), repeat=n_dims):
            weight = 1.0
            point = []
            for dim in range(n_dims):
                weight = weight * weights[dim][offsets[dim] + 1]
                point.append(bases[dim] + offsets[dim])
            result = result + weight * self.values[tuple(point)]
        return result
