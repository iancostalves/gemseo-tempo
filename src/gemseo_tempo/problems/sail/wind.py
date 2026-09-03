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
"""Synthetic wind fields over the sailing area.

:class:`~gemseo_tempo.dynamics.fields.SpatialField` interpolates a scalar,
so a wind field is a **pair** of them, one per velocity component. Each
field is generated from a closed-form expression sampled on a uniform
grid, which keeps them reproducible and lets the tests state exactly what
the field should be.

Three fields are provided: a uniform breeze, a sheared one whose strength
grows across the course, and a gyre that turns.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from gemseo_tempo.dynamics.fields import SpatialField

EXTENT = 1200.0
"""The half-width of the sampled area, in m."""

RESOLUTION = 41
"""The number of grid points along each axis."""

BASE_SPEED = 10.0
"""The nominal wind speed, in m/s."""


@dataclass
class WindField:
    """A wind field, as one interpolated component per axis."""

    east: SpatialField
    """The eastward component of the wind velocity, in m/s."""

    north: SpatialField
    """The northward component of the wind velocity, in m/s."""

    def velocity(self, x, y):
        """Return the wind velocity at a point.

        Parameters
        ----------
        x
            The eastward coordinate, in m.
        y
            The northward coordinate, in m.

        Returns
        -------
        tuple
            The eastward and northward components, in m/s.
        """
        return self.east(x, y), self.north(x, y)


def _axes():
    """Return the sampling axes shared by every field."""
    axis = jnp.linspace(-EXTENT, EXTENT, RESOLUTION)
    return axis, axis


def _build(east_values, north_values) -> WindField:
    """Wrap two sampled components into a field."""
    x_axis, y_axis = _axes()
    return WindField(
        east=SpatialField([x_axis, y_axis], east_values, method="cubic"),
        north=SpatialField([x_axis, y_axis], north_values, method="cubic"),
    )


def uniform_wind(speed: float = BASE_SPEED) -> WindField:
    """Return a steady breeze blowing towards the east.

    Parameters
    ----------
    speed
        The wind speed, in m/s.

    Returns
    -------
    WindField
        The field.
    """
    x_axis, y_axis = _axes()
    shape = (x_axis.size, y_axis.size)
    return _build(jnp.full(shape, speed), jnp.zeros(shape))


def sheared_wind(speed: float = BASE_SPEED, shear: float = 0.4) -> WindField:
    """Return an eastward breeze that strengthens towards the north.

    Parameters
    ----------
    speed
        The wind speed at the centre of the area, in m/s.
    shear
        The relative change in speed over one half-width.

    Returns
    -------
    WindField
        The field.
    """
    x_axis, y_axis = _axes()
    factor = 1.0 + shear * (y_axis / EXTENT)
    east = speed * jnp.broadcast_to(factor[None, :], (x_axis.size, y_axis.size))
    return _build(east, jnp.zeros_like(east))


def gyre_wind(speed: float = BASE_SPEED) -> WindField:
    """Return a rotating wind field.

    The velocity derives from a streamfunction proportional to
    ``sin(pi x / L) sin(pi y / L)``, so the field turns about the centre of
    the area and has no divergence.

    Parameters
    ----------
    speed
        The peak wind speed, in m/s.

    Returns
    -------
    WindField
        The field.
    """
    x_axis, y_axis = _axes()
    grid_x, grid_y = jnp.meshgrid(x_axis, y_axis, indexing="ij")
    angle_x = jnp.pi * grid_x / EXTENT
    angle_y = jnp.pi * grid_y / EXTENT
    east = -speed * jnp.sin(angle_x) * jnp.cos(angle_y)
    north = speed * jnp.cos(angle_x) * jnp.sin(angle_y)
    return _build(east, north)


WIND_FIELDS = {
    "uniform": uniform_wind,
    "sheared": sheared_wind,
    "gyre": gyre_wind,
}
"""The available wind fields, by name."""


def create_wind(name: str = "uniform", **settings) -> WindField:
    """Return a named wind field.

    Parameters
    ----------
    name
        One of :data:`WIND_FIELDS`.
    **settings
        Settings forwarded to the generator.

    Returns
    -------
    WindField
        The field.

    Raises
    ------
    ValueError
        If the name is unknown.
    """
    if name not in WIND_FIELDS:
        raise ValueError(
            f"Unknown wind field {name!r}; use one of {sorted(WIND_FIELDS)}."
        )
    return WIND_FIELDS[name](**settings)
