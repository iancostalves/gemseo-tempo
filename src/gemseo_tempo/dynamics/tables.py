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
"""Smooth tables and saturations.

Piecewise-linear lookup tables (ubiquitous in system-dynamics models such
as World3) have kinks that stall gradient-based optimizers; these helpers
replace them with C2 splines and smooth saturations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from gemseo_tempo.dynamics.interpolation import CubicSpline

if TYPE_CHECKING:
    from collections.abc import Callable


def smooth_table(x_pts, y_pts, endpoints: str = "natural") -> Callable:
    """Build a C2 spline through table points.

    Extrapolation: outside the table range the edge cubic pieces are
    evaluated as-is, so the function stays C2 at the boundary but grows
    cubically; keep table ranges wide enough to cover the reachable
    inputs, or compose with :func:`smooth_clip`.

    Parameters
    ----------
    x_pts
        The strictly increasing table abscissae; at least 4.
    y_pts
        The table ordinates.
    endpoints
        The spline endpoint condition, ``"natural"`` (default, tames
        extrapolation) or ``"not-a-knot"``.

    Returns
    -------
    Callable
        The JAX-traceable table function.
    """
    return CubicSpline(x_pts, y_pts, endpoints=endpoints)


def smooth_abs(x, eps: float = 1e-4):
    """Smooth absolute value, ``sqrt(x**2 + eps)``.

    Parameters
    ----------
    x
        The input.
    eps
        The smoothing parameter; the error at 0 is ``sqrt(eps)``.

    Returns
    -------
    Array
        The smooth absolute value, always at least ``sqrt(eps)``.
    """
    return jnp.sqrt(x**2 + eps)


def smooth_max(x, bound, eps: float = 1e-4):
    """Smooth elementwise maximum of the input and a bound.

    Parameters
    ----------
    x
        The input.
    bound
        The bound.
    eps
        The smoothing parameter of the underlying :func:`smooth_abs`.

    Returns
    -------
    Array
        The smooth maximum.
    """
    return 0.5 * (x + bound + smooth_abs(x - bound, eps))


def smooth_min(x, bound, eps: float = 1e-4):
    """Smooth elementwise minimum of the input and a bound.

    Parameters
    ----------
    x
        The input.
    bound
        The bound.
    eps
        The smoothing parameter of the underlying :func:`smooth_abs`.

    Returns
    -------
    Array
        The smooth minimum.
    """
    return 0.5 * (x + bound - smooth_abs(x - bound, eps))


def smooth_clip(x, lower, upper, eps: float = 1e-4):
    """Smooth clip of the input between two bounds.

    Parameters
    ----------
    x
        The input.
    lower
        The lower bound.
    upper
        The upper bound.
    eps
        The smoothing parameter of the underlying :func:`smooth_abs`.

    Returns
    -------
    Array
        The smoothly clipped value.
    """
    return smooth_min(smooth_max(x, lower, eps), upper, eps)
