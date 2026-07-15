# Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
# Copyright 2025 IRT Saint Exupéry, https://www.irt-saintexupery.com
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
"""JAX-traceable interpolation utilities.

Ported from NOADS ``noads/core/models/interpolation.py`` (LGPL-3.0),
trimmed to the linear and cubic paths used by gemseo-tempo.
"""

from __future__ import annotations

import jax.numpy as jnp


def interpolate_data(x, x_data, y_data, cubic: bool = False):
    """Interpolate data linearly or cubically, JAX-traceable.

    Parameters
    ----------
    x
        The evaluation points.
    x_data
        The strictly increasing data abscissae.
    y_data
        The data ordinates.
    cubic
        Whether to use a cubic spline instead of linear interpolation.

    Returns
    -------
    Array
        The interpolated values at ``x``.
    """
    if cubic:
        return CubicSpline(x_data, y_data)(x)
    return jnp.interp(x, x_data, y_data)


class CubicSpline:
    """A C2 cubic interpolating spline, JAX-traceable and differentiable.

    Reproduces the cubic path of scipy's ``InterpolatedUnivariateSpline``.
    Each piece is ``f[i](x) = a[i] + b[i] t + c[i] t^2 + d[i] t^3`` with
    ``t = x - x[i]``; the curvature coefficients ``c`` solve a tridiagonal
    system built from the chosen endpoint condition.

    Extrapolation: outside ``[x[0], x[-1]]`` the first (respectively last)
    cubic piece is evaluated as-is, so values and first two derivatives
    are continuous at the boundary but the polynomial grows cubically.
    """

    def __init__(self, x, y, endpoints: str = "not-a-knot") -> None:
        """Fit the spline through the data points.

        Parameters
        ----------
        x
            The strictly increasing data abscissae; at least 4 points.
        y
            The data ordinates, same length as ``x``.
        endpoints
            The endpoint condition, ``"not-a-knot"`` (scipy default) or
            ``"natural"`` (vanishing second derivative at both ends).

        Raises
        ------
        ValueError
            If fewer than 4 points are given or the endpoint condition is
            unknown.
        """
        x = jnp.atleast_1d(x)
        y = jnp.atleast_1d(y)
        n_data = x.shape[0]
        if n_data < 4:
            raise ValueError("A cubic spline requires at least 4 points.")
        if endpoints not in ("natural", "not-a-knot"):
            raise ValueError(f"Unknown endpoint condition {endpoints!r}.")

        h = jnp.diff(x)
        p = jnp.diff(y)

        # Special values for the first and last equations.
        zero = jnp.array([0.0])
        one = jnp.array([1.0])
        natural = endpoints == "natural"
        a00 = one if natural else jnp.array([h[1]])
        a01 = zero if natural else jnp.array([-(h[0] + h[1])])
        a02 = zero if natural else jnp.array([h[0]])
        ann = one if natural else jnp.array([h[-2]])
        # NOADS used -1 here for natural endpoints, which enforces equal
        # curvature on the last two knots instead of zero curvature at the
        # end; 0 gives the true natural condition c[n-1] = 0.
        an1 = zero if natural else jnp.array([-(h[-2] + h[-1])])
        an2 = zero if natural else jnp.array([h[-1]])

        # The tridiagonal (plus corners) system for the curvatures.
        a_matrix = jnp.diag(jnp.concatenate((a00, 2 * (h[:-1] + h[1:]), ann)))
        a_matrix += jnp.diag(jnp.concatenate((a01, h[1:])), k=1)
        a_matrix += jnp.diag(jnp.concatenate((a02, jnp.zeros(n_data - 3))), k=2)
        a_matrix += jnp.diag(jnp.concatenate((h[:-1], an1)), k=-1)
        a_matrix += jnp.diag(jnp.concatenate((jnp.zeros(n_data - 3), an2)), k=-2)

        center = 3 * (p[1:] / h[1:] - p[:-1] / h[:-1])
        s = jnp.concatenate((zero, center, zero))

        self._x = x
        self._y = y
        self._coefficients = jnp.linalg.solve(a_matrix, s)

    def __call__(self, xs):
        """Evaluate the spline.

        Parameters
        ----------
        xs
            The evaluation points.

        Returns
        -------
        Array
            The spline values at ``xs``.
        """
        x, y, c_all = self._x, self._y, self._coefficients
        index = jnp.digitize(xs, x) - 1
        index = jnp.clip(index, 0, x.shape[0] - 2)
        t = xs - x[index]
        h = jnp.diff(x)[index]
        c = c_all[index]
        c1 = c_all[index + 1]
        a = y[index]
        a1 = y[index + 1]
        b = (a1 - a) / h - (2 * c + c1) * h / 3.0
        d = (c1 - c) / (3 * h)
        return a + b * t + c * t**2 + d * t**3
