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
"""The boat polar: how fast the boat sails at a given wind angle.

The polar is authored here from an analytic shape rather than taken from
published data, which would carry its own licence. It is separable,

.. code-block::

    V(twa, tws) = shape(|twa|) * saturation(tws)

with ``shape`` a C2 spline that is zero inside the no-go cone and peaks on
a reach, and ``saturation`` a smoothly saturating factor so the boat does
not keep accelerating without limit in strong wind.

It is a *fitted* polar, and deliberately so. Where a boat's speed comes
from — a sail working on the apparent wind, balanced against the hull's
resistance — is written out in :mod:`~gemseo_tempo.problems.sail.aero` and
drawn on the problem's physics page, but solving that balance inside the
dynamics costs about six times as much per evaluation, and this problem's
uncertainty studies integrate the trajectory once per sampled wind. The
table reproduces the same polar for a problem whose subject is the
trajectory rather than the boat.

The spline needs care near the no-go cone. Interpolating a table that
drops steeply to zero makes a C2 spline **undershoot below zero**: with a
sparse table the minimum reaches -0.12, which is a boat sailing backwards
and exactly the sort of thing an optimizer finds and exploits. Two things
prevent it: the table is dense through the transition, which brings the
undershoot down to about -0.03, and the result is clamped with
:func:`~gemseo_tempo.dynamics.tables.smooth_max`, which floors it just
above zero without introducing a kink.

It also needs care at the other end, for a reason that took a Jacobian
check to find: see :data:`_SHAPE` on why the table is interpolated against
the cosine of the wind angle rather than against the angle.
"""

from __future__ import annotations

import jax.numpy as jnp

from gemseo_tempo.dynamics.tables import smooth_max
from gemseo_tempo.dynamics.tables import smooth_table

NO_GO_ANGLE = 30.0
"""The true wind angle below which the boat cannot sail, in degrees.

Asserted here, because the polar is fitted. Where it *comes from* is the
sail's lift-to-drag ratio: a sail can only pull the boat forward while
``L sin(awa) > D cos(awa)``, so the cone closes where that stops holding.
:mod:`~gemseo_tempo.problems.sail.aero` computes it from the surrogate
coefficients, and the problem's physics page draws it.
"""

TWA_TABLE = (
    0.0,
    15.0,
    30.0,
    35.0,
    40.0,
    45.0,
    55.0,
    70.0,
    90.0,
    110.0,
    130.0,
    150.0,
    165.0,
    180.0,
)
"""The true wind angles of the polar table, in degrees.

Deliberately dense between 30 and 55 degrees: that is where the boat comes
out of the no-go cone, and a sparse table there makes the spline undershoot
into negative speeds.
"""

SPEED_TABLE = (
    0.0,
    0.0,
    0.0,
    0.10,
    0.28,
    0.42,
    0.55,
    0.66,
    0.72,
    0.72,
    0.66,
    0.58,
    0.50,
    0.45,
)
"""Boat speed as a fraction of the true wind speed, at :data:`TWA_TABLE`.

The dip towards dead downwind is what makes gybing pay: running straight
at 180 degrees gives less progress downwind than reaching at about 150 and
gybing.
"""

REFERENCE_WIND = 10.0
"""The wind speed at which the saturation factor is one, in m/s."""

_COSINE_TABLE = jnp.cos(jnp.radians(jnp.asarray(TWA_TABLE)))[::-1]
"""The table angles as cosines, ascending from dead downwind to head to wind."""

_SHAPE = smooth_table(_COSINE_TABLE, jnp.asarray(SPEED_TABLE)[::-1])
"""The polar interpolated against the *cosine* of the true wind angle.

Interpolating against the angle itself and folding with ``abs`` gives the
right values and the wrong derivatives. The polar is symmetric between
tacks, so the folded curve meets its mirror image at dead downwind, where
the table still has slope: the result is a corner at exactly 180 degrees,
in the model rather than in the physics, and it sits in the middle of a
downwind leg. It is a corner the same way the lift limit was a hardcoded
bound: not visible in any value the model reports, only in the derivatives.

The cosine is even and stationary at both ends, so composing the table with
it is symmetric between tacks by construction and smooth through dead
downwind and head to wind alike. Every tabulated point is reproduced
exactly; only the shape between them changes, and only near the ends.
"""


def speed_shape(twa_degrees):
    """Return the speed fraction at a true wind angle.

    Parameters
    ----------
    twa_degrees
        The true wind angle in degrees; the polar is symmetric between
        tacks, and taking the cosine makes it so without a fold.

    Returns
    -------
    Array
        The boat speed as a fraction of the true wind speed, never
        negative.
    """
    return smooth_max(_SHAPE(jnp.cos(jnp.radians(twa_degrees))), 0.0, 1e-6)


def wind_saturation(tws):
    """Return the factor by which the true wind speed drives the boat.

    Parameters
    ----------
    tws
        The true wind speed, in m/s.

    Returns
    -------
    Array
        A factor growing linearly in light wind and saturating in strong
        wind, equal to one at :data:`REFERENCE_WIND`.
    """
    return jnp.tanh(tws / REFERENCE_WIND) / jnp.tanh(1.0)


def boat_speed(twa_degrees, tws):
    """Return the boat speed for a wind angle and a wind speed.

    The separable form: a shape in the wind angle times the wind speed,
    saturating so the boat does not accelerate without limit.

    This is a *fitted* polar, not a force balance. What a boat's speed
    actually comes from — a sail working on the apparent wind against a
    hull resisting through the water — is written out in
    :mod:`~gemseo_tempo.problems.sail.aero` and drawn on the problem's
    physics page, but it is kept out of the dynamics on purpose. Solving
    the force balance inside the right-hand side costs six Newton steps per
    evaluation, about six times this, and the sail's uncertainty studies
    integrate the trajectory once per draw: the robust design study went
    from minutes to hours. A table that reproduces the same polar is the
    better trade for a problem whose point is the *trajectory*.

    Parameters
    ----------
    twa_degrees
        The true wind angle, in degrees.
    tws
        The true wind speed, in m/s.

    Returns
    -------
    Array
        The boat speed, in m/s, never negative.
    """
    return speed_shape(twa_degrees) * tws * wind_saturation(tws)


def downwind_vmg(twa_degrees):
    """Return the progress made directly downwind, per unit wind speed.

    Parameters
    ----------
    twa_degrees
        The true wind angle, in degrees.

    Returns
    -------
    Array
        The component of the boat speed along the downwind direction.
    """
    return speed_shape(twa_degrees) * jnp.cos(jnp.radians(180.0 - jnp.abs(twa_degrees)))


def best_vmg_angle(toward: str = "downwind") -> float:
    """Return the true wind angle giving the best speed made good.

    Speed made good is the part of the boat speed that actually shortens
    the distance to a mark lying dead upwind or dead downwind. It is what
    makes tacking and gybing pay: the boat gives up heading straight at
    the mark in exchange for enough extra speed to arrive sooner.

    Parameters
    ----------
    toward
        ``"downwind"`` for a mark dead downwind, ``"upwind"`` for one dead
        upwind.

    Returns
    -------
    float
        The magnitude of the best true wind angle, in degrees.

    Raises
    ------
    ValueError
        If ``toward`` is neither ``"upwind"`` nor ``"downwind"``.
    """
    if toward not in {"upwind", "downwind"}:
        raise ValueError(f"Unknown direction {toward!r}; use 'upwind' or 'downwind'.")
    grid = jnp.linspace(0.0, 180.0, 1801)
    # Progress toward the wind's source; downwind progress is its opposite.
    made_good = speed_shape(grid) * jnp.cos(jnp.radians(grid))
    if toward == "downwind":
        made_good = -made_good
    return float(grid[int(jnp.argmax(made_good))])


def plot_polar(axes=None):
    """Draw the polar on a polar axis.

    Parameters
    ----------
    axes
        The axis to draw on; a new polar axis when omitted.

    Returns
    -------
    Axes
        The axis drawn on.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(subplot_kw={"projection": "polar"})
    angles = jnp.linspace(0.0, 180.0, 181)
    speeds = speed_shape(angles)
    radians = jnp.radians(angles)
    axes.plot(radians, speeds, color="tab:blue")
    axes.plot(-radians, speeds, color="tab:blue")
    axes.set_theta_zero_location("N")
    axes.set_title("Boat speed as a fraction of true wind speed")
    return axes
