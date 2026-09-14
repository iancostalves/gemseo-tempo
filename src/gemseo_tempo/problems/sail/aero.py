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
"""What actually drives the boat: a sail, an apparent wind, and a hull.

The boat speed used to be read off a table in the *true* wind angle. A sail
does not feel the true wind. It feels the **apparent** wind, which is the
true wind minus the boat's own velocity, and the boat's own velocity is
what the table was supposed to give — so the physics is circular, and the
table was the circle already solved and then forgotten.

This module unrolls it. With the water static, so that the boat's velocity
over the ground is also its velocity through the water:

.. code-block::

    v_A = v_W - v_B                         the velocity triangle
    L   = 0.5 rho_air S C_L(alpha) |v_A|^2
    D   = 0.5 rho_air S C_D(alpha) |v_A|^2
    F   = L sin(awa) - D cos(awa)           driving force along the heading
    R   = 0.5 rho_water S_h C_h V^2         hull resistance
    F(V) = R(V)                             solved for V

It is the same triangle an aircraft flies in a wind gradient, written
``v_K = v_W + v_A`` in the dynamic-soaring literature and drawn on
:doc:`../../problems/albatross/physics`; the sail and the albatross differ
in which side of it is known, not in the construction.

The coefficients are **surrogates**, not a measured sail. ``C_L`` is the
flat-plate ``C_Lmax sin(2 alpha)``, which peaks at forty-five degrees and
falls away either side, and ``C_D`` is the usual parabolic polar on top of
a base drag. What this buys is that the things the old table asserted now
*follow*: the no-go cone is where lift can no longer beat drag along the
heading, rather than three zeros written into a table, and running dead
downwind is slow because the sail is stalled rather than because the table
says so.
"""

from __future__ import annotations

import jax.numpy as jnp

AIR_DENSITY = 1.225
"""The density of air, in kg/m**3."""

WATER_DENSITY = 1025.0
"""The density of sea water, in kg/m**3."""

SAIL_AREA = 28.0
"""The sail area, in m**2: a cruising boat of about ten metres."""

HULL_AREA = 20.0
"""The wetted area of the hull, in m**2."""

HULL_DRAG_COEFFICIENT = 0.0040
"""The hull's skin-friction coefficient on its wetted area.

The value for a smooth hull at the Reynolds numbers a ten-metre boat
reaches. It sets the boat's speed scale against the sail's driving force,
and it is where this model is thinnest: a real hull's resistance climbs
steeply as it approaches hull speed, and skin friction alone does not.
"""

LIFT_MAX = 1.5
"""The peak lift coefficient of the sail, at forty-five degrees."""

BASE_DRAG = 0.10
"""The drag coefficient of the sail and rig at no lift."""

INDUCED_DRAG_FACTOR = 0.14
"""How much drag the lift costs, the ``k`` of ``C_D = C_D0 + k C_L**2``."""

TRIM_FRACTION = 0.5
"""How much of the apparent wind angle the sheeting angle takes up.

A sail is trimmed to the apparent wind and what is left over is the angle
of attack, ``alpha = (1 - TRIM_FRACTION) * awa``. A half is the trimmer's
own rule of thumb — set the sail to bisect the angle between the
centreline and the apparent wind — and it is also the only value that
keeps the model smooth, which is a pleasant coincidence and was not the
first value tried.

The reason is that ``alpha`` enters through ``sin(2 alpha)``, so a half
turns it into ``sin(awa)``: a function of the apparent wind angle with the
same period as the angle itself. Any other fraction gives ``sin(2(1-f)
awa)``, which does not close over a full turn, and the mismatch lands at
dead downwind where the angle wraps from ``+180`` to ``-180``. Trimming at
0.72 put a corner there of 0.043 m/s per degree — sixteen times the corner
the old table was rewritten to remove, in the middle of the downwind leg,
and invisible in every value the model reports.
"""


def angle_of_attack(awa_degrees):
    """Return the sail's angle of attack, in degrees.

    Signed, deliberately, and this is the second time this problem has
    turned on it. Folding the angle with ``abs`` to make the two tacks
    symmetric gives every value right and one derivative wrong: the folded
    curve meets its mirror image at dead downwind with slope still on it,
    putting a corner in the model at exactly the angle a downwind leg
    spends its time near. Kept signed, the lift is odd in the angle and the
    driving force even, so symmetry between tacks comes out of the algebra
    and the polar is smooth through dead downwind — the same conclusion the
    old table reached by interpolating against the cosine.

    Parameters
    ----------
    awa_degrees
        The apparent wind angle, in degrees, signed by tack.

    Returns
    -------
    Array
        The angle between the sail's chord and the apparent wind.
    """
    return (1.0 - TRIM_FRACTION) * awa_degrees


def lift_coefficient(alpha_degrees):
    """Return the sail's lift coefficient at an angle of attack.

    The flat-plate surrogate ``C_Lmax sin(2 alpha)``: zero when the sail is
    edge-on, greatest at forty-five degrees, and falling away beyond, which
    is the stall a sail running dead downwind is in.

    Parameters
    ----------
    alpha_degrees
        The angle of attack, in degrees.

    Returns
    -------
    Array
        The lift coefficient.
    """
    return LIFT_MAX * jnp.sin(2.0 * jnp.radians(alpha_degrees))


def drag_coefficient(alpha_degrees):
    """Return the sail's drag coefficient at an angle of attack.

    A parabolic polar on a base drag, plus the pressure drag of a plate
    held across the flow, which is what carries the boat dead downwind
    once the lift has gone.

    Parameters
    ----------
    alpha_degrees
        The angle of attack, in degrees.

    Returns
    -------
    Array
        The drag coefficient.
    """
    lift = lift_coefficient(alpha_degrees)
    normal = 1.0 - jnp.cos(2.0 * jnp.radians(alpha_degrees))
    return BASE_DRAG + INDUCED_DRAG_FACTOR * lift**2 + 0.5 * normal


def apparent_wind(boat_speed, twa_degrees, tws):
    """Close the velocity triangle: what the sail feels.

    The water is static, so the boat's velocity through it is its velocity
    over the ground, and the apparent wind is the true wind minus that.
    Resolved in the boat's own frame, with the heading along the first
    axis, the true wind that *blows towards* the boat has components
    ``(-tws cos(twa), -tws sin(twa))`` and the boat's own motion subtracts
    a further ``boat_speed`` along the heading.

    Parameters
    ----------
    boat_speed
        The speed through the water, in m/s.
    twa_degrees
        The true wind angle, in degrees.
    tws
        The true wind speed, in m/s.

    Returns
    -------
    tuple
        The apparent wind speed in m/s and the apparent wind angle in
        degrees, the latter measured from the heading to where the
        apparent wind comes from.
    """
    twa = jnp.radians(twa_degrees)
    # Components of the wind blowing towards the boat, in the boat frame.
    along = -tws * jnp.cos(twa) - boat_speed
    across = -tws * jnp.sin(twa)
    speed = jnp.sqrt(along**2 + across**2 + 1e-12)
    # The angle to where it comes from, which is the opposite bearing.
    angle = jnp.degrees(jnp.arctan2(-across, -along))
    return speed, angle


def driving_force(boat_speed, twa_degrees, tws):
    """Return the sail's force along the heading, in N.

    Lift acts across the apparent wind and drag along it, so the part that
    pushes the boat forward is ``L sin(awa) - D cos(awa)``: close to the
    wind the lift term carries everything and the drag term is a tax on it,
    which is exactly why a boat cannot sail closer than its best
    lift-to-drag ratio allows.

    Parameters
    ----------
    boat_speed
        The speed through the water, in m/s.
    twa_degrees
        The true wind angle, in degrees.
    tws
        The true wind speed, in m/s.

    Returns
    -------
    Array
        The driving force, in N; negative when the sail cannot pull the
        boat forward at all.
    """
    speed, awa = apparent_wind(boat_speed, twa_degrees, tws)
    alpha = angle_of_attack(awa)
    pressure = 0.5 * AIR_DENSITY * SAIL_AREA * speed**2
    lift = pressure * lift_coefficient(alpha)
    drag = pressure * drag_coefficient(alpha)
    radians = jnp.radians(awa)
    # Lift is odd in the angle and drag even, so this product is even:
    # the two tacks are equally fast without folding the angle to make
    # them so. See :func:`angle_of_attack`.
    return lift * jnp.sin(radians) - drag * jnp.cos(radians)


def hull_resistance(boat_speed):
    """Return the water's resistance to the hull, in N.

    Parameters
    ----------
    boat_speed
        The speed through the water, in m/s.

    Returns
    -------
    Array
        The resistance, in N, always opposing the motion.
    """
    return (
        0.5
        * WATER_DENSITY
        * HULL_AREA
        * HULL_DRAG_COEFFICIENT
        * boat_speed
        * jnp.abs(boat_speed)
    )


__all__ = [
    "angle_of_attack",
    "apparent_wind",
    "drag_coefficient",
    "driving_force",
    "hull_resistance",
    "lift_coefficient",
]
