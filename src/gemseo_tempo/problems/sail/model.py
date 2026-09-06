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
"""Minimum-time sailing to a mark through a varying wind field.

The boat is a point moving over the water at a speed read off the polar,

.. code-block::

    dx/dt = V cos(psi)
    dy/dt = V sin(psi)

with ``psi`` the heading it is steered on and ``V`` the boat speed at the
wind angle that heading produces.

The angle convention is the one sailors use, and it is the part worth
reading twice. The wind field gives the velocity the air is travelling
*towards*; the wind is described by where it comes *from*, which is the
opposite direction. The true wind angle is the angle between the heading
and that "from" direction, so a true wind angle of zero is sailing
straight into the wind and cannot be done, and 180 degrees is running
straight downwind.

The wind carries two parameters, and they are the reason the model has any
at all. A sailor knows the shape of the breeze over the course far better
than its strength or its exact direction: the gradient across the bay is a
feature of the bay, while whether it is blowing twelve knots or fourteen,
and whether it has veered five degrees since the start, are not known and
will not be. So the field itself stays fixed and two numbers act on it,
``wind_scale`` multiplying the whole field and ``wind_veer`` rotating it.
Passing them as parameters rather than closing over them is what lets a
whole sample of winds go through the dynamics in one call.
"""

from __future__ import annotations

import jax.numpy as jnp

from gemseo_tempo.dynamics.state_space import StateSpace
from gemseo_tempo.metadata.registry import VariableRegistry
from gemseo_tempo.metadata.variable import Variable
from gemseo_tempo.problems.sail.polar import NO_GO_ANGLE
from gemseo_tempo.problems.sail.polar import boat_speed

REGISTRY = VariableRegistry(
    [
        Variable(
            name="x",
            full_name="Eastward position",
            description="Position of the boat along the east axis.",
            unit="m",
            tags=frozenset({"state"}),
        ),
        Variable(
            name="y",
            full_name="Northward position",
            description="Position of the boat along the north axis.",
            unit="m",
            tags=frozenset({"state"}),
        ),
        Variable(
            name="psi",
            full_name="Heading",
            description="Direction the boat is steered on, from the east axis.",
            unit="rad",
            tags=frozenset({"control"}),
        ),
        Variable(
            name="twa",
            full_name="True wind angle",
            description=(
                "Angle between the heading and the direction the wind comes"
                " from; zero is head to wind, 180 degrees is dead downwind."
            ),
            unit="deg",
            tags=frozenset({"output"}),
        ),
        Variable(
            name="tws",
            full_name="True wind speed",
            description=(
                "The strength of the wind over the water at the boat's"
                " position, as opposed to the apparent wind a moving boat"
                " feels."
            ),
            unit="m/s",
            lower_bound=0.0,
            tags=frozenset({"output"}),
        ),
        Variable(
            name="speed",
            full_name="Boat speed",
            description=(
                "Speed through the water, read off the polar at the"
                " current wind angle and wind speed. It is what makes"
                " gybing pay: the boat gives up heading straight at the"
                " mark in exchange for enough of this to arrive sooner."
            ),
            unit="m/s",
            lower_bound=0.0,
            tags=frozenset({"output"}),
        ),
        Variable(
            name="t_f",
            full_name="Time to the mark",
            description="Total sailing time, the quantity minimized.",
            unit="s",
            lower_bound=0.0,
            tags=frozenset({"objective"}),
        ),
        Variable(
            name="wind_scale",
            full_name="Wind strength factor",
            description=(
                "Multiplies the whole wind field. The shape of the breeze"
                " over the course is a feature of the course and is known;"
                " how hard it is blowing on the day is not."
            ),
            lower_bound=0.0,
            distribution={
                "name": "OTNormalDistribution",
                "mu": 1.0,
                "sigma": 0.12,
            },
            tags=frozenset({"constant"}),
        ),
        Variable(
            name="wind_veer",
            full_name="Wind direction shift",
            description=(
                "Rotates the whole wind field. A shift of a few degrees is"
                " what a sailor cannot see coming and what decides which"
                " gybe was the right one."
            ),
            unit="rad",
            distribution={
                "name": "OTNormalDistribution",
                "mu": 0.0,
                "sigma": 0.175,
            },
            tags=frozenset({"constant"}),
        ),
    ]
)
"""The metadata of every variable of the problem."""


def wrap_to_pi(angle):
    """Fold an angle into ``[-pi, pi]``.

    Parameters
    ----------
    angle
        The angle, in radians.

    Returns
    -------
    Array
        The folded angle, in radians.
    """
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


def true_wind_angle(psi, east, north):
    """Return the true wind angle of a heading, in degrees.

    Parameters
    ----------
    psi
        The heading, in radians from the east axis.
    east
        The eastward component of the wind velocity, in m/s.
    north
        The northward component of the wind velocity, in m/s.

    Returns
    -------
    Array
        The signed true wind angle in degrees, in ``[-180, 180]``. Zero is
        head to wind and 180 is dead downwind; the sign says which tack.
    """
    # The field gives where the air is going; the wind comes from the
    # opposite bearing, which is what the angle is measured against.
    from_bearing = jnp.arctan2(-north, -east)
    return jnp.degrees(wrap_to_pi(psi - from_bearing))


WIND_PARAMETERS = ("wind_scale", "wind_veer")
"""The parameters acting on the wind field, in declaration order."""


def apply_wind_parameters(east, north, theta):
    """Scale and rotate a wind velocity by the uncertain parameters.

    The field itself is fixed, so the two parameters act on whatever it
    returns: a factor on the strength and a rotation of the direction.
    Rotating the velocity rotates the bearing it comes from by the same
    angle, which is what a veer is.

    Parameters
    ----------
    east
        The eastward component of the wind velocity, in m/s.
    north
        The northward component, in m/s.
    theta
        The parameters, read for ``wind_scale`` and ``wind_veer``; either
        may be absent, in which case the field passes through unchanged.

    Returns
    -------
    tuple
        The transformed components, in m/s.
    """
    scale = jnp.reshape(theta.get("wind_scale", 1.0), ())
    veer = jnp.reshape(theta.get("wind_veer", 0.0), ())
    cosine = jnp.cos(veer)
    sine = jnp.sin(veer)
    return (
        scale * (cosine * east - sine * north),
        scale * (sine * east + cosine * north),
    )


def create_state_space(wind) -> StateSpace:
    """Build the state space of the boat on a given wind field.

    Parameters
    ----------
    wind
        The :class:`~gemseo_tempo.problems.sail.wind.WindField`.

    Returns
    -------
    StateSpace
        The model, with states ``x`` and ``y``, control ``psi``, the two
        wind parameters, and the wind angle, wind speed and boat speed as
        outputs.
    """

    def _sailing(t, s, u, theta):
        """Return the wind angle, wind speed and boat speed at a point."""
        east, north = apply_wind_parameters(*wind.velocity(s["x"], s["y"]), theta)
        tws = jnp.hypot(east, north)
        twa = true_wind_angle(u["psi"], east, north)
        return twa, tws, boat_speed(twa, tws)

    def rhs(t, s, u, theta):
        """Return the rate of the position."""
        _, _, speed = _sailing(t, s, u, theta)
        return {
            "x": speed * jnp.cos(u["psi"]),
            "y": speed * jnp.sin(u["psi"]),
        }

    def output(t, s, u, theta):
        """Return the sailing quantities along the trajectory."""
        twa, tws, speed = _sailing(t, s, u, theta)
        return {"twa": twa, "tws": tws, "speed": speed}

    return StateSpace(
        states=["x", "y"],
        controls=["psi"],
        parameters=list(WIND_PARAMETERS),
        outputs=["twa", "tws", "speed"],
        rhs=rhs,
        output=output,
    )


def no_go_constraint(wind):
    """Build the path constraint keeping the boat out of the no-go cone.

    Parameters
    ----------
    wind
        The wind field.

    Returns
    -------
    Callable
        A constraint ``(t, s, u, theta) -> value`` that is negative while
        the true wind angle stays above :data:`.NO_GO_ANGLE`.
    """

    def constraint(t, s, u, theta):
        east, north = apply_wind_parameters(*wind.velocity(s["x"], s["y"]), theta)
        twa = true_wind_angle(u["psi"], east, north)
        # Sailing on either tack is fine, so only the magnitude matters.
        return NO_GO_ANGLE - jnp.sqrt(twa**2 + 1e-6)

    return constraint


def channel_constraint(half_width: float):
    """Build the path constraint keeping the boat inside a channel.

    Sailing wide of the rhumb line is exactly what gybing and beating buy
    speed with, so a limit on how far the boat may stray from the direct
    line is the constraint that genuinely trades against the objective: it
    binds on the optimum rather than merely fencing off a region the boat
    had no reason to enter.

    Parameters
    ----------
    half_width
        The distance either side of the east axis the boat may use, in m.

    Returns
    -------
    Callable
        A constraint ``(t, s, u, theta) -> value`` that is negative while
        the boat stays inside the channel.
    """

    def constraint(t, s, u, theta):
        # Smooth in y, so the corner at the axis does not stall the
        # optimizer's line search.
        return jnp.sqrt(s["y"] ** 2 + 1e-6) - half_width

    return constraint
