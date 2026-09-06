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
"""A point-mass bird soaring in a wind that changes with height.

An albatross crosses oceans for days without flapping and without thermals.
It does it by repeatedly climbing into the wind and diving back downwind
through air that is moving at different speeds at different heights. Each
turn through the gradient takes a little energy out of the wind, and over a
closed cycle that can be exactly enough to pay for the drag.

The bird is a point mass with six states. Three place it,

.. code-block::

    dx/dt = V cos(gamma) cos(psi) + W(h)
    dy/dt = V cos(gamma) sin(psi)
    dh/dt = V sin(gamma)

and three describe how it is flying: the airspeed ``V``, the flight path
angle ``gamma`` measured up from horizontal, and the heading ``psi``
measured from the ``x`` axis, which is the direction the wind blows.

The controls are the lift coefficient ``C_L`` and the bank angle ``mu``.

## The equations of motion

The forces are lift, drag and weight. What makes this problem different from
an ordinary glide is that the equations are written for the velocity relative
to the **air**, while Newton's law applies to the velocity relative to the
**ground**. The two differ by the wind, so whenever the bird changes height it
moves into air travelling at a different speed, and that appears in the
equations as an acceleration term:

.. code-block::

    Wdot = (dW/dh) (dh/dt) = (dW/dh) V sin(gamma)

Resolving along the flight path, in the vertical plane, and laterally:

.. code-block::

    dV/dt     = -D/m - g sin(gamma) - Wdot cos(gamma) cos(psi)
    dgamma/dt = (L cos(mu) - m g cos(gamma)) / (m V)
                + Wdot sin(gamma) cos(psi) / V
    dpsi/dt   = L sin(mu) / (m V cos(gamma)) + Wdot sin(psi) / (V cos(gamma))

The three ``Wdot`` terms are the entire mechanism. Set the wind gradient to
zero and they vanish, leaving the textbook gliding equations, which can only
lose energy.

## Where the energy comes from

The specific energy relative to the air is ``e = h + V^2 / (2 g)``.
Differentiating it along the equations above, the height terms cancel and
what is left is:

.. code-block::

    de/dt = -D V / (m g) - (dW/dh) V^2 sin(gamma) cos(gamma) cos(psi) / g

The first term is drag, and it is always a loss. The second is the wind
gradient, and its sign is the bird's to choose. With the wind increasing
upwards, it is positive whenever ``sin(gamma) cos(psi)`` is negative: climb
while heading into the wind, descend while heading with it. That is the
manoeuvre, and it falls out of the equations rather than being imposed.

Over a closed cycle the specific energy returns to where it started, so the
two terms integrate to the same number. The wind gradient pays for the drag
exactly, and the smallest wind for which that is possible is what the
co-design formulation goes looking for.
"""

from __future__ import annotations

import jax.numpy as jnp

from gemseo_tempo.dynamics.state_space import StateSpace
from gemseo_tempo.dynamics.tables import smooth_max
from gemseo_tempo.metadata.registry import VariableRegistry
from gemseo_tempo.metadata.variable import Variable
from gemseo_tempo.problems.albatross.statics import AIR_DENSITY
from gemseo_tempo.problems.albatross.statics import GRAVITY

ROUGHNESS = 0.03
"""The nominal roughness length of the sea surface, in m.

Poorly known, and the parameter the whole problem turns on: the wind
profile is logarithmic in the height over this length, so it sets how
steep the gradient is at the height the bird flies, and the gradient is
where all the energy comes from. It varies with the sea state over
something like a factor of six, which is why it carries a distribution and
why the dynamics take it as a parameter rather than reading this constant.
"""

REFERENCE_HEIGHT = 10.0
"""The height at which the reference wind speed is quoted, in m."""

MIN_HEIGHT = 0.5
"""The lowest height the wind profile is evaluated at, in m.

The logarithmic profile is singular at the roughness length and the bird
cannot fly into the water anyway, so the height is floored smoothly here
before the logarithm is taken.
"""

LOG_RATIO = float(jnp.log(REFERENCE_HEIGHT / ROUGHNESS))
"""The logarithm that normalizes the nominal profile to the reference height."""

REGISTRY = VariableRegistry(
    [
        Variable(
            name="x",
            full_name="Downwind position",
            description="Position along the wind direction.",
            unit="m",
            tags=frozenset({"state"}),
        ),
        Variable(
            name="y",
            full_name="Crosswind position",
            description=(
                "Displacement across the wind. The soaring cycle is closed"
                " in every state including this one, so a cycle that"
                " drifted sideways would not be a cycle."
            ),
            unit="m",
            tags=frozenset({"state"}),
        ),
        Variable(
            name="h",
            full_name="Height",
            description="Height above the sea surface.",
            unit="m",
            lower_bound=MIN_HEIGHT,
            tags=frozenset({"state"}),
        ),
        Variable(
            name="V",
            full_name="Airspeed",
            description="Speed relative to the surrounding air.",
            unit="m/s",
            lower_bound=8.0,
            upper_bound=80.0,
            tags=frozenset({"state"}),
        ),
        Variable(
            name="gamma",
            full_name="Flight path angle",
            description="Climb angle of the air-relative velocity.",
            unit="rad",
            tags=frozenset({"state"}),
        ),
        Variable(
            name="psi",
            full_name="Heading",
            description="Direction of flight, from the downwind axis.",
            unit="rad",
            tags=frozenset({"state"}),
        ),
        Variable(
            name="cl",
            full_name="Lift coefficient",
            description=(
                "How hard the wing is worked, the first of the two"
                " controls. Its upper bound is not a number in this file"
                " but the stall limit the statics chain computes for the"
                " wing being designed."
            ),
            lower_bound=-0.2,
            upper_bound=1.5,
            tags=frozenset({"control"}),
        ),
        Variable(
            name="mu",
            full_name="Bank angle",
            description="Roll of the lift vector about the flight path.",
            unit="rad",
            tags=frozenset({"control"}),
        ),
        Variable(
            name="w_ref",
            full_name="Reference wind speed",
            description="Wind speed at the reference height, the quantity minimized.",
            unit="m/s",
            lower_bound=0.0,
            upper_bound=40.0,
            distribution={
                "name": "OTNormalDistribution",
                "mu": 22.0,
                "sigma": 2.2,
            },
            tags=frozenset({"design", "objective"}),
        ),
        Variable(
            name="roughness",
            full_name="Surface roughness length",
            description=(
                "The length the logarithmic wind profile is measured"
                " against; a property of the sea state, not of the bird."
            ),
            unit="m",
            lower_bound=1e-4,
            distribution={
                "name": "OTLogNormalDistribution",
                "mu": 0.03,
                "sigma": 0.012,
            },
            tags=frozenset({"constant"}),
        ),
        Variable(
            name="specific_energy",
            full_name="Air-relative specific energy",
            description="Height plus airspeed head, the energy the drag polar sees.",
            unit="m",
            tags=frozenset({"output"}),
        ),
        Variable(
            name="ground_energy",
            full_name="Inertial specific energy",
            description="Height plus ground-speed head, the energy an observer sees.",
            unit="m",
            tags=frozenset({"output"}),
        ),
        Variable(
            name="load_factor",
            full_name="Load factor",
            description="Lift over weight.",
            tags=frozenset({"output"}),
        ),
        Variable(
            name="wind",
            full_name="Wind speed",
            description=(
                "The wind at the bird's current height, from the"
                " logarithmic profile. Its variation with height, not its"
                " value, is what the flight lives on."
            ),
            unit="m/s",
            tags=frozenset({"output"}),
        ),
        Variable(
            name="drag_power",
            full_name="Drag loss rate",
            description="Rate at which drag removes specific energy.",
            unit="m/s",
            tags=frozenset({"output"}),
        ),
        Variable(
            name="gradient_power",
            full_name="Wind gradient rate",
            description="Rate at which the wind gradient adds specific energy.",
            unit="m/s",
            tags=frozenset({"output"}),
        ),
    ]
)
"""The metadata of every variable of the dynamics."""


def wind_speed(height, w_ref, roughness=ROUGHNESS):
    """Return the wind speed at a height.

    The profile is logarithmic, which is what a boundary layer over open
    water looks like: the wind is slowed to nothing at the surface and
    recovers over the first few metres, so the gradient is steepest exactly
    where the bird flies.

    Parameters
    ----------
    height
        The height above the surface, in m.
    w_ref
        The wind speed at :data:`REFERENCE_HEIGHT`, in m/s.
    roughness
        The roughness length of the surface, in m; :data:`ROUGHNESS` when
        omitted. Note that it cancels only at the reference height, so a
        rougher sea does not merely rescale the profile, it changes its
        shape.

    Returns
    -------
    Array
        The wind speed, in m/s.
    """
    floored = smooth_max(height, MIN_HEIGHT, 1e-3)
    return w_ref * jnp.log(floored / roughness) / jnp.log(REFERENCE_HEIGHT / roughness)


def wind_gradient(height, w_ref, roughness=ROUGHNESS):
    """Return the rate of change of wind speed with height.

    Parameters
    ----------
    height
        The height above the surface, in m.
    w_ref
        The wind speed at :data:`REFERENCE_HEIGHT`, in m/s.
    roughness
        The roughness length of the surface, in m.

    Returns
    -------
    Array
        The gradient ``dW/dh``, in 1/s.
    """
    floored = smooth_max(height, MIN_HEIGHT, 1e-3)
    return w_ref / (floored * jnp.log(REFERENCE_HEIGHT / roughness))


def _scalars(theta):
    """Return the parameters as true scalars.

    Constants arrive as length-one arrays. Multiplying a per-node quantity
    by one of those broadcasts a spurious trailing axis, which survives all
    the way out to a three-dimensional Jacobian block that GEMSEO rejects.
    Reshaping them away at the boundary keeps every output one-dimensional.

    Parameters
    ----------
    theta
        The parameters keyed by name.

    Returns
    -------
    dict
        The same parameters, each reshaped to a scalar.
    """
    return {name: jnp.reshape(value, ()) for name, value in theta.items()}


def _forces(state, control, theta):
    """Return the lift and drag of the current state.

    Parameters
    ----------
    state
        The states keyed by name.
    control
        The controls keyed by name.
    theta
        The parameters keyed by name.

    Returns
    -------
    tuple
        The lift and drag, in N.
    """
    dynamic_pressure = 0.5 * AIR_DENSITY * state["V"] ** 2 * theta["wing_area"]
    lift = dynamic_pressure * control["cl"]
    drag = dynamic_pressure * (
        theta["cd0"] + theta["induced_factor"] * control["cl"] ** 2
    )
    return lift, drag


def create_state_space() -> StateSpace:
    """Build the six-state soaring dynamics.

    Returns
    -------
    StateSpace
        The model, with states ``x``, ``y``, ``h``, ``V``, ``gamma`` and
        ``psi``, controls ``cl`` and ``mu``, and the energy bookkeeping
        exposed as outputs.
    """

    def rhs(t, s, u, theta):
        """Return the rate of every state."""
        theta = _scalars(theta)
        lift, drag = _forces(s, u, theta)
        mass = theta["mass"]
        roughness = theta["roughness"]
        gradient = wind_gradient(s["h"], theta["w_ref"], roughness)

        climb_rate = s["V"] * jnp.sin(s["gamma"])
        # The air the bird is flying through is itself accelerating, because
        # changing height means moving into wind of a different speed.
        wind_rate = gradient * climb_rate

        return {
            "x": s["V"] * jnp.cos(s["gamma"]) * jnp.cos(s["psi"])
            + wind_speed(s["h"], theta["w_ref"], roughness),
            "y": s["V"] * jnp.cos(s["gamma"]) * jnp.sin(s["psi"]),
            "h": climb_rate,
            "V": -drag / mass
            - GRAVITY * jnp.sin(s["gamma"])
            - wind_rate * jnp.cos(s["gamma"]) * jnp.cos(s["psi"]),
            "gamma": (lift * jnp.cos(u["mu"]) - mass * GRAVITY * jnp.cos(s["gamma"]))
            / (mass * s["V"])
            + wind_rate * jnp.sin(s["gamma"]) * jnp.cos(s["psi"]) / s["V"],
            "psi": lift * jnp.sin(u["mu"]) / (mass * s["V"] * jnp.cos(s["gamma"]))
            + wind_rate * jnp.sin(s["psi"]) / (s["V"] * jnp.cos(s["gamma"])),
        }

    def output(t, s, u, theta):
        """Return the energies, the load factor and the wind."""
        theta = _scalars(theta)
        lift, drag = _forces(s, u, theta)
        mass = theta["mass"]
        roughness = theta["roughness"]
        wind = wind_speed(s["h"], theta["w_ref"], roughness)
        gradient = wind_gradient(s["h"], theta["w_ref"], roughness)

        # Ground speed is the air-relative velocity plus the wind, which
        # blows along x, so only the x component picks the wind up.
        along = s["V"] * jnp.cos(s["gamma"]) * jnp.cos(s["psi"]) + wind
        across = s["V"] * jnp.cos(s["gamma"]) * jnp.sin(s["psi"])
        vertical = s["V"] * jnp.sin(s["gamma"])
        ground_speed_squared = along**2 + across**2 + vertical**2

        return {
            "specific_energy": s["h"] + s["V"] ** 2 / (2.0 * GRAVITY),
            "ground_energy": s["h"] + ground_speed_squared / (2.0 * GRAVITY),
            "load_factor": lift / (mass * GRAVITY),
            "wind": wind,
            "drag_power": -drag * s["V"] / (mass * GRAVITY),
            "gradient_power": -gradient
            * s["V"] ** 2
            * jnp.sin(s["gamma"])
            * jnp.cos(s["gamma"])
            * jnp.cos(s["psi"])
            / GRAVITY,
        }

    return StateSpace(
        states=["x", "y", "h", "V", "gamma", "psi"],
        controls=["cl", "mu"],
        parameters=[
            "w_ref",
            "roughness",
            "mass",
            "wing_area",
            "induced_factor",
            "cd0",
            "cl_max",
        ],
        outputs=[
            "specific_energy",
            "ground_energy",
            "load_factor",
            "wind",
            "drag_power",
            "gradient_power",
        ],
        rhs=rhs,
        output=output,
    )
