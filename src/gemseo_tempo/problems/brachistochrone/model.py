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
"""The brachistochrone: minimum-time descent of a bead under gravity.

The bead slides from the origin to a target point, and the path angle
``gamma`` is the control. With ``y`` measured downward,

.. code-block::

    dx/dt = v cos(gamma)
    dy/dt = v sin(gamma)
    dv/dt = g sin(gamma) - mu * (N/m) - (k/m) v**2

Two forces beyond gravity, each written from the object rather than from a
convenient constant.

**Friction** is Coulomb's law, so it is proportional to the *normal* force
the wire actually exerts, which is not simply the normal component of
weight. A bead on a curved path is also being turned, and the turn costs a
centripetal force that the wire has to supply along with holding the bead
up:

.. code-block::

    N/m = g cos(gamma) - v dgamma/dt

The second term is the one usually dropped. It is kept here, which is why
this model asks its transcription for the control's time-derivative (see
:attr:`~gemseo_tempo.dynamics.state_space.StateSpace.control_rates`), and
it is floored smoothly at zero: a wire can push a bead but not pull it, and
an unfloored ``N`` turns negative in a hard turn, flips the sign of
friction and *accelerates* the bead — which an optimizer will find.

**Drag** is the standard quadratic law with the drag coefficient of the
shape the bead actually is, a circular cylinder in cross-flow. Writing the
mass as the cylinder's own volume times its density, the length cancels and
the whole group collapses to a function of one length scale:

.. code-block::

    k/m = 0.5 rho_air Cd (d L) / (rho_bead (pi/4) d**2 L)
        = 2 Cd rho_air / (pi rho_bead d)

so nothing here is a number chosen to make the figure look right. What is
uncertain is :data:`CYLINDER_DRAG_COEFFICIENT` itself, which for a cylinder
genuinely moves with Reynolds number and surface finish, and that is where
the declared distribution now sits.

The three variants differ only in which of the two forces is switched on:
``mu = 0`` with no drag gives the ideal frictionless case whose solution is
a cycloid, a nonzero ``mu`` adds friction, and the drag variant gives the
bead to the air.
"""

from __future__ import annotations

import jax.numpy as jnp

from gemseo_tempo.dynamics.state_space import StateSpace
from gemseo_tempo.dynamics.tables import smooth_max
from gemseo_tempo.metadata.registry import VariableRegistry
from gemseo_tempo.metadata.variable import Variable

CYLINDER_DRAG_COEFFICIENT = 1.2
"""The drag coefficient of a circular cylinder in cross-flow.

The textbook subcritical value, near enough constant over the Reynolds
numbers a centimetre-scale bead at a few metres per second reaches. It is
fixed rather than fitted: the point of writing drag this way is that the
number comes from the shape.
"""

AIR_DENSITY = 1.225
"""The density of air at sea level, in kg/m**3."""

BEAD_DIAMETER = 0.020
"""The diameter of the bead, in m: a 20 mm bead."""

BEAD_DENSITY = 950.0
"""The density of the bead, in kg/m**3: a dense plastic, close to water."""

GRAVITY = 9.81
"""The standard gravitational acceleration, in m/s**2."""

TARGET = (2.0, 2.0)
"""The default target point ``(x, y)``, in m."""

NORMAL_FORCE_SMOOTHING = 1.0e-3
"""How sharply the normal force is floored at zero, in m/s**2.

Small against ``g``, so it changes nothing where the wire is pushing, and
enough to leave the floor differentiable where it stops.
"""


def drag_over_mass(drag_coefficient) -> float:
    """Return the quadratic drag group ``k/m`` of the bead, in 1/m.

    Writing the mass as the cylinder's own volume times its density makes
    the length cancel, so the group depends on the diameter alone:

    .. code-block::

        k/m = 0.5 rho_air Cd (d L) / (rho_bead (pi/4) d**2 L)
            = 2 Cd rho_air / (pi rho_bead d)

    Parameters
    ----------
    drag_coefficient
        The drag coefficient; zero for the variants flying in vacuum.

    Returns
    -------
    float
        The drag coefficient divided by the bead mass, in 1/m.
    """
    return (
        2.0 * drag_coefficient * AIR_DENSITY / (jnp.pi * BEAD_DENSITY * BEAD_DIAMETER)
    )


VARIANTS = {
    "ideal": {"mu": 0.0, "drag_coefficient": 0.0},
    "friction": {"mu": 0.3, "drag_coefficient": 0.0},
    "drag": {"mu": 0.0, "drag_coefficient": CYLINDER_DRAG_COEFFICIENT},
}
"""The constants of each physical variant, beyond gravity.

The ideal bead is both frictionless and in vacuum; the other two switch on
one force each, so that what each costs can be read separately.
"""

REGISTRY = VariableRegistry(
    [
        Variable(
            name="x",
            full_name="Horizontal position",
            description="Horizontal coordinate of the bead.",
            unit="m",
            tags=frozenset({"state"}),
        ),
        Variable(
            name="y",
            full_name="Depth",
            description="Vertical coordinate of the bead, measured downward.",
            unit="m",
            tags=frozenset({"state"}),
        ),
        Variable(
            name="v",
            full_name="Speed",
            description="Speed of the bead along its path.",
            unit="m/s",
            lower_bound=0.0,
            tags=frozenset({"state"}),
        ),
        Variable(
            name="elapsed",
            full_name="Elapsed time",
            description=(
                "Time since release, a state when the wire is integrated"
                " along its length rather than in time."
            ),
            unit="s",
            lower_bound=0.0,
            tags=frozenset({"state"}),
        ),
        Variable(
            name="gamma",
            full_name="Path angle",
            description="Angle of the path below the horizontal.",
            unit="rad",
            lower_bound=0.0,
            upper_bound=1.5707963267948966,
            tags=frozenset({"control"}),
        ),
        Variable(
            name="g",
            full_name="Gravitational acceleration",
            description=(
                "What drives the bead down the wire. Known to three"
                " decimal places, and in the list of uncertainties on"
                " purpose: a screening ought to be able to say that a"
                " well-known parameter does not matter."
            ),
            unit="m/s**2",
            tags=frozenset({"constant"}),
            # Known to a few parts in a thousand, and included precisely so
            # a screening has something it should rank last.
            distribution={
                "name": "OTNormalDistribution",
                "mu": 9.81,
                "sigma": 0.02,
            },
        ),
        Variable(
            name="mu",
            full_name="Coulomb friction coefficient",
            description=(
                "Sliding friction against the wire, opposing the motion"
                " in proportion to the normal force. Zero in the classical"
                " problem, which is why that one has a closed-form answer."
            ),
            unit="dimensionless",
            lower_bound=0.0,
            # Uniform rather than normal: the range is what is known about a
            # sliding contact, and a normal would put mass on a negative
            # friction coefficient.
            distribution={
                "name": "OTUniformDistribution",
                "minimum": 0.02,
                "maximum": 0.08,
            },
            tags=frozenset({"constant"}),
        ),
        Variable(
            name="drag_coefficient",
            full_name="Drag coefficient",
            description=(
                "The drag coefficient of the bead, a circular cylinder in"
                " cross-flow. Fixed by the shape rather than fitted, and"
                " uncertain where a cylinder's really is: a subcritical"
                " cylinder runs about 1.0 to 1.3 with Reynolds number and"
                " surface finish. The quadratic drag the dynamics feel"
                " follows from it and the bead's geometry."
            ),
            unit="dimensionless",
            lower_bound=0.0,
            distribution={
                "name": "OTUniformDistribution",
                "minimum": 1.0,
                "maximum": 1.3,
            },
            tags=frozenset({"constant"}),
        ),
        Variable(
            name="t_f",
            full_name="Descent time",
            description="Total time of descent, the quantity minimized.",
            unit="s",
            lower_bound=0.0,
            tags=frozenset({"objective"}),
        ),
    ]
)
"""The metadata of every variable of the problem."""


def normal_over_mass(speed, angle, turn_rate, gravity):
    """Return the normal force the wire exerts, per unit mass.

    Newton's law perpendicular to the path. Gravity contributes
    ``g cos(gamma)``; turning the bead costs ``v dgamma/dt``, which the
    wire also has to supply, and which the usual textbook treatment drops.

    The result is floored smoothly at zero because a wire pushes and does
    not pull. Left unfloored, a hard enough turn drives the normal force
    negative, friction changes sign, and the model quietly says that
    rubbing against the wire makes the bead faster.

    Parameters
    ----------
    speed
        The speed along the path, in m/s.
    angle
        The path angle below the horizontal, in rad.
    turn_rate
        The rate of change of the path angle, in rad/s.
    gravity
        The gravitational acceleration, in m/s**2.

    Returns
    -------
    Array
        The normal force per unit mass, in m/s**2, never negative.
    """
    return smooth_max(
        gravity * jnp.cos(angle) - speed * turn_rate, 0.0, NORMAL_FORCE_SMOOTHING
    )


def brachistochrone_rhs(t, s, u, theta):
    """Return the state rates of the bead.

    Parameters
    ----------
    t
        The time.
    s
        The states ``x``, ``y`` and ``v``.
    u
        The control ``gamma`` and its rate ``gamma_rate``.
    theta
        The constants ``g``, ``mu`` and ``drag_coefficient``.

    Returns
    -------
    dict
        The rate of each state.
    """
    angle = u["gamma"]
    speed = s["v"]
    normal = normal_over_mass(speed, angle, u["gamma_rate"], theta["g"])
    return {
        "x": speed * jnp.cos(angle),
        "y": speed * jnp.sin(angle),
        "v": theta["g"] * jnp.sin(angle)
        - theta["mu"] * normal
        - drag_over_mass(theta["drag_coefficient"]) * speed**2,
    }


def frictionless_rhs(t, s, u, theta):
    """Return the state rates of a bead that does not rub.

    The same model as :func:`brachistochrone_rhs` with the friction term
    dropped rather than multiplied by a zero coefficient, so that nothing
    asks for the control's rate and the indirect transcription stays
    available.

    Parameters
    ----------
    t
        The time.
    s
        The states ``x``, ``y`` and ``v``.
    u
        The control ``gamma``.
    theta
        The constants ``g`` and ``drag_coefficient``; ``mu`` is ignored.

    Returns
    -------
    dict
        The rate of each state.
    """
    angle = u["gamma"]
    speed = s["v"]
    return {
        "x": speed * jnp.cos(angle),
        "y": speed * jnp.sin(angle),
        "v": theta["g"] * jnp.sin(angle)
        - drag_over_mass(theta["drag_coefficient"]) * speed**2,
    }


def create_state_space(friction: bool = True) -> StateSpace:
    """Build the state space of the bead.

    The same model covers every variant: the ideal case is recovered by
    setting ``mu`` and ``drag_coefficient`` to zero.

    Parameters
    ----------
    friction
        Whether the bead rubs against the wire. Only a model with friction
        needs the normal force, and only the normal force needs the
        control's time-derivative, so a frictionless variant is built
        without asking for it. That is not a detail: a model that asks for
        a control rate cannot be transcribed by indirect shooting, since
        the control then enters the Hamiltonian through its own derivative,
        and the frictionless bead is exactly the case the indirect method
        is demonstrated on.

    Returns
    -------
    StateSpace
        The model, with states ``x``, ``y``, ``v`` and control ``gamma``.
    """
    return StateSpace(
        states=["x", "y", "v"],
        controls=["gamma"],
        parameters=["g", "mu", "drag_coefficient"],
        rhs=brachistochrone_rhs if friction else frictionless_rhs,
        control_rates=["gamma"] if friction else (),
    )


def variant_constants(variant: str, gravity: float = GRAVITY) -> dict[str, float]:
    """Return the constants of a named variant.

    Parameters
    ----------
    variant
        The variant name, one of :data:`VARIANTS`.
    gravity
        The gravitational acceleration.

    Returns
    -------
    dict
        The constants, ready to be passed to the transcription.

    Raises
    ------
    ValueError
        If the variant name is unknown.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; use one of {sorted(VARIANTS)}.")
    return {"g": gravity, **VARIANTS[variant]}
