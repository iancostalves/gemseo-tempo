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
    dv/dt = g (sin(gamma) - mu cos(gamma)) - (k/m) v**2

The three physical variants differ only in their constants: ``mu = 0`` and
``k/m = 0`` give the ideal frictionless case whose solution is a cycloid,
a nonzero ``mu`` adds Coulomb friction, and a nonzero ``k/m`` adds
quadratic drag.
"""

from __future__ import annotations

import jax.numpy as jnp

from gemseo_tempo.dynamics.state_space import StateSpace
from gemseo_tempo.metadata.registry import VariableRegistry
from gemseo_tempo.metadata.variable import Variable

VARIANTS = {
    "ideal": {"mu": 0.0, "k_over_m": 0.0},
    "friction": {"mu": 0.3, "k_over_m": 0.0},
    "drag": {"mu": 0.0, "k_over_m": 0.05},
}
"""The constants of each physical variant, beyond gravity."""

GRAVITY = 9.81
"""The standard gravitational acceleration, in m/s**2."""

TARGET = (2.0, 2.0)
"""The default target point ``(x, y)``, in m."""

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
            name="k_over_m",
            full_name="Drag coefficient over mass",
            description="Quadratic drag coefficient divided by the bead mass.",
            unit="1/m",
            lower_bound=0.0,
            distribution={
                "name": "OTUniformDistribution",
                "minimum": 0.03,
                "maximum": 0.07,
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


def brachistochrone_rhs(t, s, u, theta):
    """Return the state rates of the bead.

    Parameters
    ----------
    t
        The time.
    s
        The states ``x``, ``y`` and ``v``.
    u
        The control ``gamma``.
    theta
        The constants ``g``, ``mu`` and ``k_over_m``.

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
        "v": theta["g"] * (jnp.sin(angle) - theta["mu"] * jnp.cos(angle))
        - theta["k_over_m"] * speed**2,
    }


def create_state_space() -> StateSpace:
    """Build the state space of the bead.

    The same model covers every variant: the ideal case is recovered by
    setting ``mu`` and ``k_over_m`` to zero.

    Returns
    -------
    StateSpace
        The model, with states ``x``, ``y``, ``v`` and control ``gamma``.
    """
    return StateSpace(
        states=["x", "y", "v"],
        controls=["gamma"],
        parameters=["g", "mu", "k_over_m"],
        rhs=brachistochrone_rhs,
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
