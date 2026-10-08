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
"""The brachistochrone as a wire: a shape, integrated along its arc length.

The time formulation schedules the path angle in time, which is the right
description of a vehicle steering itself and the wrong one of a wire. Once
the wire is built its shape is fixed, and a bead meeting more friction than
it was designed for runs along the same wire, only slower. Freezing an
angle schedule in time and changing the friction would instead bend the
bead onto another path.

So the wire is described by its angle along its own length, and the arc
length replaces the time as the independent variable. Through the square
root ``zeta = sqrt(s)``, which keeps the start regular, the equations read

.. code-block::

    dx/dzeta = 2 zeta cos(gamma)
    dy/dzeta = 2 zeta sin(gamma)
    dv/dzeta = 2 zeta (g sin(gamma) - mu N/m - (k/m) v**2) / v
    dt/dzeta = 2 zeta / v

with ``N/m`` the normal force of the time model,
:func:`~gemseo_tempo.problems.brachistochrone.model.normal_over_mass`,
turning included. A wire turns the bead as fast as the bead runs along its
curvature, ``dgamma/dt = v dgamma/ds``, and the control's rate the model
reads is ``dgamma/dzeta = 2 zeta dgamma/ds``; so the turning term is
``v**2 (dgamma/dzeta) / (2 zeta)``, which vanishes at the release, where the
speed grows like ``zeta``.

Released from rest the speed grows like ``zeta``, so ``2 zeta / v`` stays
finite and tends to ``sqrt(2 / a)`` at the start, with ``a`` the initial
acceleration: that limit, and the vanishing turning term, are written
out, so the integrator never divides zero by zero. The time taken is then
a state, read at the end of the wire, and the end point is pure geometry:
however the physics changes, the bead arrives where the wire goes.

The scenario's free final "time" is the square root of the wire length, and
the angle is a control over the normalized ``zeta``: a cubic spline
through its nodes, so that its rate, which the normal force reads, is
continuous. A piecewise-linear angle would hand the friction a curvature
that jumps from one interval to the next, and friction that grows with the
turn rate then pays the optimizer to alternate steep and flat intervals: it
settles on a sawtooth and stalls short of the end point, where the spline
converges in a few dozen iterations to the smooth wire.

Unlike the time model, the wire never drops friction, not even for the
ideal bead. Only indirect shooting has reason to want it dropped, and a
wire is shot directly; meanwhile the wire is what the uncertainty studies
fly, starting from the ideal variant's constants with an uncertain ``mu``
laid over them, and a model that ignored it would report a sensitivity of
exactly zero.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy

from gemseo_tempo.dynamics.controls import CubicControl
from gemseo_tempo.dynamics.integrators import DiffraxIntegrator
from gemseo_tempo.dynamics.state_space import StateSpace
from gemseo_tempo.outputs.functionals import Mayer
from gemseo_tempo.problems.brachistochrone.model import GRAVITY
from gemseo_tempo.problems.brachistochrone.model import REGISTRY
from gemseo_tempo.problems.brachistochrone.model import TARGET
from gemseo_tempo.problems.brachistochrone.model import drag_over_mass
from gemseo_tempo.problems.brachistochrone.model import normal_over_mass
from gemseo_tempo.problems.brachistochrone.model import variant_constants
from gemseo_tempo.scenario.dynamic_scenario import DynamicScenario
from gemseo_tempo.scenario.formulations import OptimalControl
from gemseo_tempo.transcriptions.base import BoundarySpec
from gemseo_tempo.transcriptions.single_shooting import SingleShooting

DESCENT_TIME = "descent_time"
"""The name of the time the bead takes, the objective."""

_TINY = 1e-12
"""A floor keeping a guarded division away from zero."""


def wire_rhs(zeta, s, u, theta):
    """Return the rates of the bead along the wire.

    Parameters
    ----------
    zeta
        The square root of the arc length.
    s
        The states ``x``, ``y``, ``v`` and ``elapsed``.
    u
        The control ``gamma``, the angle of the wire, and its rate
        ``gamma_rate`` with respect to ``zeta``.
    theta
        The constants ``g``, ``mu`` and ``drag_coefficient``.

    Returns
    -------
    dict
        The rate of each state with respect to ``zeta``.
    """
    angle = u["gamma"]
    speed = s["v"]
    started = zeta > 0.0
    # Both branches are evaluated, so each is kept finite where it is not
    # used, or its gradient would be a NaN multiplied by zero.
    safe_zeta = jnp.where(started, zeta, 1.0)
    turn_rate = jnp.where(started, speed * u["gamma_rate"] / (2.0 * safe_zeta), 0.0)
    normal = normal_over_mass(speed, angle, turn_rate, theta["g"])
    push = theta["g"] * jnp.sin(angle) - theta["mu"] * normal
    safe_speed = jnp.where(started, jnp.maximum(speed, _TINY), 1.0)
    safe_push = jnp.where(started, 1.0, jnp.maximum(push, _TINY))
    ratio = jnp.where(started, 2.0 * zeta / safe_speed, jnp.sqrt(2.0 / safe_push))
    return {
        "x": 2.0 * zeta * jnp.cos(angle),
        "y": 2.0 * zeta * jnp.sin(angle),
        "v": ratio * (push - drag_over_mass(theta["drag_coefficient"]) * speed**2),
        "elapsed": ratio,
    }


def create_wire_state_space() -> StateSpace:
    """Build the state space of the bead on a wire.

    Returns
    -------
    StateSpace
        The model, with states ``x``, ``y``, ``v`` and ``elapsed``, the
        wire angle ``gamma`` as its control, and the angle's rate read for
        the normal force.
    """
    return StateSpace(
        states=["x", "y", "v", "elapsed"],
        controls=["gamma"],
        parameters=["g", "mu", "drag_coefficient"],
        rhs=wire_rhs,
        control_rates=["gamma"],
    )


def create_wire_scenario(
    size: int = 20,
    variant: str = "ideal",
    constants: dict[str, float] | None = None,
    integrator=None,
    target: tuple[float, float] = TARGET,
    gravity: float = GRAVITY,
) -> DynamicScenario:
    """Build the wire, by single shooting along its length.

    Parameters
    ----------
    size
        The number of angle intervals along the wire.
    variant
        The physical variant whose constants are the starting ones.
    constants
        Constants laid over the variant's.
    integrator
        The integrator; a fixed step of 1/400 of the wire when omitted,
        smooth in every input, which a sampled study needs.
    target
        The end of the wire ``(x, y)``, in m.
    gravity
        The gravitational acceleration, in m/s**2.

    Returns
    -------
    DynamicScenario
        The scenario, starting from a wire that falls steeply and flattens.
    """
    nodes = jnp.linspace(0.0, 1.0, size + 1)
    return DynamicScenario(
        state_space=create_wire_state_space(),
        integrator=integrator or DiffraxIntegrator(dt0=1.0 / 400, n_save=size + 1),
        controls=(
            CubicControl(
                "gamma", nodes, default_params=jnp.linspace(1.5, 0.2, size + 1)
            ),
        ),
        functionals=(
            Mayer(DESCENT_TIME, lambda t_final, states, theta: states["elapsed"]),
        ),
        boundary=BoundarySpec(
            s0={"x": 0.0, "y": 0.0, "v": 0.0, "elapsed": 0.0},
            fix_final={"x": True, "y": True},
            s1={"x": target[0], "y": target[1]},
            free_final_time=True,
            t_f=float(numpy.sqrt(numpy.hypot(*target) * 1.3)),
        ),
        transcription=SingleShooting(),
        constants={**variant_constants(variant, gravity), **(constants or {})},
        registry=REGISTRY,
        t0=0.0,
        t1=1.0,
    )


def wire_bounds(scenario: DynamicScenario) -> dict[str, tuple[float, float]]:
    """Return the box bounds of a wire's design variables.

    Parameters
    ----------
    scenario
        The wire scenario.

    Returns
    -------
    dict
        The angle between level and vertical, and a length from the
        straight line to four times it, by its square root. The straight
        line is the scenario's own, from its start to its target.
    """
    boundary = scenario.definition["boundary"]
    straight = float(
        numpy.hypot(
            boundary.s1["x"] - boundary.s0["x"], boundary.s1["y"] - boundary.s0["y"]
        )
    )
    bounds = {"t_f": (float(numpy.sqrt(straight)), float(numpy.sqrt(4 * straight)))}
    for name in scenario.problem.design_variables:
        if name.startswith("control."):
            bounds[name] = (0.0, float(numpy.pi / 2.0))
    return bounds


def create_wire_optimization(**settings):
    """Build the wire and the GEMSEO scenario that designs it.

    Parameters
    ----------
    **settings
        Forwarded to :func:`create_wire_scenario`.

    Returns
    -------
    tuple
        The :class:`.DynamicScenario` and the GEMSEO scenario built on it.
    """
    scenario = create_wire_scenario(**settings)
    formulation = OptimalControl(scenario, bounds=wire_bounds(scenario))
    return scenario, formulation.create_gemseo_scenario()
