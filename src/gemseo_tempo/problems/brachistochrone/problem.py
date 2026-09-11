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
"""Builders for the brachistochrone, one per transcription.

The interesting part here is the initial guess. A naive linear ramp
between the boundary states leaves the speed at zero for the whole
trajectory, which makes ``dx/dt = v cos(gamma)`` degenerate and stalls
collocation badly (over a hundred percent error on a coarse grid). So a
guess is instead built by *simulating* the dynamics once under a guessed
control law: cheap, needs no optimizer, and yields a trajectory that is
consistent with the equations of motion. Every transcription is seeded
from that same trajectory, which is what makes comparisons between them
fair.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

import jax.numpy as jnp
import numpy

from gemseo_tempo.dynamics.controls import PiecewiseLinearControl
from gemseo_tempo.dynamics.integrators import DiffraxIntegrator
from gemseo_tempo.outputs.functionals import Mayer
from gemseo_tempo.problems.brachistochrone.model import GRAVITY
from gemseo_tempo.problems.brachistochrone.model import REGISTRY
from gemseo_tempo.problems.brachistochrone.model import TARGET
from gemseo_tempo.problems.brachistochrone.model import create_state_space
from gemseo_tempo.problems.brachistochrone.model import variant_constants
from gemseo_tempo.scenario.dynamic_scenario import DynamicScenario
from gemseo_tempo.scenario.formulations import OptimalControl
from gemseo_tempo.transcriptions.base import BoundarySpec
from gemseo_tempo.transcriptions.collocation import HermiteSimpson
from gemseo_tempo.transcriptions.indirect import IndirectShooting

HALF_PI = float(numpy.pi / 2.0)
"""The upper bound of the path angle, in rad."""


@dataclass
class Guess:
    """A starting point, described by a few physical parameters.

    The path angle is guessed as the affine law
    ``gamma(tau) = level + slope * tau`` over the normalized time, and the
    descent time by ``t_f``. Sampling these three numbers gives meaningful
    variety without the absurd trajectories that random node values would
    produce.
    """

    gamma_level: float = 0.6
    """The path angle at the start, in rad."""

    gamma_slope: float = 0.6
    """The increase of the path angle over the descent, in rad."""

    t_f: float = 1.0
    """The guessed descent time, in s."""

    costates: dict[str, float] = field(default_factory=dict)
    """The initial costates, used only by the indirect transcription."""

    def angles(self, count: int):
        """Sample the guessed path-angle law.

        Parameters
        ----------
        count
            The number of nodes.

        Returns
        -------
        Array
            The path angle at ``count`` evenly spaced normalized times.
        """
        return self.gamma_level + self.gamma_slope * jnp.linspace(0.0, 1.0, count)


def optimal_path_angle(t, s, lam, theta):
    """Return the path angle minimizing the Hamiltonian.

    Stationarity of ``H = lam^T f`` in ``gamma`` gives
    ``tan(gamma) = (lam_y v + lam_v g) / (lam_x v - lam_v g mu)``, which is
    solved here in closed form. Quadratic drag does not enter, having no
    dependence on the path angle.

    Parameters
    ----------
    t
        The time.
    s
        The states.
    lam
        The costates.
    theta
        The constants.

    Returns
    -------
    dict
        The path angle.
    """
    speed = s["v"]
    gravity = theta["g"]
    return {
        "gamma": jnp.arctan2(
            lam["y"] * speed + lam["v"] * gravity,
            lam["x"] * speed - lam["v"] * gravity * theta["mu"],
        )
    }


def simulate_guess(
    guess: Guess,
    constants: dict[str, float],
    n_save: int = 61,
):
    """Integrate the dynamics under the guessed control law.

    Parameters
    ----------
    guess
        The starting point.
    constants
        The variant constants.
    n_save
        The number of points of the returned trajectory.

    Returns
    -------
    Trajectory
        A trajectory consistent with the equations of motion, spanning
        ``[0, guess.t_f]`` in physical time.
    """
    times = jnp.linspace(0.0, guess.t_f, 21)
    control = PiecewiseLinearControl("gamma", times, default_params=guess.angles(21))
    integrator = DiffraxIntegrator(rtol=1e-8, atol=1e-8, n_save=n_save)
    return integrator.integrate(
        create_state_space(),
        {"x": 0.0, "y": 0.0, "v": 0.0},
        (control,),
        {"gamma": control.default_params},
        constants,
        0.0,
        guess.t_f,
    )


def create_scenario(
    transcription=None,
    variant: str = "ideal",
    size: int = 20,
    guess: Guess | None = None,
    target: tuple[float, float] = TARGET,
    gravity: float = GRAVITY,
    integrator=None,
    **scenario_settings,
) -> DynamicScenario:
    """Build the brachistochrone scenario for one transcription.

    Parameters
    ----------
    transcription
        The transcription; Hermite-Simpson collocation when omitted.
    variant
        The physical variant, ``"ideal"``, ``"friction"`` or ``"drag"``.
    size
        The discretization: collocation segments, shooting control
        intervals, or integrator save intervals for the indirect method.
    guess
        The starting point; the default guess when omitted.
    target
        The target point ``(x, y)``, in m.
    gravity
        The gravitational acceleration, in m/s**2.
    **scenario_settings
        Extra settings forwarded to :class:`.DynamicScenario`.

    Returns
    -------
    DynamicScenario
        The scenario, already seeded with the guess.
    """
    guess = guess or Guess()
    constants = variant_constants(variant, gravity)
    transcription = transcription or HermiteSimpson(num_segments=size)
    indirect = isinstance(transcription, IndirectShooting)

    boundary = BoundarySpec(
        s0={"x": 0.0, "y": 0.0, "v": 0.0},
        fix_final={"x": True, "y": True},
        s1={"x": target[0], "y": target[1]},
        free_final_time=True,
        t_f=guess.t_f,
    )
    # With a free final time the dynamics are integrated over the
    # normalized time, so the control is parameterized on [0, 1].
    nodes = jnp.linspace(0.0, 1.0, size + 1)
    controls = (
        ()
        if indirect
        else (
            PiecewiseLinearControl(
                "gamma", nodes, default_params=guess.angles(size + 1)
            ),
        )
    )
    scenario = DynamicScenario(
        state_space=create_state_space(),
        integrator=integrator
        or DiffraxIntegrator(rtol=1e-9, atol=1e-9, n_save=size + 1),
        controls=controls,
        # Not ``t_f``: with a free final time that name is already a
        # design variable, and GEMSEO keys its history dataset by name, so
        # an objective sharing one with an input gives every column twice.
        # Its post-processing then indexes a two-column array as if it were
        # one and fails, which is how this was found. The albatross avoids
        # the same collision by calling its objective ``w_ref_objective``.
        functionals=(Mayer("descent_time", lambda t_final, states, theta: t_final),),
        boundary=boundary,
        transcription=transcription,
        constants=constants,
        registry=REGISTRY,
        t0=0.0,
        t1=1.0,
        **scenario_settings,
    )
    _seed(scenario, transcription, guess, constants)
    return scenario


def _seed(scenario, transcription, guess, constants):
    """Write the guess into the design-variable defaults."""
    defaults = scenario.problem.input_defaults
    if isinstance(transcription, IndirectShooting):
        for name, value in guess.costates.items():
            defaults[f"costate.{name}"] = jnp.asarray([value], dtype=float)
        return
    if isinstance(transcription, HermiteSimpson):
        # Collocation carries node states of its own, so it needs a
        # trajectory that actually satisfies the dynamics.
        scenario.problem.set_initial_guess(simulate_guess(guess, constants))


def create_optimization(
    transcription=None,
    variant: str = "ideal",
    size: int = 20,
    guess: Guess | None = None,
    **kwargs,
):
    """Build the GEMSEO scenario ready to be executed.

    Parameters
    ----------
    transcription
        The transcription; Hermite-Simpson collocation when omitted.
    variant
        The physical variant.
    size
        The discretization.
    guess
        The starting point.
    **kwargs
        Extra settings forwarded to :func:`create_scenario`.

    Returns
    -------
    tuple
        The :class:`.DynamicScenario` and the GEMSEO scenario built on it.
    """
    scenario = create_scenario(
        transcription=transcription,
        variant=variant,
        size=size,
        guess=guess,
        **kwargs,
    )
    # Generous but finite box bounds on everything. Under collocation the
    # states are design variables, and left unbounded they make a design
    # space half of whose bounds are infinite: SLSQP does not mind, but
    # every diagnostic that normalizes a design vector does, and GEMSEO's
    # history views divide by a range that is then not a number. The
    # limits below are five times the target and ten times the free-fall
    # speed that reaches it, so they describe the problem without ever
    # binding on it.
    span = max(abs(TARGET[0]), abs(TARGET[1]))
    reach = 5.0 * span
    fastest = 10.0 * float(numpy.sqrt(2.0 * GRAVITY * span))
    bounds = {"t_f": (0.05, 10.0)}
    for name in scenario.problem.design_variables:
        if name.startswith(("control.", "state.gamma", "initial.gamma")):
            bounds[name] = (0.0, HALF_PI)
        elif name.startswith(("state.v", "initial.v")):
            bounds[name] = (0.0, fastest)
        elif name.startswith(("state.", "initial.")):
            bounds[name] = (-reach, reach)
    formulation = OptimalControl(scenario, bounds=bounds)
    return scenario, formulation.create_gemseo_scenario()
