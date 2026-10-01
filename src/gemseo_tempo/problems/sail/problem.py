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
"""Builders for the sailing problem.

Two legs are available. On the **downwind** leg the mark lies straight
downwind, and the interesting result is that sailing at it directly is not
the quickest way there: the boat does better to reach off to one side and
gybe back, which the optimizer finds without being told. On the **upwind**
leg the mark lies straight into the wind, so the boat cannot sail at it at
all and has to beat.

Both legs are seeded by simulating the dynamics once under the heading law
:func:`default_guess` reads off the polar, so collocation starts from a
trajectory that satisfies the equations of motion and shooting starts from
a final time of the right order.

Two points about the setup are worth stating, because both were arrived at
by measurement rather than by choice.

The heading is bounded to a single turn centred on the bearing to the
mark. It enters the dynamics only through its sine and cosine, so wider
bounds re-admit every trajectory shifted by a multiple of ``2*pi``, and
the optimizer drifts through a wrap instead of gybing.

The no-go cone needs no path constraint, so ``enforce_no_go`` is off by
default. The polar already gives a boat pointed inside the cone
essentially no speed, which is the physical reason it cannot go there, and
adding a hard constraint on top only forbids the brief crossing that a
tack is made of. The constraint that does bind usefully is the channel:
sailing wide of the direct line is what gybing buys speed with, so
limiting how far the boat may stray trades directly against the objective.

The downwind leg has one further wrinkle. Gybing to the north and gybing
to the south are equally good, and heading straight at the mark sits
exactly on the symmetric ridge between them, where the gradient gives the
optimizer no reason to prefer either. The guess therefore starts on one
tack rather than on that axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from gemseo_tempo.dynamics.controls import CubicControl
from gemseo_tempo.dynamics.integrators import DiffraxIntegrator
from gemseo_tempo.outputs.constraints import PathConstraint
from gemseo_tempo.outputs.functionals import Mayer
from gemseo_tempo.problems.sail.model import REGISTRY
from gemseo_tempo.problems.sail.model import channel_constraint
from gemseo_tempo.problems.sail.model import create_state_space
from gemseo_tempo.problems.sail.model import no_go_constraint
from gemseo_tempo.problems.sail.polar import best_vmg_angle
from gemseo_tempo.problems.sail.polar import boat_speed
from gemseo_tempo.problems.sail.wind import create_wind
from gemseo_tempo.scenario.dynamic_scenario import DynamicScenario
from gemseo_tempo.scenario.formulations import OptimalControl
from gemseo_tempo.transcriptions.base import BoundarySpec
from gemseo_tempo.transcriptions.collocation import HermiteSimpson

LEG_LENGTH = 800.0
"""The distance to the mark, in m."""

NOMINAL_WIND_PARAMETERS = {"wind_scale": 1.0, "wind_veer": 0.0}
"""The wind parameters of the nominal field: as generated, unrotated."""

SAIL_ADJOINT = "Direct"
"""The adjoint this problem is differentiated through, and not the default.

The library's default RecursiveCheckpoint adjoint returns inexact
derivatives here, and how inexact depends on how the reverse pass is
wrapped. Measured at the downwind optimum against the exact sensitivity of
the along-track miss to the wind strength (1241.15, which central
differences reproduce to eight digits at two step sizes):

=========================== =================== ======
route                       RecursiveCheckpoint Direct
=========================== =================== ======
one cotangent (``grad``)    exact               exact
``jacrev``, several outputs +0.61%              exact
that, under a ``vmap``      -8.95%              exact
=========================== =================== ======

Only the first route is safe under the checkpointed adjoint, and it is the
one route GEMSEO does not take: a discipline Jacobian is a ``jacrev`` over
every output at once. So the Jacobian this problem hands the optimizer is
the middle row.

What singles this problem out is the wind field. It is the only one whose
right-hand side reads an interpolated
:class:`~gemseo_tempo.dynamics.fields.SpatialField`, whose transpose is a
scatter-add, and that is what the checkpointed reverse pass mishandles once
it is itself batched. The brachistochrone and the albatross are exact under
both adjoints at every route measured.

The Direct adjoint unrolls the solve instead of checkpointing it and is
exact everywhere measured. It costs about a third more to solve the
downwind leg (23.0 s against 17.0 s at the default discretization), which
is the price of an answer that is right.

A caution about how this was found, because the first diagnosis was wrong.
The two adjoints also disagreed at the *starting guess*, and that
disagreement was not the adjoint at all: the polar had a corner at dead
downwind (see :data:`~gemseo_tempo.problems.sail.polar._SHAPE`), the guess
sails straight through it, and the two adjoints were landing on opposite
sides of it. With the polar smooth, the guess is exact under both adjoints
and only the optimum separates them. A one-sided derivative at a kink and a
wrong derivative look identical from a single number.
"""


def sail_integrator(**settings) -> DiffraxIntegrator:
    """Build an integrator for this problem, with the adjoint it needs.

    Use this rather than :class:`.DiffraxIntegrator` directly. The default
    adjoint gives wrong Jacobians on this problem, for the reasons
    :data:`SAIL_ADJOINT` records, and :func:`create_scenario` refuses an
    integrator carrying it.

    Parameters
    ----------
    **settings
        Settings forwarded to :class:`.DiffraxIntegrator`; ``adjoint``
        defaults to :data:`SAIL_ADJOINT`.

    Returns
    -------
    DiffraxIntegrator
        The integrator.
    """
    settings.setdefault("adjoint", SAIL_ADJOINT)
    return DiffraxIntegrator(**settings)


CORRIDOR_HALF_WIDTH = 500.0
"""How far north or south of the rhumb line the boat may sail, in m.

A race is sailed in a bounded area. Without one the optimum is free to
wander as far across the wind as a gybe angle takes it, which on the
upwind leg is a long way and nothing like a race.
"""

LEGS = {
    # The wind blows east, so the mark to the east is dead downwind and the
    # mark to the west is dead upwind.
    "downwind": (LEG_LENGTH, 0.0),
    "upwind": (-LEG_LENGTH, 0.0),
}
"""The mark position of each leg, in m."""


@dataclass
class Guess:
    """A starting point, as an affine heading law and a time.

    The heading is guessed as ``psi(tau) = level + slope * tau`` over the
    normalized time, so a nonzero slope sweeps the boat through a single
    tack or gybe.
    """

    psi_level: float = 0.7
    """The heading at the start, in rad from the east axis."""

    psi_slope: float = -1.4
    """The change in heading over the leg, in rad."""

    t_f: float = 220.0
    """The guessed time to the mark, in s."""

    def headings(self, count: int):
        """Sample the guessed heading law.

        Parameters
        ----------
        count
            The number of nodes.

        Returns
        -------
        Array
            The heading at ``count`` evenly spaced normalized times.
        """
        return self.psi_level + self.psi_slope * jnp.linspace(0.0, 1.0, count)


def default_guess(leg: str = "downwind", wind_name: str = "uniform") -> Guess:
    """Build the starting point of a leg from the polar itself.

    The best speed made good fixes both halves of the guess. Its true wind
    angle gives the two headings that achieve it, one per tack, and the
    guess sweeps from one to the other, so the boat starts out already
    doing the thing the leg calls for. The made-good speed then gives the
    time, ``distance / vmg``, which is the quickest the leg could possibly
    be sailed and so a sound scale for the free final time.

    Hand-picked guesses do not survive a change of polar or wind, and a
    poor one is not harmless: single shooting starting a downwind leg 38%
    slow in the final time stalls on its first linesearch, while the same
    problem started here converges to within 0.4% of collocation.

    Parameters
    ----------
    leg
        The leg name.
    wind_name
        The wind field name.

    Returns
    -------
    Guess
        The starting point.
    """
    mark = LEGS[leg]
    wind = create_wind(wind_name)
    east, north = wind.velocity(0.0, 0.0)
    tws = float(jnp.hypot(east, north))
    from_bearing = float(jnp.arctan2(-north, -east))
    bearing = float(jnp.arctan2(mark[1], mark[0]))

    angle = best_vmg_angle("downwind" if leg == "downwind" else "upwind")
    speed = float(boat_speed(angle, tws))
    made_good = speed * abs(float(jnp.cos(jnp.radians(angle))))

    # The two tacks are symmetric about the wind axis. Express both in the
    # turn centred on the bearing to the mark, which is the window the
    # design bounds use.
    def _in_window(psi: float) -> float:
        return bearing + float(
            jnp.arctan2(jnp.sin(psi - bearing), jnp.cos(psi - bearing))
        )

    offset = float(jnp.radians(angle))
    start = _in_window(from_bearing + offset)
    end = _in_window(from_bearing - offset)
    return Guess(
        psi_level=start,
        psi_slope=end - start,
        t_f=float(jnp.hypot(*mark)) / made_good,
    )


def straight_run_time(leg: str = "downwind", wind_name: str = "uniform") -> float:
    """Return the time to sail straight at the mark.

    This is the honest baseline the optimizer has to beat: point the boat
    at the mark and hold that heading.

    Parameters
    ----------
    leg
        The leg name.
    wind_name
        The wind field name.

    Returns
    -------
    float
        The time in s, or infinity when the mark lies inside the no-go
        cone and the boat simply cannot sail there directly.
    """
    mark = LEGS[leg]
    wind = create_wind(wind_name)
    east, north = wind.velocity(0.0, 0.0)
    tws = float(jnp.hypot(east, north))
    heading = jnp.arctan2(mark[1], mark[0])
    from_bearing = jnp.arctan2(-north, -east)
    twa = jnp.degrees(
        jnp.arctan2(jnp.sin(heading - from_bearing), jnp.cos(heading - from_bearing))
    )
    speed = float(boat_speed(twa, tws))
    distance = float(jnp.hypot(*mark))
    if speed <= 0.01 * tws:
        return float("inf")
    return distance / speed


def create_scenario(
    transcription=None,
    leg: str = "downwind",
    size: int = 25,
    guess: Guess | None = None,
    wind_name: str = "uniform",
    enforce_no_go: bool = False,
    channel_half_width: float | None = CORRIDOR_HALF_WIDTH,
    integrator=None,
    check_adjoint: bool = True,
    **scenario_settings,
) -> DynamicScenario:
    """Build the sailing scenario for one transcription.

    Parameters
    ----------
    transcription
        The transcription; Hermite-Simpson collocation when omitted. The
        indirect method is not applicable, since the problem carries a
        path constraint.
    leg
        ``"downwind"`` or ``"upwind"``.
    size
        The discretization: collocation segments or control intervals.
    guess
        The starting point; the default for the leg when omitted.
    wind_name
        The wind field, one of
        :data:`~gemseo_tempo.problems.sail.wind.WIND_FIELDS`.
    enforce_no_go
        Whether to add the no-go path constraint; off by default, since
        the polar already denies the boat any speed inside the cone and
        the constraint only forbids the crossing a tack is made of.
    channel_half_width
        The distance either side of the rhumb line the boat may use, in
        m: the race area, :data:`CORRIDOR_HALF_WIDTH`, by default, and
        unconstrained when ``None``.
    integrator
        The execution model of the dynamics; an adaptive integrator with
        the adjoint this problem needs when omitted. Worth passing when
        the trajectory is to be evaluated many times over, where a fixed
        step makes every evaluation cost the same; build it with
        :func:`sail_integrator` so it carries the right adjoint.
    check_adjoint
        Whether to refuse an integrator whose adjoint gives wrong
        Jacobians here. Only the test recording *why* it is wrong turns
        this off.
    **scenario_settings
        Extra settings forwarded to :class:`.DynamicScenario`.

    Returns
    -------
    DynamicScenario
        The scenario, seeded with the guess.

    Raises
    ------
    ValueError
        If the leg is unknown, or the integrator carries an adjoint known
        to be wrong on this problem.
    """
    if leg not in LEGS:
        raise ValueError(f"Unknown leg {leg!r}; use one of {sorted(LEGS)}.")
    if (
        check_adjoint
        and integrator is not None
        and str(getattr(integrator, "adjoint_method", SAIL_ADJOINT)) != SAIL_ADJOINT
    ):
        msg = (
            "This problem must be differentiated through the"
            f" {SAIL_ADJOINT} adjoint; see SAIL_ADJOINT for why the default"
            " returns wrong Jacobians here. Build the integrator with"
            " sail_integrator() to get it right."
        )
        raise ValueError(msg)
    guess = guess or default_guess(leg, wind_name)
    wind = create_wind(wind_name)
    state_space = create_state_space(wind)
    transcription = transcription or HermiteSimpson(num_segments=size)
    mark = LEGS[leg]

    path_constraints = []
    if enforce_no_go:
        path_constraints.append(PathConstraint("no_go", no_go_constraint(wind)))
    if channel_half_width is not None:
        path_constraints.append(
            PathConstraint("channel", channel_constraint(channel_half_width))
        )
    path_constraints = tuple(path_constraints)

    boundary = BoundarySpec(
        s0={"x": 0.0, "y": 0.0},
        fix_final={"x": True, "y": True},
        s1={"x": mark[0], "y": mark[1]},
        free_final_time=True,
        t_f=guess.t_f,
    )
    # The final time is free, so the dynamics run on the normalized time
    # and the heading is parameterized on [0, 1].
    nodes = jnp.linspace(0.0, 1.0, size + 1)
    control = CubicControl("psi", nodes, default_params=guess.headings(size + 1))
    scenario = DynamicScenario(
        state_space=state_space,
        integrator=integrator
        or DiffraxIntegrator(
            rtol=1e-8, atol=1e-8, n_save=size + 1, adjoint=SAIL_ADJOINT
        ),
        controls=(control,),
        # Not ``t_f``: with a free final time that name is already a
        # design variable, and GEMSEO keys its history dataset by name, so
        # an objective sharing one with an input gives every column twice.
        # Its post-processing then indexes a two-column array as if it were
        # one and fails, which is how this was found. The albatross avoids
        # the same collision by calling its objective ``w_ref_objective``.
        functionals=(Mayer("arrival_time", lambda t_final, states, theta: t_final),),
        constraints=path_constraints,
        boundary=boundary,
        transcription=transcription,
        registry=REGISTRY,
        constants=NOMINAL_WIND_PARAMETERS,
        t0=0.0,
        t1=1.0,
        **scenario_settings,
    )
    if isinstance(transcription, HermiteSimpson):
        scenario.problem.set_initial_guess(simulate_guess(state_space, guess, size))
    return scenario


def simulate_guess(state_space, guess: Guess, size: int = 25):
    """Integrate the dynamics under the guessed heading law.

    Parameters
    ----------
    state_space
        The boat model.
    guess
        The starting point.
    size
        The discretization, which sets the trajectory resolution.

    Returns
    -------
    Trajectory
        A trajectory consistent with the equations of motion, spanning
        ``[0, guess.t_f]``.
    """
    times = jnp.linspace(0.0, guess.t_f, size + 1)
    control = CubicControl("psi", times, default_params=guess.headings(size + 1))
    integrator = DiffraxIntegrator(rtol=1e-7, atol=1e-7, n_save=size + 1)
    return integrator.integrate(
        state_space,
        {"x": 0.0, "y": 0.0},
        (control,),
        {"psi": control.default_params},
        dict(NOMINAL_WIND_PARAMETERS),
        0.0,
        guess.t_f,
    )


def create_optimization(
    transcription=None,
    leg: str = "downwind",
    size: int = 25,
    guess: Guess | None = None,
    **kwargs,
):
    """Build the GEMSEO scenario ready to be executed.

    Parameters
    ----------
    transcription
        The transcription; collocation when omitted.
    leg
        ``"downwind"`` or ``"upwind"``.
    size
        The discretization.
    guess
        The starting point.
    **kwargs
        Extra settings forwarded to :func:`create_scenario`.

    Returns
    -------
    tuple
        The :class:`.DynamicScenario` and the GEMSEO scenario on it.
    """
    scenario = create_scenario(
        transcription=transcription, leg=leg, size=size, guess=guess, **kwargs
    )
    # One full turn centred on the bearing to the mark. The heading enters
    # the dynamics only through its sine and cosine, so every bound wider
    # than a turn admits the same trajectories again, shifted by a multiple
    # of 2*pi. Those copies are what the optimizer wanders into: given two
    # turns it drifts through a wrap instead of gybing, and the downwind
    # solution degrades from a 10% gain to 3%.
    mark = LEGS[leg]
    bearing = float(jnp.arctan2(mark[1], mark[0]))
    half_turn = float(jnp.pi)
    bounds = {"t_f": (10.0, 4000.0)}
    for name in scenario.problem.design_variables:
        if name.startswith(("control.psi", "state.psi")):
            bounds[name] = (bearing - half_turn, bearing + half_turn)
    formulation = OptimalControl(scenario, bounds=bounds)
    return scenario, formulation.create_gemseo_scenario()
