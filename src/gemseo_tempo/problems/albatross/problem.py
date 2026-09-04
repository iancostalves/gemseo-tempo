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
"""Builders for the albatross problem.

Two formulations are built here, and they ask opposite questions about the
same cycle.

**Energy** (:func:`create_energy_scenario`) fixes the wind and the bird and
asks how much the cycle can gain. The bird has to come back to the height and
attitude it started at, but not to the same speed, so whatever it gains shows
up as airspeed. Against a glide in still air, which can only lose, the
difference is the energy taken out of the wind.

**Co-design** (:func:`create_codesign_scenario`) asks the harder question:
what is the weakest wind in which this bird can fly forever? Now the cycle
must close completely, in speed as well as height and attitude, so the wind
gradient has to pay for the drag exactly. The wind strength, the cycle, the
duration and the wing are all free, and the objective is simply the wind.

## The starting point

A dynamic soaring cycle is not something an optimizer stumbles onto. The
guess is therefore built rather than picked, by **inverse dynamics**: pick the
shape of the cycle, then solve for the controls that would fly it.

The shape is the manoeuvre the energy equation asks for, which is to climb
while heading into the wind and descend while heading with it. Sweeping the
heading through a full turn and putting the climb angle a quarter cycle out of
phase with it does exactly that:

.. code-block::

    psi(t)   = 2 pi t / T
    gamma(t) = -A cos(psi)

Height follows by integrating ``V sin(gamma)``, which closes on itself because
the climb angle averages to zero. With the path known, the two rotational
equations of motion are inverted for the lift and the bank that would produce
it, and the lift becomes a lift coefficient. The airspeed is held constant in
the guess, so the speed equation is not satisfied and the collocation defects
start non-zero, which is the optimizer's job to fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy

from gemseo_tempo.dynamics.controls import CubicControl
from gemseo_tempo.dynamics.integrators import DiffraxIntegrator
from gemseo_tempo.dynamics.integrators import Trajectory
from gemseo_tempo.outputs.constraints import PathConstraint
from gemseo_tempo.outputs.constraints import PeriodicityConstraint
from gemseo_tempo.outputs.constraints import TerminalConstraint
from gemseo_tempo.outputs.functionals import Mayer
from gemseo_tempo.problems.albatross.model import MIN_HEIGHT
from gemseo_tempo.problems.albatross.model import REGISTRY as DYNAMICS_REGISTRY
from gemseo_tempo.problems.albatross.model import create_state_space
from gemseo_tempo.problems.albatross.model import wind_gradient
from gemseo_tempo.problems.albatross.model import wind_speed
from gemseo_tempo.problems.albatross.statics import AIR_DENSITY
from gemseo_tempo.problems.albatross.statics import DESIGN_DEFAULTS
from gemseo_tempo.problems.albatross.statics import GRAVITY
from gemseo_tempo.problems.albatross.statics import REGISTRY as DESIGN_REGISTRY
from gemseo_tempo.problems.albatross.statics import create_statics
from gemseo_tempo.scenario.dynamic_scenario import DynamicScenario
from gemseo_tempo.scenario.formulations import ControlAndDesign
from gemseo_tempo.scenario.formulations import OptimalControl
from gemseo_tempo.transcriptions.base import CONSTANT_PREFIX
from gemseo_tempo.transcriptions.base import TIMES
from gemseo_tempo.transcriptions.base import TRAJECTORY_PREFIX
from gemseo_tempo.transcriptions.base import BoundarySpec
from gemseo_tempo.transcriptions.collocation import HermiteSimpson

if TYPE_CHECKING:
    from collections.abc import Sequence

REGISTRY = DYNAMICS_REGISTRY
for _name in DESIGN_REGISTRY:
    REGISTRY.register(DESIGN_REGISTRY[_name], overwrite=True)
"""The metadata of the dynamics and the wing design together."""

NOMINAL_WIND = 12.0
"""The reference wind speed of the fixed-wind formulation, in m/s."""

MAX_LOAD_FACTOR = 4.5
"""The structural limit on lift over weight."""

CYCLE_HEIGHT_LIMIT = 2.0
"""The lowest the bird may fly, in m."""

WIND_BOUNDS = (1.0, 40.0)
"""The range the reference wind is searched over, in m/s."""

WING_BOUNDS = {
    "wing_area": (0.5, 0.9),
    "wing_span": (2.5, 3.7),
    "section": (0.0, 1.0),
}
"""The wing designs considered, in m^2 and m.

The observed range of a wandering albatross rather than a wide box. Within
this model the wind a cycle needs falls monotonically with aspect ratio, so
whatever bound is given will be the answer; making it the real bird's range
at least makes the answer mean something.
"""

TOPOLOGIES = ("loop", "weave")
"""The two cycle shapes.

A **loop** turns all the way round, so the heading gains a full turn and the
ground track closes as an O. A **weave** turns one way and back, so the
heading returns to itself and the track is an S. The periodicity offset is
the only thing that distinguishes them.
"""

WEAVE_HEADING_SWING = 1.6
"""The heading amplitude of the guessed weave, in rad."""

MISSIONS = ("energy", "distance", "time")
"""What the cycle is being flown for.

``energy`` banks whatever it extracts as airspeed and closes only in height
and attitude. ``distance`` and ``time`` close in speed as well, so the cycle
is energy-neutral and is spending exactly what it earns; the difference is
what it buys with it.
"""

MISSION_MAXIMIZES = {"energy": True, "distance": True, "time": False}
"""Whether each mission's objective is maximized."""

TURN = 2.0 * float(jnp.pi)
"""The heading gained over one cycle: the bird comes back on the same
heading having turned all the way round, not having stayed put."""


@dataclass
class Guess:
    """The shape of the starting cycle.

    The defaults describe a fast, shallow loop: a full turn every nine
    seconds at 32 m/s, rising and falling either side of 17 m. That keeps
    the lift coefficient near 0.6, the bank under 85 degrees and the load
    factor under 3, all comfortably inside the bird's envelope.
    """

    period: float = 9.0
    """The cycle duration, in s."""

    climb_amplitude: float = 0.30
    """The peak climb angle, in rad."""

    airspeed: float = 32.0
    """The airspeed held through the guessed cycle, in m/s."""

    mean_height: float = 17.0
    """The height the cycle is centred on, in m."""


def _design_values(design: dict[str, float] | None = None) -> dict[str, float]:
    """Return the wing design the statics chain runs on.

    Parameters
    ----------
    design
        Overrides of :data:`.DESIGN_DEFAULTS`.

    Returns
    -------
    dict
        The design constants, defaults filled in.
    """
    values = dict(DESIGN_DEFAULTS)
    values.update(design or {})
    return values


def soaring_cycle(
    guess: Guess | None = None,
    w_ref: float = NOMINAL_WIND,
    design: dict[str, float] | None = None,
    count: int = 41,
    topology: str = "loop",
):
    """Build a closed cycle and the controls that would fly it.

    Parameters
    ----------
    guess
        The cycle shape; the default when omitted.
    w_ref
        The reference wind speed, in m/s.
    design
        Overrides of :data:`.DESIGN_DEFAULTS`.
    count
        The number of samples over the cycle.
    topology
        ``"loop"`` to turn all the way round, ``"weave"`` to turn one way
        and back.

    Returns
    -------
    tuple
        The sample times, the states keyed by name, and the controls keyed
        by name.

    Raises
    ------
    ValueError
        If the topology is unknown.
    """
    if topology not in TOPOLOGIES:
        raise ValueError(
            f"Unknown topology {topology!r}; use one of {list(TOPOLOGIES)}."
        )
    guess = guess or Guess()
    values = _design_values(design)
    # The mass now comes out of the statics chain, since the wing weighs
    # something and the wing is a design variable.
    mass = float(create_statics()(values)["mass"])

    times = jnp.linspace(0.0, guess.period, count)
    phase = 2.0 * jnp.pi * times / guess.period
    if topology == "loop":
        # The heading winds through a full turn, and the climb angle lags it
        # by a quarter cycle: climbing while headed into the wind, descending
        # while headed with it.
        heading = phase
        climb = -guess.climb_amplitude * jnp.cos(phase)
        heading_rate = 2.0 * jnp.pi / guess.period * jnp.ones_like(times)
        climb_accel = (
            guess.climb_amplitude * jnp.sin(phase) * (2.0 * jnp.pi / guess.period)
        )
    else:
        # The heading swings either side of dead upwind and comes back, so
        # it gains nothing over the cycle. The climb peaks where the heading
        # is most nearly upwind, which is where climbing pays best.
        heading = jnp.pi + WEAVE_HEADING_SWING * jnp.sin(phase)
        climb = guess.climb_amplitude * jnp.cos(phase)
        heading_rate = (
            WEAVE_HEADING_SWING * jnp.cos(phase) * 2.0 * jnp.pi / guess.period
        )
        climb_accel = (
            -guess.climb_amplitude * jnp.sin(phase) * (2.0 * jnp.pi / guess.period)
        )
    speed = guess.airspeed * jnp.ones_like(times)

    # Height follows the climb angle, and closes because it averages to zero.
    climb_rate = speed * jnp.sin(climb)
    height = jnp.concatenate(
        [
            jnp.zeros(1),
            jnp.cumsum(0.5 * (climb_rate[1:] + climb_rate[:-1]) * jnp.diff(times)),
        ]
    )
    height = height - jnp.mean(height) + guess.mean_height

    # Invert the two rotational equations for the lift and the bank.
    wind_rate = wind_gradient(height, w_ref) * climb_rate
    lift_vertical = mass * (
        speed * climb_accel + GRAVITY * jnp.cos(climb)
    ) - mass * wind_rate * jnp.sin(climb) * jnp.cos(heading)
    lift_lateral = mass * speed * jnp.cos(
        climb
    ) * heading_rate - mass * wind_rate * jnp.sin(heading)
    lift = jnp.hypot(lift_vertical, lift_lateral)
    bank = jnp.arctan2(lift_lateral, lift_vertical)
    cl = lift / (0.5 * AIR_DENSITY * speed**2 * values["wing_area"])

    # The along-track positions are free, so they only have to be consistent.
    ground_east = speed * jnp.cos(climb) * jnp.cos(heading) + wind_speed(height, w_ref)
    ground_north = speed * jnp.cos(climb) * jnp.sin(heading)
    east = jnp.concatenate(
        [
            jnp.zeros(1),
            jnp.cumsum(0.5 * (ground_east[1:] + ground_east[:-1]) * jnp.diff(times)),
        ]
    )
    north = jnp.concatenate(
        [
            jnp.zeros(1),
            jnp.cumsum(0.5 * (ground_north[1:] + ground_north[:-1]) * jnp.diff(times)),
        ]
    )

    states = {
        "x": east,
        "y": north,
        "h": height,
        "V": speed,
        "gamma": climb,
        "psi": heading,
    }
    return times, states, {"cl": cl, "mu": bank}


def _controls(times, controls):
    """Build the control interpolants from sampled histories.

    Parameters
    ----------
    times
        The knot times.
    controls
        The control samples keyed by name.

    Returns
    -------
    tuple
        The :class:`.CubicControl` of each control.
    """
    return (
        CubicControl("cl", times, default_params=controls["cl"]),
        CubicControl("mu", times, default_params=controls["mu"]),
    )


def _path_constraints():
    """Return the load-factor and height path constraints.

    Returns
    -------
    tuple
        The constraints, each negative while satisfied.
    """

    def load_factor(t, s, u, theta):
        area = jnp.reshape(theta["wing_area"], ())
        mass = jnp.reshape(theta["mass"], ())
        dynamic_pressure = 0.5 * AIR_DENSITY * s["V"] ** 2 * area
        lift = dynamic_pressure * u["cl"]
        return lift / (mass * GRAVITY) - MAX_LOAD_FACTOR

    def height(t, s, u, theta):
        return CYCLE_HEIGHT_LIMIT - s["h"]

    def lift_limit(t, s, u, theta):
        # The wing stalls above its maximum lift coefficient, and this is
        # the constraint that actually binds: the cycle sits on it for most
        # of its length, because turning hard is what the manoeuvre needs.
        # It has to be a path constraint rather than a bound on the knots,
        # since the control is interpolated and overshoots between them.
        return u["cl"] - jnp.reshape(theta["cl_max"], ())

    return (
        PathConstraint("load_factor_limit", load_factor),
        PathConstraint("height_limit", height),
        PathConstraint("lift_limit", lift_limit),
    )


def create_energy_scenario(
    transcription=None,
    size: int = 20,
    guess: Guess | None = None,
    w_ref: float = NOMINAL_WIND,
    design: dict[str, float] | None = None,
    integrator=None,
    mission: str = "energy",
    topology: str = "loop",
    course: float = 0.0,
    target_distance: float | None = None,
    **scenario_settings,
) -> DynamicScenario:
    """Build a fixed-wind cycle, for one of three missions.

    The ``energy`` mission returns to the same height and attitude after one
    period with its speed left free, so whatever it extracts from the wind
    appears there. The other two close in speed as well, making the cycle
    energy-neutral, and ask what that neutral cycle can be spent on:
    ``distance`` covers as much ground as it can in a fixed time, and
    ``time`` covers a fixed distance as fast as it can.

    Parameters
    ----------
    transcription
        The transcription; Hermite-Simpson collocation when omitted.
    size
        The discretization: collocation segments or control intervals.
    guess
        The starting cycle.
    w_ref
        The reference wind speed, in m/s. Zero gives the gliding baseline.
    design
        Overrides of :data:`.DESIGN_DEFAULTS`.
    integrator
        The integrator; an adaptive one when omitted. A fixed-step
        integrator makes the whole map a smooth function of its inputs,
        which is what finite-difference checks need: adaptive stepping
        changes its step sequence discontinuously as the inputs move, and
        the resulting noise swamps a 1e-7 probe.
    mission
        One of :data:`MISSIONS`.
    topology
        ``"loop"`` or ``"weave"``; see :data:`TOPOLOGIES`.
    course
        The bearing progress is measured along, in rad from the downwind
        axis. Only the ``distance`` and ``time`` missions use it.
    target_distance
        The distance the ``time`` mission must cover, in m. Required by
        that mission and ignored by the others.
    **scenario_settings
        Extra settings forwarded to :class:`.DynamicScenario`.

    Returns
    -------
    DynamicScenario
        The scenario, seeded with the guessed cycle.

    Raises
    ------
    ValueError
        If the mission is unknown, or ``time`` is asked for without a
        target distance.
    """
    if mission not in MISSIONS:
        raise ValueError(f"Unknown mission {mission!r}; use one of {list(MISSIONS)}.")
    if mission == "time" and target_distance is None:
        raise ValueError("The 'time' mission needs a target_distance to cover.")

    guess = guess or Guess()
    times, states, controls = soaring_cycle(guess, w_ref, design, size + 1, topology)
    values = _design_values(design)
    state_space = create_state_space()
    transcription = transcription or HermiteSimpson(num_segments=size)

    def made_good(final_states):
        """Return the ground distance covered along the course bearing."""
        return jnp.cos(course) * final_states["x"] + jnp.sin(course) * final_states["y"]

    free_time = mission == "time"
    # A free duration puts the dynamics on the normalized clock, so the
    # control knots move with it.
    knots = jnp.linspace(0.0, 1.0, size + 1) if free_time else times
    closed = (
        ["h", "gamma", "psi"] if mission == "energy" else ["h", "V", "gamma", "psi"]
    )
    offsets = {"psi": TURN if topology == "loop" else 0.0}

    if mission == "energy":
        objective = Mayer(
            "energy",
            lambda t_final, s1, th: s1["h"] + s1["V"] ** 2 / (2.0 * GRAVITY),
        )
        extra = ()
    elif mission == "distance":
        objective = Mayer("distance", lambda t_final, s1, th: made_good(s1))
        extra = ()
    else:
        objective = Mayer("duration", lambda t_final, s1, th: t_final)
        extra = (
            TerminalConstraint(
                "distance_target",
                lambda t_final, s1, th: made_good(s1) - target_distance,
            ),
        )

    boundary = BoundarySpec(
        s0={name: float(values[0]) for name, values in states.items()},
        free_final_time=free_time,
        t_f=guess.period if free_time else None,
    )
    scenario = DynamicScenario(
        state_space=state_space,
        integrator=integrator
        or DiffraxIntegrator(rtol=1e-8, atol=1e-8, n_save=size + 1),
        controls=_controls(knots, controls),
        functionals=(objective,),
        constraints=(
            PeriodicityConstraint("cycle", closed, offsets=offsets),
            *_path_constraints(),
            *extra,
        ),
        boundary=boundary,
        transcription=transcription,
        registry=REGISTRY,
        statics=create_statics(),
        constants={"w_ref": w_ref, **values},
        t0=0.0,
        t1=1.0 if free_time else guess.period,
        **scenario_settings,
    )
    if isinstance(transcription, HermiteSimpson):
        scenario.problem.set_initial_guess(_trajectory(times, states, controls))
    return scenario


def create_codesign_scenario(
    transcription=None,
    size: int = 20,
    guess: Guess | None = None,
    design: dict[str, float] | None = None,
    integrator=None,
    target_distance: float | None = None,
) -> DynamicScenario:
    """Build the minimum-wind, energy-neutral cycle.

    Everything is free: the controls, the duration, the state the cycle
    starts from, and the wing itself. The cycle must close completely, so
    the wind gradient has to pay for the drag exactly, and the objective is
    the wind speed that makes that possible.

    Parameters
    ----------
    transcription
        The transcription; Hermite-Simpson collocation when omitted, since
        periodicity conditions well with node states.
    size
        The discretization.
    guess
        The starting cycle.
    design
        Overrides of :data:`.DESIGN_DEFAULTS`.
    integrator
        The integrator; an adaptive one when omitted. A fixed step makes the
        run time predictable, which is what a CI budget needs.
    target_distance
        A ground distance to cover, in m. Naming one turns the single
        objective into two, the wind and the time taken, and adds the
        terminal constraint that the distance is actually covered. That is
        the trade a Pareto front is drawn over: a bird that will accept
        more wind can cross faster.

    Returns
    -------
    DynamicScenario
        The scenario, seeded with the guessed cycle.
    """
    guess = guess or Guess()
    times, states, controls = soaring_cycle(guess, NOMINAL_WIND, design, size + 1)
    values = _design_values(design)
    state_space = create_state_space()
    transcription = transcription or HermiteSimpson(num_segments=size)

    if target_distance is None:
        objective = Mayer("w_ref_objective", lambda t_final, s1, th: th["w_ref"])
        crossing = ()
    else:
        objective = Mayer(
            "performance",
            lambda t_final, s1, th: jnp.stack([jnp.reshape(th["w_ref"], ()), t_final]),
        )
        crossing = (
            TerminalConstraint(
                "distance_target",
                lambda t_final, s1, th: s1["x"] - target_distance,
            ),
        )

    # The duration is free, so the cycle is parameterized on [0, 1].
    normalized = jnp.linspace(0.0, 1.0, size + 1)
    boundary = BoundarySpec(
        s0={name: float(values[0]) for name, values in states.items()},
        fix_initial={"h": False, "V": False, "gamma": False, "psi": False},
        free_final_time=True,
        t_f=guess.period,
    )
    scenario = DynamicScenario(
        state_space=state_space,
        integrator=integrator
        or DiffraxIntegrator(rtol=1e-8, atol=1e-8, n_save=size + 1),
        controls=_controls(normalized, controls),
        functionals=(objective,),
        constraints=(
            PeriodicityConstraint(
                "cycle", ["h", "V", "gamma", "psi"], offsets={"psi": TURN}
            ),
            *_path_constraints(),
            *crossing,
        ),
        boundary=boundary,
        transcription=transcription,
        registry=REGISTRY,
        statics=create_statics(),
        constants={"w_ref": NOMINAL_WIND, **values},
        t0=0.0,
        t1=1.0,
    )
    if isinstance(transcription, HermiteSimpson):
        # The warm start interpolates on physical node times, so the
        # trajectory is handed over on the cycle's own clock even though
        # the control knots live on the normalized one.
        scenario.problem.set_initial_guess(_trajectory(times, states, controls))
    return scenario


def optimal_trajectory(scenario, gemseo_scenario):
    """Recover the trajectory a solved scenario settled on.

    The transcription publishes every state, control and output on the save
    grid, so the cycle is read back from the discipline rather than
    re-simulated, which would reintroduce the integration error the
    transcription just worked to remove.

    Parameters
    ----------
    scenario
        The dynamic scenario that was optimized.
    gemseo_scenario
        The executed GEMSEO scenario.

    Returns
    -------
    Trajectory
        The states, controls and outputs on the save grid.
    """
    state_space = create_state_space()
    discipline = scenario.disciplines[0]
    optimum = gemseo_scenario.optimization_result.x_opt_as_dict
    discipline.execute(
        {
            name: value
            for name, value in optimum.items()
            if name in discipline.input_grammar.names
        }
    )
    data = discipline.io.data
    states = {}
    controls = {}
    outputs = {}
    for name in discipline.output_grammar.names:
        if not name.startswith(TRAJECTORY_PREFIX):
            continue
        short = name[len(TRAJECTORY_PREFIX) :]
        target = (
            states
            if short in state_space.states
            else controls
            if short in state_space.controls
            else outputs
        )
        target[short] = jnp.asarray(data[name])
    return Trajectory(
        ts=jnp.asarray(data[TIMES]),
        states=states,
        controls=controls,
        outputs=outputs,
    )


def _trajectory(times, states, controls):
    """Wrap sampled histories in a trajectory for seeding.

    Parameters
    ----------
    times
        The sample times.
    states
        The states keyed by name.
    controls
        The controls keyed by name.

    Returns
    -------
    Trajectory
        The trajectory the transcription seeds itself from.
    """
    return Trajectory(ts=times, states=states, controls=controls, outputs={})


def create_energy_optimization(
    transcription=None,
    size: int = 20,
    guess: Guess | None = None,
    w_ref: float = NOMINAL_WIND,
    design: dict[str, float] | None = None,
    **scenario_settings,
):
    """Build the GEMSEO scenario of the maximum-energy cycle.

    Parameters
    ----------
    transcription
        The transcription.
    size
        The discretization.
    guess
        The starting cycle.
    w_ref
        The reference wind speed, in m/s.
    design
        Overrides of :data:`.DESIGN_DEFAULTS`.
    **scenario_settings
        Forwarded to :func:`create_energy_scenario`, which is how the
        mission and the topology are chosen.

    Returns
    -------
    tuple
        The :class:`.DynamicScenario` and the GEMSEO scenario on it.
    """
    scenario = create_energy_scenario(
        transcription, size, guess, w_ref, design, **scenario_settings
    )
    bounds = _bounds(scenario)
    bounds["t_f"] = (3.0, 30.0)
    formulation = OptimalControl(
        scenario,
        maximize=MISSION_MAXIMIZES[scenario_settings.get("mission", "energy")],
        bounds=bounds,
    )
    return scenario, formulation.create_gemseo_scenario()


def create_codesign_optimization(
    transcription=None,
    size: int = 20,
    guess: Guess | None = None,
    design: dict[str, float] | None = None,
    design_names: Sequence[str] = ("w_ref", "wing_area", "wing_span", "section"),
    integrator=None,
    target_distance: float | None = None,
):
    """Build the GEMSEO scenario of the minimum-wind cycle.

    Parameters
    ----------
    transcription
        The transcription.
    size
        The discretization.
    guess
        The starting cycle.
    design
        Overrides of :data:`.DESIGN_DEFAULTS`.
    design_names
        The constants to optimize. The default frees the wing along with
        the wind, which is the simultaneous co-design; passing just
        ``("w_ref",)`` holds the wing and optimizes the trajectory alone,
        which is what the sequential route does once its wing is chosen.

    Returns
    -------
    tuple
        The :class:`.DynamicScenario` and the GEMSEO scenario on it.
    """
    scenario = create_codesign_scenario(
        transcription, size, guess, design, integrator, target_distance
    )
    bounds = _bounds(scenario)
    bounds["constant.w_ref"] = WIND_BOUNDS
    for name, limits in WING_BOUNDS.items():
        bounds[CONSTANT_PREFIX + name] = limits
    formulation = ControlAndDesign(scenario, list(design_names), bounds=bounds)
    return scenario, formulation.create_gemseo_scenario()


def solve_pareto_front(
    size: int = 12,
    target_distance: float = 300.0,
    n_points: int = 6,
    sub_optim_max_iter: int = 250,
):
    """Trade the weakest wind against the fastest crossing.

    Two things a soaring bird might want, which cannot both be had. A cycle
    that merely sustains itself can do so in very little wind but goes
    nowhere in a hurry; crossing the same ground faster means accepting
    stronger wind. The front is drawn with mNBI, which solves a
    sub-optimization per point rather than scalarizing, so the curve is the
    real trade rather than a weighted-sum sample of it.

    Parameters
    ----------
    size
        The discretization.
    target_distance
        The ground distance each cycle must cover, in m.
    n_points
        The number of Pareto points requested.
    sub_optim_max_iter
        The iteration budget of each sub-optimization.

    Returns
    -------
    ndarray
        The distinct Pareto points as rows of ``[w_ref, crossing time]``,
        sorted by wind.
    """
    from gemseo.algos.pareto.pareto_front import ParetoFront

    _, gemseo_scenario = create_codesign_optimization(
        size=size, target_distance=target_distance
    )
    gemseo_scenario.execute(
        algo_name="MNBI",
        n_sub_optim=n_points,
        sub_optim_algo="SLSQP",
        sub_optim_max_iter=sub_optim_max_iter,
        max_iter=20000,
    )
    front = ParetoFront.from_optimization_problem(
        gemseo_scenario.formulation.optimization_problem
    )
    points = numpy.asarray(front.f_optima)
    # mNBI returns each anchor several times, and the repeats differ in the
    # last few digits rather than exactly, so they have to be clustered
    # rather than deduplicated. Walking the sorted list and keeping a point
    # only when it has moved by more than a tolerance leaves the curve.
    ordered = points[numpy.argsort(points[:, 0])]
    kept = [ordered[0]]
    for candidate in ordered[1:]:
        moved = abs(candidate[0] - kept[-1][0]) / max(abs(kept[-1][0]), 1.0)
        if moved > 1e-3:
            kept.append(candidate)
    return numpy.asarray(kept)


def _bounds(scenario) -> dict[str, tuple[float, float]]:
    """Return the design-variable bounds of a scenario.

    Parameters
    ----------
    scenario
        The dynamic scenario.

    Returns
    -------
    dict
        The bounds, keyed by grammar name.
    """
    bounds: dict[str, tuple[float, float]] = {}
    for name in scenario.problem.design_variables:
        if name.startswith(("control.cl", "state.cl")):
            bounds[name] = (-0.2, 2.5)
        elif name.startswith(("control.mu", "state.mu")):
            bounds[name] = (-1.5, 1.5)
        elif name.startswith("state.h"):
            bounds[name] = (MIN_HEIGHT, 200.0)
        elif name.startswith("state.V"):
            bounds[name] = (10.0, 80.0)
        elif name.startswith(("state.x", "state.y")):
            # A cycle covers a few hundred metres, so these are generous
            # rather than binding. They exist because a design space with
            # unbounded components cannot be sampled, which is what
            # multi-start needs to do.
            bounds[name] = (-3000.0, 3000.0)
        elif name.startswith("state.gamma"):
            bounds[name] = (-0.5 * float(jnp.pi), 0.5 * float(jnp.pi))
        elif name.startswith("state.psi"):
            # Wide enough for a loop to wind through a full turn either way.
            bounds[name] = (-4.0 * float(jnp.pi), 4.0 * float(jnp.pi))
        elif name == "t_f":
            bounds[name] = (3.0, 30.0)
    return bounds


def gliding_baseline(size: int = 20, guess: Guess | None = None) -> float:
    """Return the best energy change achievable with no wind gradient.

    This is the honest baseline for the soaring cycle: the same problem in
    still air, where the bird can only trade height for speed and drag takes
    its cut either way.

    Parameters
    ----------
    size
        The discretization.
    guess
        The starting cycle.

    Returns
    -------
    float
        The specific energy at the end of the cycle, in m.
    """
    _, gemseo_scenario = create_energy_optimization(size=size, guess=guess, w_ref=0.0)
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=200)
    # The formulation maximizes, so GEMSEO stores the objective negated.
    return -float(gemseo_scenario.optimization_result.f_opt)


__all__ = [
    "CYCLE_HEIGHT_LIMIT",
    "MAX_LOAD_FACTOR",
    "MISSIONS",
    "MISSION_MAXIMIZES",
    "TOPOLOGIES",
    "NOMINAL_WIND",
    "TURN",
    "WIND_BOUNDS",
    "WING_BOUNDS",
    "Guess",
    "create_codesign_optimization",
    "create_codesign_scenario",
    "create_energy_optimization",
    "create_energy_scenario",
    "gliding_baseline",
    "optimal_trajectory",
    "soaring_cycle",
    "solve_pareto_front",
]
