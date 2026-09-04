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

import jax.numpy as jnp

from gemseo_tempo.dynamics.controls import CubicControl
from gemseo_tempo.dynamics.integrators import DiffraxIntegrator
from gemseo_tempo.dynamics.integrators import Trajectory
from gemseo_tempo.outputs.constraints import PathConstraint
from gemseo_tempo.outputs.constraints import PeriodicityConstraint
from gemseo_tempo.outputs.functionals import Mayer
from gemseo_tempo.problems.albatross.model import DRAG_COEFFICIENT
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
from gemseo_tempo.transcriptions.base import TIMES
from gemseo_tempo.transcriptions.base import TRAJECTORY_PREFIX
from gemseo_tempo.transcriptions.base import BoundarySpec
from gemseo_tempo.transcriptions.collocation import HermiteSimpson

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

    Returns
    -------
    tuple
        The sample times, the states keyed by name, and the controls keyed
        by name.
    """
    guess = guess or Guess()
    values = _design_values(design)
    # The mass now comes out of the statics chain, since the wing weighs
    # something and the wing is a design variable.
    mass = float(create_statics()(values)["mass"])

    times = jnp.linspace(0.0, guess.period, count)
    heading = 2.0 * jnp.pi * times / guess.period
    # A quarter cycle out of phase with the heading: climbing while headed
    # into the wind, descending while headed with it.
    climb = -guess.climb_amplitude * jnp.cos(heading)
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
    heading_rate = 2.0 * jnp.pi / guess.period
    climb_accel = guess.climb_amplitude * jnp.sin(heading) * heading_rate
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

    return (
        PathConstraint("load_factor_limit", load_factor),
        PathConstraint("height_limit", height),
    )


def create_energy_scenario(
    transcription=None,
    size: int = 20,
    guess: Guess | None = None,
    w_ref: float = NOMINAL_WIND,
    design: dict[str, float] | None = None,
    integrator=None,
    **scenario_settings,
) -> DynamicScenario:
    """Build the fixed-wind, maximum-energy cycle.

    The bird starts from a given state and must return to the same height
    and attitude after one period. Its speed at the end is free, so any
    energy it extracts from the wind appears there.

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
    **scenario_settings
        Extra settings forwarded to :class:`.DynamicScenario`.

    Returns
    -------
    DynamicScenario
        The scenario, seeded with the guessed cycle.
    """
    guess = guess or Guess()
    times, states, controls = soaring_cycle(guess, w_ref, design, size + 1)
    values = _design_values(design)
    state_space = create_state_space()
    transcription = transcription or HermiteSimpson(num_segments=size)

    boundary = BoundarySpec(
        s0={name: float(values[0]) for name, values in states.items()},
        free_final_time=False,
    )
    scenario = DynamicScenario(
        state_space=state_space,
        integrator=integrator
        or DiffraxIntegrator(rtol=1e-8, atol=1e-8, n_save=size + 1),
        controls=_controls(times, controls),
        functionals=(
            Mayer(
                "energy",
                lambda t_final, s1, th: s1["h"] + s1["V"] ** 2 / (2.0 * GRAVITY),
            ),
        ),
        # The cycle closes in height and attitude, but deliberately not in
        # speed: that is where the extracted energy shows up.
        constraints=(
            PeriodicityConstraint(
                "cycle", ["h", "gamma", "psi"], offsets={"psi": TURN}
            ),
            *_path_constraints(),
        ),
        boundary=boundary,
        transcription=transcription,
        registry=REGISTRY,
        statics=create_statics(),
        constants={"w_ref": w_ref, **values},
        t0=0.0,
        t1=guess.period,
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
        integrator=DiffraxIntegrator(rtol=1e-8, atol=1e-8, n_save=size + 1),
        controls=_controls(normalized, controls),
        functionals=(Mayer("w_ref_objective", lambda t_final, s1, th: th["w_ref"]),),
        constraints=(
            PeriodicityConstraint(
                "cycle", ["h", "V", "gamma", "psi"], offsets={"psi": TURN}
            ),
            *_path_constraints(),
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

    Returns
    -------
    tuple
        The :class:`.DynamicScenario` and the GEMSEO scenario on it.
    """
    scenario = create_energy_scenario(transcription, size, guess, w_ref, design)
    formulation = OptimalControl(scenario, maximize=True, bounds=_bounds(scenario))
    return scenario, formulation.create_gemseo_scenario()


def create_codesign_optimization(
    transcription=None,
    size: int = 20,
    guess: Guess | None = None,
    design: dict[str, float] | None = None,
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

    Returns
    -------
    tuple
        The :class:`.DynamicScenario` and the GEMSEO scenario on it.
    """
    scenario = create_codesign_scenario(transcription, size, guess, design)
    bounds = _bounds(scenario)
    bounds["constant.w_ref"] = (1.0, 40.0)
    # The observed range of a wandering albatross rather than a wide box.
    # Within this model the wind a cycle needs falls monotonically with
    # aspect ratio, so whatever bound is given will be the answer; making
    # it the real bird's range at least makes the answer mean something.
    bounds["constant.wing_area"] = (0.5, 0.9)
    bounds["constant.wing_span"] = (2.5, 3.7)
    formulation = ControlAndDesign(
        scenario, ["w_ref", "wing_area", "wing_span"], bounds=bounds
    )
    return scenario, formulation.create_gemseo_scenario()


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
            bounds[name] = (-0.2, 1.4)
        elif name.startswith(("control.mu", "state.mu")):
            bounds[name] = (-1.5, 1.5)
        elif name.startswith("state.h"):
            bounds[name] = (MIN_HEIGHT, 200.0)
        elif name.startswith("state.V"):
            bounds[name] = (10.0, 80.0)
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
    "DRAG_COEFFICIENT",
    "MAX_LOAD_FACTOR",
    "NOMINAL_WIND",
    "TURN",
    "Guess",
    "create_codesign_optimization",
    "create_codesign_scenario",
    "create_energy_optimization",
    "create_energy_scenario",
    "gliding_baseline",
    "optimal_trajectory",
    "soaring_cycle",
]
