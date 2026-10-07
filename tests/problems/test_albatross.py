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
"""Tests of the albatross dynamic soaring problem."""

from __future__ import annotations

import logging
import re
import time

import jax
import jax.numpy as jnp
import numpy
import pytest
from matplotlib import pyplot

from gemseo_tempo.dynamics import CubicControl
from gemseo_tempo.dynamics import DiffraxIntegrator
from gemseo_tempo.dynamics.integrators import Trajectory
from gemseo_tempo.outputs import PeriodicityConstraint
from gemseo_tempo.problems.albatross import DESIGN_DEFAULTS
from gemseo_tempo.problems.albatross import MIN_HEIGHT
from gemseo_tempo.problems.albatross import ROUGHNESS
from gemseo_tempo.problems.albatross import Guess
from gemseo_tempo.problems.albatross import create_codesign_optimization
from gemseo_tempo.problems.albatross import create_codesign_scenario
from gemseo_tempo.problems.albatross import create_energy_optimization
from gemseo_tempo.problems.albatross import create_energy_scenario
from gemseo_tempo.problems.albatross import create_state_space
from gemseo_tempo.problems.albatross import create_statics
from gemseo_tempo.problems.albatross import energy_budget
from gemseo_tempo.problems.albatross import gliding_baseline
from gemseo_tempo.problems.albatross import optimal_trajectory
from gemseo_tempo.problems.albatross import plot_controls
from gemseo_tempo.problems.albatross import plot_cycle_shapes
from gemseo_tempo.problems.albatross import plot_energy
from gemseo_tempo.problems.albatross import plot_energy_analysis
from gemseo_tempo.problems.albatross import plot_frames
from gemseo_tempo.problems.albatross import plot_maneuver_pattern
from gemseo_tempo.problems.albatross import plot_pareto
from gemseo_tempo.problems.albatross import plot_polar
from gemseo_tempo.problems.albatross import plot_scenario
from gemseo_tempo.problems.albatross import plot_section_trade
from gemseo_tempo.problems.albatross import plot_trajectory
from gemseo_tempo.problems.albatross import plot_wind_profile
from gemseo_tempo.problems.albatross import schematic_cycle
from gemseo_tempo.problems.albatross import soaring_cycle
from gemseo_tempo.problems.albatross import solve_pareto_front
from gemseo_tempo.problems.albatross import wind_gradient
from gemseo_tempo.problems.albatross import wind_speed
from gemseo_tempo.problems.albatross.plots import _phases
from gemseo_tempo.problems.albatross.plots import _series
from gemseo_tempo.problems.albatross.plots import _unit_vectors
from gemseo_tempo.problems.albatross.problem import WING_BOUNDS
from gemseo_tempo.problems.albatross.sequential import glide_ratio
from gemseo_tempo.transcriptions import SingleShooting

logging.disable(logging.CRITICAL)

SIZE = 12
"""The discretization of the optimization tests, kept small for speed."""

STRONG_WIND = 22.0
"""A wind the cycle gains energy in, in m/s."""


def solve_energy(w_ref, size=SIZE):
    """Optimize the fixed-wind cycle and return the scenario and result.

    Parameters
    ----------
    w_ref
        The reference wind speed, in m/s.
    size
        The discretization.

    Returns
    -------
    tuple
        The dynamic scenario and the executed GEMSEO scenario.
    """
    scenario, gemseo_scenario = create_energy_optimization(size=size, w_ref=w_ref)
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=300)
    return scenario, gemseo_scenario


@pytest.fixture(scope="module")
def soaring():
    """Return the cycle optimized in a strong wind."""
    return solve_energy(STRONG_WIND)


@pytest.fixture(scope="module")
def gliding():
    """Return the same cycle with no wind at all."""
    return solve_energy(0.0)


@pytest.fixture(scope="module")
def codesign():
    """Return the minimum-wind co-design optimum."""
    scenario, gemseo_scenario = create_codesign_optimization(size=SIZE)
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=500)
    return scenario, gemseo_scenario


# ------------------------------------------------------------- the statics


def test_wing_geometry_matches_the_closed_forms():
    """Aspect ratio and induced factor are what the definitions say."""
    outputs = create_statics()(DESIGN_DEFAULTS)
    area = DESIGN_DEFAULTS["wing_area"]
    span = DESIGN_DEFAULTS["wing_span"]
    assert float(outputs["aspect_ratio"]) == pytest.approx(span**2 / area)
    assert float(outputs["induced_factor"]) == pytest.approx(
        1.0 / (numpy.pi * DESIGN_DEFAULTS["oswald"] * span**2 / area)
    )


def test_the_default_bird_is_an_albatross():
    """The defaults describe a real animal, not an arbitrary one."""
    outputs = create_statics()(DESIGN_DEFAULTS)
    assert float(outputs["mass"]) == pytest.approx(8.5, abs=0.05)
    assert 15.0 < float(outputs["aspect_ratio"]) < 22.0
    # Stall speed of order 10 m/s: fast for a bird, slow for an aircraft.
    assert 8.0 < float(outputs["stall_speed"]) < 15.0


def test_a_longer_wing_costs_mass():
    """The structural law is what stops the aspect ratio running away."""
    chain = create_statics()
    stubby = dict(DESIGN_DEFAULTS, wing_area=0.8, wing_span=3.2)
    lanky = dict(DESIGN_DEFAULTS, wing_area=0.5, wing_span=4.2)
    assert float(chain(lanky)["aspect_ratio"]) > float(chain(stubby)["aspect_ratio"])
    assert float(chain(lanky)["wing_mass"]) > float(chain(stubby)["wing_mass"])


def test_the_statics_chain_is_jittable():
    """The chain composes into a traced program, not just a callable."""
    chain = create_statics()
    jitted = jax.jit(chain.__call__)
    assert float(jitted(DESIGN_DEFAULTS)["mass"]) == pytest.approx(
        float(chain(DESIGN_DEFAULTS)["mass"])
    )


# ---------------------------------------------------------------- the wind


def test_wind_profile_is_physical():
    """The wind grows with height, and its gradient is positive and falling."""
    heights = numpy.linspace(MIN_HEIGHT, 60.0, 200)
    speeds = numpy.array([float(wind_speed(h, 12.0)) for h in heights])
    gradients = numpy.array([float(wind_gradient(h, 12.0)) for h in heights])
    assert numpy.all(numpy.diff(speeds) > 0.0)
    assert numpy.all(gradients > 0.0)
    # Steepest near the surface, which is why the bird flies low.
    assert numpy.all(numpy.diff(gradients) < 0.0)
    # The profile is singular at the roughness length; the smooth floor
    # keeps it finite and close to its value at the lowest flyable height.
    at_the_surface = float(wind_speed(ROUGHNESS, 12.0))
    assert numpy.isfinite(at_the_surface)
    assert at_the_surface == pytest.approx(
        float(wind_speed(MIN_HEIGHT, 12.0)), rel=0.05
    )


def test_the_wind_gradient_is_the_derivative_of_the_profile():
    """The analytic gradient agrees with differentiating the profile."""
    derivative = jax.grad(lambda h: wind_speed(h, 12.0))
    for height in (2.0, 5.0, 15.0, 40.0):
        assert float(wind_gradient(height, 12.0)) == pytest.approx(
            float(derivative(height)), rel=1e-3
        )


def test_no_wind_means_no_gradient():
    assert float(wind_gradient(10.0, 0.0)) == 0.0
    assert float(wind_speed(10.0, 0.0)) == 0.0


# ----------------------------------------------------------- the energetics


def test_gliding_only_ever_loses_energy(gliding):
    """With no gradient there is nothing to extract, and drag still bites."""
    scenario, gemseo_scenario = gliding
    trajectory = optimal_trajectory(scenario, gemseo_scenario)
    budget = energy_budget(trajectory)
    assert budget["gradient_work"] == pytest.approx(0.0, abs=1e-9)
    assert budget["drag_work"] < 0.0
    assert budget["air_change"] < 0.0


@pytest.mark.parametrize("w_ref", [0.0, 12.0])
def test_the_energy_budget_closes(w_ref):
    """The rate of change of energy is exactly drag plus wind-gradient work.

    This is the check a sign error cannot survive: with the wind terms the
    wrong way round the cycle would still appear to gain energy, but the
    gain would not match the work that produced it.

    It is asserted on a directly integrated trajectory rather than an
    optimized one, on a fine grid, so what is being measured is the model
    and not the quadrature.
    """
    state_space = create_state_space()
    statics = create_statics()(DESIGN_DEFAULTS)
    theta = {
        "w_ref": w_ref,
        "roughness": ROUGHNESS,
        "mass": float(statics["mass"]),
        "wing_area": DESIGN_DEFAULTS["wing_area"],
        "induced_factor": float(statics["induced_factor"]),
        "cd0": float(statics["cd0"]),
        "cl_max": float(statics["cl_max"]),
    }
    knots = jnp.linspace(0.0, 12.0, 9)
    controls = (
        CubicControl("cl", knots, default_params=0.7 * jnp.ones(9)),
        CubicControl("mu", knots, default_params=0.3 * jnp.ones(9)),
    )
    trajectory = DiffraxIntegrator(rtol=1e-12, atol=1e-12, n_save=2001).integrate(
        state_space,
        {"x": 0.0, "y": 0.0, "h": 20.0, "V": 25.0, "gamma": -0.05, "psi": 0.6},
        controls,
        {
            name: control.default_params
            for name, control in zip(("cl", "mu"), controls, strict=True)
        },
        theta,
        0.0,
        12.0,
    )
    times = numpy.asarray(trajectory.ts)
    measured = numpy.gradient(_series(trajectory, "specific_energy"), times)
    accounted = _series(trajectory, "drag_power") + _series(
        trajectory, "gradient_power"
    )
    # Trim the ends, where the one-sided finite difference is least accurate.
    error = numpy.max(numpy.abs(measured - accounted)[3:-3])
    assert error / numpy.max(numpy.abs(accounted)) < 5e-3


def test_the_wind_gradient_is_what_pays_for_the_flight(soaring):
    """Energy comes from the gradient, and only from the gradient."""
    scenario, gemseo_scenario = soaring
    trajectory = optimal_trajectory(scenario, gemseo_scenario)
    budget = energy_budget(trajectory)
    assert budget["gradient_work"] > 0.0
    assert budget["drag_work"] < 0.0
    # In a strong enough wind the gradient more than covers the drag.
    assert budget["gradient_work"] > -budget["drag_work"]
    assert budget["air_change"] > 0.0


def test_the_two_frames_tell_different_stories(soaring):
    """The inertial frame swings far more than the air-relative one.

    Both describe the same flight. The bird is carried along by air that
    moves faster higher up, so its speed over the water changes much more
    than its speed through the air, and the gap between the two is exactly
    the wind it is riding.
    """
    scenario, gemseo_scenario = soaring
    trajectory = optimal_trajectory(scenario, gemseo_scenario)
    air = _series(trajectory, "specific_energy")
    ground = _series(trajectory, "ground_energy")
    assert numpy.ptp(ground) > numpy.ptp(air)
    # Ground speed exceeds airspeed on average, because the wind helps more
    # of the time than it hinders over a cycle that is flown deliberately.
    assert numpy.mean(ground) > numpy.mean(air)


def test_soaring_beats_gliding(soaring, gliding):
    """The whole point: a wind gradient turns a losing cycle into a winning one."""
    soaring_scenario, soaring_gemseo = soaring
    gliding_scenario, gliding_gemseo = gliding
    soaring_energy = -float(soaring_gemseo.optimization_result.f_opt)
    gliding_energy = -float(gliding_gemseo.optimization_result.f_opt)
    assert soaring_energy > gliding_energy
    # The margin is large, not marginal.
    assert soaring_energy > 1.2 * gliding_energy


# --------------------------------------------------------- the formulations


def test_the_cycle_closes(soaring):
    """Height and attitude return to where they started, to 1e-5."""
    _, gemseo_scenario = soaring
    result = gemseo_scenario.optimization_result
    database = gemseo_scenario.formulation.optimization_problem.database
    residuals = numpy.asarray(
        database.get_function_value("constraint.cycle", result.x_opt)
    )
    assert numpy.max(numpy.abs(residuals)) < 1e-5


def test_the_heading_comes_back_around(soaring):
    """The bird turns through a full circle rather than staying put.

    The heading is periodic only modulo a turn, which is why the constraint
    carries an offset. Without it the cycle could not close at all.
    """
    scenario, gemseo_scenario = soaring
    trajectory = optimal_trajectory(scenario, gemseo_scenario)
    heading = _series(trajectory, "psi")
    assert heading[-1] - heading[0] == pytest.approx(2.0 * numpy.pi, abs=1e-4)


def test_codesign_finds_a_weaker_wind_than_the_fixed_cycle(codesign):
    """Freeing the cycle and the wing buys a lot of wind.

    The fixed-wind formulation flies one prescribed cycle shape and needs
    around 18 m/s before it breaks even. Letting the optimizer choose the
    cycle, its duration, where it starts and the wing itself brings that
    down by a wide margin.
    """
    _, gemseo_scenario = codesign
    minimum_wind = float(gemseo_scenario.optimization_result.f_opt)
    assert 4.0 < minimum_wind < 14.0
    assert minimum_wind < 0.75 * 17.7


def test_codesign_uses_the_span_it_is_given(codesign):
    """The wing goes to the highest aspect ratio the bounds allow.

    Within this model the wind a cycle needs falls monotonically with
    aspect ratio, so the design bound is active at the optimum. That is a
    property of the model rather than a failure of the optimizer, and the
    bounds are the observed range of a real bird for exactly that reason.
    """
    _, gemseo_scenario = codesign
    optimum = gemseo_scenario.optimization_result.x_opt_as_dict
    area = float(optimum["constant.wing_area"][0])
    span = float(optimum["constant.wing_span"][0])
    assert span == pytest.approx(3.7, abs=1e-3)
    assert (
        span**2 / area
        > DESIGN_DEFAULTS["wing_span"] ** 2 / (DESIGN_DEFAULTS["wing_area"])
    )


def test_the_codesign_cycle_closes_completely(codesign):
    """Speed closes too, so the cycle is genuinely energy-neutral."""
    scenario, gemseo_scenario = codesign
    result = gemseo_scenario.optimization_result
    database = gemseo_scenario.formulation.optimization_problem.database
    residuals = numpy.asarray(
        database.get_function_value("constraint.cycle", result.x_opt)
    )
    assert numpy.max(numpy.abs(residuals)) < 1e-5
    trajectory = optimal_trajectory(scenario, gemseo_scenario)
    air = _series(trajectory, "specific_energy")
    assert air[-1] == pytest.approx(air[0], abs=1e-3)


def test_the_split_assembly_jacobians_are_right():
    """Finite differences agree with the analytic Jacobian, split apart.

    The split assembly is the one that differentiates the statics chain
    separately from the dynamics, so it is where a mistake in the coupling
    between the wing and the equations of motion would show up.
    """
    scenario = create_energy_scenario(
        transcription=SingleShooting(),
        size=6,
        assembly="split",
        integrator=DiffraxIntegrator(dt0=0.01, n_save=7),
    )
    for discipline in scenario.disciplines:
        assert discipline.check_jacobian(threshold=1e-3, step=1e-7), discipline.name


# --------------------------------------------------------------- the guards


def test_periodicity_offsets_must_name_periodic_states():
    with pytest.raises(ValueError, match="not periodic"):
        PeriodicityConstraint("cycle", ["h", "V"], offsets={"psi": 1.0})


def test_a_missing_series_is_reported():
    trajectory = Trajectory(
        ts=jnp.zeros(3), states={"h": jnp.zeros(3)}, controls={}, outputs={}
    )
    assert _series(trajectory, "h") is not None
    with pytest.raises(KeyError, match="No series named"):
        _series(trajectory, "not_a_series")


def test_the_guessed_cycle_is_closed_and_inside_the_envelope():
    """The starting point is flyable before any optimizer touches it."""
    _, states, controls = soaring_cycle(Guess(), 12.0, None, 41)
    height = numpy.asarray(states["h"])
    assert height.min() > MIN_HEIGHT
    assert float(states["h"][-1]) == pytest.approx(float(states["h"][0]), abs=1e-6)
    assert float(states["psi"][-1] - states["psi"][0]) == pytest.approx(2.0 * numpy.pi)
    lift_coefficient = numpy.asarray(controls["cl"])
    assert 0.0 < lift_coefficient.min()
    assert lift_coefficient.max() < 1.4


# ---------------------------------------------------------------- the plots


def test_every_figure_draws(soaring):
    scenario, gemseo_scenario = soaring
    trajectory = optimal_trajectory(scenario, gemseo_scenario)
    assert plot_trajectory(trajectory) is not None
    assert plot_controls(trajectory) is not None
    assert plot_energy(trajectory) is not None
    axes = plot_energy_analysis(trajectory)
    assert len(axes) == 3


def test_gliding_baseline_is_the_energy_a_glide_ends_with():
    """The still-air baseline is a real optimization, not an assumed number."""
    baseline = gliding_baseline(size=6)
    # A glide starts near 69 m of specific energy and can only lose.
    assert 0.0 < baseline < 69.2


def test_the_codesign_scenario_accepts_shooting():
    """Collocation is the default, but it is not the only option.

    Shooting has no node grid to seed, so this exercises the branch that
    skips the warm start.
    """
    scenario = create_codesign_scenario(transcription=SingleShooting(), size=6)
    assert scenario.problem.design_variables


def test_every_figure_accepts_an_existing_axis(soaring):
    scenario, gemseo_scenario = soaring
    trajectory = optimal_trajectory(scenario, gemseo_scenario)

    figure = pyplot.figure()
    spatial = figure.add_subplot(projection="3d")
    assert plot_trajectory(trajectory, spatial) is spatial

    _, flat = pyplot.subplots()
    assert plot_controls(trajectory, flat) is flat
    _, another = pyplot.subplots()
    assert plot_energy(trajectory, another) is another
    _, three = pyplot.subplots(3, 1)
    assert plot_energy_analysis(trajectory, three) is three


# ------------------------------------------------------ the section trade


def test_the_section_trades_cruise_against_lift():
    """A cleaner section glides further and stalls earlier, and vice versa."""
    chain = create_statics()
    sections = [0.0, 0.5, 1.0]
    ratios, lifts = [], []
    for section in sections:
        outputs = chain(dict(DESIGN_DEFAULTS, section=section))
        ratios.append(glide_ratio({"section": section}))
        lifts.append(float(outputs["cl_max"]))
    # Glide ratio falls as the section gets thicker; maximum lift rises.
    assert ratios[0] > ratios[1] > ratios[2]
    assert lifts[0] < lifts[1] < lifts[2]


def test_the_lift_limit_is_an_active_constraint(codesign):
    """The cycle spends most of its length on the maximum lift coefficient.

    This is worth asserting rather than assuming. The lift limit was once
    a hard-coded bound disconnected from ``cl_max``, and because it never
    reached the design constant the sensitivity of the mission to maximum
    lift measured as exactly zero, which read like physics and was an
    artefact.
    """
    scenario, gemseo_scenario = codesign
    trajectory = optimal_trajectory(scenario, gemseo_scenario)
    lift = _series(trajectory, "cl")
    optimum = gemseo_scenario.optimization_result.x_opt_as_dict
    design = {
        name.removeprefix("constant."): float(value[0])
        for name, value in optimum.items()
        if name.startswith("constant.") and name != "constant.w_ref"
    }
    limit = float(create_statics()(dict(DESIGN_DEFAULTS, **design))["cl_max"])
    assert lift.max() == pytest.approx(limit, abs=1e-3)
    assert (lift > limit - 0.01).sum() > lift.size // 2


# ---------------------------------------------------------- the missions


@pytest.mark.parametrize("mission", ["energy", "distance", "time"])
def test_each_mission_converges_and_closes_what_it_should(mission):
    """Every mission builds, converges, and closes its own periodicity."""
    extra = {"target_distance": 250.0} if mission == "time" else {}
    scenario, gemseo_scenario = create_energy_optimization(
        size=SIZE, w_ref=STRONG_WIND, mission=mission, **extra
    )
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=300)
    trajectory = optimal_trajectory(scenario, gemseo_scenario)
    energy = _series(trajectory, "specific_energy")

    height = _series(trajectory, "h")
    assert height[-1] == pytest.approx(height[0], abs=1e-4)
    if mission == "energy":
        # Speed is deliberately left open, which is where the gain shows up.
        assert energy[-1] > energy[0]
    else:
        # Closing in speed too makes the cycle exactly energy-neutral.
        assert energy[-1] == pytest.approx(energy[0], abs=1e-3)
    if mission == "time":
        assert _series(trajectory, "x")[-1] == pytest.approx(250.0, abs=1e-2)


def test_a_neutral_cycle_covers_more_ground_than_an_energy_hungry_one():
    """Spending the surplus on progress buys progress."""
    banked = create_energy_optimization(size=SIZE, w_ref=STRONG_WIND, mission="energy")
    spent = create_energy_optimization(size=SIZE, w_ref=STRONG_WIND, mission="distance")
    for _, gemseo_scenario in (banked, spent):
        gemseo_scenario.execute(algo_name="SLSQP", max_iter=300)
    covered = _series(optimal_trajectory(*banked), "x")[-1]
    further = -float(spent[1].optimization_result.f_opt)
    assert further > covered


def test_unknown_missions_and_topologies_are_rejected():
    with pytest.raises(ValueError, match="Unknown mission"):
        create_energy_scenario(size=6, mission="glory")
    with pytest.raises(ValueError, match="Unknown topology"):
        soaring_cycle(topology="spiral")
    with pytest.raises(ValueError, match="needs a target_distance"):
        create_energy_scenario(size=6, mission="time")


# -------------------------------------------------------- the topologies


@pytest.mark.parametrize(
    ("topology", "gained"), [("loop", 2.0 * numpy.pi), ("weave", 0.0)]
)
def test_the_topology_decides_what_the_heading_does(topology, gained):
    """A loop turns all the way round; a weave turns one way and back."""
    _, states, _ = soaring_cycle(topology=topology, count=41)
    heading = numpy.asarray(states["psi"])
    assert heading[-1] - heading[0] == pytest.approx(gained, abs=1e-6)
    if topology == "weave":
        # It still has to swing, or there is no manoeuvre at all.
        assert numpy.ptp(heading) > 1.0


def test_a_weave_banks_energy_as_a_loop_does():
    """An S pays for itself: it reverses heading, just not all the way round.

    It climbs while turned into the wind and descends while turned away
    from it, which is all the gradient term asks, and comes back on the
    heading it started on rather than a full turn later.
    """
    scenario, gemseo_scenario = create_energy_optimization(
        size=SIZE, w_ref=STRONG_WIND, topology="weave", mission="energy"
    )
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=300)
    trajectory = optimal_trajectory(scenario, gemseo_scenario)
    energy = _series(trajectory, "specific_energy")
    heading = _series(trajectory, "psi")
    assert energy[-1] > energy[0] + 1.0
    assert heading[-1] - heading[0] == pytest.approx(0.0, abs=1e-4)
    assert numpy.ptp(heading) > 1.0


@pytest.mark.parametrize("topology", ["loop", "weave"])
def test_the_codesign_seed_closes_the_way_its_topology_does(topology):
    """The heading offset and the seed agree, so the seed starts closed.

    Were the weave checked against a full turn, its seed would miss by one;
    were the loop checked against none, it would miss by one the other way.
    """
    scenario = create_codesign_scenario(size=6, topology=topology)
    outputs = scenario.problem.jax_function(scenario.problem.input_defaults)
    closure = numpy.asarray(outputs["constraint.cycle"])
    assert closure.shape == (4,)
    # Height, airspeed, climb angle and heading, in that order.
    assert numpy.allclose(closure, 0.0, atol=1e-6)


def test_a_displacement_closes_the_position_as_well():
    """A hover returns to where it started over the sea, not just to its state.

    The seed drifts with the wind, so the two added residuals start open by
    that drift, less whatever displacement was asked for.
    """
    hover = create_codesign_scenario(size=6, displacement=(0.0, 0.0))
    passage = create_codesign_scenario(size=6, displacement=(5.0, -3.0))
    closures = [
        numpy.asarray(
            scenario.problem.jax_function(scenario.problem.input_defaults)[
                "constraint.cycle"
            ]
        )
        for scenario in (hover, passage)
    ]
    assert closures[0].shape == (6,)
    assert abs(closures[0][4]) > 10.0
    assert numpy.allclose(closures[0][4:] - closures[1][4:], (5.0, -3.0))


# ------------------------------------------------------- the model figures


def test_the_model_figures_draw():
    assert len(plot_wind_profile()) == 2
    assert len(plot_polar()) == 2
    assert plot_section_trade() is not None


def test_the_model_figures_accept_existing_axes():
    _, pair = pyplot.subplots(1, 2)
    assert plot_wind_profile(axes=pair) is pair
    _, another = pyplot.subplots(1, 2)
    assert plot_polar(axes=another) is another
    _, single = pyplot.subplots()
    assert plot_section_trade(single) is single


# ------------------------------------------------ the three-dimensional views


def _as_arrays(trajectory, **scalars):
    """Flatten a trajectory the way a stored result loads: arrays by name."""
    arrays = {"ts": numpy.asarray(trajectory.ts)}
    for group in (trajectory.states, trajectory.controls, trajectory.outputs):
        arrays.update({name: numpy.asarray(value) for name, value in group.items()})
    arrays.update(scalars)
    return arrays


@pytest.mark.parametrize(("shape", "count"), [("S", 2), ("O", 1)])
def test_the_scenario_names_every_phase_of_the_cycle(shape, count):
    axes = plot_scenario(shape, count, w_ref=22.0)
    texts = [text.get_text() for text in axes.texts]
    for phase in ("windward climb", "upper turn", "leeward descent", "lower turn"):
        assert texts.count(phase) == 1
    assert "wind W(h), 22.0 m/s at 10 m" in texts
    assert f"{shape} cycle" in axes.get_title()


@pytest.mark.parametrize(("shape", "turns"), [("O", 1.0), ("S", 0.0)])
def test_the_schematic_cycles_only_climb_into_the_wind(shape, turns):
    """Drawn, not solved, but drawn so that the wind gradient always pays.

    The loop gains a full turn of heading and the snake gives it back; both
    climb only while heading upwind, which makes the gradient term of the
    energy equation non-negative all the way round.
    """
    cycle = schematic_cycle(shape)
    heading = cycle["psi"]
    assert heading[-1] - heading[0] == pytest.approx(2.0 * numpy.pi * turns)
    assert cycle["h"][0] == pytest.approx(cycle["h"][-1])
    assert cycle["h"].min() > 0.0
    assert cycle["gradient_power"].min() > -1e-12


def test_only_the_two_shapes_are_drawn():
    with pytest.raises(ValueError, match="Unknown shape 'L'"):
        schematic_cycle("L")


def test_the_cycle_shapes_draw():
    assert len(plot_cycle_shapes(count=2)) == 2
    figure = pyplot.figure()
    pair = [figure.add_subplot(2, 1, k + 1, projection="3d") for k in range(2)]
    assert plot_cycle_shapes(axes=pair) is pair


def test_the_views_accept_a_trajectory_as_well_as_stored_arrays(soaring):
    scenario, gemseo_scenario = soaring
    trajectory = optimal_trajectory(scenario, gemseo_scenario)
    figure = pyplot.figure()
    spatial = figure.add_subplot(projection="3d")
    assert plot_scenario(axes=spatial) is spatial
    assert len(plot_frames(trajectory)) == 3
    _, three = pyplot.subplots(1, 3)
    assert plot_frames(_as_arrays(trajectory), axes=three) is not None
    figure = pyplot.figure()
    pair = [figure.add_subplot(1, 2, k + 1, projection="3d") for k in range(2)]
    assert plot_maneuver_pattern({"a": trajectory, "b": trajectory}, 2, pair) is pair
    assert len(plot_maneuver_pattern({"a": _as_arrays(trajectory)})) == 1
    with pytest.raises(KeyError, match="No series named"):
        _series(_as_arrays(trajectory), "not_a_series")


def test_the_frames_read_the_stored_wind_fixed_track(soaring):
    """A stored cycle carries its wind-fixed track; it is drawn, not redone."""
    scenario, gemseo_scenario = soaring
    cycle = _as_arrays(optimal_trajectory(scenario, gemseo_scenario))
    cycle["x_air"] = cycle["x"] - 1.0
    axes = plot_frames(cycle)
    drawn = axes[2].lines[1].get_xdata()
    assert numpy.allclose(drawn, cycle["x"] - 1.0)


def test_the_ribbon_is_widened_against_the_stored_span():
    """A stored cycle carries its wing span, and the stated widening uses it."""

    def widening(span):
        axes = plot_maneuver_pattern({"a": {**schematic_cycle("O"), "wing_span": span}})
        return float(re.search(r"ribbon ([0-9.]+)×", axes[0].get_title()).group(1))

    assert widening(2.0) == pytest.approx(2.0 * widening(4.0), rel=0.05)


def test_the_ribbon_follows_the_wing_and_the_bank_tilts_it():
    """The span is level with no bank, and banking rolls it about the path."""
    velocity, lift, span = _unit_vectors([0.0, 0.0], [0.0, 0.0], [0.0, 0.5])
    assert numpy.allclose(span[0], (0.0, 1.0, 0.0))
    assert numpy.allclose(lift[0], (0.0, 0.0, 1.0))
    for vectors in (velocity, lift, span):
        assert numpy.allclose(numpy.linalg.norm(vectors, axis=1), 1.0)
    assert numpy.allclose(numpy.sum(velocity * span, axis=1), 0.0)
    assert numpy.allclose(numpy.sum(lift * span, axis=1), 0.0)
    # A positive bank leans the lift to the left, the way the heading turns.
    assert lift[1][1] > 0.0


def test_a_cycle_without_a_windward_climb_still_has_its_phases():
    """A glide never climbs into the wind; the phases fall back to the gain."""
    count = 8
    cycle = {
        "gamma": -0.1 * numpy.ones(count),
        "psi": numpy.zeros(count),
        "h": numpy.linspace(20.0, 5.0, count),
        "gradient_power": numpy.arange(count, dtype=float),
    }
    phases = _phases(cycle)
    assert phases["windward climb"] == count - 1
    assert phases["upper turn"] == 0
    assert phases["lower turn"] == count - 1


# ------------------------------------------- aspect ratio reaches the lift


def test_the_nominal_wing_keeps_its_section_properties():
    """Both finite-wing corrections are exactly one at the reference bird."""
    outputs = create_statics()(DESIGN_DEFAULTS)
    assert float(outputs["cd0"]) == pytest.approx(float(outputs["cd0_section"]))
    assert float(outputs["cl_max"]) == pytest.approx(float(outputs["cl_max_section"]))


def test_a_narrower_wing_flies_at_a_lower_reynolds_number():
    """Chord is area over span, and Reynolds follows it."""
    chain = create_statics()
    wide = chain(dict(DESIGN_DEFAULTS, wing_area=0.9, wing_span=2.5))
    narrow = chain(dict(DESIGN_DEFAULTS, wing_area=0.5, wing_span=3.7))
    assert float(wide["chord"]) == pytest.approx(0.9 / 2.5)
    assert float(narrow["chord"]) == pytest.approx(0.5 / 3.7)
    assert float(narrow["reynolds"]) < float(wide["reynolds"])


def test_a_low_reynolds_wing_loses_lift_and_gains_drag():
    """The penalty that stops aspect ratio running away.

    Aspect ratio was modelled as affecting induced drag alone, so nothing
    charged a long wing for the narrow chord it implies. At bird scale that
    chord sets a Reynolds number where maximum lift falls and profile drag
    rises, which is why small flyers do not have sailplane wings.
    """
    chain = create_statics()
    wide = chain(dict(DESIGN_DEFAULTS, wing_area=0.9, wing_span=2.5))
    narrow = chain(dict(DESIGN_DEFAULTS, wing_area=0.5, wing_span=3.7))
    assert float(narrow["cl_max"]) < float(wide["cl_max"])
    assert float(narrow["cd0"]) > float(wide["cd0"])


def test_a_longer_wing_gets_a_little_more_lift_from_its_slope():
    """The finite-wing effect on its own, with Reynolds held still.

    Comparing at a fixed chord isolates the lift-curve slope: a longer wing
    flies in less of its own downwash and holds slightly more lift. It is a
    few percent, which is why it cannot carry the argument alone.
    """
    chain = create_statics()
    chord = DESIGN_DEFAULTS["wing_area"] / DESIGN_DEFAULTS["wing_span"]
    short = chain(dict(DESIGN_DEFAULTS, wing_span=3.0, wing_area=3.0 * chord))
    long = chain(dict(DESIGN_DEFAULTS, wing_span=4.0, wing_area=4.0 * chord))
    assert float(long["reynolds"]) == pytest.approx(float(short["reynolds"]))
    assert float(long["cl_max"]) > float(short["cl_max"])
    assert float(long["cl_max"]) < 1.15 * float(short["cl_max"])


def test_the_co_design_wing_area_is_interior():
    """The coupling is what finally takes a wing variable off its bound.

    Through two earlier passes every wing variable sat on a bound, because
    nothing charged aspect ratio for anything but structure. With the chord
    penalty in place the area settles between its limits.
    """
    _, gemseo_scenario = create_codesign_optimization(size=SIZE)
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=600)
    area = float(
        gemseo_scenario.optimization_result.x_opt_as_dict["constant.wing_area"][0]
    )
    lower, upper = WING_BOUNDS["wing_area"]
    assert lower + 1e-3 < area < upper - 1e-3


# --------------------------------------------------- closing out the sprint


def test_multi_start_confirms_the_seeded_guess_is_close_to_the_best():
    """The reported optima are not artefacts of one lucky start.

    Every number in this problem comes from a single inverse-dynamics
    guess, which is a strong start and therefore a suspicious one. Random
    starts over a Latin hypercube are the check, and what they establish is
    that the basin is broad: three of them land within a fraction of a
    percent of the seeded optimum, in either direction.

    Which direction is luck, and this test used to claim otherwise. It
    asserted that wider search never does worse, which is not a property
    three random starts have on a problem with several local optima:
    ``MultiStart`` returns the best of what it happened to find, and a
    strong seeded guess can beat that. The claim held until an unrelated
    change perturbed the arithmetic in its last bits, which was enough to
    send one start to a different optimum. The agreement is the evidence
    here; the sign of the difference is not.
    """
    _, seeded_scenario = create_codesign_optimization(size=10)
    seeded_scenario.execute(algo_name="SLSQP", max_iter=400)
    seeded = float(seeded_scenario.optimization_result.f_opt)

    _, sampled_scenario = create_codesign_optimization(size=10)
    sampled_scenario.execute(
        algo_name="MultiStart",
        n_start=3,
        opt_algo_name="SLSQP",
        opt_algo_max_iter=250,
        doe_algo_name="LHS",
        max_iter=3000,
    )
    sampled = float(sampled_scenario.optimization_result.f_opt)
    # Both are cycles that close, and they agree closely enough that the
    # seeded guess can be trusted to stand for the problem.
    assert sampled == pytest.approx(seeded, rel=0.05)


def test_fixed_stepping_gives_the_same_answer_faster():
    """The CI budget, and the check that meeting it costs nothing.

    A fixed step makes the run time predictable, which is what a time budget
    needs, and it turns out to be quicker as well: an adaptive controller
    spends steps chasing tolerance the optimizer does not need.
    """
    start = time.perf_counter()
    _, adaptive = create_codesign_optimization(size=SIZE)
    adaptive.execute(algo_name="SLSQP", max_iter=600)
    adaptive_time = time.perf_counter() - start

    start = time.perf_counter()
    _, fixed = create_codesign_optimization(
        size=SIZE, integrator=DiffraxIntegrator(dt0=0.005, n_save=SIZE + 1)
    )
    fixed.execute(algo_name="SLSQP", max_iter=600)
    fixed_time = time.perf_counter() - start

    assert float(fixed.optimization_result.f_opt) == pytest.approx(
        float(adaptive.optimization_result.f_opt), rel=1e-3
    )
    # The sprint asks for five minutes; this leaves that alone entirely.
    assert fixed_time < 300.0
    assert adaptive_time < 300.0


def test_the_pareto_front_trades_wind_against_speed():
    """Weaker wind or a faster crossing, and the front says how much of each."""
    front = solve_pareto_front(
        size=10, target_distance=250.0, n_points=4, sub_optim_max_iter=250
    )
    assert front.shape[1] == 2
    assert front.shape[0] >= 2
    # Sorted by wind, and the crossing time falls as the wind rises.
    assert numpy.all(numpy.diff(front[:, 0]) > 0.0)
    assert numpy.all(numpy.diff(front[:, 1]) < 0.0)


def test_the_pareto_figure_draws():
    front = numpy.array([[9.0, 14.0], [12.0, 11.0], [20.0, 8.0]])
    axes = plot_pareto(front, wings={"cruise": (10.0, 12.5)})
    assert axes is not None
    _, supplied = pyplot.subplots()
    assert plot_pareto(front, axes=supplied) is supplied


def test_shooting_design_variables_are_bounded_too():
    """Single shooting names its unknowns differently, and still gets bounds.

    Its design variables are the free initial states rather than node
    values, so this exercises the path through the bound table that
    collocation never takes.
    """
    scenario, _ = create_codesign_optimization(
        transcription=SingleShooting(), size=6, design_names=("w_ref",)
    )
    assert any(
        name.startswith("initial.") for name in scenario.problem.design_variables
    )
