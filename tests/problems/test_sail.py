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
"""Tests of the sailing problem package."""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy
import pytest

from gemseo_tempo.dynamics.integrators import DiffraxIntegrator
from gemseo_tempo.problems.sail import CORRIDOR_HALF_WIDTH
from gemseo_tempo.problems.sail import LEGS
from gemseo_tempo.problems.sail import NO_GO_ANGLE
from gemseo_tempo.problems.sail import best_vmg_angle
from gemseo_tempo.problems.sail import boat_speed
from gemseo_tempo.problems.sail import create_optimization
from gemseo_tempo.problems.sail import create_scenario
from gemseo_tempo.problems.sail import create_wind
from gemseo_tempo.problems.sail import default_guess
from gemseo_tempo.problems.sail import downwind_vmg
from gemseo_tempo.problems.sail import no_go_constraint
from gemseo_tempo.problems.sail import plot_polar
from gemseo_tempo.problems.sail import speed_shape
from gemseo_tempo.problems.sail import straight_run_time
from gemseo_tempo.problems.sail import true_wind_angle
from gemseo_tempo.problems.sail.model import apply_wind_parameters
from gemseo_tempo.problems.sail.problem import sail_integrator
from gemseo_tempo.problems.sail.wind import EXTENT
from gemseo_tempo.transcriptions import IndirectShooting
from gemseo_tempo.transcriptions import SingleShooting

logging.disable(logging.CRITICAL)

SIZE = 12
"""The discretization used by the optimization tests, kept small for speed."""

REFERENCE_WIND_SPEED = 10.0
"""The wind speed of the uniform field, in m/s."""


def solve(transcription=None, **settings):
    """Optimize the downwind leg and return the GEMSEO result.

    Parameters
    ----------
    transcription
        The transcription; collocation when omitted.
    **settings
        Extra settings forwarded to the scenario builder.

    Returns
    -------
    OptimizationResult
        The result.
    """
    _, gemseo_scenario = create_optimization(
        transcription=transcription, leg="downwind", size=SIZE, **settings
    )
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=300)
    return gemseo_scenario.optimization_result


@pytest.fixture(scope="module")
def collocation_result():
    """Return the downwind optimum found by collocation."""
    return solve()


@pytest.fixture(scope="module")
def shooting_result():
    """Return the downwind optimum found by single shooting."""
    return solve(SingleShooting())


def test_polar_is_physical():
    """The polar is non-negative, symmetric, and dead inside the no-go cone."""
    angles = numpy.linspace(-180.0, 180.0, 721)
    speeds = numpy.array([float(speed_shape(angle)) for angle in angles])
    assert speeds.min() >= 0.0
    # Symmetric in the sign of the angle: the two tacks are equally good.
    assert numpy.allclose(speeds, speeds[::-1], atol=1e-9)
    inside = numpy.abs(angles) < NO_GO_ANGLE - 5.0
    assert speeds[inside].max() < 0.02


def test_polar_grows_with_wind():
    """More wind is never slower."""
    winds = numpy.linspace(1.0, 25.0, 25)
    speeds = numpy.array([float(boat_speed(120.0, wind)) for wind in winds])
    assert numpy.all(numpy.diff(speeds) > 0.0)


def test_gybing_pays_on_the_polar():
    """Reaching off at the best angle beats running dead downwind.

    This is the property the whole problem rests on, so it is asserted on
    the polar itself: if it fails, the cause is the boat model rather than
    the optimizer.

    The margin is 5% rather than the 10% this asked for while the polar
    was a table. Computing the speed from a sail and a hull instead of
    reading it off a list puts the gain at about 8%, against the table's
    12%: the same finding, smaller, and now a consequence rather than an
    assumption.
    """
    angle = best_vmg_angle("downwind")
    assert NO_GO_ANGLE < angle < 180.0
    made_good = float(boat_speed(angle, REFERENCE_WIND_SPEED)) * abs(
        numpy.cos(numpy.radians(angle))
    )
    dead_downwind = float(boat_speed(180.0, REFERENCE_WIND_SPEED))
    assert made_good > 1.05 * dead_downwind


def test_best_vmg_angle_rejects_an_unknown_direction():
    with pytest.raises(ValueError, match="upwind"):
        best_vmg_angle("sideways")


def test_true_wind_angle_conventions():
    """Zero is head to wind and 180 is dead downwind."""
    # Wind blowing east: it comes from the west, at a bearing of pi.
    east, north = 5.0, 0.0
    assert float(true_wind_angle(numpy.pi, east, north)) == pytest.approx(0.0)
    assert abs(float(true_wind_angle(0.0, east, north))) == pytest.approx(180.0)


def test_wind_fields_have_the_advertised_structure():
    """The uniform field is constant and the sheared one varies only with y."""
    uniform = create_wind("uniform")
    sheared = create_wind("sheared")

    assert float(uniform.velocity(0.0, 0.0)[0]) == pytest.approx(
        float(uniform.velocity(300.0, -200.0)[0]), rel=1e-6
    )
    # Shear: the same y gives the same wind, a different y does not.
    assert float(sheared.velocity(0.0, 100.0)[0]) == pytest.approx(
        float(sheared.velocity(400.0, 100.0)[0]), rel=1e-6
    )
    assert float(sheared.velocity(0.0, 100.0)[0]) != pytest.approx(
        float(sheared.velocity(0.0, -100.0)[0]), rel=1e-6
    )


def test_the_gyre_turns_without_diverging():
    """The gyre comes from a streamfunction, so it turns and does not diverge.

    The origin is a stagnation point between four cells rather than the
    centre of one, so the rotation is measured at a cell centre, half an
    extent along each axis.
    """
    gyre = create_wind("gyre")
    step = 5.0

    def flow(x, y):
        east, north = gyre.velocity(x, y)
        return float(east), float(north)

    centre = EXTENT / 2.0
    east_x = (flow(centre + step, centre)[0] - flow(centre - step, centre)[0]) / (
        2.0 * step
    )
    north_y = (flow(centre, centre + step)[1] - flow(centre, centre - step)[1]) / (
        2.0 * step
    )
    north_x = (flow(centre + step, centre)[1] - flow(centre - step, centre)[1]) / (
        2.0 * step
    )
    east_y = (flow(centre, centre + step)[0] - flow(centre, centre - step)[0]) / (
        2.0 * step
    )
    assert east_x + north_y == pytest.approx(0.0, abs=1e-6)
    assert abs(north_x - east_y) > 1e-3


def test_straight_run_is_impossible_upwind():
    """The boat cannot sail at a mark that lies dead upwind."""
    assert straight_run_time("downwind") < 1e3
    assert straight_run_time("upwind") == float("inf")


def test_default_guess_follows_the_polar():
    """The guess starts on a tack at the best angle, not at the mark."""
    guess = default_guess("downwind")
    start = numpy.degrees(guess.psi_level)
    end = numpy.degrees(guess.psi_level + guess.psi_slope)
    # Symmetric about the bearing to the mark, which is due east.
    assert start == pytest.approx(-end, abs=1e-6)
    assert abs(start) > 10.0
    # The time is the quickest the leg could be sailed, so it is optimistic.
    assert guess.t_f < straight_run_time("downwind")


def test_unknown_leg_is_rejected():
    with pytest.raises(ValueError, match="Unknown leg"):
        create_scenario(leg="sideways")


def test_indirect_shooting_takes_the_channel_as_a_penalty():
    """A path constraint enters the Hamiltonian rather than the optimizer.

    Indirect shooting solves the necessary conditions, so the channel is not
    handed to the optimizer as a constraint: it is folded into the running
    cost as a penalty, and the residual is all there is to drive to zero.
    """
    scenario = create_scenario(
        transcription=IndirectShooting(),
        leg="downwind",
        size=SIZE,
        channel_half_width=150.0,
    )
    assert scenario.problem.constraint_names == []
    assert "residual" in scenario.problem.output_names


def test_tacking_emerges(collocation_result):
    """The optimizer gybes, and gybing beats sailing straight at the mark.

    Nothing in the setup asks for a gybe. The boat is only told to reach a
    mark that lies dead downwind in the least time, and the heading it
    comes back with leaves the dead-downwind axis on both tacks.
    """
    baseline = straight_run_time("downwind")
    assert float(collocation_result.f_opt) < 0.96 * baseline

    headings = numpy.degrees(collocation_result.x_opt_as_dict["control.psi"])
    # The mark is due east, so a heading of zero is straight at it.
    assert headings.max() > 10.0
    assert headings.min() < -10.0


def test_the_transcriptions_agree(collocation_result, shooting_result):
    """Collocation and shooting find the same downwind optimum."""
    collocation = float(collocation_result.f_opt)
    shooting = float(shooting_result.f_opt)
    assert abs(collocation - shooting) / shooting < 0.02


def test_a_wide_channel_does_not_bind():
    """A limit the optimum already respects leaves it untouched."""
    free = solve()
    wide = solve(channel_half_width=400.0)
    assert float(wide.f_opt) == pytest.approx(float(free.f_opt), rel=1e-4)


def test_a_narrow_channel_binds_and_is_respected():
    """Squeezing the channel costs time, and the boat stays inside it.

    The comparison is warm-started rather than run from two cold starts,
    and it has to be. A narrow channel's optimum is feasible for the
    unconstrained problem, so the unconstrained optimum cannot be slower —
    but only if both solves are in the same basin, and this leg has
    several. Cold-started, the free solve lands 0.07% *behind* the
    constrained one, which says nothing about channels and everything
    about where the solver started; see the local optima discussed on
    :doc:`the problem's page <../../docs/problems/sail/optimization>`.
    Starting the free solve from the narrow answer removes the question.
    """
    narrow = solve(channel_half_width=80.0)
    across = numpy.abs(narrow.x_opt_as_dict["state.y"])
    assert across.max() <= 80.0 + 1.0

    scenario, gemseo_scenario = create_optimization(leg="downwind", size=SIZE)
    current = gemseo_scenario.design_space.get_current_value(as_dict=True)
    gemseo_scenario.design_space.set_current_value(
        {
            name: numpy.reshape(value, numpy.shape(current[name]))
            for name, value in narrow.x_opt_as_dict.items()
            if name in current
        }
    )
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=300)
    free = gemseo_scenario.optimization_result
    assert float(free.f_opt) <= float(narrow.f_opt)


def test_every_leg_has_a_mark():
    assert set(LEGS) == {"downwind", "upwind"}
    for mark in LEGS.values():
        assert float(numpy.hypot(*mark)) > 0.0


def test_downwind_vmg_peaks_at_the_best_angle():
    """The made-good curve agrees with the angle the search returns."""
    angles = numpy.linspace(NO_GO_ANGLE, 180.0, 601)
    made_good = numpy.array([float(downwind_vmg(angle)) for angle in angles])
    best = float(angles[int(numpy.argmax(made_good))])
    assert best == pytest.approx(best_vmg_angle("downwind"), abs=0.5)
    # Running dead downwind makes good exactly the boat speed.
    assert float(downwind_vmg(180.0)) == pytest.approx(
        float(speed_shape(180.0)), rel=1e-6
    )


def test_the_best_upwind_angle_is_a_beat():
    """Beating means sailing well off the wind, but not across it."""
    angle = best_vmg_angle("upwind")
    assert NO_GO_ANGLE < angle < 90.0


def test_plot_polar_draws_both_tacks():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    axes = plot_polar()
    # One line per tack, mirrored about the wind axis.
    assert len(axes.get_lines()) == 2


def test_enforce_no_go_adds_a_path_constraint():
    """The no-go cone can be constrained explicitly when it is wanted."""
    free = create_scenario(leg="downwind", size=SIZE)
    fenced = create_scenario(leg="downwind", size=SIZE, enforce_no_go=True)
    assert not [name for name, _ in free.problem.constraint_names if "no_go" in name]
    assert [name for name, _ in fenced.problem.constraint_names if "no_go" in name]


def test_unknown_wind_field_is_rejected():
    with pytest.raises(ValueError, match="Unknown wind field"):
        create_wind("hurricane")


def test_plot_polar_accepts_an_existing_axis():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot

    _, axes = pyplot.subplots(subplot_kw={"projection": "polar"})
    assert plot_polar(axes) is axes


def test_the_no_go_constraint_marks_the_cone():
    """The constraint is positive inside the cone and negative outside it."""
    wind = create_wind("uniform")
    constraint = no_go_constraint(wind)
    position = {"x": 0.0, "y": 0.0}

    def value(heading):
        return float(constraint(0.0, position, {"psi": heading}, {}))

    # The wind blows east, so it comes from the west: a heading of pi is
    # straight into it, and a heading of zero is dead downwind.
    assert value(numpy.pi) > 0.0
    assert value(0.0) < 0.0
    # Just outside the cone the constraint has already turned.
    assert value(numpy.pi - numpy.radians(NO_GO_ANGLE + 5.0)) < 0.0


# ------------------------------------------- the wind parameters and the adjoint


def test_the_wind_parameters_scale_and_turn_the_field():
    """The field is fixed and the two parameters act on what it returns."""
    wind = create_wind("uniform")
    east, north = wind.velocity(0.0, 0.0)
    doubled = apply_wind_parameters(east, north, {"wind_scale": 2.0})
    assert float(doubled[0]) == pytest.approx(2.0 * float(east))
    quarter = apply_wind_parameters(east, north, {"wind_veer": numpy.pi / 2.0})
    # A quarter turn takes an eastward wind to a northward one.
    assert float(quarter[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(quarter[1]) == pytest.approx(float(east))
    # Absent parameters leave the field alone, so a field with no
    # uncertainty declared still integrates.
    same = apply_wind_parameters(east, north, {})
    assert float(same[0]) == pytest.approx(float(east))


def test_the_sail_jacobian_is_exact():
    """The Jacobian this problem hands the optimizer matches finite differences.

    This is the test whose absence let a wrong one through. The library's
    default RecursiveCheckpoint adjoint returns an incorrect Jacobian on
    this problem, and only on this one: it is wrong when the reverse pass
    is batched over several outputs, which is exactly what ``jax.jacrev``
    and therefore every GEMSEO discipline Jacobian does. The error is not
    small and not only quantitative. Under that adjoint the cross-track
    miss appears to respond to the wind *strength*, which the symmetry of
    the leg forbids.

    The problem defaults to the Direct adjoint for this reason, and this
    test is what says so is still true.
    """
    scenario = create_scenario(
        transcription=SingleShooting(),
        size=8,
        integrator=sail_integrator(dt0=1.0 / 200, n_save=9),
    )
    discipline = scenario.disciplines[0]
    base = {
        name: numpy.atleast_1d(numpy.asarray(value, dtype=float))
        for name, value in scenario.problem.input_defaults.items()
    }
    discipline.linearize(base, compute_all_jacobians=True)
    step = 1e-6
    for output in ("constraint.final.x", "constraint.final.y"):
        for name in ("constant.wind_scale", "constant.wind_veer", "t_f"):
            analytic = float(numpy.ravel(discipline.jac[output][name])[0])
            up = dict(base)
            down = dict(base)
            up[name] = base[name] + step
            down[name] = base[name] - step
            difference = (
                float(numpy.ravel(discipline.execute(up)[output])[0])
                - float(numpy.ravel(discipline.execute(down)[output])[0])
            ) / (2.0 * step)
            assert analytic == pytest.approx(difference, rel=1e-5, abs=1e-5)


def test_the_default_adjoint_is_inexact_here_and_the_chosen_one_is_not():
    """The reason the problem does not take the library default.

    Kept as a test rather than a comment because it is the evidence: were
    diffrax to fix the batched reverse pass, this test fails and the
    override can be dropped.

    The route probed is a ``vmap`` over a ``jacrev``, which is the one
    that fails by a wide enough margin to assert on cheaply. A plain
    ``jacrev``, which is what a discipline Jacobian is, fails by 0.6% and
    only once the leg is solved; asserting on that would cost a solve.
    """
    exact = _cross_derivative("Direct")
    checkpointed = _cross_derivative("RecursiveCheckpoint")
    reference = _finite_difference()
    assert exact == pytest.approx(reference, rel=1e-8)
    assert abs(checkpointed - reference) > 0.01 * abs(reference)


def _wind_scale_function(adjoint: str):
    """Return the along-track miss as a function of the wind parameters."""
    scenario = create_scenario(
        transcription=SingleShooting(),
        size=8,
        integrator=DiffraxIntegrator(dt0=1.0 / 100, n_save=9, adjoint=adjoint),
        check_adjoint=False,
    )
    problem = scenario.problem
    defaults = dict(problem.input_defaults)

    def value(row):
        inputs = dict(defaults)
        inputs["constant.wind_scale"] = jnp.reshape(row[0], (1,))
        inputs["constant.wind_veer"] = jnp.reshape(row[1], (1,))
        return jnp.reshape(problem.jax_function(inputs)["constraint.final.x"], ())

    return value


def _cross_derivative(adjoint: str) -> float:
    """Differentiate through a vmap over a jacrev, the route that fails."""
    value = _wind_scale_function(adjoint)

    def stacked(row):
        return jnp.stack([value(row)])

    sample = jnp.asarray([[1.0, 0.0]])
    return float(jax.jit(jax.vmap(jax.jacrev(stacked)))(sample)[0, 0, 0])


def _finite_difference() -> float:
    """The reference the derivative has to reproduce."""
    value = _wind_scale_function("Direct")
    row = jnp.asarray([1.0, 0.0])
    step = 1e-5
    return (float(value(row.at[0].add(step))) - float(value(row.at[0].add(-step)))) / (
        2.0 * step
    )


def test_a_wrong_adjoint_is_refused_rather_than_used():
    """The guard, which is what keeps the fix from being bypassed by accident.

    Passing an integrator is the ordinary way to fix the step size for a
    sampling study, and doing so used to hand the problem the default
    adjoint again without a word.
    """
    with pytest.raises(ValueError, match="must be differentiated through"):
        create_scenario(
            transcription=SingleShooting(),
            size=6,
            integrator=DiffraxIntegrator(dt0=1.0 / 100, n_save=7),
        )


# ------------------------------------------------- the sail's aerodynamics


def test_the_velocity_triangle_closes_by_hand():
    """One case anyone can check on paper.

    Running dead downwind at 4 m/s in 10 m/s of wind, the sail feels the
    difference: 6 m/s from dead astern. That is the whole content of the
    triangle, and it is where a sign error would show first.
    """
    from gemseo_tempo.problems.sail.aero import apparent_wind

    speed, angle = apparent_wind(4.0, 180.0, 10.0)
    assert float(speed) == pytest.approx(6.0, rel=1e-9)
    assert abs(float(angle)) == pytest.approx(180.0, abs=1e-9)

    # Hard on the wind, the boat's own speed adds instead of subtracting.
    speed, angle = apparent_wind(4.0, 0.0, 10.0)
    assert float(speed) == pytest.approx(14.0, rel=1e-9)
    assert float(angle) == pytest.approx(0.0, abs=1e-9)

    # On a beam reach the two are perpendicular, so the triangle is
    # Pythagoras and the apparent wind comes from ahead of the beam.
    speed, angle = apparent_wind(4.0, 90.0, 3.0)
    assert float(speed) == pytest.approx(5.0, rel=1e-9)
    assert float(angle) == pytest.approx(numpy.degrees(numpy.arctan2(3.0, 4.0)))


def test_the_force_balance_has_a_speed_to_find():
    """The explanation behind the polar has to hold up on its own.

    The dynamics read a fitted polar, not this balance — see
    :func:`~gemseo_tempo.problems.sail.polar.boat_speed` for why — but the
    physics page draws these curves and says the polar's shape comes from
    them, so they have to actually have a root where a boat sails and none
    where it cannot.
    """
    from scipy.optimize import brentq

    from gemseo_tempo.problems.sail.aero import driving_force
    from gemseo_tempo.problems.sail.aero import hull_resistance

    def net(speed, twa, tws):
        return float(driving_force(speed, twa, tws) - hull_resistance(speed))

    for tws in (3.0, 10.0, 25.0):
        for twa in (60.0, 90.0, 130.0, 180.0):
            assert net(1e-9, twa, tws) > 0.0, (twa, tws)
            speed = brentq(net, 1e-9, 60.0, args=(twa, tws))
            assert 0.1 < speed < 3.0 * tws
        # Dead into the wind there is no root at all: the sail cannot pull
        # the boat forward against its own drag.
        assert net(1e-9, 0.0, tws) < 0.0


def test_the_boat_cannot_sail_into_the_wind():
    """And the two accounts of *how close* it can sail are consistent.

    The tabulated polar declares a 30 degree no-go cone; the aerodynamic
    surrogate on the physics page derives one, from where the sail's lift
    stops beating its own drag along the heading, and gets 18.4 degrees —
    the same at every wind speed, as the theory says it should be, since
    the cone is a property of the polar rather than of the day.

    They do not have to agree, and they do not: the table is the more
    conservative of the two. What must hold is the direction of the
    disagreement. A table claiming the boat points *higher* than a flat
    plate could would be claiming something the physics does not support.
    """
    from gemseo_tempo.problems.sail.aero import driving_force

    # The table floors just above zero rather than at it, so that the
    # spline's undershoot cannot become a boat sailing backwards.
    assert float(boat_speed(0.0, 10.0)) < 0.01
    assert float(speed_shape(NO_GO_ANGLE - 5.0)) < 0.02

    grid = numpy.arange(1.0, 60.0, 0.05)
    cones = set()
    for tws in (5.0, 10.0, 20.0):
        drive = numpy.asarray(driving_force(0.0, grid, tws))
        cones.add(round(float(grid[numpy.argmax(drive > 0.0)]), 1))
    assert cones == {18.4}, cones
    assert max(cones) < NO_GO_ANGLE


def test_more_wind_is_never_slower_and_gybing_still_pays():
    """The two properties every finding on the page rests on."""
    from gemseo_tempo.problems.sail.polar import boat_speed

    speeds = [float(boat_speed(110.0, tws)) for tws in (4.0, 8.0, 12.0, 16.0)]
    assert speeds == sorted(speeds)

    angles = numpy.linspace(120.0, 180.0, 121)
    made_good = [
        float(boat_speed(float(angle), 10.0)) * numpy.cos(numpy.radians(180.0 - angle))
        for angle in angles
    ]
    best = int(numpy.argmax(made_good))
    assert 140.0 < angles[best] < 165.0
    assert made_good[best] > float(boat_speed(180.0, 10.0))


def test_the_polar_has_no_corner_at_dead_downwind():
    """The defect this problem keeps finding, tested the way that sees it.

    A symmetric difference across a corner reads zero, so it cannot tell a
    smooth minimum from a V. The one-sided slope can: approaching dead
    downwind it must fall towards zero rather than settle on a value.
    """
    import jax

    from gemseo_tempo.problems.sail.polar import boat_speed

    slope = jax.grad(boat_speed, argnums=0)
    slopes = [abs(float(slope(angle, 10.0))) for angle in (179.0, 179.9, 179.99)]
    assert slopes == sorted(slopes, reverse=True)
    assert slopes[-1] < 0.1 * slopes[0]


def test_the_race_is_sailed_inside_a_corridor_by_default():
    """North and south are bounded unless asked otherwise."""
    bounded = create_scenario(size=4)
    free = create_scenario(size=4, channel_half_width=None)
    names = [name for name, _ in bounded.problem.constraint_names]
    assert "constraint.channel" in names
    assert "constraint.channel" not in [n for n, _ in free.problem.constraint_names]
    assert CORRIDOR_HALF_WIDTH == 500.0
