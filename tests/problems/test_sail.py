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
    """
    angle = best_vmg_angle("downwind")
    assert NO_GO_ANGLE < angle < 180.0
    made_good = float(boat_speed(angle, REFERENCE_WIND_SPEED)) * abs(
        numpy.cos(numpy.radians(angle))
    )
    dead_downwind = float(boat_speed(180.0, REFERENCE_WIND_SPEED))
    assert made_good > 1.1 * dead_downwind


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
    """Squeezing the channel costs time, and the boat stays inside it."""
    free = solve()
    narrow = solve(channel_half_width=80.0)
    assert float(narrow.f_opt) > float(free.f_opt)
    across = numpy.abs(narrow.x_opt_as_dict["state.y"])
    assert across.max() <= 80.0 + 1.0


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


def test_the_race_is_sailed_inside_a_corridor_by_default():
    """North and south are bounded unless asked otherwise."""
    bounded = create_scenario(size=4)
    free = create_scenario(size=4, channel_half_width=None)
    names = [name for name, _ in bounded.problem.constraint_names]
    assert "constraint.channel" in names
    assert "constraint.channel" not in [n for n, _ in free.problem.constraint_names]
    assert CORRIDOR_HALF_WIDTH == 500.0
