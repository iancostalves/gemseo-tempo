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
"""What compiling costs, and closing the cycle with an inner solve."""

from __future__ import annotations

import logging

import numpy
import pytest

from gemseo_tempo.benchmarking.nested import build_inner_problem
from gemseo_tempo.benchmarking.nested import implicit_gradient
from gemseo_tempo.benchmarking.nested import robustness_map
from gemseo_tempo.benchmarking.nested import solve_initial_speed
from gemseo_tempo.benchmarking.suites import BRACHISTOCHRONE
from gemseo_tempo.benchmarking.timing import Timing
from gemseo_tempo.benchmarking.timing import format_table
from gemseo_tempo.benchmarking.timing import measure
from gemseo_tempo.benchmarking.timing import sweep

logging.disable(logging.CRITICAL)

PHYSICAL_START = 36.0
"""An initial speed whose cycle stays above the water."""


# --------------------------------------------------------------- timing


def test_timing_separates_compilation_from_execution():
    """Three times, not one, and a compiled evaluation is much faster.

    Sprint 9 reported the compile fix as a single speedup number, which
    conflates a cost that grows with problem size and a saving that is per
    evaluation.
    """
    timing = measure(BRACHISTOCHRONE, "collocation", 10, repeats=2)
    assert timing.compile_seconds > 0.0
    assert timing.compiled_seconds > 0.0
    assert timing.eager_seconds > timing.compiled_seconds
    assert timing.speedup > 2.0
    assert timing.n_design_variables > 0


def test_the_crossover_says_when_compiling_pays():
    """Compiling repays itself within a handful of evaluations here.

    Any optimization run makes far more than that, so on these problems
    compiling always pays. The quantity is reported rather than assumed
    because it need not be true of a cheaper problem.
    """
    timing = measure(BRACHISTOCHRONE, "collocation", 10, repeats=2)
    assert 0.0 < timing.crossover < 200.0


def test_a_compilation_that_never_pays_is_reported_as_such():
    """A saving of nothing is an infinite crossover, not a division error."""
    timing = Timing(
        suite="s",
        level="collocation",
        size=1,
        compile_seconds=1.0,
        compiled_seconds=2.0,
        eager_seconds=2.0,
        n_design_variables=1,
    )
    assert timing.crossover == float("inf")
    assert "never" in format_table([timing])


def test_a_timing_sweep_covers_every_size():
    timings = sweep(BRACHISTOCHRONE, "collocation", (3, 10))
    assert [timing.size for timing in timings] == [3, 10]
    table = format_table(timings)
    assert table.count("\n") == len(timings) + 1
    assert "crossover" in table
    assert all("speedup" in timing.as_dict() for timing in timings)


# --------------------------------------------------------------- nested


@pytest.fixture(scope="module")
def inner():
    """Return the speed-periodicity residual of one albatross cycle."""
    return build_inner_problem(size=16)


def test_the_inner_solve_closes_the_cycle_in_speed(inner):
    """Started from a physical speed, Newton finds the periodic one."""
    speed, converged = solve_initial_speed(inner, PHYSICAL_START)
    assert converged
    assert 20.0 < speed < 60.0
    assert abs(float(inner.residual(speed, inner.params))) < 1e-6


def test_the_implicit_gradient_matches_finite_differences(inner):
    """The derivative comes from the implicit function theorem.

    Differentiating through the Newton iterations would work too, but its
    cost grows with the number of steps and its accuracy falls with them.
    This checks the cheap exact route agrees with a central difference.
    """
    gradient = implicit_gradient(inner, PHYSICAL_START)
    assert gradient.shape == inner.params["cl"].shape

    step = 1e-5
    moved = {}
    for sign in (1.0, -1.0):
        params = dict(inner.params)
        params["cl"] = inner.params["cl"].at[0].add(sign * step)
        problem = type(inner)(
            residual=inner.residual,
            trajectory=inner.trajectory,
            speed_guess=inner.speed_guess,
            params=params,
        )
        moved[sign] = solve_initial_speed(problem, PHYSICAL_START)[0]
    difference = (moved[1.0] - moved[-1.0]) / (2.0 * step)
    assert gradient[0] == pytest.approx(difference, rel=1e-3)


def test_the_inner_solve_reports_where_it_can_be_started(inner):
    """Its robustness is physical, not numerical, and it is mapped.

    Started too slow the prescribed cycle flies the bird under the water,
    and the integration underneath the root finder fails outright rather
    than returning a poor number. That is reported, because inside an
    optimizer it is a discipline that did not evaluate.
    """
    rows = robustness_map(inner, (28.0, PHYSICAL_START))
    below, above = rows
    assert not below["physical"]
    assert not below["converged"]
    assert above["physical"]
    assert above["converged"]
    # Every start that converges agrees on the answer.
    solved = [row["solved"] for row in rows if row["converged"]]
    assert numpy.allclose(solved, solved[0], atol=1e-6)


def test_a_trajectory_that_will_not_integrate_is_reported_not_raised(inner):
    """A probe that fails becomes a row, not an exception.

    The integrations here survive even an absurd initial speed, so the guard
    is exercised with a stub. It matters because a sweep must not die
    partway through on one bad cell, and a design far from nominal can make
    the integration fail outright rather than return a poor number.
    """

    def refuses(v0, params):
        raise RuntimeError("the solver gave up")

    broken = type(inner)(
        residual=inner.residual,
        trajectory=refuses,
        speed_guess=inner.speed_guess,
        params=inner.params,
    )
    row = robustness_map(broken, (PHYSICAL_START,))[0]
    assert numpy.isnan(row["min_height"])
    assert not row["physical"]


def test_perturbation_leaves_an_empty_input_alone():
    """Nothing to nudge is not an error, it is a copy."""
    from gemseo_tempo.benchmarking.timing import _perturbed

    empty = {"nothing": numpy.zeros(0)}
    assert _perturbed(empty, 0)["nothing"].size == 0

    mixed = {"empty": numpy.zeros(0), "real": numpy.ones(2)}
    nudged = _perturbed(mixed, 3)
    assert numpy.array_equal(nudged["empty"], mixed["empty"])
    assert not numpy.array_equal(nudged["real"], mixed["real"])
