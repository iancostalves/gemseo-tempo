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
"""Sizing the bird before flying it, against doing both at once."""

from __future__ import annotations

import logging

import pytest

from gemseo_tempo.problems.albatross import create_codesign_optimization
from gemseo_tempo.problems.albatross.problem import WING_BOUNDS
from gemseo_tempo.problems.albatross.sequential import Sensitivity
from gemseo_tempo.problems.albatross.sequential import glide_ratio
from gemseo_tempo.problems.albatross.sequential import minimum_wind
from gemseo_tempo.problems.albatross.sequential import misalignment_study
from gemseo_tempo.problems.albatross.sequential import sensitivity_study
from gemseo_tempo.problems.albatross.sequential import sequential_codesign
from gemseo_tempo.problems.albatross.sequential import size_the_wing
from gemseo_tempo.problems.albatross.sequential import total_mass
from gemseo_tempo.problems.albatross.sequential import weights_from

logging.disable(logging.CRITICAL)

SIZE = 10
"""The discretization of these tests, kept small: they re-fly the bird a lot."""

MEASURED = {
    "glide_ratio": Sensitivity("glide_ratio", 19.75, -1.117),
    "mass": Sensitivity("mass", 8.50, -0.0045),
    "cl_max": Sensitivity("cl_max", 1.50, 0.5312),
}
"""Elasticities measured at ``size=12``, so the tests below need not re-run them."""


@pytest.fixture(scope="module")
def simultaneous():
    """Return the minimum wind found by optimizing everything at once."""
    _, gemseo_scenario = create_codesign_optimization(size=SIZE)
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=600)
    return float(gemseo_scenario.optimization_result.f_opt)


def test_the_mission_cares_most_about_gliding():
    """Glide ratio dominates, and maximum lift is no longer invisible.

    The maximum lift coefficient once measured as exactly zero here, which
    looked like a finding and was a wiring fault: the limit the optimizer
    actually met was a hard-coded bound that ``cl_max`` never reached. Wired
    up and coupled to the section, it now registers, and it registers with
    the opposite sign: for a mission that only wants the weakest wind, the
    drag a thicker section costs outweighs the lift it buys.
    """
    measured = sensitivity_study(size=SIZE, relative_step=0.08, max_iter=300)
    assert measured["glide_ratio"].elasticity < -0.5
    assert measured["cl_max"].elasticity > 0.05
    assert abs(measured["glide_ratio"].elasticity) > abs(measured["cl_max"].elasticity)
    assert abs(measured["mass"].elasticity) < abs(measured["cl_max"].elasticity)


def test_the_weights_reward_what_lowers_the_wind():
    """A quantity that lowers the wind earns a positive weight, and vice versa."""
    weights = weights_from(MEASURED)
    assert weights["glide_ratio"] > 0.0
    # Maximum lift costs more than it buys here, so the merit is told to
    # trade it away rather than chase it.
    assert weights["cl_max"] < 0.0
    assert weights["glide_ratio"] > abs(weights["cl_max"])


def test_the_sizing_step_returns_a_wing_inside_its_bounds():
    wing = size_the_wing(weights_from(MEASURED))
    assert set(wing) == set(WING_BOUNDS)
    for name, (lower, upper) in WING_BOUNDS.items():
        assert lower - 1e-6 <= wing[name] <= upper + 1e-6
    # Sized against the mission's own priorities, it out-glides the nominal.
    assert glide_ratio(wing) > glide_ratio(None)


def test_sequential_matches_simultaneous_when_the_merit_is_right(simultaneous):
    """Sizing first costs nothing when it is sized for the right thing.

    This is the result rather than the expected 2 percent margin. Both
    routes reach the same wing, because the figure of merit the sizing step
    was given is the one the mission actually rewards.
    """
    wing = size_the_wing(weights_from(MEASURED))
    sequential = minimum_wind(wing, SIZE, 600)
    assert sequential == pytest.approx(simultaneous, rel=0.02)


def test_sizing_for_the_wrong_thing_costs_a_great_deal(simultaneous):
    """Over-weighting mass buys a lighter bird that cannot fly as cheaply."""
    study = misalignment_study(
        [0.5, 4.0], reference_wind=simultaneous, size=SIZE, max_iter=400
    )
    aligned, misaligned = study
    assert aligned.penalty < 0.05
    assert misaligned.penalty > 0.3
    assert total_mass(misaligned.wing) < total_mass(aligned.wing)
    assert glide_ratio(misaligned.wing) < glide_ratio(aligned.wing)


def test_the_whole_route_reports_every_step():
    """The result carries what was measured, what was chosen, and the outcome."""
    result = sequential_codesign(size=8, relative_step=0.1, max_iter=250)
    assert set(result.sensitivities) == {"glide_ratio", "mass", "cl_max"}
    assert set(result.wing) == set(WING_BOUNDS)
    assert 4.0 < result.minimum_wind < 16.0
