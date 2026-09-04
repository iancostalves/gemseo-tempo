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
from gemseo_tempo.problems.albatross.sequential import CRUISE_WEIGHTS
from gemseo_tempo.problems.albatross.sequential import DYNAMICS_WEIGHTS
from gemseo_tempo.problems.albatross.sequential import Sensitivity
from gemseo_tempo.problems.albatross.sequential import compare_sizing_routes
from gemseo_tempo.problems.albatross.sequential import cross_country_time
from gemseo_tempo.problems.albatross.sequential import glide_ratio
from gemseo_tempo.problems.albatross.sequential import minimum_wind
from gemseo_tempo.problems.albatross.sequential import misalignment_study
from gemseo_tempo.problems.albatross.sequential import sensitivity_study
from gemseo_tempo.problems.albatross.sequential import sequential_codesign
from gemseo_tempo.problems.albatross.sequential import size_the_wing
from gemseo_tempo.problems.albatross.sequential import total_mass
from gemseo_tempo.problems.albatross.sequential import turn_capability
from gemseo_tempo.problems.albatross.sequential import weights_from

logging.disable(logging.CRITICAL)

SIZE = 10
"""The discretization of these tests, kept small: they re-fly the bird a lot."""

MEASURED = {
    "glide_ratio": Sensitivity("glide_ratio", 19.75, -0.986),
    "mass": Sensitivity("mass", 8.50, 0.0441),
    "cl_max": Sensitivity("cl_max", 1.50, 0.6179),
}
"""Elasticities measured at ``size=12``, so the tests below need not re-run them."""


def _simultaneous_wing():
    """Return the wing the simultaneous co-design run settles on.

    Returns
    -------
    dict
        The wing area, span and section.
    """
    _, gemseo_scenario = create_codesign_optimization(size=SIZE)
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=600)
    optimum = gemseo_scenario.optimization_result.x_opt_as_dict
    return {name: float(optimum["constant." + name][0]) for name in WING_BOUNDS}


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


def test_the_two_sizing_routes_pick_different_birds(simultaneous):
    """Sizing for cruise and for turning gives two quite different wings.

    Conventional practice sizes for cruise without saying so, and the wing
    that comes out is a glider: all the span it is allowed, the thinnest
    section, and almost no capacity to turn. Sizing for the manoeuvre
    instead gives the opposite bird.
    """
    cruise = size_the_wing(CRUISE_WEIGHTS)
    dynamics = size_the_wing(DYNAMICS_WEIGHTS)
    assert glide_ratio(cruise) > glide_ratio(dynamics)
    assert turn_capability(dynamics) > 2.0 * turn_capability(cruise)
    assert cruise["wing_area"] < dynamics["wing_area"]


def test_co_design_beats_both_sizing_routes_on_both_missions(simultaneous):
    """Neither merit reproduces the co-design wing, on either mission.

    This is the argument for optimizing the wing and the cycle together,
    and it is stronger than a weighting being mis-set. Sizing for cruise is
    close on minimum wind, which is a cruise-like objective, and loses more
    on the crossing. Sizing for the manoeuvre loses heavily on both.
    """
    reference_time = cross_country_time(_simultaneous_wing(), SIZE, 500)
    for weights in (CRUISE_WEIGHTS, DYNAMICS_WEIGHTS):
        wing = size_the_wing(weights)
        assert minimum_wind(wing, SIZE, 500) > simultaneous
        assert cross_country_time(wing, SIZE, 500) > reference_time


def test_the_sizing_merit_can_only_land_on_a_corner():
    """No weighting reaches the co-design wing, and here is why.

    The vehicle merit is close to linear in the normalized quantities over
    a box, so its optimum sits at a vertex and jumps between vertices as the
    weights move. The mission's optimum is interior. That is a structural
    reason a sequential route cannot match co-design here, rather than a
    matter of tuning: the wing area the co-design run settles on is not
    available to any weighting.
    """
    areas = {
        size_the_wing({"glide_ratio": 1.0, "mass": 0.0, "turn": weight})["wing_area"]
        for weight in (0.0, 0.2, 0.4, 1.0)
    }
    lower, upper = WING_BOUNDS["wing_area"]
    # Every sized wing sits on one bound or the other, never between them.
    for area in areas:
        assert min(abs(area - lower), abs(area - upper)) < 1e-3
    # The co-design wing does not.
    interior = _simultaneous_wing()["wing_area"]
    assert lower + 1e-3 < interior < upper - 1e-3


def test_sizing_for_the_wrong_thing_costs_a_great_deal(simultaneous):
    """Over-weighting mass buys a lighter bird that cannot fly as cheaply."""
    study = misalignment_study(
        [0.5, 4.0], reference_wind=simultaneous, size=SIZE, max_iter=400
    )
    aligned, misaligned = study
    assert misaligned.penalty > aligned.penalty + 0.3
    assert total_mass(misaligned.wing) < total_mass(aligned.wing)
    assert glide_ratio(misaligned.wing) < glide_ratio(aligned.wing)


def test_the_whole_route_reports_every_step():
    """The result carries what was measured, what was chosen, and the outcome."""
    result = sequential_codesign(size=8, relative_step=0.1, max_iter=250)
    assert set(result.sensitivities) == {"glide_ratio", "mass", "cl_max"}
    assert set(result.wing) == set(WING_BOUNDS)
    assert 4.0 < result.minimum_wind < 20.0


def test_comparing_the_routes_reports_both_missions():
    """The whole two-by-two in one call, cruise first."""
    routes = compare_sizing_routes(size=8, max_iter=250)
    assert [route.name for route in routes] == ["cruise", "dynamics"]
    for route in routes:
        assert set(route.wing) == set(WING_BOUNDS)
        assert route.minimum_wind > 0.0
        assert route.cross_country_time > 0.0
    cruise, dynamics = routes
    # The cruise wing glides better; the dynamics wing turns harder.
    assert glide_ratio(cruise.wing) > glide_ratio(dynamics.wing)
    assert turn_capability(dynamics.wing) > turn_capability(cruise.wing)
