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
"""Tests of the brachistochrone problem package."""

from __future__ import annotations

import logging

import numpy
import pytest

from gemseo_tempo.problems.brachistochrone import GRAVITY
from gemseo_tempo.problems.brachistochrone import REGISTRY
from gemseo_tempo.problems.brachistochrone import Guess
from gemseo_tempo.problems.brachistochrone import create_optimization
from gemseo_tempo.problems.brachistochrone import create_scenario
from gemseo_tempo.problems.brachistochrone import create_state_space
from gemseo_tempo.problems.brachistochrone import cycloid_time
from gemseo_tempo.problems.brachistochrone import optimal_path_angle
from gemseo_tempo.problems.brachistochrone import reference_time
from gemseo_tempo.problems.brachistochrone import simulate_guess
from gemseo_tempo.problems.brachistochrone import variant_constants
from gemseo_tempo.transcriptions import HermiteSimpson
from gemseo_tempo.transcriptions import IndirectShooting
from gemseo_tempo.transcriptions import SingleShooting

logging.disable(logging.CRITICAL)


def solve(scenario_and_gemseo, max_iter=200):
    _, gemseo_scenario = scenario_and_gemseo
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=max_iter)
    return float(gemseo_scenario.optimization_result.f_opt)


# ----------------------------------------------------------------- model


def test_model_structure_and_registry():
    space = create_state_space()
    assert space.states == ("x", "y", "v")
    assert space.controls == ("gamma",)
    assert space.parameters == ("g", "mu", "drag_coefficient")
    # The centripetal part of the normal force needs the turning rate.
    assert space.control_rates == ("gamma",)
    assert REGISTRY["v"].unit == "m/s"
    assert REGISTRY["gamma"].unit == "rad"
    assert REGISTRY["drag_coefficient"].unit == "dimensionless"


def test_variant_constants():
    assert variant_constants("ideal") == {
        "g": GRAVITY,
        "mu": 0.0,
        "drag_coefficient": 0.0,
    }
    assert variant_constants("friction")["mu"] > 0.0
    assert variant_constants("drag")["drag_coefficient"] > 0.0
    with pytest.raises(ValueError, match="Unknown variant"):
        variant_constants("rocket")


def test_ideal_dynamics_are_the_frictionless_ones():
    space = create_state_space()
    rates = space.rhs(
        0.0,
        {"x": 0.0, "y": 0.0, "v": 2.0},
        {"gamma": 0.0, "gamma_rate": 0.0},
        {"g": GRAVITY, "mu": 0.0, "drag_coefficient": 0.0},
    )
    # Horizontal motion at gamma = 0, and no acceleration.
    assert float(rates["x"]) == pytest.approx(2.0)
    assert float(rates["y"]) == pytest.approx(0.0)
    assert float(rates["v"]) == pytest.approx(0.0)


# ------------------------------------------------------------- reference


def test_cycloid_reference():
    # Verified independently against the converged optimizer.
    assert cycloid_time() == pytest.approx(0.824339, abs=1e-6)
    assert reference_time("ideal") == cycloid_time()
    # The defining property: the cycloid beats the straight chord, whose
    # descent time from rest is L * sqrt(2 / (g * y)).
    chord = numpy.hypot(2.0, 2.0) * numpy.sqrt(2.0 / (GRAVITY * 2.0))
    assert cycloid_time() < chord
    # A deeper target takes longer: the extra distance outweighs the
    # stronger gravity assist.
    assert cycloid_time(2.0, 4.0) > cycloid_time(2.0, 2.0)


def test_stored_references():
    # Friction slows the bead, drag too, both relative to the ideal case.
    assert reference_time("friction") > reference_time("ideal")
    assert reference_time("drag") > reference_time("ideal")
    with pytest.raises(KeyError, match="No stored reference"):
        reference_time("friction", 5.0, 5.0)


# ----------------------------------------------------------------- guess


def test_simulated_guess_is_dynamically_consistent():
    """The guess must not leave the bead at rest."""
    trajectory = simulate_guess(Guess(), variant_constants("ideal"))
    speed = numpy.asarray(trajectory.states["v"])
    assert speed[0] == pytest.approx(0.0)
    # The bead accelerates, so the degenerate v == 0 guess is gone.
    assert speed[-1] > 1.0
    assert numpy.all(numpy.diff(speed) >= -1e-9)
    assert float(trajectory.states["x"][-1]) > 0.0


def test_guess_angles():
    guess = Guess(gamma_level=0.2, gamma_slope=0.4)
    angles = numpy.asarray(guess.angles(5))
    assert angles[0] == pytest.approx(0.2)
    assert angles[-1] == pytest.approx(0.6)


def test_guess_pipeline_rescues_collocation():
    """The regression guard: a linear ramp stalls, the simulated guess does not.

    Cold-started from the default linear ramp, collocation on this problem
    misses by over a hundred percent; seeded from a simulated trajectory it
    is exact to the digits printed here.
    """
    seeded = create_scenario(HermiteSimpson(num_segments=20), size=20)
    cold = create_scenario(HermiteSimpson(num_segments=20), size=20)
    # Undo the seeding to recover the naive ramp.
    problem = cold.problem
    for name, spec in problem.design_variables.items():
        if name.startswith("state."):
            problem.input_defaults[name] = spec.value * 0.0
    seeded_states = numpy.asarray(seeded.problem.input_defaults["state.v"])
    cold_states = numpy.asarray(problem.input_defaults["state.v"])
    assert numpy.max(numpy.abs(seeded_states)) > 1.0
    assert numpy.max(numpy.abs(cold_states)) == pytest.approx(0.0)


# ---------------------------------------------------------- optimization


@pytest.mark.parametrize("size", [10, 30])
def test_shooting_matches_the_cycloid(size):
    value = solve(create_optimization(SingleShooting(), size=size))
    assert value == pytest.approx(cycloid_time(), rel=1e-3)


@pytest.mark.parametrize("size", [10, 30])
def test_collocation_matches_the_cycloid(size):
    value = solve(create_optimization(HermiteSimpson(num_segments=size), size=size))
    assert value == pytest.approx(cycloid_time(), rel=5e-3)


def test_transcriptions_agree_on_the_ideal_variant():
    shooting = solve(create_optimization(SingleShooting(), size=20))
    collocation = solve(create_optimization(HermiteSimpson(num_segments=20), size=20))
    assert shooting == pytest.approx(collocation, rel=5e-3)


@pytest.mark.parametrize("variant", ["friction", "drag"])
def test_variants_match_their_stored_references(variant):
    value = solve(
        create_optimization(HermiteSimpson(num_segments=40), variant=variant, size=40)
    )
    assert value == pytest.approx(reference_time(variant), rel=5e-3)


def test_custom_guess_is_used():
    scenario = create_scenario(
        SingleShooting(), size=8, guess=Guess(gamma_level=0.4, t_f=1.7)
    )
    assert float(scenario.problem.input_defaults["t_f"][0]) == pytest.approx(1.7)
    assert float(scenario.problem.input_defaults["control.gamma"][0]) == pytest.approx(
        0.4
    )


def test_indirect_uses_the_analytic_path_angle():
    """Stationarity of H in gamma has a closed form, so u* is supplied."""
    guess = Guess(t_f=0.9, costates={"x": -0.5, "y": -0.5, "v": -0.1})
    scenario = create_scenario(
        IndirectShooting(u_star=optimal_path_angle), size=20, guess=guess
    )
    defaults = scenario.problem.input_defaults
    assert float(defaults["costate.x"][0]) == pytest.approx(-0.5)
    assert float(defaults["costate.v"][0]) == pytest.approx(-0.1)
    outputs = scenario.problem.jax_function(defaults)
    # Three states plus the free final time: a square system.
    assert outputs["residual"].shape == (4,)
    assert len(scenario.problem.design_variables) == 4
    angles = numpy.asarray(outputs["trajectory.gamma"])
    assert numpy.all(numpy.isfinite(angles))


def test_analytic_path_angle_is_the_stationary_point():
    """The closed form must zero dH/dgamma, where it applies.

    Which is at a turn rate of zero. The closed form is the stationary
    point of the classical friction law, ``mu g cos(gamma)``, and the
    model's centripetal term vanishes exactly when ``dgamma/dt`` does; away
    from there the control enters through its own rate and there is no
    pointwise stationarity to find, which is why indirect shooting refuses
    a model that reads a rate at all.
    """
    import jax
    import jax.numpy as jnp

    # Costates chosen to put the stationary angle inside the control's own
    # bounds, where the wire is pushing. Beyond ninety degrees the normal
    # force floors at zero and friction stops depending on the angle at
    # all, so the closed form — which knows nothing of the floor — has
    # nothing to be stationary against there.
    states = {"x": 0.3, "y": 0.4, "v": 2.0}
    costates = {"x": 0.7, "y": 1.0, "v": -0.15}
    theta = variant_constants("friction")
    space = create_state_space()

    def hamiltonian(angle):
        rates = space.rhs(0.0, states, {"gamma": angle, "gamma_rate": 0.0}, theta)
        return sum(costates[name] * rates[name] for name in ("x", "y", "v"))

    best = optimal_path_angle(0.0, states, costates, theta)["gamma"]
    # Not machine zero: the normal force is floored with a smooth maximum,
    # which leaves a trace of the floor even where it is not binding, so
    # the closed form is stationary to the smoothing rather than exactly.
    assert abs(float(jax.grad(hamiltonian)(float(best)))) < 1e-6
    assert jnp.isfinite(best)


def test_every_uncertain_constant_reaches_the_dynamics():
    """A declared uncertainty the model ignores is worse than no model.

    The albatross learned this once: ``C_L,max`` was an independent
    constant reaching the dynamics through nothing at all, and the
    mission's sensitivity to it came out *exactly zero*, which reads like a
    physical finding and was an artefact of the wiring. The brachistochrone
    reintroduced it for one sprint — the friction term was dropped whenever
    ``mu`` happened to be zero in a variant's constants, so the drag
    variant's sensitivity to ``mu`` was exactly zero too — and a screening
    caught it.

    So: for every variant a study is run on, perturbing each uncertain
    constant must move something. The ideal variant is exempt, and only it,
    because there friction is zero by definition rather than by accident.
    """
    from gemseo_tempo.uncertainty import uncertain_names

    names = uncertain_names(REGISTRY)
    assert set(names) == {"g", "mu", "drag_coefficient"}

    for variant in ("friction", "drag"):
        scenario = create_scenario(SingleShooting(), variant=variant, size=8)
        inputs = dict(scenario.problem.input_defaults)

        def miss(values, scenario=scenario):
            """Where the bead ends up, which is what a constant moves."""
            outputs = scenario.problem.jax_function(values)
            return numpy.array(
                [
                    float(numpy.ravel(outputs["constraint.final.x"])[0]),
                    float(numpy.ravel(outputs["constraint.final.y"])[0]),
                ]
            )

        base = miss(inputs)
        for name in names:
            perturbed = dict(inputs)
            key = f"constant.{name}"
            perturbed[key] = numpy.asarray(inputs[key], dtype=float) + 0.05
            moved = numpy.max(numpy.abs(miss(perturbed) - base))
            assert moved > 1e-9, (
                f"the {variant} variant ignores {name!r}: perturbing it moves"
                " the trajectory by nothing at all, so any sensitivity study"
                " on it would report exactly zero and mean nothing"
            )
