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
"""Sizing the bird before flying it: the sequential route.

The co-design formulation optimizes the wing and the cycle at once. That is
not how the work is usually organized. In practice the vehicle is settled
first, against figures of merit an engineer can reason about, and only then is
a trajectory flown on it. This module builds that route so the two can be
compared honestly, which is the point of having both.

Three steps.

**Which parameters matter** (:func:`sensitivity_study`). The mission is
re-optimized with each vehicle-level figure of merit nudged in turn, and the
response of the minimum wind is recorded as an elasticity. Re-optimizing
rather than differentiating at the optimum means what is measured is the real
effect on the mission, including the optimizer's freedom to fly the perturbed
bird differently.

**Choose a wing** (:func:`size_the_wing`). Those elasticities become the
weights of a vehicle-level objective, and the wing is chosen against that
alone. No trajectory appears in this step, which is exactly the point: the
designer is working with glide ratio and mass, not with cycles.

**Fly it** (:func:`sequential_codesign`). The trajectory is optimized on the
chosen wing, and the wind it needs is what the sequential route delivers.

The weights are measured rather than asserted, but a weighted sum over
normalized quantities is still a modelling choice. Both the elasticities and
the normalization are reported in the result, so what the wing was actually
asked for stays visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

import jax.numpy as jnp

from gemseo_tempo.dynamics.integrators import DiffraxIntegrator
from gemseo_tempo.outputs.functionals import Mayer
from gemseo_tempo.problems.albatross.model import create_state_space
from gemseo_tempo.problems.albatross.problem import REGISTRY
from gemseo_tempo.problems.albatross.problem import WING_BOUNDS
from gemseo_tempo.problems.albatross.problem import Guess
from gemseo_tempo.problems.albatross.problem import _controls
from gemseo_tempo.problems.albatross.problem import _design_values
from gemseo_tempo.problems.albatross.problem import create_codesign_optimization
from gemseo_tempo.problems.albatross.problem import soaring_cycle
from gemseo_tempo.problems.albatross.statics import DESIGN_DEFAULTS
from gemseo_tempo.problems.albatross.statics import create_statics
from gemseo_tempo.scenario.dynamic_scenario import DynamicScenario
from gemseo_tempo.scenario.formulations import SystemDesign
from gemseo_tempo.transcriptions.base import BoundarySpec

VEHICLE_PARAMETERS = {
    "glide_ratio": "oswald",
    "mass": "body_mass",
    "cl_max": "section",
}
"""The figures of merit, and the design constant each is moved through.

The first two are separable: the span efficiency moves the glide ratio without
touching the mass, and the body mass moves the mass without touching the
aerodynamics.

The maximum lift coefficient is not, and deliberately so. It is moved through
the wing section, which raises the profile drag at the same time, so its
elasticity is the **net** effect of walking along the cruise-against-manoeuvre
trade rather than the effect of free extra lift. That is the honest quantity:
nothing in the model, or in a real wing, offers more maximum lift at no cost.
"""


def glide_ratio(design: dict[str, float] | None = None) -> float:
    """Return the best glide ratio of a design.

    Parameters
    ----------
    design
        Overrides of :data:`.DESIGN_DEFAULTS`.

    Returns
    -------
    float
        The maximum lift-to-drag ratio.
    """
    statics = create_statics()(_design_values(design))
    induced = float(statics["induced_factor"])
    return 1.0 / (2.0 * float(jnp.sqrt(float(statics["cd0"]) * induced)))


def total_mass(design: dict[str, float] | None = None) -> float:
    """Return the all-up mass of a design.

    Parameters
    ----------
    design
        Overrides of :data:`.DESIGN_DEFAULTS`.

    Returns
    -------
    float
        The mass, in kg.
    """
    return float(create_statics()(_design_values(design))["mass"])


def minimum_wind(
    design: dict[str, float] | None = None,
    size: int = 20,
    max_iter: int = 500,
) -> float:
    """Return the weakest wind a cycle closes in, on a fixed wing.

    Parameters
    ----------
    design
        Overrides of :data:`.DESIGN_DEFAULTS`.
    size
        The discretization.
    max_iter
        The optimizer iteration budget.

    Returns
    -------
    float
        The minimum reference wind, in m/s.
    """
    _, gemseo_scenario = create_codesign_optimization(
        size=size, design=design, design_names=("w_ref",)
    )
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=max_iter)
    return float(gemseo_scenario.optimization_result.f_opt)


@dataclass
class Sensitivity:
    """How the mission responds to one vehicle-level figure of merit."""

    name: str
    """The figure of merit."""

    baseline: float
    """Its nominal value."""

    elasticity: float
    """The relative change in minimum wind per relative change in it.

    Negative means more of it is better: raising the glide ratio lowers the
    wind the bird needs. The maximum lift coefficient comes out **positive**
    for this mission, because reaching it means a thicker section, and the
    drag that costs outweighs the lift it buys when nothing rewards turning
    tightly.
    """


def sensitivity_study(
    size: int = 20,
    relative_step: float = 0.08,
    max_iter: int = 500,
) -> dict[str, Sensitivity]:
    """Measure what the mission cares about, by re-flying perturbed birds.

    Parameters
    ----------
    size
        The discretization.
    relative_step
        The fractional perturbation applied either side of nominal.
    max_iter
        The optimizer iteration budget of each re-solve.

    Returns
    -------
    dict
        A :class:`Sensitivity` per entry of :data:`VEHICLE_PARAMETERS`.
    """
    measured = {name: _measure(name) for name in VEHICLE_PARAMETERS}
    results = {}
    for merit, constant in VEHICLE_PARAMETERS.items():
        nominal = DESIGN_DEFAULTS[constant]
        low = {constant: nominal * (1.0 - relative_step)}
        high = {constant: nominal * (1.0 + relative_step)}
        wind_low = minimum_wind(low, size, max_iter)
        wind_high = minimum_wind(high, size, max_iter)
        merit_low = measured[merit](low)
        merit_high = measured[merit](high)

        # A central difference in relative terms, so the three figures of
        # merit are comparable despite their different units and scales.
        mean_wind = 0.5 * (wind_low + wind_high)
        mean_merit = 0.5 * (merit_low + merit_high)
        spread = merit_high - merit_low
        elasticity = (
            0.0
            if abs(spread) < 1e-12
            else ((wind_high - wind_low) / mean_wind) / (spread / mean_merit)
        )
        results[merit] = Sensitivity(
            name=merit, baseline=measured[merit](None), elasticity=elasticity
        )
    return results


def _measure(merit: str):
    """Return the function evaluating one figure of merit.

    Parameters
    ----------
    merit
        The name, a key of :data:`VEHICLE_PARAMETERS`.

    Returns
    -------
    Callable
        A function from a design override dict to the value.
    """
    if merit == "glide_ratio":
        return glide_ratio
    if merit == "mass":
        return total_mass
    return lambda design: float(create_statics()(_design_values(design))["cl_max"])


def _vehicle_scenario(weights: dict[str, float], size: int = 6) -> DynamicScenario:
    """Build the sizing problem, whose objective is the wing alone.

    No cycle is closed and no path is constrained: the trajectory is present
    only because the wing reaches the objective through the statics chain,
    which is what makes this a :class:`.SystemDesign` rather than a bare
    algebraic optimization.

    Parameters
    ----------
    weights
        The weight of the glide ratio, the mass and the maximum lift
        coefficient in the merit.
    size
        The (small) discretization of the inert trajectory.

    Returns
    -------
    DynamicScenario
        The sizing scenario.
    """
    guess = Guess()
    times, states, controls = soaring_cycle(guess, 0.0, None, size + 1)
    nominal_ratio = glide_ratio(None)
    nominal_mass = total_mass(None)
    nominal_lift = float(create_statics()(_design_values(None))["cl_max"])

    def merit(t_final, final_states, theta):
        """Return the weighted vehicle merit, in normalized units."""
        induced = jnp.reshape(theta["induced_factor"], ())
        profile = jnp.reshape(theta["cd0"], ())
        ratio = 1.0 / (2.0 * jnp.sqrt(profile * induced))
        mass = jnp.reshape(theta["mass"], ())
        lift = jnp.reshape(theta["cl_max"], ())
        return (
            weights["glide_ratio"] * ratio / nominal_ratio
            - weights["mass"] * mass / nominal_mass
            + weights.get("cl_max", 0.0) * lift / nominal_lift
        )

    return DynamicScenario(
        state_space=create_state_space(),
        integrator=DiffraxIntegrator(dt0=0.5, n_save=size + 1),
        controls=_controls(times, controls),
        functionals=(Mayer("vehicle_merit", merit),),
        constraints=(),
        boundary=BoundarySpec(
            s0={name: float(values[0]) for name, values in states.items()}
        ),
        statics=create_statics(),
        registry=REGISTRY,
        constants={"w_ref": 0.0, **_design_values(None)},
        t0=0.0,
        t1=guess.period,
    )


def weights_from(sensitivities: dict[str, Sensitivity]) -> dict[str, float]:
    """Turn measured elasticities into merit weights.

    Parameters
    ----------
    sensitivities
        The output of :func:`sensitivity_study`.

    Returns
    -------
    dict
        The weight of each figure of merit.
    """
    # A negative elasticity means more of that quantity lowers the wind, so
    # its weight in a merit to be maximized is the elasticity negated.
    return {
        "glide_ratio": -sensitivities["glide_ratio"].elasticity,
        "mass": sensitivities["mass"].elasticity,
        "cl_max": -sensitivities["cl_max"].elasticity,
    }


def size_the_wing(
    weights: dict[str, float],
    size: int = 6,
    max_iter: int = 100,
) -> dict[str, float]:
    """Choose a wing against the weighted vehicle merit.

    Parameters
    ----------
    weights
        The weight of the glide ratio, the mass and the maximum lift
        coefficient in the merit.
        :func:`weights_from` derives them from a sensitivity study; passing
        them directly is how a designer's own priorities are expressed.
    size
        The discretization of the inert trajectory.
    max_iter
        The optimizer iteration budget.

    Returns
    -------
    dict
        The chosen ``wing_area`` and ``wing_span``.
    """
    scenario = _vehicle_scenario(weights, size)
    bounds = {"constant." + name: limits for name, limits in WING_BOUNDS.items()}
    formulation = SystemDesign(
        scenario, list(WING_BOUNDS), maximize=True, bounds=bounds
    )
    gemseo_scenario = formulation.create_gemseo_scenario()
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=max_iter)
    optimum = gemseo_scenario.optimization_result.x_opt_as_dict
    return {name: float(optimum["constant." + name][0]) for name in WING_BOUNDS}


@dataclass
class SequentialResult:
    """What the sequential route produced, and what it was asked for."""

    sensitivities: dict[str, Sensitivity]
    """The measured response of the mission to each figure of merit."""

    wing: dict[str, float] = field(default_factory=dict)
    """The wing the sizing step chose."""

    minimum_wind: float = 0.0
    """The wind that wing then needs, in m/s."""


def sequential_codesign(
    size: int = 20,
    relative_step: float = 0.08,
    max_iter: int = 500,
) -> SequentialResult:
    """Run the whole sequential route and report every step.

    Parameters
    ----------
    size
        The discretization of the trajectory runs.
    relative_step
        The perturbation of the sensitivity study.
    max_iter
        The optimizer iteration budget.

    Returns
    -------
    SequentialResult
        The sensitivities, the wing, and the wind it needs.
    """
    sensitivities = sensitivity_study(size, relative_step, max_iter)
    wing = size_the_wing(weights_from(sensitivities))
    return SequentialResult(
        sensitivities=sensitivities,
        wing=wing,
        minimum_wind=minimum_wind(wing, size, max_iter),
    )


@dataclass
class Misalignment:
    """One point of the study of what a wrong merit costs."""

    mass_weight: float
    """The weight the designer put on mass."""

    wing: dict[str, float]
    """The wing that weighting chose."""

    minimum_wind: float
    """The wind that wing needs, in m/s."""

    penalty: float
    """The fractional excess over the simultaneous optimum."""


def misalignment_study(
    mass_weights,
    reference_wind: float,
    glide_weight: float = 0.943,
    size: int = 12,
    max_iter: int = 500,
) -> list[Misalignment]:
    """Measure what it costs to size a vehicle against the wrong merit.

    The sequential route is only as good as the figures of merit it is given.
    When they line up with what the mission wants it loses nothing at all;
    when they do not, the wing is chosen for something the mission does not
    reward and the trajectory pays for it afterwards. Sweeping the weight on
    mass turns that from an argument into a number.

    Parameters
    ----------
    mass_weights
        The weights on mass to try, against a fixed weight on glide ratio.
    reference_wind
        The simultaneous co-design optimum to measure the penalty against,
        in m/s.
    glide_weight
        The weight on the glide ratio, held fixed across the sweep.
    size
        The discretization of the trajectory runs.
    max_iter
        The optimizer iteration budget.

    Returns
    -------
    list
        One :class:`Misalignment` per weight, in the order given.
    """
    results = []
    for weight in mass_weights:
        wing = size_the_wing({"glide_ratio": glide_weight, "mass": weight})
        wind = minimum_wind(wing, size, max_iter)
        results.append(
            Misalignment(
                mass_weight=float(weight),
                wing=wing,
                minimum_wind=wind,
                penalty=(wind - reference_wind) / reference_wind,
            )
        )
    return results


__all__ = [
    "VEHICLE_PARAMETERS",
    "SequentialResult",
    "Misalignment",
    "Sensitivity",
    "glide_ratio",
    "minimum_wind",
    "misalignment_study",
    "sensitivity_study",
    "sequential_codesign",
    "size_the_wing",
    "weights_from",
    "total_mass",
]
