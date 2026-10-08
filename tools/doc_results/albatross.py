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
"""The albatross's results: shapes, missions, wings, the robust cycle."""

from __future__ import annotations

import functools
import itertools
import time

import jax.numpy as jnp
import numpy

from .common import _benchmark_transcriptions
from .common import _compile
from .common import _samples
from .common import _solve
from .common import _solve_checked
from .common import _trajectory_arrays
from .common import _write_arrays
from .common import _write_formulation
from .common import _write_json
from .propagation import _gradient_cost
from .propagation import _propagation_study
from .propagation import _spread
from .propagation import _statistic_of_design


def generate_albatross(smoke: bool = False) -> None:
    """Compute the albatross paper's results.

    Parameters
    ----------
    smoke
        Whether to run the tiny version, for the tests.
    """
    from gemseo_tempo.dynamics.integrators import DiffraxIntegrator
    from gemseo_tempo.problems.albatross import create_codesign_optimization
    from gemseo_tempo.problems.albatross import create_energy_optimization
    from gemseo_tempo.problems.albatross import energy_budget
    from gemseo_tempo.scenario.trajectories import optimal_trajectory

    print("albatross")
    size = 6 if smoke else 20
    budget = 40 if smoke else 500

    from gemseo_tempo.problems.albatross import create_codesign_scenario
    from gemseo_tempo.problems.albatross import create_energy_scenario

    checks = {}

    # The soaring cycle at a fixed wind, and the wing it would choose.
    scenario, gemseo_scenario = create_energy_optimization(size=size)
    _write_formulation("albatross/formulation_energy.json", scenario, gemseo_scenario)
    _solve(gemseo_scenario, budget)
    cycle = optimal_trajectory(scenario, gemseo_scenario)
    _write_arrays("albatross/cycle_energy.npz", **_trajectory_arrays(cycle))
    _write_json("albatross/energy_budget.json", energy_budget(cycle))
    checks["energy"] = _write_dense_cycle(
        "albatross/cycle_energy_dense.npz",
        create_energy_scenario,
        size,
        cycle,
        gemseo_scenario,
        smoke,
    )

    scenario, gemseo_scenario = create_codesign_optimization(size=size)
    _write_formulation("albatross/formulation_codesign.json", scenario, gemseo_scenario)
    minimum_wind, _, feasible = _solve(gemseo_scenario, budget)
    codesign = optimal_trajectory(scenario, gemseo_scenario)
    _write_arrays("albatross/cycle_codesign.npz", **_trajectory_arrays(codesign))
    checks["codesign"] = _write_dense_cycle(
        "albatross/cycle_codesign_dense.npz",
        create_codesign_scenario,
        size,
        codesign,
        gemseo_scenario,
        smoke,
    )
    _write_json("albatross/dense_check.json", checks)
    _write_json(
        "albatross/codesign.json",
        {
            "minimum_wind": minimum_wind,
            "feasible": feasible,
            "budget": energy_budget(codesign),
        },
    )

    _generate_cycle_shapes(size, budget, smoke)
    _generate_design_strategies(size, budget, smoke)
    _benchmark_transcriptions(
        "albatross",
        lambda transcription, count: create_codesign_optimization(
            transcription=transcription,
            size=count,
            design_names=("w_ref",),
            # A fixed step, since SLSQP's trial points can make adaptive
            # stepping run out of steps, and a hundred save points, since
            # shooting checks the path constraints only where it saves:
            # with one per interval the cycle dipped through the sea
            # between them and claimed 1.6 m/s.
            integrator=DiffraxIntegrator(dt0=1.0 / 400, n_save=101),
        ),
        (10, 20) if smoke else (10, 20, 40),
        budget,
        indirect={
            **ALBATROSS_INDIRECT,
            "integrator": DiffraxIntegrator(dt0=1.0 / 400, n_save=9),
        },
        indirect_budget=20,
    )
    _generate_missions(size, budget, smoke)
    _generate_hong_replication(smoke)
    _generate_albatross_robust(smoke)


CYCLE_SHAPES = {"loop": "O, a loop", "weave": "S, a snake"}
"""Barnes's two families, keyed by the topology that makes each."""


SEED_RANGES = {
    "period": (6.0, 12.0),
    "airspeed": (22.0, 36.0),
    "mean_height": (10.0, 22.0),
    "swing": (0.3, 0.9),
}
"""The box the starting cycles are drawn from.

The first three are fields of Guess. The last is how far the cycle rises
and falls either side of its mean height, as a fraction of the room above
the height limit, from which the climb angle follows; sampling the angle
directly would send fast, slow cycles into the sea.
"""


HOVER_STEPS = (0.8, 0.6, 0.5, 0.47, 0.44, 0.41, 0.38, 0.35, 0.3, 0.2, 0.1, 0.0)
"""The fractions of its drift a passage is shortened through towards a hover."""


def _generate_cycle_shapes(size: int, budget: int, smoke: bool) -> None:
    """Find the weakest wind each shape of cycle needs, and what hovering costs.

    The wing is held at the real bird's, so that shapes are compared rather
    than wings. Each shape is started from the default guess and from a
    Latin hypercube of guesses over :data:`SEED_RANGES`, every one built by
    inverse dynamics in that shape; a start counts only if its answer
    measurably meets every constraint. The best is kept, re-flown densely
    and stored with how many starts reached it.

    A hover cannot be found that way: from a guess that drifts with the
    wind, the optimizer stalls. So the best passage is shortened instead,
    its displacement over the sea scaled through :data:`HOVER_STEPS`, each
    solve starting from the last, and the weakest wind is recorded at every
    step reached.

    Parameters
    ----------
    size
        The discretization.
    budget
        The iteration budget of each start.
    smoke
        Whether to run the tiny version, for the tests.
    """
    from scipy.stats import qmc

    from gemseo_tempo.problems.albatross import create_codesign_optimization
    from gemseo_tempo.problems.albatross import create_codesign_scenario
    from gemseo_tempo.problems.albatross.problem import CYCLE_HEIGHT_LIMIT
    from gemseo_tempo.problems.albatross.problem import Guess
    from gemseo_tempo.scenario.trajectories import optimal_trajectory

    unit = qmc.LatinHypercube(d=len(SEED_RANGES), seed=0).random(1 if smoke else 8)
    seeds = [Guess()]
    for row in unit:
        drawn = {
            name: low + fraction * (high - low)
            for (name, (low, high)), fraction in zip(
                SEED_RANGES.items(), row, strict=True
            )
        }
        # A climb angle a held over a period T at airspeed V lifts the bird
        # by at most a V T / 2 pi either side of its mean.
        rise = drawn.pop("swing") * (drawn["mean_height"] - CYCLE_HEIGHT_LIMIT)
        climb = 2.0 * numpy.pi * rise / (drawn["airspeed"] * drawn["period"])
        seeds.append(Guess(climb_amplitude=float(climb), **drawn))

    shapes = {}
    for topology, label in CYCLE_SHAPES.items():
        winds, best = [], None
        for seed in seeds:
            scenario, gemseo_scenario = create_codesign_optimization(
                size=size, guess=seed, design_names=("w_ref",), topology=topology
            )
            wind, design, met, _ = _solve_checked(scenario, gemseo_scenario, budget)
            winds.append(wind if met else None)
            if met and (best is None or wind < best[0]):
                best = (wind, design, scenario, gemseo_scenario)
        if best is None:
            shapes[topology] = {"label": label, "starts": len(seeds), "winds": winds}
            continue
        wind, design, scenario, gemseo_scenario = best
        cycle = optimal_trajectory(scenario, gemseo_scenario)
        drift = numpy.array([cycle.states["x"][-1], cycle.states["y"][-1]])
        heading = numpy.asarray(cycle.states["psi"])
        shapes[topology] = {
            "label": label,
            "starts": len(seeds),
            "winds": winds,
            "minimum_wind": wind,
            "converged": sum(w is not None for w in winds),
            "at_best": sum(w is not None and w <= 1.001 * wind for w in winds),
            "period": float(
                gemseo_scenario.optimization_result.x_opt_as_dict["t_f"][0]
            ),
            "heading_swing": float(numpy.ptp(heading)),
            "height_max": float(numpy.max(cycle.states["h"])),
            "travel": drift.tolist(),
            "dense_check": _write_dense_cycle(
                f"albatross/cycle_{topology}_dense.npz",
                functools.partial(create_codesign_scenario, topology=topology),
                size,
                cycle,
                gemseo_scenario,
                smoke,
            ),
        }

        # Towards a hover: the same cycle, asked to end ever nearer its start.
        steps = [{"fraction": 1.0, "minimum_wind": wind}]
        for fraction in HOVER_STEPS[-1:] if smoke else HOVER_STEPS:
            scenario, gemseo_scenario = create_codesign_optimization(
                size=size,
                design_names=("w_ref",),
                topology=topology,
                displacement=tuple(fraction * drift),
            )
            wind, candidate, met, _ = _solve_checked(
                scenario, gemseo_scenario, 4 * budget, warm=design
            )
            if not met:
                break
            design = candidate
            steps.append({"fraction": fraction, "minimum_wind": wind})
        shapes[topology]["hover"] = steps
    _write_json("albatross/shapes.json", shapes)


ALBATROSS_INDIRECT = {"control_grid": 8, "segments": 10}
"""How the albatross is shot indirectly: a grid over lift and bank, and
enough segments that the costates never grow far within one."""


MISSION_WIND = 22.0
"""The wind the missions are compared in, in m/s: strong enough that a loop
banks energy, so that there is a surplus to spend."""


MISSION_DISTANCE = 250.0
"""The ground distance the ``time`` mission covers as fast as it can, in m."""


def _generate_missions(size: int, budget: int, smoke: bool) -> None:
    """Fly each topology for energy and for distance, in the same wind.

    Parameters
    ----------
    size
        The discretization.
    budget
        The iteration budget.
    smoke
        Whether to run the tiny version, for the tests.
    """
    from gemseo_tempo.problems.albatross import create_energy_optimization
    from gemseo_tempo.problems.albatross import energy_budget
    from gemseo_tempo.problems.albatross.problem import TOPOLOGIES
    from gemseo_tempo.scenario.trajectories import optimal_trajectory

    missions = {"w_ref": MISSION_WIND, "target_distance": MISSION_DISTANCE}
    for topology, mission in itertools.product(
        TOPOLOGIES, ("energy", "distance", "time")
    ):
        extra = {"target_distance": MISSION_DISTANCE} if mission == "time" else {}
        scenario, gemseo_scenario = create_energy_optimization(
            size=size, w_ref=MISSION_WIND, mission=mission, topology=topology, **extra
        )
        objective, _, met, _ = _solve_checked(scenario, gemseo_scenario, budget)
        cycle = optimal_trajectory(scenario, gemseo_scenario)
        missions[f"{topology}_{mission}"] = {
            "met": met,
            "objective": objective,
            "air_change": energy_budget(cycle)["air_change"],
            "distance": float(cycle.states["x"][-1]),
            "duration": float(numpy.asarray(cycle.ts)[-1]),
        }
    _write_json("albatross/missions.json", missions)


def _generate_design_strategies(size: int, budget: int, smoke: bool) -> None:
    """Fly the real wing, a wing sized first, and a wing designed with the cycle.

    Three ways of arriving at the bird, each flown as an O and as an S: the
    wandering albatross's own wing with only the cycle optimized; the
    sequential route, which sizes the wing for its glide ratio and only
    then optimizes the cycle on it; and co-design, which optimizes both at
    once. Co-design starts from the sequential answer, so that it is
    compared with it rather than with a worse local optimum: from the
    default guess the S came back infeasible, at more than twice the wind.

    Parameters
    ----------
    size
        The discretization.
    budget
        The iteration budget.
    smoke
        Whether to run the tiny version, for the tests.
    """
    from gemseo_tempo.problems.albatross import create_codesign_optimization
    from gemseo_tempo.problems.albatross.problem import TOPOLOGIES
    from gemseo_tempo.problems.albatross.problem import WING_BOUNDS
    from gemseo_tempo.problems.albatross.sequential import CRUISE_WEIGHTS
    from gemseo_tempo.problems.albatross.sequential import glide_ratio
    from gemseo_tempo.problems.albatross.sequential import size_the_wing
    from gemseo_tempo.problems.albatross.statics import DESIGN_DEFAULTS

    cruise_wing = size_the_wing(CRUISE_WEIGHTS)
    strategies = {}
    for topology in TOPOLOGIES:
        solutions = {}
        for label, design, names in (
            ("fixed", None, ("w_ref",)),
            ("sequential", cruise_wing, ("w_ref",)),
            ("codesign", None, ("w_ref", *WING_BOUNDS)),
        ):
            _, gemseo_scenario = create_codesign_optimization(
                size=size, design=design, design_names=names, topology=topology
            )
            warm = None
            if label == "codesign":
                warm = {
                    **solutions["sequential"],
                    **{
                        "constant." + key: numpy.atleast_1d(cruise_wing[key])
                        for key in WING_BOUNDS
                    },
                }
            wind, solution, met = _solve(gemseo_scenario, budget, warm=warm)
            solutions[label] = solution
            wing = {
                key: float(solution["constant." + key][0])
                if "constant." + key in solution
                else float((design or DESIGN_DEFAULTS)[key])
                for key in WING_BOUNDS
            }
            strategies[f"{topology}_{label}"] = {
                "minimum_wind": wind,
                "feasible": met,
                "wing": wing,
                "aspect_ratio": wing["wing_span"] ** 2 / wing["wing_area"],
                "glide_ratio": float(glide_ratio(wing)),
            }
    _write_json("albatross/design_strategies.json", strategies)


DENSE_POINTS = 401
"""The save points of a re-simulated cycle, enough to draw it smoothly."""


def _write_dense_cycle(
    name, builder, size, nodes, gemseo_scenario, smoke: bool = False
) -> dict:
    """Fly an optimal cycle's controls through an accurate integrator.

    Collocation solves for the states at its nodes and never integrates
    the dynamics between them. Handing the optimal controls, initial state
    and design to a single-shooting twin of the same scenario, integrated
    to a tight tolerance on a fine grid, gives the trajectory the bird would
    actually fly, which is what the figures draw, and measures how far it
    departs from the nodes, which is what the page reports as the
    discretization error.

    Parameters
    ----------
    name
        The path relative to ``docs/_data``.
    builder
        The scenario builder the optimum came from, which accepts
        ``transcription``, ``size`` and ``integrator``.
    size
        The discretization the optimum was solved at.
    nodes
        The optimal trajectory at the collocation nodes.
    gemseo_scenario
        The executed GEMSEO scenario.
    smoke
        Whether the optimum is a smoke one, flown with fixed steps.

    Returns
    -------
    dict
        The largest departure of each state from the nodes, and how far
        the flown cycle misses closing in height and airspeed.
    """
    from gemseo_tempo.dynamics.integrators import DiffraxIntegrator
    from gemseo_tempo.problems.albatross.statics import DESIGN_DEFAULTS
    from gemseo_tempo.transcriptions import SingleShooting

    optimum = gemseo_scenario.optimization_result.x_opt_as_dict
    if smoke:
        # Six segments are too few for the nodes to hold the cycle up: the
        # smoke optimum, flown, dives below the height floor of the wind
        # profile, and adaptive stepping runs out of steps at any tolerance
        # on the kink there. A fixed step always gets through, which is all
        # the smoke run asks of it.
        integrator = DiffraxIntegrator(dt0=1e-3, n_save=DENSE_POINTS, max_steps=100_000)
    else:
        integrator = DiffraxIntegrator(rtol=1e-10, atol=1e-10, n_save=DENSE_POINTS)
    twin = builder(transcription=SingleShooting(), size=size, integrator=integrator)
    inputs = dict(twin.problem.input_defaults)
    for key, value in optimum.items():
        if key in inputs:
            inputs[key] = jnp.reshape(jnp.asarray(value), jnp.shape(inputs[key]))
    # A free initial state is a node of the collocation grid and a
    # separate input of the shooting one.
    for key in inputs:
        if key.startswith("initial."):
            state = optimum["state." + key.split(".", 1)[1]]
            inputs[key] = jnp.reshape(jnp.asarray(state)[0], jnp.shape(inputs[key]))
    outputs = twin.problem.jax_function(inputs)

    arrays = {"ts": numpy.asarray(outputs["times"])}
    for key, value in outputs.items():
        if key.startswith("trajectory."):
            arrays[key.split(".", 1)[1]] = numpy.asarray(value)
    # The wind-fixed frame moves with the local wind, so a position in it
    # is the ground position less the wind the bird has been carried by.
    carried = numpy.concatenate(
        [
            [0.0],
            numpy.cumsum(
                0.5
                * (arrays["wind"][1:] + arrays["wind"][:-1])
                * numpy.diff(arrays["ts"])
            ),
        ]
    )
    arrays["x_air"] = arrays["x"] - carried
    design = {
        key: float(numpy.ravel(inputs["constant." + key])[0])
        for key in ("w_ref", "wing_area", "wing_span")
        if "constant." + key in inputs
    }
    arrays["wing_span"] = design.get("wing_span", DESIGN_DEFAULTS["wing_span"])
    arrays["w_ref"] = design["w_ref"]
    _write_arrays(name, **arrays)

    departure = {}
    node_times = numpy.asarray(nodes.ts)
    for state, series in nodes.states.items():
        flown = numpy.interp(node_times, arrays["ts"], arrays[state])
        departure[state] = float(numpy.max(numpy.abs(flown - numpy.asarray(series))))
    return {
        "departure": departure,
        "closure": {
            "h": float(arrays["h"][-1] - arrays["h"][0]),
            "V": float(arrays["V"][-1] - arrays["V"][0]),
        },
        "extent": {
            state: float(numpy.ptp(numpy.asarray(nodes.states[state])))
            for state in ("x", "y", "h", "V")
        },
    }


HONG_2023 = {
    "minimum_wind": 6.96,
    "mass": 8.5,
    "wing_area": 0.65,
    "cd0": 0.02,
    "induced_factor": 0.02,
    "cl_max": 1.5,
    "load_factor_max": 3.0,
    "height_min": 1.5,
    "bank_max_deg": 70.0,
}
"""The neutral-boundary-layer case of Hong et al. (2023), Table 1 and Sec. III.

Their minimum reference wind for an energy-neutral cycle in the neutral
logarithmic profile, with the same roughness length (0.03 m) and reference
height (10 m) as this library's wind model, and the bird, drag polar and
limits it was obtained with.
"""


def _generate_hong_replication(smoke: bool) -> None:
    """Fly Hong et al.'s neutral case through this library's dynamics.

    The equations of motion are the same; the statics chain is replaced by
    their constant drag polar and bird, and the limits by theirs. What
    cannot be matched is recorded on the page: their controls are a
    degree-five trigonometric series with rate limits, solved by
    trapezoidal collocation, where these are cubic splines through free
    knots under Hermite-Simpson collocation. Their heading closes without
    a turn, so their cycle is an S; it is solved here as an S, which is the
    like-for-like comparison, and as an O, which is not. Each is solved at
    two discretizations to show the number has converged.

    Parameters
    ----------
    smoke
        Whether to run the tiny version, for the tests.
    """
    from gemseo_tempo.dynamics.controls import CubicControl
    from gemseo_tempo.dynamics.integrators import DiffraxIntegrator
    from gemseo_tempo.outputs.constraints import PathConstraint
    from gemseo_tempo.outputs.constraints import PeriodicityConstraint
    from gemseo_tempo.outputs.functionals import Mayer
    from gemseo_tempo.problems.albatross.model import ROUGHNESS
    from gemseo_tempo.problems.albatross.model import create_state_space
    from gemseo_tempo.problems.albatross.problem import NOMINAL_WIND
    from gemseo_tempo.problems.albatross.problem import REGISTRY
    from gemseo_tempo.problems.albatross.problem import TOPOLOGIES
    from gemseo_tempo.problems.albatross.problem import TURN
    from gemseo_tempo.problems.albatross.problem import Guess
    from gemseo_tempo.problems.albatross.problem import _bounds
    from gemseo_tempo.problems.albatross.problem import _trajectory
    from gemseo_tempo.problems.albatross.problem import soaring_cycle
    from gemseo_tempo.problems.albatross.statics import AIR_DENSITY
    from gemseo_tempo.problems.albatross.statics import GRAVITY
    from gemseo_tempo.scenario import ControlAndDesign
    from gemseo_tempo.scenario import DynamicScenario
    from gemseo_tempo.scenario.trajectories import optimal_trajectory
    from gemseo_tempo.transcriptions import BoundarySpec
    from gemseo_tempo.transcriptions import HermiteSimpson

    reference = HONG_2023
    bird = {
        key: reference[key]
        for key in ("mass", "wing_area", "cd0", "induced_factor", "cl_max")
    }
    cl_max = reference["cl_max"]
    bank_max = float(numpy.radians(reference["bank_max_deg"]))

    def scalar(value):
        return jnp.reshape(value, ())

    def load_factor(t, s, u, theta):
        lift = 0.5 * AIR_DENSITY * s["V"] ** 2 * scalar(theta["wing_area"]) * u["cl"]
        return lift / (scalar(theta["mass"]) * GRAVITY) - reference["load_factor_max"]

    shapes = {}
    for topology, size in itertools.product(TOPOLOGIES, (6,) if smoke else (20, 30)):
        guess = Guess()
        times, states, controls = soaring_cycle(
            guess, NOMINAL_WIND, None, size + 1, topology
        )
        # Their bank limit is a little tighter than the seed banks.
        controls = {
            "cl": jnp.clip(controls["cl"], 0.0, cl_max),
            "mu": jnp.clip(controls["mu"], -bank_max, bank_max),
        }
        normalized = jnp.linspace(0.0, 1.0, size + 1)
        scenario = DynamicScenario(
            state_space=create_state_space(),
            integrator=DiffraxIntegrator(rtol=1e-8, atol=1e-8, n_save=size + 1),
            controls=(
                CubicControl("cl", normalized, default_params=controls["cl"]),
                CubicControl("mu", normalized, default_params=controls["mu"]),
            ),
            functionals=(
                Mayer("w_ref_objective", lambda t_final, s1, th: th["w_ref"]),
            ),
            constraints=(
                PeriodicityConstraint(
                    "cycle",
                    ["h", "V", "gamma", "psi"],
                    offsets={"psi": TURN if topology == "loop" else 0.0},
                ),
                PathConstraint("load_factor_limit", load_factor),
                PathConstraint(
                    "height_limit",
                    lambda t, s, u, th: reference["height_min"] - s["h"],
                ),
                # The controls are splines, which overshoot between knots,
                # so the control limits are path constraints as well as
                # bounds on the knots.
                PathConstraint("lift_limit", lambda t, s, u, th: u["cl"] - cl_max),
                PathConstraint("lift_floor", lambda t, s, u, th: -u["cl"]),
                PathConstraint(
                    "bank_limit", lambda t, s, u, th: u["mu"] ** 2 - bank_max**2
                ),
            ),
            boundary=BoundarySpec(
                s0={key: float(values[0]) for key, values in states.items()},
                fix_initial={"h": False, "V": False, "gamma": False, "psi": False},
                free_final_time=True,
                t_f=guess.period,
            ),
            transcription=HermiteSimpson(num_segments=size),
            registry=REGISTRY,
            constants={"w_ref": NOMINAL_WIND, "roughness": ROUGHNESS, **bird},
            t0=0.0,
            t1=1.0,
        )
        scenario.problem.set_initial_guess(_trajectory(times, states, controls))
        bounds = _bounds(scenario)
        bounds["constant.w_ref"] = (0.0, 40.0)
        bounds["control.cl"] = (0.0, cl_max)
        bounds["control.mu"] = (-bank_max, bank_max)
        # Hong et al. put no floor on the airspeed. This library's generic
        # one, 10 m/s, binds on their bird and costs 0.07 m/s of wind.
        bounds["state.V"] = (2.0, 80.0)
        gemseo_scenario = ControlAndDesign(
            scenario, ["w_ref"], bounds=bounds
        ).create_gemseo_scenario()
        minimum_wind, _, feasible = _solve(gemseo_scenario, 60 if smoke else 800)
        cycle = optimal_trajectory(scenario, gemseo_scenario)
        shapes.setdefault(topology, {"runs": []})["runs"].append(
            {
                "size": size,
                "minimum_wind": minimum_wind,
                "feasible": feasible,
                "period": float(
                    gemseo_scenario.optimization_result.x_opt_as_dict["t_f"][0]
                ),
                "height_max": float(numpy.max(cycle.states["h"])),
                "height_min": float(numpy.min(cycle.states["h"])),
                "airspeed_min": float(numpy.min(cycle.states["V"])),
                "airspeed_max": float(numpy.max(cycle.states["V"])),
                "load_factor_max": float(numpy.max(cycle.outputs["load_factor"])),
            }
        )
    for shape in shapes.values():
        shape["minimum_wind"] = shape["runs"][-1]["minimum_wind"]
        shape["difference"] = shape["minimum_wind"] / reference["minimum_wind"] - 1.0
    _write_json("albatross/hong_2023.json", {"reference": reference, **shapes})


def _generate_albatross_robust(smoke: bool) -> None:
    """Propagate the wind through the energy cycle, then design for it.

    The energy mission at the forecast wind of 22 m/s, flown by single
    shooting so that a frozen cycle responds to a different wind. The
    robust cycle maximizes the mean less twice the deviation of the energy
    it ends with, over a fixed sample of the wind speed and the roughness
    of the sea, starting from the deterministic cycle.

    Parameters
    ----------
    smoke
        Whether to run the tiny version, for the tests.
    """
    from gemseo_tempo.dynamics.integrators import DiffraxIntegrator
    from gemseo_tempo.problems.albatross import create_energy_optimization
    from gemseo_tempo.problems.albatross import create_energy_scenario
    from gemseo_tempo.problems.albatross.model import REGISTRY
    from gemseo_tempo.scenario.formulations import OptimalControl
    from gemseo_tempo.transcriptions import SingleShooting
    from gemseo_tempo.uncertainty import uncertain_names
    from gemseo_tempo.uncertainty.robust import MARGIN
    from gemseo_tempo.uncertainty.robust import robust_discipline

    # Forty segments: twenty leave the collocation cycle 6 m short of closing
    # when its controls are flown by integration, too far for shooting to
    # recover from; forty leave it 1 m short, and shooting converges.
    size = 10 if smoke else 40
    budget = 60 if smoke else 150
    factor = -2.0
    quantity = "energy"
    names = uncertain_names(REGISTRY)
    design_sample, check_sample = _samples(
        REGISTRY, names, (8, 128) if smoke else (64, 2000)
    )

    settings = {
        "transcription": SingleShooting(),
        "size": size,
        "w_ref": 22.0,
        # A hundred save points, since shooting checks the lift limit and the
        # height floor only where it saves.
        "integrator": DiffraxIntegrator(dt0=1.0 / 400, n_save=101, adjoint="Direct"),
        # Some three hundred constraint rows against eighty-two inputs:
        # forward mode, through the Direct adjoint, which allows it. The
        # robust discipline takes forward mode for its constraints and
        # reverse mode for its one statistic, group by group, and one
        # integrator serves both. Measured at this size, per Jacobian:
        # constraints 6.7 s forward through Direct, 26 s reverse through it,
        # 13 s reverse through the checkpointing adjoint; the statistic
        # 9.2 s reverse through Direct, 5.9 s through checkpointing. Direct
        # makes 16 s in all, checkpointing 19 s.
        "differentiation_method": "forward",
    }

    def build():
        return create_energy_optimization(**settings)

    # Single shooting from the default guess does not find a feasible cycle:
    # nine seconds of unstable flight make it badly conditioned. Collocation
    # does, and its controls, on the same knots, start shooting beside it.
    # Measured: from collocation's controls, shooting meets every constraint
    # in under eighty iterations.
    _, collocation = create_energy_optimization(size=size, w_ref=22.0)
    _, found, _ = _solve(collocation, budget)
    controls = {
        key: value for key, value in found.items() if key.startswith("control.")
    }

    scenario, gemseo_scenario = build()
    study = _propagation_study(
        "albatross",
        scenario,
        gemseo_scenario,
        REGISTRY,
        quantity,
        smoke,
        budget,
        warm=controls,
    )
    deterministic = {
        key: numpy.atleast_1d(value)
        for key, value in gemseo_scenario.optimization_result.x_opt_as_dict.items()
    }

    scenario, reference = build()
    formulation = OptimalControl(
        scenario,
        maximize=True,
        # The deterministic mission's own box.
        bounds={
            key: (
                reference.design_space.get_lower_bound(key),
                reference.design_space.get_upper_bound(key),
            )
            for key in reference.design_space.variable_names
        },
        disciplines=[
            robust_discipline(scenario, names, design_sample, quantity, factor=factor)
        ],
        objective=f"{quantity}.{MARGIN}",
    )
    gemseo_scenario = formulation.create_gemseo_scenario()
    compile_seconds = _compile(formulation.disciplines[0], True)
    started = time.perf_counter()
    objective, robust, met, _ = _solve_checked(
        scenario, gemseo_scenario, budget, warm=deterministic
    )
    seconds = time.perf_counter() - started
    calls = len(gemseo_scenario.formulation.optimization_problem.database)

    # What one gradient of the margin costs at the robust cycle, by each
    # adjoint; for one output, reverse mode through either is exact.
    functions = {}
    for adjoint, method in (("Direct", "forward"), ("RecursiveCheckpoint", "reverse")):
        measured = create_energy_scenario(
            **{
                **settings,
                "integrator": DiffraxIntegrator(
                    dt0=1.0 / 400, n_save=101, adjoint=adjoint
                ),
                "differentiation_method": method,
            }
        )
        measured.problem.input_defaults.update(robust)
        functions[adjoint], point = _statistic_of_design(
            measured, names, design_sample, quantity, MARGIN, factor=factor
        )
    designs = {
        "deterministic": {
            "objective": study["objective"],
            **_spread(
                create_energy_scenario(**settings),
                names,
                check_sample,
                quantity,
                deterministic,
            ),
        },
        "robust": {
            "objective": objective,
            "met": met,
            "compile_seconds": compile_seconds,
            "seconds": seconds,
            "calls": calls,
            **_spread(
                create_energy_scenario(**settings),
                names,
                check_sample,
                quantity,
                robust,
            ),
        },
    }
    _write_json(
        "albatross/robust.json",
        {
            "factor": factor,
            "design_sample": len(design_sample),
            "check_sample": len(check_sample),
            "designs": designs,
            "gradient_cost": _gradient_cost(functions, point),
        },
    )
