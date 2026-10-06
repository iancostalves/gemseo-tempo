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
"""Computing everything the documentation's figures are drawn from.

The documentation draws its figures at build time, so a reader sees the
code that made the picture. What that code must not do is solve anything:
an optimizer in a documentation build is an hour of continuous integration
and a page that breaks whenever a solver has an off day, which this
library has seen happen.

So every optimization happens here, once, and the answers are stored under
``docs/_data``. The figures load them. Direct function evaluations, a polar
curve or a wind profile, are left to the page, because they cost
milliseconds and cannot go stale.

This is the same arrangement ``tools/generate_brachistochrone_references.py``
uses for the reference descent times, and it has the same weakness: run by
hand, the stored answers can drift from the code that would now produce
them. ``tests/test_doc_data.py`` is the guard, checking the stored results
against quantities cheap enough to recompute on every test run.

Usage::

    python tools/generate_doc_results.py                 # everything
    python tools/generate_doc_results.py --only sail     # one problem
    python tools/generate_doc_results.py --smoke         # tiny, for tests
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import logging
import time
from pathlib import Path

import jax.numpy as jnp
import numpy

logging.disable(logging.CRITICAL)

DATA = Path(__file__).resolve().parents[1] / "docs" / "_data"
"""Where the stored results live."""


PROBLEMS = ("brachistochrone", "sail", "albatross")
"""The problems, in the order the documentation introduces them."""


def _write_arrays(name: str, **arrays) -> Path:
    """Store arrays as a compressed archive.

    Parameters
    ----------
    name
        The path relative to ``docs/_data``.
    **arrays
        The arrays to store.

    Returns
    -------
    pathlib.Path
        Where they were written.
    """
    path = DATA / name
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy.savez_compressed(
        path, **{key: numpy.asarray(value) for key, value in arrays.items()}
    )
    print(f"  {name}")
    return path


def _write_json(name: str, payload) -> Path:
    """Store a table or a set of scalars as JSON.

    Parameters
    ----------
    name
        The path relative to ``docs/_data``.
    payload
        Anything json-serializable.

    Returns
    -------
    pathlib.Path
        Where it was written.
    """
    path = DATA / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"  {name}")
    return path


def _write_formulation(name: str, scenario, gemseo_scenario) -> Path:
    """Store the problem statement the optimizer is handed.

    The ``{formulation}`` directive renders this record, so a problem page
    shows the design variables, bounds and constraints of the scenario the
    ``create_*_optimization`` function actually builds. The bounds are read
    off the GEMSEO design space rather than restated, since that is where
    the builders put them; the constraint sizes are measured by
    :func:`.describe_problem`, by evaluating the problem once.

    Parameters
    ----------
    name
        The path relative to ``docs/_data``.
    scenario
        The dynamic scenario.
    gemseo_scenario
        The GEMSEO scenario built on it, not yet executed.

    Returns
    -------
    pathlib.Path
        Where it was written.
    """
    from gemseo_tempo.scenario.description import describe_problem

    problem = gemseo_scenario.formulation.optimization_problem
    space = problem.design_space
    measured = describe_problem(scenario)

    def side(values, reduce):
        value = float(reduce(numpy.asarray(values, dtype=float)))
        return value if numpy.isfinite(value) else None

    return _write_json(
        name,
        {
            "objective": measured.objective,
            "maximize": not problem.minimize_objective,
            "design_variables": [
                {
                    "name": variable,
                    "size": int(space.get_size(variable)),
                    "lower_bound": side(space.get_lower_bound(variable), numpy.min),
                    "upper_bound": side(space.get_upper_bound(variable), numpy.max),
                }
                for variable in space.variable_names
            ],
            "constraints": [
                {"name": item.name, "kind": item.kind, "size": item.size}
                for item in measured.constraints
            ],
        },
    )


def _solve(gemseo_scenario, max_iter: int, warm=None):
    """Run one optimization, optionally from a given design.

    Parameters
    ----------
    gemseo_scenario
        The GEMSEO scenario.
    max_iter
        The iteration budget.
    warm
        A design to start from, as ``{name: value}``.

    Returns
    -------
    tuple
        The objective, the optimal design, and whether it is feasible.
    """
    if warm:
        current = gemseo_scenario.design_space.get_current_value(as_dict=True)
        gemseo_scenario.design_space.set_current_value(
            {
                name: numpy.reshape(value, numpy.shape(current[name]))
                for name, value in warm.items()
                if name in current
            }
        )
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=max_iter)
    result = gemseo_scenario.optimization_result
    design = {
        name: numpy.atleast_1d(value) for name, value in result.x_opt_as_dict.items()
    }
    return float(result.f_opt), design, bool(result.is_feasible)


def _solve_checked(scenario, gemseo_scenario, max_iter: int, warm=None, tolerance=1e-3):
    """Solve, and judge the result by measuring it rather than by the flag.

    GEMSEO reports a feasibility flag, and on these problems it cannot be
    relied on: the same sail design came back ``False`` at eight hundred
    iterations and ``True`` at two thousand, with an identical design
    vector and terminal misses of order 1e-7. What a figure claims is that
    the trajectory met its constraints, so that is what is checked here,
    reading the problem's own constraint names and kinds.

    Parameters
    ----------
    scenario
        The dynamic scenario, for evaluating the constraints at the answer.
    gemseo_scenario
        The GEMSEO scenario to run.
    max_iter
        The iteration budget.
    warm
        A design to start from.
    tolerance
        How far a constraint may miss, in its own units. The same figure
        serves for both kinds, which matters: an *active* inequality sits
        at zero at the optimum, so demanding it be non-positive to within
        a micron asks a boat to stay inside a 150 m channel to the nearest
        micron. A first attempt did exactly that and rejected a design
        whose terminal misses were 4e-8.

    Returns
    -------
    tuple
        The objective, the design, whether every constraint is met, and
        the outputs at that design.
    """
    objective, design, _ = _solve(gemseo_scenario, max_iter, warm=warm)
    outputs = scenario.problem.jax_function(
        {
            **scenario.problem.input_defaults,
            **design,
        }
    )
    equalities = [
        float(numpy.max(numpy.abs(numpy.asarray(outputs[name]))))
        for name, kind in scenario.problem.constraint_names
        if kind == "eq"
    ]
    inequalities = [
        float(numpy.max(numpy.asarray(outputs[name])))
        for name, kind in scenario.problem.constraint_names
        if kind == "ineq"
    ]
    met = (
        max(equalities, default=0.0) <= tolerance
        and max(inequalities, default=0.0) <= tolerance
    )
    return objective, design, met, outputs


def _trajectory_arrays(trajectory) -> dict:
    """Flatten a trajectory into arrays ready to store.

    Parameters
    ----------
    trajectory
        The trajectory.

    Returns
    -------
    dict
        ``ts`` plus every state, control and output by name.
    """
    arrays = {"ts": numpy.asarray(trajectory.ts)}
    for group in (trajectory.states, trajectory.controls, trajectory.outputs):
        for name, series in group.items():
            arrays[name] = numpy.asarray(series)
    return arrays


# --------------------------------------------------------- brachistochrone


def generate_brachistochrone(smoke: bool = False) -> None:
    """Compute the guided tour's results.

    Parameters
    ----------
    smoke
        Whether to run the tiny version, for the tests.
    """
    from gemseo_tempo.problems.brachistochrone import VARIANTS
    from gemseo_tempo.problems.brachistochrone import create_optimization
    from gemseo_tempo.problems.brachistochrone import cycloid_time
    from gemseo_tempo.problems.brachistochrone import reference_time
    from gemseo_tempo.scenario.trajectories import optimal_trajectory

    print("brachistochrone")
    size = 8 if smoke else 20
    budget = 120 if smoke else 300

    # The three variants, on the same grid, with the analytic cycloid.
    optima = {}
    for variant in VARIANTS:
        scenario, gemseo_scenario = create_optimization(variant=variant, size=size)
        objective, design, feasible = _solve(gemseo_scenario, budget)
        trajectory = optimal_trajectory(scenario, gemseo_scenario)
        _write_arrays(
            f"brachistochrone/variant_{variant}.npz", **_trajectory_arrays(trajectory)
        )
        optima[variant] = {
            "time": objective,
            "reference": float(reference_time(variant)),
            "feasible": feasible,
        }
    _write_json("brachistochrone/variants.json", optima)

    # The cycloid itself, sampled finely, as the answer to reproduce. theta_f
    # solves the same transcendental aspect-ratio equation cycloid_time()
    # does; it is not pi in general (only the target's aspect ratio would
    # make it so), so it is solved for here rather than assumed, the way an
    # earlier version of this generator did, which quietly stored a curve
    # that reached (pi, 2) instead of the actual (2, 2) target.
    import math

    from scipy.optimize import brentq

    from gemseo_tempo.problems.brachistochrone.model import TARGET

    x_target, y_target = TARGET
    ratio = x_target / y_target
    theta_final = brentq(
        lambda theta: (theta - math.sin(theta)) / (1.0 - math.cos(theta)) - ratio,
        1e-6,
        2.0 * math.pi - 1e-6,
    )
    radius = y_target / (1.0 - math.cos(theta_final))
    theta = numpy.linspace(0.0, theta_final, 200)
    _write_arrays(
        "brachistochrone/cycloid.npz",
        x=radius * (theta - numpy.sin(theta)),
        y=radius * (1.0 - numpy.cos(theta)),
        time=numpy.asarray(cycloid_time()),
    )

    # The three transcriptions on the ideal variant, against the cycloid.
    _benchmark_transcriptions(
        "brachistochrone",
        lambda transcription, count: create_optimization(
            transcription=transcription, variant="ideal", size=count
        ),
        (5, 10) if smoke else (5, 10, 20, 40, 80),
        budget,
        reference=float(cycloid_time()),
        indirect={"control_grid": 32},
    )

    _generate_wire_uncertainty(smoke)


# ------------------------------------------------------------------- sail


def _generate_upwind(size: int, budget: int, integrator) -> None:
    """Sail the upwind leg, where the mark cannot be sailed at straight.

    The polar's guess sweeps the heading steadily from one tack to the
    other, which spends much of the leg pointing close to head to wind; a
    single tack at half-way starts nearer the answer. Both are solved and
    the faster of those that meet the constraints is kept.

    Parameters
    ----------
    size
        The discretization.
    budget
        The iteration budget.
    integrator
        The integrator of the single shooting.
    """
    from gemseo_tempo.problems.sail import create_optimization
    from gemseo_tempo.problems.sail import default_guess
    from gemseo_tempo.problems.sail.problem import LEGS
    from gemseo_tempo.scenario.trajectories import optimal_trajectory
    from gemseo_tempo.transcriptions import SingleShooting

    guess = default_guess("upwind")
    tau = numpy.linspace(0.0, 1.0, size + 1)
    one_tack = {
        "control.psi": numpy.where(
            tau < 0.5, guess.psi_level, guess.psi_level + guess.psi_slope
        ),
        "t_f": numpy.array([1.05 * guess.t_f]),
    }
    solves = []
    for start in (None, one_tack):
        scenario, gemseo_scenario = create_optimization(
            transcription=SingleShooting(),
            leg="upwind",
            size=size,
            integrator=integrator,
        )
        objective, _, met, _ = _solve_checked(
            scenario, gemseo_scenario, budget, warm=start
        )
        solves.append((not met, objective, met, scenario, gemseo_scenario))
    _, objective, met, scenario, gemseo_scenario = min(
        solves, key=lambda solve: solve[:2]
    )
    track = optimal_trajectory(scenario, gemseo_scenario)
    _write_arrays("sail/track_upwind.npz", **_trajectory_arrays(track))
    _write_json(
        "sail/upwind.json",
        {
            "optimized": objective,
            "met": met,
            "mark": list(LEGS["upwind"]),
            "across": float(numpy.max(numpy.abs(numpy.asarray(track.states["y"])))),
        },
    )


def generate_sail(smoke: bool = False) -> None:
    """Compute the sailing paper's results.

    Parameters
    ----------
    smoke
        Whether to run the tiny version, for the tests.
    """
    from gemseo_tempo.problems.sail import create_optimization
    from gemseo_tempo.problems.sail.problem import LEGS
    from gemseo_tempo.problems.sail.problem import sail_integrator
    from gemseo_tempo.problems.sail.problem import straight_run_time
    from gemseo_tempo.scenario.trajectories import optimal_trajectory
    from gemseo_tempo.transcriptions import HermiteSimpson
    from gemseo_tempo.transcriptions import SingleShooting

    print("sail")
    size = 10 if smoke else 25
    budget = 200 if smoke else 800

    def integrator(count):
        return sail_integrator(rtol=1e-8, atol=1e-8, n_save=count + 1)

    # The downwind leg, unconstrained: the gybe nobody asked for.
    scenario, gemseo_scenario = create_optimization(
        transcription=SingleShooting(), size=size, integrator=integrator(size)
    )
    _write_formulation("sail/formulation.json", scenario, gemseo_scenario)
    free_time, free_design, _ = _solve(gemseo_scenario, budget)
    trajectory = optimal_trajectory(scenario, gemseo_scenario)
    _write_arrays("sail/track_downwind.npz", **_trajectory_arrays(trajectory))

    # The same leg sailed straight at the mark, which is what it beats.
    _write_json(
        "sail/downwind.json",
        {
            "optimized": free_time,
            "straight": float(straight_run_time("downwind")),
            "mark": list(LEGS["downwind"]),
        },
    )

    _generate_upwind(size, budget, integrator(size))

    # Collocation on the same leg, which finds a different and worse gybe.
    _, collocation_scenario = create_optimization(
        transcription=HermiteSimpson(num_segments=size),
        size=size,
        integrator=integrator(size),
    )
    collocation_time, _, _ = _solve(collocation_scenario, budget)

    # The channel, warm-started from the free optimum, then from the design
    # one step looser. No single starting strategy wins at every width, so
    # each is the better of the two.
    widths = (250.0, 80.0) if smoke else (250.0, 150.0, 80.0)
    drawn = widths[-1] if smoke else 150.0
    channel: list[dict] = []
    previous = free_design
    for width in widths:
        best = None
        for label, start in (("free", free_design), ("chained", previous)):
            inner, inner_gemseo = create_optimization(
                transcription=SingleShooting(),
                size=size,
                integrator=integrator(size),
                channel_half_width=width,
            )
            objective, solution, met, outputs = _solve_checked(
                inner, inner_gemseo, budget, warm=start
            )
            if not met or (best is not None and objective >= best["time"]):
                continue
            best = {
                "half_width": width,
                "time": objective,
                "across": float(
                    numpy.max(numpy.abs(numpy.asarray(outputs["trajectory.y"])))
                ),
                "start": label,
                "solution": solution,
                "trajectory": optimal_trajectory(inner, inner_gemseo),
            }
        if best is None:
            msg = (
                f"no design at half-width {width} m both reached the mark"
                " and stayed inside the channel, from either start; the"
                " channel study cannot be reported."
            )
            raise RuntimeError(msg)
        previous = best.pop("solution")
        trajectory = best.pop("trajectory")
        channel.append(best)
        if width == drawn:
            _write_arrays("sail/track_channel.npz", **_trajectory_arrays(trajectory))
    _write_json(
        "sail/channel.json",
        {
            "unconstrained": {"time": free_time, "across": None},
            "widths": channel,
            "collocation": collocation_time,
        },
    )

    _benchmark_transcriptions(
        "sail",
        lambda transcription, count: create_optimization(
            transcription=transcription, size=count, integrator=integrator(count)
        ),
        (10, 25) if smoke else (10, 25, 50, 100),
        budget,
        indirect={"control_grid": 64},
        indirect_budget=20,
    )
    _write_sail_vmg()
    _generate_sail_uncertainty(size, smoke)


def _write_sail_vmg() -> None:
    """Store the speed-made-good bound the downwind optimum is checked against.

    Everything here is read off the polar, with no optimizer: the angle
    that maximizes the speed made good towards (or away from) the wind, the
    boat speed there, and the time a leg would take if the boat could sail
    that angle for its whole length. The optimized track is then compared
    with it on the page, which is the verification this problem has in
    place of a closed-form solution.
    """
    from gemseo_tempo.problems.sail.polar import REFERENCE_WIND
    from gemseo_tempo.problems.sail.polar import best_vmg_angle
    from gemseo_tempo.problems.sail.polar import boat_speed
    from gemseo_tempo.problems.sail.problem import LEGS
    from gemseo_tempo.problems.sail.problem import straight_run_time

    record = {"wind": float(REFERENCE_WIND)}
    for leg in ("downwind", "upwind"):
        angle = float(best_vmg_angle(leg))
        speed = float(boat_speed(angle, REFERENCE_WIND))
        made_good = speed * abs(float(numpy.cos(numpy.radians(angle))))
        distance = float(numpy.hypot(*LEGS[leg]))
        straight = float(straight_run_time(leg))
        record[leg] = {
            "angle": angle,
            # How far either tack points off the bearing to the mark.
            "offset": angle if leg == "upwind" else 180.0 - angle,
            "speed": speed,
            "made_good": made_good,
            "ideal_time": distance / made_good,
            # Dead upwind the boat does not move, so there is no straight run.
            "straight_time": straight if numpy.isfinite(straight) else None,
        }
    downwind = record["downwind"]
    downwind["straight_speed"] = float(boat_speed(180.0, REFERENCE_WIND))
    downwind["speed_gain"] = downwind["made_good"] / downwind["straight_speed"] - 1.0
    downwind["time_gain"] = 1.0 - downwind["ideal_time"] / downwind["straight_time"]
    track = numpy.load(DATA / "sail" / "track_downwind.npz")
    downwind["optimized_angle"] = float(numpy.median(numpy.abs(track["twa"])))
    downwind["optimized_across"] = float(numpy.max(numpy.abs(track["y"])))
    downwind["gybe_across"] = (
        0.5 * LEGS["downwind"][0] * float(numpy.tan(numpy.radians(downwind["offset"])))
    )
    _write_json("sail/vmg.json", record)


def _generate_sail_uncertainty(size: int, smoke: bool) -> None:
    """Propagate the wind through the downwind leg, then design for it.

    Single shooting, so that a frozen heading schedule responds to a
    different wind; the quantity is how far along the course the boat
    stands from the mark when the leg's time is up.

    Parameters
    ----------
    size
        The discretization.
    smoke
        Whether to run the tiny version.
    """
    from gemseo_tempo.problems.sail import create_optimization
    from gemseo_tempo.problems.sail.model import REGISTRY
    from gemseo_tempo.problems.sail.problem import sail_integrator
    from gemseo_tempo.transcriptions import SingleShooting

    scenario, gemseo_scenario = create_optimization(
        transcription=SingleShooting(),
        size=size,
        integrator=sail_integrator(dt0=1.0 / 300, n_save=size + 1),
    )
    _propagation_study(
        "sail",
        scenario,
        gemseo_scenario,
        REGISTRY,
        "constraint.final.x",
        smoke,
        budget=200 if smoke else 800,
    )
    _generate_sail_margin(size, smoke)


def _generate_sail_margin(size: int, smoke: bool) -> None:
    """Compare a safety margin against carrying the probability.

    Every design is warm-started from the one nearest it, and each width is
    solved from more than one start, because a looser channel cannot have a
    worse optimum than a tighter one and a table that says otherwise is
    reporting the solver rather than the physics.

    Parameters
    ----------
    size
        The discretization.
    smoke
        Whether to run the tiny version.
    """
    from gemseo_tempo.problems.sail import create_optimization
    from gemseo_tempo.problems.sail import create_scenario
    from gemseo_tempo.problems.sail.model import REGISTRY
    from gemseo_tempo.problems.sail.problem import sail_integrator
    from gemseo_tempo.scenario.formulations import OptimalControl
    from gemseo_tempo.transcriptions import SingleShooting
    from gemseo_tempo.uncertainty import make_parameter_space
    from gemseo_tempo.uncertainty import nominal_values
    from gemseo_tempo.uncertainty import uncertain_names
    from gemseo_tempo.uncertainty.robust import SHORTFALL
    from gemseo_tempo.uncertainty.robust import assess
    from gemseo_tempo.uncertainty.robust import robust_discipline

    half_width = 150.0
    target = 0.95
    channel = "constraint.channel"
    budget = 200 if smoke else 800
    names = uncertain_names(REGISTRY)
    space = make_parameter_space(REGISTRY, names)
    design_sample = numpy.asarray(space.compute_samples(32 if smoke else 256))
    check_sample = numpy.asarray(space.compute_samples(256 if smoke else 4000))
    nominal = nominal_values(REGISTRY, names)

    def build(width):
        return create_optimization(
            transcription=SingleShooting(),
            size=size,
            integrator=sail_integrator(dt0=1.0 / 300, n_save=size + 1),
            channel_half_width=width,
        )

    def measure(design, objective, label):
        """Freeze a design into a fresh scenario and see what it achieves."""
        scenario = create_scenario(
            transcription=SingleShooting(),
            size=size,
            integrator=sail_integrator(dt0=1.0 / 300, n_save=size + 1),
            channel_half_width=half_width,
        )
        for name, value in design.items():
            scenario.problem.input_defaults[name] = value
        for name, value in nominal.items():
            scenario.problem.input_defaults[f"constant.{name}"] = numpy.atleast_1d(
                value
            )
        assessment = assess(
            scenario,
            names,
            check_sample,
            channel,
            objective=objective,
            name=label,
            reduction=jnp.max,
        )
        outputs = scenario.problem.jax_function(dict(scenario.problem.input_defaults))
        return {
            "name": label,
            "objective": assessment.objective,
            "reliability": assessment.reliability,
            "quantile": assessment.quantile,
            "worst": assessment.worst,
            "across": float(
                numpy.max(numpy.abs(numpy.asarray(outputs["trajectory.y"])))
            ),
        }

    _, free_gemseo = create_optimization(
        transcription=SingleShooting(),
        size=size,
        integrator=sail_integrator(dt0=1.0 / 300, n_save=size + 1),
    )
    _, free_design, _ = _solve(free_gemseo, budget)

    records = []
    starts = [free_design]
    margins = (0.0, 50.0) if smoke else (0.0, 25.0, 50.0, 75.0)
    # Tightest first, so that every looser margin also starts from the
    # designs of the tighter ones, which it is free to keep: a looser margin
    # cannot have a slower optimum, and solved the other way round, a 50 m
    # margin once came back slower than a 75 m one.
    for margin in sorted(margins, reverse=True):
        best = None
        for start in starts:
            inner, gemseo_scenario = build(half_width - margin)
            objective, design, met, _ = _solve_checked(
                inner, gemseo_scenario, budget, warm=start
            )
            if met and (best is None or objective < best[0]):
                best = (objective, design)
        if best is None:
            msg = (
                f"no design at a {margin:.0f} m margin met its constraints"
                " from any start; the margin study cannot be reported."
            )
            raise RuntimeError(msg)
        label = "deterministic" if margin == 0.0 else f"margin {margin:.0f} m"
        records.insert(0, measure(best[1], best[0], label))
        starts.append(best[1])

    # The probability carried in the constraints rather than padded against.
    best = None
    for start in starts:
        scenario = create_scenario(
            transcription=SingleShooting(),
            size=size,
            integrator=sail_integrator(dt0=1.0 / 300, n_save=size + 1),
            channel_half_width=half_width,
        )
        bounds = {"t_f": (10.0, 4000.0)}
        for name in scenario.problem.design_variables:
            if name.startswith("control.psi"):
                bounds[name] = (-numpy.pi, numpy.pi)
        formulation = OptimalControl(
            scenario,
            bounds=bounds,
            disciplines=[
                robust_discipline(
                    scenario,
                    names,
                    design_sample,
                    channel,
                    reduction=jnp.max,
                    smoothing=8.0,
                    target=target,
                )
            ],
            extra_constraints=[(f"{channel}.shortfall", "ineq")],
        )
        gemseo_scenario = formulation.create_gemseo_scenario()
        compile_seconds = _compile(formulation.disciplines[0], True)
        started = time.perf_counter()
        objective, design, met, _ = _solve_checked(
            scenario, gemseo_scenario, budget, warm=start
        )
        loop = {
            "compile_seconds": compile_seconds,
            "seconds": time.perf_counter() - started,
            "calls": len(gemseo_scenario.formulation.optimization_problem.database),
        }
        if met and (best is None or objective < best[0]):
            best = (objective, design, loop, scenario)
    if best is not None:
        records.append(
            {
                **measure(best[1], best[0], f"P(inside) >= {target:.0%}"),
                **best[2],
            }
        )
        # What one gradient of the reliability costs, at the design found.
        scenario = best[3]
        scenario.problem.input_defaults.update(best[1])
        function, point = _statistic_of_design(
            scenario,
            names,
            design_sample,
            channel,
            SHORTFALL,
            reduction=jnp.max,
            smoothing=8.0,
            target=target,
        )
        # The sail integrates through the Direct adjoint only.
        cost = _gradient_cost({"Direct": function}, point)
    else:
        cost = None

    _write_json(
        "sail/margin.json",
        {
            "half_width": half_width,
            "target": target,
            "design_sample": len(design_sample),
            "designs": records,
            "gradient_cost": cost,
        },
    )


def _estimate_record(estimate, comparison) -> dict:
    """Flatten one estimate and its error into a storable record.

    Parameters
    ----------
    estimate
        The estimate.
    comparison
        The comparison it belongs to, for the errors.

    Returns
    -------
    dict
        The record.
    """
    mean_error, deviation_error = comparison.relative_error(estimate)
    return {
        "name": estimate.name,
        "mean": estimate.mean,
        "standard_deviation": estimate.standard_deviation,
        "n_evaluations": estimate.n_evaluations,
        "n_gradients": estimate.n_gradients,
        "cost": estimate.cost,
        "seconds": estimate.seconds,
        "mean_error": mean_error,
        "deviation_error": deviation_error,
    }


def _propagation_study(
    name, scenario, gemseo_scenario, registry, quantity, smoke, budget=400, warm=None
) -> dict:
    """Propagate the uncertainty through an optimum, by every estimator.

    The design is frozen at the optimum and the uncertain constants are
    drawn from the registry's distributions. Each estimator is then scored
    against a large sample, and the ones that need derivatives are run
    twice, with the exact derivatives automatic differentiation provides
    and with central finite differences, which is the comparison that shows
    what the derivatives are worth.

    Parameters
    ----------
    name
        The problem, which names the stored file.
    scenario
        The dynamic scenario, single-shooting so the trajectory responds.
    gemseo_scenario
        Its GEMSEO scenario, solved here at the nominal constants.
    registry
        The variable metadata carrying the distributions.
    quantity
        The scalar output propagated.
    smoke
        Whether to run the tiny version, for the tests.
    budget
        The iteration budget of the nominal solve.
    warm
        A design to start the nominal solve from.

    Returns
    -------
    dict
        What is stored. Each estimate carries its cost in evaluations of
        the model, counting a reverse pass as one and a finite-difference
        gradient as the two evaluations per uncertain variable it takes.
    """
    from gemseo_tempo.uncertainty import freeze_design
    from gemseo_tempo.uncertainty import looped_evaluator
    from gemseo_tempo.uncertainty import make_parameter_space
    from gemseo_tempo.uncertainty import measure_speedup
    from gemseo_tempo.uncertainty import nominal_values
    from gemseo_tempo.uncertainty import uncertain_names
    from gemseo_tempo.uncertainty.estimators import Comparison
    from gemseo_tempo.uncertainty.estimators import finite_differences
    from gemseo_tempo.uncertainty.estimators import from_scenario
    from gemseo_tempo.uncertainty.estimators import monte_carlo
    from gemseo_tempo.uncertainty.estimators import polynomial_chaos
    from gemseo_tempo.uncertainty.estimators import taylor

    names = uncertain_names(registry)
    objective, _, feasible, _ = _solve_checked(
        scenario, gemseo_scenario, 60 if smoke else budget, warm=warm
    )
    freeze_design(scenario, gemseo_scenario, nominal_values(registry, names))
    space = make_parameter_space(registry, names)
    sample = numpy.asarray(space.compute_samples(32 if smoke else 256))
    speedup = measure_speedup(scenario, names, [quantity], sample)

    exact = from_scenario(scenario, registry, names, quantity)
    fd_step = _propagation_step(exact)
    differenced = finite_differences(exact, relative_step=fd_step)
    looped = looped_evaluator(scenario, names, [quantity])
    size = 64 if smoke else 256
    reference = monte_carlo(
        exact, n_samples=256 if smoke else 20000, seed=1, name="reference"
    )
    # Named by slugs, which is how the pages refer to them.
    estimates = [
        monte_carlo(exact, n_samples=size, name="sampling_vmapped"),
        monte_carlo(
            exact,
            n_samples=size,
            name="sampling_looped",
            evaluate=lambda draws: numpy.asarray(looped(draws)).ravel(),
        ),
    ]
    for label, propagation in (("ad", exact), ("fd", differenced)):
        for order, second in ((1, False), (2, True)):
            estimate = taylor(propagation, second_order=second)
            estimate.name = f"taylor{order}_{label}"
            estimates.append(estimate)
    for label, propagation, enhanced in (
        ("chaos", exact, False),
        ("chaos_ad", exact, True),
        ("chaos_fd", differenced, True),
    ):
        estimate = polynomial_chaos(
            propagation, n_training=12, degree=2, gradient_enhanced=enhanced
        )
        estimate.name = label
        estimates.append(estimate)
    comparison = Comparison(reference=reference, estimates=tuple(estimates))
    records = {}
    for estimate in estimates:
        record = _estimate_record(estimate, comparison)
        per_gradient = 2 * len(names) if estimate.name.endswith("_fd") else 1
        record["model_calls"] = (
            estimate.n_evaluations + per_gradient * estimate.n_gradients
        )
        records[estimate.name] = record
    study = {
        "quantity": quantity,
        "names": list(names),
        "objective": objective,
        "feasible": feasible,
        "fd_step": fd_step,
        "speedup": {
            "n_samples": speedup.n_samples,
            "looped_seconds": speedup.looped_seconds,
            "batched_seconds": speedup.batched_seconds,
            "speedup": speedup.speedup,
        },
        "reference": _estimate_record(reference, comparison),
        "estimates": records,
    }
    _write_json(f"{name}/propagation.json", study)
    return study


def _propagation_step(propagation, steps=(1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)):
    """Return the finite-difference step closest to the exact derivative.

    Measured at the mean, against the derivative automatic differentiation
    gives there, so that the differenced estimators are compared at their
    best rather than at an arbitrary step.

    Parameters
    ----------
    propagation
        The propagation with exact derivatives.
    steps
        The steps tried, as fractions of each standard deviation.

    Returns
    -------
    float
        The step whose central differences come closest.
    """
    from gemseo_tempo.uncertainty.estimators import finite_differences

    mean = propagation.mean_point
    exact = numpy.asarray(propagation.jacobian(mean))
    scale = max(float(numpy.abs(exact).max()), numpy.finfo(float).tiny)

    def error(step):
        differenced = finite_differences(propagation, relative_step=step)
        return float(numpy.abs(differenced.jacobian(mean) - exact).max()) / scale

    return min(steps, key=error)


def _statistic_of_design(scenario, names, sample, quantity, statistic, **settings):
    """Return one statistic of a quantity as a function of a flat design.

    Parameters
    ----------
    scenario
        The dynamic scenario, whose design variables make the vector.
    names
        The uncertain names.
    sample
        The fixed sample the statistic is taken over.
    quantity
        The output.
    statistic
        Which statistic, one of the names
        :func:`~gemseo_tempo.uncertainty.robust.statistics_function`
        returns.
    **settings
        Forwarded to that function.

    Returns
    -------
    tuple
        The function of the flat design vector, and the current design.
    """
    from gemseo_tempo.uncertainty.robust import statistics_function

    statistics = statistics_function(scenario, names, sample, quantity, **settings)
    defaults = scenario.problem.input_defaults
    design = list(scenario.problem.design_variables)
    shapes = {key: numpy.shape(defaults[key]) for key in design}
    sizes = [int(numpy.prod(shapes[key])) or 1 for key in design]
    point = jnp.concatenate([jnp.ravel(jnp.asarray(defaults[key])) for key in design])

    def function(flat):
        values, start = {}, 0
        for key, count in zip(design, sizes, strict=True):
            values[key] = jnp.reshape(flat[start : start + count], shapes[key])
            start += count
        return jnp.reshape(statistics(values)[statistic], ())

    return function, point


def _gradient_cost(functions, point, repeats: int = 5) -> dict:
    """Time one gradient by AD and by finite differences, each at its best.

    Measured on the bare JAX function, compiled first, since GEMSEO caches
    a linearization until its inputs change and timing it twice at one
    point would time the cache. Automatic differentiation is timed in
    reverse mode, one pass for one output, through every adjoint given,
    the fastest counting; all of them are exact for a scalar. Forward mode
    is left out: it takes one pass per input, and on the albatross's
    margin, eighty-two inputs through sixty-four draws, it had not
    finished compiling and running once in twenty-five minutes.
    Finite differences take the step that comes closest to that exact
    gradient and are timed batched into one vectorized call and one
    evaluation at a time, the faster counting.

    Parameters
    ----------
    functions
        The scalar function of a flat vector, by the adjoint its
        integration is differentiated through.
    point
        Where to differentiate it.
    repeats
        How many times each is timed; the fastest counts.

    Returns
    -------
    dict
        The design size, the seconds of one evaluation and of one gradient
        by each route, and the configuration each route was fastest in.
    """
    import jax

    from gemseo_tempo.benchmarking.differentiation import _differences
    from gemseo_tempo.benchmarking.differentiation import _fastest
    from gemseo_tempo.benchmarking.differentiation import batched_differences
    from gemseo_tempo.benchmarking.differentiation import best_step

    routes = {
        f"reverse, {adjoint}": jax.jit(jax.grad(function))
        for adjoint, function in functions.items()
    }
    timings = {}
    for name, route in routes.items():
        route(point).block_until_ready()
        timings[name] = _fastest(lambda route=route: route(point), repeats)
    ad = min(timings, key=timings.get)
    exact = numpy.asarray(routes[ad](point))[None, :]

    function = functions.get("Direct", next(iter(functions.values())))
    value = jax.jit(function)
    batched = jax.jit(batched_differences(lambda x: jnp.atleast_1d(function(x))))
    step, _ = best_step(batched, point, exact)
    value(point).block_until_ready()
    batched(point, step).block_until_ready()
    fd = {
        "batched": _fastest(lambda: batched(point, step), repeats),
        "sequential": _fastest(
            lambda: _differences(lambda x: jnp.atleast_1d(value(x)), point, step),
            repeats,
        ),
    }
    variant = min(fd, key=fd.get)
    return {
        "n_design": int(point.size),
        "evaluation_seconds": _fastest(lambda: value(point), repeats),
        "ad_seconds": timings[ad],
        "ad_route": ad,
        "fd_seconds": fd[variant],
        "fd_variant": variant,
        "fd_step": step,
    }


def _compile(discipline, gradient: bool) -> float:
    """Compile a JAX discipline before an optimization is timed.

    Compilation happens once, on the first call, and is a fixed cost that
    has nothing to do with how many iterations follow; timed inside the
    loop it would be charged to whichever route happened to call first.

    Parameters
    ----------
    discipline
        The discipline.
    gradient
        Whether its Jacobian will be used, and so compiled too.

    Returns
    -------
    float
        The seconds compilation took.
    """
    started = time.perf_counter()
    discipline.execute()
    if gradient:
        discipline.linearize()
    return time.perf_counter() - started


def _spread(scenario, names, sample, quantity, design) -> dict:
    """Return what a frozen design does over a sample of the uncertainty.

    Parameters
    ----------
    scenario
        A scenario of the problem, at the nominal constants.
    names
        The uncertain names.
    sample
        The sample, independent of the one any design was optimized on.
    quantity
        The output.
    design
        The design to freeze.

    Returns
    -------
    dict
        The nominal value, and the mean, deviation and 5th and 95th
        percentiles over the sample.
    """
    from gemseo_tempo.uncertainty import batch_evaluator

    defaults = scenario.problem.input_defaults
    defaults.update({key: numpy.atleast_1d(value) for key, value in design.items()})
    values = numpy.asarray(batch_evaluator(scenario, names, [quantity])(sample)).ravel()
    nominal = scenario.problem.jax_function(dict(defaults))[quantity]
    return {
        "nominal": float(numpy.asarray(nominal).ravel()[0]),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "p05": float(numpy.quantile(values, 0.05)),
        "p95": float(numpy.quantile(values, 0.95)),
    }


def _samples(registry, names, sizes, seed: int = 0) -> list:
    """Draw reproducible samples of the uncertain inputs.

    Parameters
    ----------
    registry
        The variable metadata carrying the distributions.
    names
        The uncertain names.
    sizes
        The size of each sample.
    seed
        The seed of the first.

    Returns
    -------
    list
        One array per size, each from its own seed.
    """
    import openturns

    from gemseo_tempo.uncertainty import make_parameter_space

    space = make_parameter_space(registry, names)
    samples = []
    for offset, size in enumerate(sizes):
        openturns.RandomGenerator.SetSeed(seed + offset)
        samples.append(numpy.asarray(space.compute_samples(size)))
    return samples


def _generate_wire_uncertainty(smoke: bool) -> None:
    """Propagate the uncertainty through the optimal wire, then design for it.

    Friction, drag and gravity are uncertain; the wire, once built, is not.
    Three wires are compared on a sample none of them was designed on: the
    cycloid, designed for the ideal bead; the wire designed for the mean
    physics; and the wire minimizing the mean plus twice the deviation of
    the descent time over a fixed sample. The last is solved twice from the
    same start, each derivative at its best: exact, in the cheaper mode of
    each output group, through the checkpointed adjoint, which is exact on
    this problem and the fastest measured on it; and by forward differences
    batched into one vectorized call, at the step that comes closest to the
    exact gradient there. The cost of one gradient of that objective is
    measured as the wire is refined.

    Parameters
    ----------
    smoke
        Whether to run the tiny version, for the tests.
    """
    from gemseo_tempo.dynamics.integrators import DiffraxIntegrator
    from gemseo_tempo.problems.brachistochrone import REGISTRY
    from gemseo_tempo.problems.brachistochrone import create_wire_optimization
    from gemseo_tempo.problems.brachistochrone import create_wire_scenario
    from gemseo_tempo.problems.brachistochrone.wire import DESCENT_TIME
    from gemseo_tempo.problems.brachistochrone.wire import wire_bounds
    from gemseo_tempo.scenario.formulations import OptimalControl
    from gemseo_tempo.uncertainty import nominal_values
    from gemseo_tempo.uncertainty import uncertain_names
    from gemseo_tempo.uncertainty.robust import MARGIN
    from gemseo_tempo.uncertainty.robust import robust_discipline

    size = 10 if smoke else 20
    budget = 100 if smoke else 400
    factor = 2.0
    names = uncertain_names(REGISTRY)
    nominal = nominal_values(REGISTRY, names)
    design_sample, check_sample = _samples(
        REGISTRY, names, (16, 256) if smoke else (64, 4000)
    )

    scenario, gemseo_scenario = create_wire_optimization(size=size, constants=nominal)
    _propagation_study(
        "brachistochrone", scenario, gemseo_scenario, REGISTRY, DESCENT_TIME, smoke
    )

    def spread(design):
        return _spread(
            create_wire_scenario(size=size, constants=nominal),
            names,
            check_sample,
            DESCENT_TIME,
            design,
        )

    designs = {}
    for label, constants in (("ideal", None), ("nominal", nominal)):
        _, gemseo_scenario = create_wire_optimization(size=size, constants=constants)
        started = time.perf_counter()
        _, design, met = _solve(gemseo_scenario, budget)
        designs[label] = {
            "met": met,
            "seconds": time.perf_counter() - started,
            "calls": len(gemseo_scenario.formulation.optimization_problem.database),
            **spread(design),
        }
    costs = []
    for count in (10, 20) if smoke else (10, 20, 40, 80):
        functions = {}
        for adjoint in ("Direct", "RecursiveCheckpoint"):
            functions[adjoint], point = _statistic_of_design(
                create_wire_scenario(
                    size=count,
                    constants=nominal,
                    integrator=DiffraxIntegrator(
                        dt0=1.0 / 400, n_save=count + 1, adjoint=adjoint
                    ),
                ),
                names,
                design_sample,
                DESCENT_TIME,
                MARGIN,
                factor=factor,
            )
        costs.append({"size": count, **_gradient_cost(functions, point)})
    # Measured at the start of the robust solve, on its own discretization.
    fd_step = next(cost["fd_step"] for cost in costs if cost["size"] == size)

    for label, modes in (("robust", ("auto", "auto")), ("robust_fd", ("fd", "fd"))):
        scenario = create_wire_scenario(size=size, constants=nominal)
        formulation = OptimalControl(
            scenario,
            bounds=wire_bounds(scenario),
            disciplines=[
                robust_discipline(
                    scenario,
                    names,
                    design_sample,
                    DESCENT_TIME,
                    factor=factor,
                    modes=modes,
                    fd_step=fd_step,
                )
            ],
            objective=f"{DESCENT_TIME}.{MARGIN}",
        )
        gemseo_scenario = formulation.create_gemseo_scenario()
        compile_seconds = _compile(formulation.disciplines[0], True)
        started = time.perf_counter()
        objective, design, met, _ = _solve_checked(scenario, gemseo_scenario, budget)
        designs[label] = {
            "objective": objective,
            "met": met,
            "modes": list(modes),
            "compile_seconds": compile_seconds,
            "seconds": time.perf_counter() - started,
            "calls": len(gemseo_scenario.formulation.optimization_problem.database),
            **spread(design),
        }
    _write_json(
        "brachistochrone/robust.json",
        {
            "factor": factor,
            "design_sample": len(design_sample),
            "check_sample": len(check_sample),
            "fd_step": fd_step,
            "designs": designs,
            "gradient_cost": costs,
        },
    )


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


# -------------------------------------------------------- differentiation


def _measure_configuration(build, adjoints, jit, repeats):
    """Measure the three modes on one configuration of one problem.

    Parameters
    ----------
    build
        ``adjoint -> scenario``, the problem at this configuration.
    adjoints
        The adjoint each mode differentiates the integration through.
    jit
        Whether to compile.
    repeats
        How many times each Jacobian is timed.

    Returns
    -------
    list
        One record per mode.
    """
    from gemseo_tempo.benchmarking.differentiation import as_records
    from gemseo_tempo.benchmarking.differentiation import flat_function
    from gemseo_tempo.benchmarking.differentiation import measure_modes

    if not jit:
        # Without compiling, the checkpointed variant only repeats what
        # reverse mode shows, at the cost the sweep can least afford.
        adjoints = {
            mode: a for mode, a in adjoints.items() if mode != "reverse_checkpointed"
        }
    problems = {adjoint: build(adjoint).problem for adjoint in set(adjoints.values())}
    functions, point = {}, None
    for mode, adjoint in adjoints.items():
        problem = problems[adjoint]
        inputs = list(problem.design_variables)
        outputs = [problem.objective_name] + [
            name for name, _ in problem.constraint_names
        ]
        functions[mode], point = flat_function(problem, inputs, outputs)
    # A generic point, a thousandth away from the guess: the sail's guess
    # heads exactly along a breakpoint of its polar table, where a one-sided
    # difference straddles a kink and an exact derivative picks one side.
    offset = numpy.random.default_rng(0).standard_normal(point.size)
    point = point * (1.0 + 1e-3 * offset)
    return as_records(measure_modes(functions, point, jit=jit, repeats=repeats))


def _differentiation_study(name, builder, adjoints, sweeps, eager, smoke) -> dict:
    """Sweep the inputs and the outputs, by every mode, compiled and not.

    Parameters
    ----------
    name
        The problem, which names the stored file.
    builder
        ``(size, n_save, adjoint) -> scenario``: the control intervals set
        the inputs, the save points the outputs of any path constraint.
    adjoints
        The adjoint each mode differentiates the integration through.
    sweeps
        ``{"inputs": (sizes, n_save), "outputs": (size, saves)}``; either
        may be left out.
    eager
        The largest size and save count also measured without compiling,
        whose cost grows quickly, as ``(size, n_save)``.
    smoke
        Whether to run the tiny version, for the tests.

    Returns
    -------
    dict
        What is stored.
    """
    repeats = 1 if smoke else 3
    study = {}
    if "inputs" in sweeps:
        sizes, n_save = sweeps["inputs"]
        records = []
        for size in sizes[:2] if smoke else sizes:
            for jit in (True, False) if size <= eager[0] else (True,):
                records += _measure_configuration(
                    lambda adjoint, size=size: builder(size, n_save, adjoint),
                    adjoints,
                    jit,
                    repeats,
                )
        study["inputs"] = records
    if "outputs" in sweeps:
        size, saves = sweeps["outputs"]
        records = []
        for n_save in saves[:2] if smoke else saves:
            for jit in (True, False) if n_save <= eager[1] else (True,):
                records += _measure_configuration(
                    lambda adjoint, n_save=n_save: builder(size, n_save, adjoint),
                    adjoints,
                    jit,
                    repeats,
                )
        study["outputs"] = records
    _write_json(f"{name}/differentiation.json", study)
    return study


def _chain_rule(products, seeds, data) -> dict:
    """Carry total derivatives through a chain of stage Jacobians.

    Parameters
    ----------
    products
        ``(jacobian, wrt, of)`` per stage, in execution order: the Jacobian
        of the concatenated ``of`` outputs with respect to the concatenated
        ``wrt`` inputs.
    seeds
        The total derivative of each design variable, its identity block.
    data
        The values, which give each output's size.

    Returns
    -------
    dict
        The total derivative of every quantity, by name.
    """
    derivatives = dict(seeds)
    for jacobian, wrt, of in products:
        rows = jacobian @ numpy.vstack([derivatives[name] for name in wrt])
        start = 0
        for name in of:
            count = int(numpy.size(data[name]))
            derivatives[name] = rows[start : start + count]
            start += count
    return derivatives


def _chain_modes_study(smoke: bool) -> dict:
    """Differentiate the albatross's design chain with a mode per stage.

    The split assembly makes three stages of the energy cycle: the statics,
    which turn the wing into the aerodynamic constants; the dynamics, which
    fly the cycle; and the outputs, which read its energy and constraints.
    Each stage is differentiated in its own mode, every combination of the
    eight, on the bare JAX function of the stage, and the chain Jacobian is
    their product. Finite differences of the whole chain stand alongside,
    at their best configuration.

    Parameters
    ----------
    smoke
        Whether to run the tiny version, for the tests.

    Returns
    -------
    dict
        What is stored.
    """
    import jax

    from gemseo_tempo.benchmarking.differentiation import FD
    from gemseo_tempo.benchmarking.differentiation import REVERSE
    from gemseo_tempo.benchmarking.differentiation import _fastest
    from gemseo_tempo.benchmarking.differentiation import _measure_differences
    from gemseo_tempo.benchmarking.differentiation import _memory
    from gemseo_tempo.dynamics.integrators import DiffraxIntegrator
    from gemseo_tempo.problems.albatross import create_energy_scenario
    from gemseo_tempo.problems.albatross.problem import WING_BOUNDS
    from gemseo_tempo.transcriptions import SingleShooting

    size, n_save = (5, 11) if smoke else (20, 21)
    repeats = 1 if smoke else 3
    wing = ["constant." + key for key in WING_BOUNDS]
    modes = ("forward", "reverse")

    def flatten(values, names):
        return jnp.concatenate([jnp.ravel(values[name]) for name in names])

    def stage(discipline, wrt, of, data):
        """Return the stage as a function of one vector, and its point."""
        shapes = {name: jnp.shape(data[name]) for name in wrt}

        def function(flat):
            values, start = dict(data), 0
            for name in wrt:
                count = int(numpy.prod(shapes[name])) or 1
                values[name] = jnp.reshape(flat[start : start + count], shapes[name])
                start += count
            return flatten(discipline.jax_out_func(values), of)

        return function, flatten(data, wrt)

    records, reference = [], None
    for combination in itertools.product(modes, repeat=3):
        methods = dict(
            zip(("statics", "dynamics", "outputs"), combination, strict=True)
        )
        # The Direct adjoint whatever the mode, so that only the mode varies:
        # it is exact both ways, where the checkpointing one is not always.
        scenario = create_energy_scenario(
            transcription=SingleShooting(),
            size=size,
            w_ref=22.0,
            integrator=DiffraxIntegrator(
                dt0=1.0 / 400, n_save=n_save, adjoint="Direct"
            ),
            assembly="split",
            differentiation_method=methods,
        )
        problem = scenario.problem
        disciplines = dict(
            zip(("statics", "dynamics", "outputs"), scenario.disciplines, strict=True)
        )
        data = dict(problem.input_defaults)
        for discipline in disciplines.values():
            inputs = {
                name: data.get(name, discipline.default_input_data.get(name))
                for name in discipline.input_grammar.names
            }
            data.update(discipline.jax_out_func(inputs))
        controls = [
            name for name in problem.design_variables if name.startswith("control.")
        ]
        design = controls + wing
        final = [problem.objective_name] + [n for n, _ in problem.constraint_names]
        consumers = {
            "statics": set(disciplines["dynamics"].input_grammar.names)
            | set(disciplines["outputs"].input_grammar.names),
            "dynamics": set(disciplines["outputs"].input_grammar.names),
            "outputs": set(final),
        }
        # The total derivative of every quantity with respect to the design,
        # seeded with the identity on the design itself.
        sizes = {name: int(numpy.size(data[name])) for name in design}
        identity = numpy.eye(sum(sizes.values()))
        seeds, start = {}, 0
        for name in design:
            seeds[name] = identity[start : start + sizes[name]]
            start += sizes[name]
        record = {"modes": methods}
        products, known = [], set(design)
        for label, discipline in disciplines.items():
            wrt = [n for n in discipline.input_grammar.names if n in known]
            of = [n for n in discipline.output_grammar.names if n in consumers[label]]
            known.update(of)
            function, point = stage(discipline, wrt, of, data)
            transform = jax.jacfwd if methods[label] == "forward" else jax.jacrev
            started = time.perf_counter()
            compiled = jax.jit(transform(function)).lower(point).compile()
            record[f"{label}_compile"] = time.perf_counter() - started
            record[f"{label}_seconds"] = _fastest(
                lambda c=compiled, p=point: c(p), repeats
            )
            record[f"{label}_memory"] = _memory(compiled)
            jacobian = numpy.atleast_2d(numpy.asarray(compiled(point)))
            record[f"{label}_shape"] = list(jacobian.shape)
            products.append((jacobian, wrt, of))
        total = _chain_rule(products, seeds, data)
        chain_jacobian = numpy.vstack([total[name] for name in final])
        record["product_seconds"] = _fastest(
            lambda products=products, seeds=seeds, data=data: _chain_rule(
                products, seeds, data
            ),
            repeats,
        )
        record["seconds"] = record["product_seconds"] + sum(
            record[f"{label}_seconds"] for label in disciplines
        )
        record["memory"] = max(record[f"{label}_memory"] for label in disciplines)
        if reference is None:
            # The whole chain as one function of the design, differenced, and
            # differentiated in reverse through the Direct adjoint: the
            # reference the products are checked against.
            def whole(flat, problem=problem, design=design, final=final):
                values = dict(problem.input_defaults)
                start = 0
                for name in design:
                    count = int(numpy.size(values[name]))
                    values[name] = jnp.reshape(
                        flat[start : start + count], jnp.shape(values[name])
                    )
                    start += count
                return flatten(problem.jax_function(values), final)

            point = flatten(problem.input_defaults, design)
            reference = numpy.asarray(jax.jit(jax.jacrev(whole))(point))
            # At their best: batched or one evaluation at a time, whichever
            # is faster, at the step closest to the exact Jacobian.
            jacobians = {REVERSE: reference}
            measured = _measure_differences(
                whole, point, True, repeats, 1e-6, jacobians
            )
            scale = float(numpy.max(numpy.abs(reference)))
            fd = {
                "modes": {"chain": "fd"},
                "seconds": measured["seconds"],
                "memory": measured["memory"],
                "step": measured["step"],
                "variant": measured["variant"],
                "error": float(numpy.max(numpy.abs(jacobians[FD] - reference))) / scale,
            }
        scale = float(numpy.max(numpy.abs(reference)))
        record["error"] = (
            float(numpy.max(numpy.abs(chain_jacobian - reference))) / scale
        )
        records.append(record)
    study = {"size": size, "n_save": n_save, "chains": records, "fd": fd}
    _write_json("albatross/chain_modes.json", study)
    return study


def _generate_differentiation(name: str, smoke: bool) -> None:
    """Run one problem's differentiation studies.

    Parameters
    ----------
    name
        The problem.
    smoke
        Whether to run the tiny version, for the tests.
    """
    from gemseo_tempo.dynamics.integrators import DiffraxIntegrator
    from gemseo_tempo.transcriptions import SingleShooting

    # Every mode through the Direct adjoint, which is exact in both
    # directions, and reverse mode once more through the checkpointing one,
    # the library's default, whose saving is memory and whose risk is
    # precision.
    checkpoint = dict.fromkeys(("fd", "forward", "reverse"), "Direct")
    checkpoint["reverse_checkpointed"] = "RecursiveCheckpoint"
    if name == "brachistochrone":
        from gemseo_tempo.problems.brachistochrone import create_scenario

        def builder(size, n_save, adjoint):
            return create_scenario(
                transcription=SingleShooting(),
                variant="ideal",
                size=size,
                integrator=DiffraxIntegrator(
                    dt0=1.0 / 400, n_save=n_save, adjoint=adjoint
                ),
            )

        _differentiation_study(
            name,
            builder,
            checkpoint,
            {"inputs": ((5, 10, 20, 40, 80, 160), 21)},
            (80, 0),
            smoke,
        )
    elif name == "sail":
        from gemseo_tempo.problems.sail import create_scenario
        from gemseo_tempo.problems.sail.problem import sail_integrator

        def builder(size, n_save, adjoint):
            return create_scenario(
                transcription=SingleShooting(),
                size=size,
                integrator=sail_integrator(dt0=1.0 / 300, n_save=n_save),
            )

        # The sail's derivatives go through the Direct adjoint either way.
        _differentiation_study(
            name,
            builder,
            dict.fromkeys(("fd", "forward", "reverse"), "Direct"),
            {
                "inputs": ((5, 10, 25, 50, 100, 200), 26),
                "outputs": (25, (11, 41, 161, 641)),
            },
            (50, 161),
            smoke,
        )
    else:
        from gemseo_tempo.problems.albatross import create_energy_scenario

        def builder(size, n_save, adjoint):
            return create_energy_scenario(
                transcription=SingleShooting(),
                size=size,
                w_ref=22.0,
                integrator=DiffraxIntegrator(
                    dt0=1.0 / 400, n_save=n_save, adjoint=adjoint
                ),
            )

        _differentiation_study(
            name,
            builder,
            checkpoint,
            {
                "inputs": ((5, 10, 20, 40, 80), 21),
                "outputs": (20, (11, 41, 161, 641)),
            },
            (20, 41),
            smoke,
        )
        _chain_modes_study(smoke)


def main(argv=None) -> int:
    """Generate the stored results.

    Parameters
    ----------
    argv
        The command-line arguments.

    Returns
    -------
    int
        Zero.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=PROBLEMS, default=list(PROBLEMS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)

    global DATA  # noqa: PLW0603
    if arguments.output is not None:
        DATA = arguments.output

    generators = {
        "brachistochrone": generate_brachistochrone,
        "sail": generate_sail,
        "albatross": generate_albatross,
    }
    for problem in arguments.only:
        started = time.perf_counter()
        generators[problem](smoke=arguments.smoke)
        print(f"  ({time.perf_counter() - started:.0f} s)")
    return 0


# -------------------------------------------------------------- albatross


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


def _benchmark_transcriptions(
    name,
    build,
    sizes,
    budget,
    reference=None,
    indirect=None,
    indirect_budget=None,
    best=min,
) -> dict:
    """Solve one problem by each transcription, with SLSQP, and time it.

    The direct transcriptions are solved over a range of grids from the
    problem's default guess. The indirect one is warm-started from the
    finest collocation solution by :func:`indirect_from_direct`, since
    that is the only way an indirect method is ever started in practice,
    and its time excludes that of the solution it starts from.

    Parameters
    ----------
    name
        The problem, which names the stored file.
    build
        ``(transcription, size) -> (scenario, gemseo_scenario)``.
    sizes
        The grids of the direct transcriptions.
    budget
        The iteration budget of every solve.
    reference
        The answer the errors are measured against; the best one any
        converged solve reached when omitted.
    indirect
        The settings of the indirect shooting, or ``None`` to skip it.
    indirect_budget
        The iteration budget of the indirect solve; ``budget`` when
        omitted. Where the shooting cannot converge, a small one records
        how far it gets without spending an hour finding out.
    best
        Which converged answer is the best: ``min`` for a minimization.

    Returns
    -------
    dict
        The reference and one row per solve.
    """
    from gemseo_tempo.scenario import OptimalControl
    from gemseo_tempo.transcriptions import HermiteSimpson
    from gemseo_tempo.transcriptions import SingleShooting
    from gemseo_tempo.transcriptions import indirect_from_direct

    rows, finest = [], None
    for label in ("shooting", "collocation"):
        for size in sizes:
            transcription = (
                SingleShooting()
                if label == "shooting"
                else HermiteSimpson(num_segments=size)
            )
            scenario, gemseo_scenario = build(transcription, size)
            started = time.perf_counter()
            objective, _, met, _ = _solve_checked(scenario, gemseo_scenario, budget)
            rows.append(
                {
                    "transcription": label,
                    "size": size,
                    "objective": objective,
                    "met": met,
                    "seconds": time.perf_counter() - started,
                    "calls": len(
                        gemseo_scenario.formulation.optimization_problem.database
                    ),
                }
            )
            if label == "collocation":
                finest = (scenario, gemseo_scenario, objective)
    if indirect is not None:
        scenario, gemseo_scenario, _ = finest
        started = time.perf_counter()
        shooting = indirect_from_direct(scenario, gemseo_scenario, **indirect)
        seed = shooting.problem.jax_function(shooting.problem.input_defaults)
        solved = OptimalControl(shooting).create_gemseo_scenario()
        solved.execute(algo_name="SLSQP", max_iter=indirect_budget or budget)
        optimum = solved.optimization_result.x_opt_as_dict
        outputs = shooting.problem.jax_function(
            {
                **shooting.problem.input_defaults,
                **{key: numpy.atleast_1d(value) for key, value in optimum.items()},
            }
        )
        residual = float(numpy.max(numpy.abs(numpy.asarray(outputs["residual"]))))
        rows.append(
            {
                "transcription": "indirect",
                "size": None,
                "objective": float(outputs["ocp_cost"][0]),
                "met": bool(residual < 1e-6),
                "residual": residual,
                "seed_residual": float(
                    numpy.max(numpy.abs(numpy.asarray(seed["residual"])))
                ),
                "unknowns": int(
                    sum(
                        spec.size for spec in shooting.problem.design_variables.values()
                    )
                ),
                "seconds": time.perf_counter() - started,
                "calls": len(solved.formulation.optimization_problem.database),
            }
        )
    if reference is None:
        # Without a closed form, the best answer any method reached, since
        # these problems have local optima and the finest grid need not be
        # the best one.
        reference = best(row["objective"] for row in rows if row["met"])
    for row in rows:
        row["error"] = abs(row["objective"] - reference) / abs(reference)
    study = {"reference": reference, "rows": rows}
    _write_json(f"{name}/transcriptions.json", study)
    return study


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


def _write_dense_cycle(name, builder, size, nodes, gemseo_scenario) -> dict:
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
    twin = builder(
        transcription=SingleShooting(),
        size=size,
        integrator=DiffraxIntegrator(rtol=1e-10, atol=1e-10, n_save=DENSE_POINTS),
    )
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


if __name__ == "__main__":
    raise SystemExit(main())
