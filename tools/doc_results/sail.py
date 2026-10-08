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
"""The sail's results: the legs, the channel, the margin."""

from __future__ import annotations

import time

import jax.numpy as jnp
import numpy

from .common import _benchmark_transcriptions
from .common import _compile
from .common import _solve
from .common import _solve_checked
from .common import _trajectory_arrays
from .common import _write_arrays
from .common import _write_formulation
from .common import _write_json
from .common import data_path
from .propagation import _gradient_cost
from .propagation import _propagation_study
from .propagation import _statistic_of_design


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
        best["design"] = previous
        best["trajectory"] = trajectory
        channel.append(best)

    # A second pass, from the tight end back out. A tighter channel's
    # optimum is feasible for every looser one, so it can only improve a
    # looser answer, and starting each width from the next-tighter design
    # is what turns "the solver found a basin" into "the study is
    # monotone". Without it the 150 m entry landed 2 s *behind* the 80 m
    # one on this polar, which is a statement about starting points rather
    # than about channels.
    for index in range(len(channel) - 2, -1, -1):
        record = channel[index]
        inner, inner_gemseo = create_optimization(
            transcription=SingleShooting(),
            size=size,
            integrator=integrator(size),
            channel_half_width=record["half_width"],
        )
        objective, solution, met, outputs = _solve_checked(
            inner, inner_gemseo, budget, warm=channel[index + 1]["design"]
        )
        if met and objective < record["time"]:
            record.update(
                time=objective,
                across=float(
                    numpy.max(numpy.abs(numpy.asarray(outputs["trajectory.y"])))
                ),
                start="from the tighter channel",
                design=solution,
                trajectory=optimal_trajectory(inner, inner_gemseo),
            )

    for record in channel:
        if record["half_width"] == drawn:
            _write_arrays(
                "sail/track_channel.npz", **_trajectory_arrays(record["trajectory"])
            )
        del record["design"], record["trajectory"]
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
    track = numpy.load(data_path() / "sail" / "track_downwind.npz")
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
