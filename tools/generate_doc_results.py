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


def _write_history(name: str, gemseo_scenario) -> Path:
    """Store an optimization history, so a figure can replay it.

    GEMSEO's post-processing reads a problem from a path, so a stored
    history is enough to draw the objective, the constraints, the design
    variables and their sensitivities without solving anything again.

    Parameters
    ----------
    name
        The path relative to ``docs/_data``.
    gemseo_scenario
        The executed GEMSEO scenario.

    Returns
    -------
    pathlib.Path
        Where it was written.
    """
    path = DATA / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    gemseo_scenario.formulation.optimization_problem.to_hdf(path)
    print(f"  {name} ({path.stat().st_size // 1024} kB)")
    return path


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
    from gemseo_tempo.dynamics.integrators import DiffraxIntegrator
    from gemseo_tempo.problems.brachistochrone import VARIANTS
    from gemseo_tempo.problems.brachistochrone import create_optimization
    from gemseo_tempo.problems.brachistochrone import cycloid_time
    from gemseo_tempo.problems.brachistochrone import reference_time
    from gemseo_tempo.scenario.trajectories import optimal_trajectory
    from gemseo_tempo.transcriptions import HermiteSimpson
    from gemseo_tempo.transcriptions import SingleShooting

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

    # The cycloid itself, sampled finely, as the answer to reproduce.
    theta = numpy.linspace(0.0, numpy.pi, 200)
    radius = 2.0 / (1.0 - numpy.cos(numpy.pi))
    _write_arrays(
        "brachistochrone/cycloid.npz",
        x=radius * (theta - numpy.sin(theta)),
        y=radius * (1.0 - numpy.cos(theta)),
        time=numpy.asarray(cycloid_time()),
    )

    # The three transcriptions on the ideal variant, and how the error of
    # each falls as the grid is refined.
    sizes = (5, 10) if smoke else (5, 10, 20, 40, 80)
    exact = float(cycloid_time())
    convergence: dict[str, list[float]] = {}
    for label, transcription in (
        ("collocation", HermiteSimpson),
        ("shooting", SingleShooting),
    ):
        errors = []
        for count in sizes:
            build = {"num_segments": count} if transcription is HermiteSimpson else {}
            _, gemseo_scenario = create_optimization(
                transcription=transcription(**build), variant="ideal", size=count
            )
            objective, _, _ = _solve(gemseo_scenario, budget)
            errors.append(abs(objective - exact) / exact)
        convergence[label] = errors
    _write_json(
        "brachistochrone/convergence.json",
        {"sizes": list(sizes), "errors": convergence, "exact": exact},
    )

    # One history, kept so the optimizer can be watched rather than
    # described. Collocation, because its design vector is the whole
    # trajectory and the variable history is worth looking at.
    scenario, gemseo_scenario = create_optimization(variant="drag", size=size)
    _solve(gemseo_scenario, budget)
    _write_history("brachistochrone/history_collocation.h5", gemseo_scenario)

    # Fixed against adaptive stepping: the same problem, both ways.
    stepping = {}
    for label, integrator in (
        ("adaptive 1e-8", DiffraxIntegrator(rtol=1e-8, atol=1e-8, n_save=size + 1)),
        ("fixed 1/200", DiffraxIntegrator(dt0=1.0 / 200, n_save=size + 1)),
        ("fixed 1/20", DiffraxIntegrator(dt0=1.0 / 20, n_save=size + 1)),
    ):
        started = time.perf_counter()
        _, gemseo_scenario = create_optimization(
            transcription=SingleShooting(),
            variant="drag",
            size=size,
            integrator=integrator,
        )
        objective, _, feasible = _solve(gemseo_scenario, budget)
        stepping[label] = {
            "time": objective,
            "seconds": time.perf_counter() - started,
            "feasible": feasible,
        }
    _write_json("brachistochrone/stepping.json", stepping)

    # The optimizer study, for the benchmark subsection asking whether the
    # choice of optimizer matters. Reuses the same one-factor-at-a-time
    # machinery as the transcription study already blessed into
    # benchmarks/baselines/brachistochrone.json.
    _write_benchmark_study(
        "brachistochrone",
        "optimizer",
        sizes=(sizes if smoke else (10, 30, 100)),
        n_starts=2 if smoke else 10,
        max_iter=budget,
    )

    _generate_ad_vs_fd(smoke)


def _generate_ad_vs_fd(smoke: bool) -> None:
    """Measure the cost of a gradient by AD against by finite differences.

    Not GEMSEO's own ``derivative`` study (dense/sparse AD against a
    forced finite-difference discipline), which times whole optimizations
    and so measures how many SLSQP iterations each mode needs to
    converge — answering "does the choice still find the answer", not
    "what does one gradient cost". Both modes solve in the same handful
    of iterations here, since a numerically sound gradient is a numerically
    sound gradient whichever way it was obtained, so that study's own
    figure cannot show what this one exists to show.

    This measures the one thing the transcription guide's argument turns
    on directly: the cost of one gradient of the objective, with respect
    to the whole design vector, growing the vector by refining the grid.
    Reverse-mode AD through the whole trajectory costs one pass regardless
    of the number of design variables; central finite differences cost one
    extra pass *per variable*. Everything is measured on the bare JAX
    function rather than through a GEMSEO discipline, because GEMSEO caches
    a linearization until its inputs change, and timing a second call at
    the same point would measure the cache instead of either derivative.

    Parameters
    ----------
    smoke
        Whether to run the tiny version, for the tests.
    """
    import time

    import jax
    import jax.numpy as jnp

    from gemseo_tempo.problems.brachistochrone import create_optimization

    sizes = (10, 40) if smoke else (10, 40, 100, 200, 400)
    records = []
    for size in sizes:
        scenario, _ = create_optimization(variant="ideal", size=size)
        problem = scenario.problem
        defaults = dict(problem.input_defaults)
        names = list(problem.design_variables)
        shapes = {name: jnp.shape(defaults[name]) for name in names}
        n_design = sum(int(numpy.prod(shapes[name])) for name in names)

        def unflatten(flat, names=names, shapes=shapes, defaults=defaults):
            """Rebuild the input dictionary from one flat design vector."""
            out = dict(defaults)
            index = 0
            for name in names:
                count = int(numpy.prod(shapes[name])) or 1
                out[name] = jnp.reshape(flat[index : index + count], shapes[name])
                index += count
            return out

        def objective(flat, problem=problem, unflatten=unflatten):
            """Return the objective as a function of the flat design vector."""
            outputs = problem.jax_function(unflatten(flat))
            return jnp.reshape(outputs[problem.objective_name], ())

        design_point = jnp.concatenate(
            [jnp.ravel(jnp.asarray(defaults[name])) for name in names]
        )

        gradient = jax.jit(jax.grad(objective))
        gradient(design_point).block_until_ready()
        started = time.perf_counter()
        gradient(design_point).block_until_ready()
        ad_seconds = time.perf_counter() - started

        evaluate = jax.jit(objective)
        evaluate(design_point).block_until_ready()
        started = time.perf_counter()
        base = float(evaluate(design_point))
        step = 1e-6
        finite_gradient = numpy.empty(n_design)
        for index in range(n_design):
            perturbed = float(evaluate(design_point.at[index].add(step)))
            finite_gradient[index] = (perturbed - base) / step
        fd_seconds = time.perf_counter() - started

        records.append(
            {
                "size": size,
                "n_design": n_design,
                "ad_seconds": ad_seconds,
                "fd_seconds": fd_seconds,
            }
        )
    _write_json("brachistochrone/ad_vs_fd.json", records)


def _write_benchmark_study(
    problem: str, study: str, sizes, n_starts: int, max_iter: int
) -> None:
    """Run one benchmark study and store its figure and its records.

    ``plot_study`` renders and saves a complete figure by itself and
    closes it, which does not fit the "define a function, let
    ``plot_directive`` capture the open figure" pattern the rest of this
    project's figures use. So the PNG is rendered here, once, and the page
    embeds it as a plain image; only pure-geometry schematics and
    loaded-array plots go through ``plot_directive``. See
    :mod:`tempo_docdata` for the full rule.

    Parameters
    ----------
    problem
        The suite name, matching a key of
        :data:`gemseo_tempo.benchmarking.suites.SUITES`.
    study
        The study name, a key of
        :data:`gemseo_tempo.benchmarking.studies.STUDIES`.
    sizes
        The discretizations to sweep.
    n_starts
        The number of starting points per cell.
    max_iter
        The iteration budget of each run.
    """
    from gemseo_tempo.benchmarking.plots import plot_study
    from gemseo_tempo.benchmarking.studies import run_study
    from gemseo_tempo.benchmarking.suites import SUITES

    suite = SUITES[problem]
    records = run_study(
        study, sizes=sizes, n_starts=n_starts, max_iter=max_iter, suite=suite
    )
    image_path = DATA / problem / f"{study}_study.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    plot_study(records, study, image_path, sizes=sizes, problem=problem.title())
    print(f"  {problem}/{study}_study.png")
    _write_json(
        f"{problem}/{study}_study.json",
        [record.as_dict() for record in records],
    )


# ------------------------------------------------------------------- sail


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
    free_time, free_design, _ = _solve(gemseo_scenario, budget)
    trajectory = optimal_trajectory(scenario, gemseo_scenario)
    _write_arrays("sail/track_downwind.npz", **_trajectory_arrays(trajectory))
    _write_history("sail/history_downwind.h5", gemseo_scenario)

    # The same leg sailed straight at the mark, which is what it beats.
    _write_json(
        "sail/downwind.json",
        {
            "optimized": free_time,
            "straight": float(straight_run_time("downwind")),
            "mark": list(LEGS["downwind"]),
        },
    )

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

    _generate_sail_uncertainty(size, smoke)


def _generate_sail_uncertainty(size: int, smoke: bool) -> None:
    """Compute the sailing paper's uncertainty results.

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
    from gemseo_tempo.uncertainty import batch_evaluator
    from gemseo_tempo.uncertainty import freeze_design
    from gemseo_tempo.uncertainty import looped_evaluator
    from gemseo_tempo.uncertainty import make_parameter_space
    from gemseo_tempo.uncertainty import measure_speedup
    from gemseo_tempo.uncertainty import nominal_values
    from gemseo_tempo.uncertainty import uncertain_names
    from gemseo_tempo.uncertainty.estimators import Comparison
    from gemseo_tempo.uncertainty.estimators import from_scenario
    from gemseo_tempo.uncertainty.estimators import monte_carlo
    from gemseo_tempo.uncertainty.estimators import polynomial_chaos
    from gemseo_tempo.uncertainty.estimators import taylor

    arrival = "constraint.final.x"
    names = uncertain_names(REGISTRY)
    scenario, gemseo_scenario = create_optimization(
        transcription=SingleShooting(),
        size=size,
        integrator=sail_integrator(dt0=1.0 / 300, n_save=size + 1),
    )
    _solve(gemseo_scenario, 200 if smoke else 800)
    freeze_design(scenario, gemseo_scenario, nominal_values(REGISTRY, names))

    space = make_parameter_space(REGISTRY, names)
    sample = numpy.asarray(space.compute_samples(64 if smoke else 512))
    speedup = measure_speedup(scenario, names, [arrival], sample)
    _write_json(
        "sail/speedup.json",
        {
            "n_samples": speedup.n_samples,
            "looped_seconds": speedup.looped_seconds,
            "batched_seconds": speedup.batched_seconds,
            "speedup": speedup.speedup,
            "max_difference": speedup.max_difference,
        },
    )

    # The response surface the estimators are arguing about, on a grid so a
    # figure can show why first order struggles with the mean.
    grid = 9 if smoke else 41
    scale = numpy.linspace(0.7, 1.3, grid)
    veer = numpy.linspace(-0.5, 0.5, grid)
    mesh_scale, mesh_veer = numpy.meshgrid(scale, veer, indexing="ij")
    points = numpy.column_stack([mesh_scale.ravel(), mesh_veer.ravel()])
    surface = numpy.asarray(
        batch_evaluator(scenario, names, [arrival])(points)
    ).reshape(grid, grid)
    _write_arrays("sail/response.npz", scale=scale, veer=veer, arrival=surface)

    propagation = from_scenario(scenario, REGISTRY, names, arrival)
    looped = looped_evaluator(scenario, names, [arrival])
    reference = monte_carlo(
        propagation, n_samples=512 if smoke else 50000, name="reference"
    )
    estimates = [
        monte_carlo(propagation, n_samples=512, name="sampling, vmapped"),
        monte_carlo(
            propagation,
            n_samples=512,
            name="sampling, looped",
            evaluate=lambda draws: numpy.asarray(looped(draws)).ravel(),
        ),
        taylor(propagation),
        taylor(propagation, second_order=True),
        polynomial_chaos(propagation, n_training=32),
        polynomial_chaos(propagation, n_training=32, gradient_enhanced=True),
        polynomial_chaos(propagation, n_training=12, degree=3),
        polynomial_chaos(propagation, n_training=12, degree=3, gradient_enhanced=True),
    ]
    comparison = Comparison(reference=reference, estimates=tuple(estimates))
    _write_json(
        "sail/estimators.json",
        {
            "reference": _estimate_record(reference, comparison),
            "estimates": [
                _estimate_record(estimate, comparison) for estimate in estimates
            ],
        },
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
    for margin in margins:
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
        records.append(measure(best[1], best[0], label))
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
        objective, design, met, _ = _solve_checked(
            scenario, formulation.create_gemseo_scenario(), budget, warm=start
        )
        if met and (best is None or objective < best[0]):
            best = (objective, design)
    if best is not None:
        records.append(measure(best[1], best[0], f"P(inside) >= {target:.0%}"))

    _write_json(
        "sail/margin.json",
        {"half_width": half_width, "target": target, "designs": records},
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
    from gemseo_tempo.transcriptions import SingleShooting

    print("albatross")
    size = 6 if smoke else 20
    budget = 40 if smoke else 500

    # The soaring cycle at a fixed wind, and the wing it would choose.
    scenario, gemseo_scenario = create_energy_optimization(size=size)
    _solve(gemseo_scenario, budget)
    cycle = optimal_trajectory(scenario, gemseo_scenario)
    _write_arrays("albatross/cycle_energy.npz", **_trajectory_arrays(cycle))
    _write_json("albatross/energy_budget.json", energy_budget(cycle))

    scenario, gemseo_scenario = create_codesign_optimization(size=size)
    minimum_wind, _, feasible = _solve(gemseo_scenario, budget)
    codesign = optimal_trajectory(scenario, gemseo_scenario)
    _write_arrays("albatross/cycle_codesign.npz", **_trajectory_arrays(codesign))
    _write_history("albatross/history_codesign.h5", gemseo_scenario)
    _write_json(
        "albatross/codesign.json",
        {
            "minimum_wind": minimum_wind,
            "feasible": feasible,
            "budget": energy_budget(codesign),
        },
    )

    # What a cycle takes out of the wind, swept over the wind strength.
    winds = (0.0, 12.0) if smoke else (0.0, 8.0, 12.0, 16.0, 18.0, 22.0, 26.0)
    sweep = []
    for wind in winds:
        _, inner_gemseo = create_energy_optimization(size=size, w_ref=wind)
        _solve(inner_gemseo, budget)
        inner, _ = create_energy_optimization(size=size, w_ref=wind)
        trajectory = optimal_trajectory(inner, inner_gemseo)
        budget_of = energy_budget(trajectory)
        sweep.append({"w_ref": wind, **budget_of})
    _write_json("albatross/energy_sweep.json", sweep)

    _generate_albatross_uncertainty(size, smoke, DiffraxIntegrator, SingleShooting)


def _generate_albatross_uncertainty(size, smoke, DiffraxIntegrator, SingleShooting):
    """Compute the albatross paper's uncertainty results.

    Single shooting, because under collocation the states are design
    variables and a frozen design cannot respond to a changed wind at all.

    Parameters
    ----------
    size
        The discretization.
    smoke
        Whether to run the tiny version.
    DiffraxIntegrator
        The integrator class, passed in to keep the import local.
    SingleShooting
        The transcription class, likewise.
    """
    from gemseo_tempo.problems.albatross import create_energy_optimization
    from gemseo_tempo.problems.albatross.model import REGISTRY
    from gemseo_tempo.uncertainty import freeze_design
    from gemseo_tempo.uncertainty import looped_evaluator
    from gemseo_tempo.uncertainty import make_parameter_space
    from gemseo_tempo.uncertainty import measure_speedup
    from gemseo_tempo.uncertainty import nominal_values
    from gemseo_tempo.uncertainty import uncertain_names
    from gemseo_tempo.uncertainty.estimators import Comparison
    from gemseo_tempo.uncertainty.estimators import from_scenario
    from gemseo_tempo.uncertainty.estimators import monte_carlo
    from gemseo_tempo.uncertainty.estimators import polynomial_chaos
    from gemseo_tempo.uncertainty.estimators import taylor
    from gemseo_tempo.uncertainty.sensitivity import screen
    from gemseo_tempo.uncertainty.sensitivity import sobol_indices

    shooting_size = 6 if smoke else 10
    # The energy the cycle banks, which is the mission's own objective and
    # is what a wrong forecast actually costs: a bird that planned for
    # 22 m/s and met something else ends its cycle with more or less in
    # hand. A first attempt used the cycle-closure residual instead,
    # reduced by the worst of its three components, and that was a poor
    # choice twice over. It is not what the design was optimized for, and
    # a max of absolute values has a kink exactly where a converged design
    # sits, so every estimator that needs a derivative was being asked to
    # differentiate a corner: second-order Taylor came back wrong by six
    # orders of magnitude.
    quantity = "energy"
    reductions = None

    names = uncertain_names(REGISTRY)
    scenario, gemseo_scenario = create_energy_optimization(
        transcription=SingleShooting(),
        size=shooting_size,
        w_ref=22.0,
        integrator=DiffraxIntegrator(dt0=1.0 / 400, n_save=shooting_size + 1),
    )
    objective, _, feasible = _solve(gemseo_scenario, 150 if smoke else 400)
    freeze_design(scenario, gemseo_scenario, nominal_values(REGISTRY, names))

    space = make_parameter_space(REGISTRY, names)
    sample = numpy.asarray(space.compute_samples(32 if smoke else 256))
    speedup = measure_speedup(
        scenario, names, [quantity], sample, reductions=reductions
    )

    screening = screen(
        scenario,
        REGISTRY,
        names,
        quantity,
        n_trajectories=6 if smoke else 20,
        reductions=reductions,
    )
    indices = sobol_indices(
        scenario,
        REGISTRY,
        names,
        quantity,
        n_samples=64 if smoke else 512,
        reductions=reductions,
    )

    propagation = from_scenario(
        scenario, REGISTRY, names, quantity, reductions=reductions
    )
    looped = looped_evaluator(scenario, names, [quantity], reductions=reductions)
    reference = monte_carlo(
        propagation, n_samples=128 if smoke else 4096, name="reference"
    )
    estimates = [
        monte_carlo(propagation, n_samples=128, name="sampling, vmapped"),
        monte_carlo(
            propagation,
            n_samples=128,
            name="sampling, looped",
            evaluate=lambda draws: numpy.asarray(looped(draws)).ravel(),
        ),
        taylor(propagation),
        taylor(propagation, second_order=True),
        polynomial_chaos(propagation, n_training=32),
        polynomial_chaos(propagation, n_training=32, gradient_enhanced=True),
    ]
    comparison = Comparison(reference=reference, estimates=tuple(estimates))
    _write_json(
        "albatross/uncertainty.json",
        {
            "objective": objective,
            "feasible": feasible,
            "size": shooting_size,
            "speedup": {
                "n_samples": speedup.n_samples,
                "looped_seconds": speedup.looped_seconds,
                "batched_seconds": speedup.batched_seconds,
                "speedup": speedup.speedup,
                "max_difference": speedup.max_difference,
            },
            "screening": {
                "ranking": screening.ranking,
                "mean_effect": list(screening.mean_effect),
                "n_evaluations": screening.n_evaluations,
            },
            "sobol": {
                "names": list(indices.names),
                "first_order": list(indices.first_order),
                "total": list(indices.total),
                "n_evaluations": indices.n_evaluations,
                "seconds": indices.seconds,
            },
            "reference": _estimate_record(reference, comparison),
            "estimates": [
                _estimate_record(estimate, comparison) for estimate in estimates
            ],
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
