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
"""What every study shares: writing results, solving, sampling."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy

_DATA = Path(__file__).resolve().parents[2] / "docs" / "_data"
"""Where the stored results live, unless redirected."""


def data_path() -> Path:
    """Return where the stored results are written.

    Returns
    -------
    Path
        The data directory, ``docs/_data`` unless
        :func:`set_data_path` redirected it.
    """
    return _DATA


def set_data_path(path: Path) -> None:
    """Redirect where the stored results are written.

    Parameters
    ----------
    path
        The directory to write to instead of ``docs/_data``.
    """
    global _DATA  # noqa: PLW0603
    _DATA = path


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
    path = data_path() / name
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
    path = data_path() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ending in a newline, as the end-of-file hook requires.
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
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
