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
"""Running one benchmark configuration and recording what happened.

Some cells of these studies are expected to do badly: finite differences
over a thousand design variables, or a gradient-free algorithm on a fine
grid. That is the point of the comparison, but it means a run must never
be able to hang, and a run that exhausted its budget must be
distinguishable from one that converged. Every run therefore carries a
hard iteration cap and reports a status.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field

from gemseo.core.discipline.discipline import Discipline

from gemseo_tempo.problems.brachistochrone.problem import Guess
from gemseo_tempo.problems.brachistochrone.problem import create_optimization
from gemseo_tempo.problems.brachistochrone.problem import optimal_path_angle
from gemseo_tempo.problems.brachistochrone.reference import reference_time
from gemseo_tempo.transcriptions.collocation import HermiteSimpson
from gemseo_tempo.transcriptions.indirect import IndirectShooting
from gemseo_tempo.transcriptions.single_shooting import SingleShooting

CONVERGED = "converged"
"""The run reached the reference within tolerance."""

INACCURATE = "inaccurate"
"""The run finished but missed the reference."""

BUDGET = "budget-exhausted"
"""The run used its whole iteration budget."""

FAILED = "failed"
"""The run raised."""

ACCURACY = 1e-2
"""The relative error below which a run counts as converged."""


@dataclass
class Record:
    """The outcome of one benchmark run."""

    study: str
    level: str
    size: int
    seed: int
    calls: int
    seconds: float
    objective: float
    error: float
    status: str
    n_design_variables: int = 0
    variant: str = "ideal"
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Return the record as a plain dictionary.

        Returns
        -------
        dict
            The fields of the record.
        """
        return asdict(self)


def classify(error: float, calls: int, max_iter: int) -> str:
    """Return the status of a finished run.

    Parameters
    ----------
    error
        The relative error against the reference.
    calls
        The number of discipline calls used.
    max_iter
        The iteration budget.

    Returns
    -------
    str
        :data:`CONVERGED` when the reference was reached,
        :data:`BUDGET` when the budget ran out first, and
        :data:`INACCURATE` when the run stopped early on a wrong answer.
    """
    if error <= ACCURACY:
        return CONVERGED
    if calls >= max_iter:
        return BUDGET
    return INACCURATE


def build_transcription(level: str, size: int):
    """Return the transcription and control settings of a level.

    Parameters
    ----------
    level
        The level name.
    size
        The discretization.

    Returns
    -------
    tuple
        The transcription and the effective size to build the problem with.

    Raises
    ------
    ValueError
        If the level is unknown.
    """
    if level == "shooting":
        return SingleShooting(), size
    if level == "collocation":
        return HermiteSimpson(num_segments=size), size
    if level == "collocation-sparse":
        return HermiteSimpson(num_segments=size, sparse_jacobian=True), size
    if level == "indirect":
        # The indirect method has no path discretization; the abscissa is
        # its integrator save resolution instead.
        return IndirectShooting(u_star=optimal_path_angle), size
    raise ValueError(f"Unknown level {level!r}.")


def run_one(
    study: str,
    level: str,
    size: int,
    guess: Guess,
    seed: int,
    algo_name: str = "SLSQP",
    derivative: str = "auto",
    variant: str = "ideal",
    max_iter: int = 200,
) -> Record:
    """Run one configuration and record the outcome.

    Parameters
    ----------
    study
        The study name, recorded with the result.
    level
        The level of the varied factor.
    size
        The discretization.
    guess
        The starting point.
    seed
        The index of the starting point.
    algo_name
        The GEMSEO optimizer name.
    derivative
        ``"auto"`` for automatic differentiation, or ``"fd"`` to force
        finite differences on the assembled discipline.
    variant
        The physical variant.
    max_iter
        The iteration budget.

    Returns
    -------
    Record
        The outcome, never raising: a failure is reported as a status.
    """
    target = reference_time(variant)
    started = time.perf_counter()
    try:
        transcription, effective = build_transcription(level, size)
        scenario, gemseo_scenario = create_optimization(
            transcription=transcription,
            variant=variant,
            size=effective,
            guess=guess,
        )
        n_design = sum(spec.size for spec in scenario.problem.design_variables.values())
        if derivative == "fd":
            for discipline in scenario.disciplines:
                discipline.set_jacobian_approximation(
                    Discipline.ApproximationMode.FINITE_DIFFERENCES
                )
                discipline.linearization_mode = (
                    Discipline.LinearizationMode.FINITE_DIFFERENCES
                )
        gemseo_scenario.execute(algo_name=algo_name, max_iter=max_iter)
        problem = gemseo_scenario.formulation.optimization_problem
        calls = len(problem.database)
        objective = float(gemseo_scenario.optimization_result.f_opt)
        error = abs(objective - target) / target
        status = classify(error, calls, max_iter)
    except Exception as error_raised:  # noqa: BLE001
        return Record(
            study=study,
            level=level,
            size=size,
            seed=seed,
            calls=0,
            seconds=time.perf_counter() - started,
            objective=float("nan"),
            error=float("nan"),
            status=FAILED,
            variant=variant,
            extra={"error": type(error_raised).__name__},
        )
    return Record(
        study=study,
        level=level,
        size=size,
        seed=seed,
        calls=calls,
        seconds=time.perf_counter() - started,
        objective=objective,
        error=error,
        status=status,
        n_design_variables=n_design,
        variant=variant,
    )
