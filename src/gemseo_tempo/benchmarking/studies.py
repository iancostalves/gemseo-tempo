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
"""The one-factor-at-a-time studies.

A full factorial over transcription, derivative mode, optimizer and grid
size would be mostly wasted effort: the interesting question is what each
factor costs on its own. So a single baseline is fixed — Hermite-Simpson
collocation, SLSQP, dense automatic differentiation — and each study
varies exactly one factor away from it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from gemseo_tempo.benchmarking.runner import Record
from gemseo_tempo.benchmarking.runner import run_one
from gemseo_tempo.benchmarking.sampling import sample_guesses

SIZES = (3, 10, 30, 100)
"""The discretizations swept by every study."""

N_STARTS = 25
"""The number of starting points per cell."""


@dataclass(frozen=True)
class Level:
    """One level of the factor a study varies."""

    name: str
    """The label shown on the figure."""

    level: str
    """The transcription level passed to the runner."""

    algo_name: str = "SLSQP"
    """The optimizer."""

    derivative: str = "auto"
    """The derivative mode, ``"auto"`` or ``"fd"``."""


BASELINE = Level("collocation", "collocation", "SLSQP", "auto")
"""Collocation, SLSQP and dense automatic differentiation."""

STUDIES = {
    "transcription": (
        Level("shooting", "shooting"),
        BASELINE,
        Level("indirect", "indirect"),
    ),
    "derivative": (
        Level("finite differences", "collocation", derivative="fd"),
        Level("dense AD", "collocation"),
        Level("sparse AD", "collocation-sparse"),
    ),
    # Only constrained solvers can be compared here: this problem carries
    # terminal equalities (and collocation defects), which L-BFGS-B, TNC and
    # Nelder-Mead reject outright rather than solve badly.
    "optimizer": (
        Level("SLSQP", "collocation", algo_name="SLSQP"),
        Level("NLOPT_SLSQP", "collocation", algo_name="NLOPT_SLSQP"),
        Level("COBYLA (gradient-free)", "collocation", algo_name="NLOPT_COBYLA"),
    ),
}
"""The three studies, each varying one factor away from the baseline."""


def run_study(
    study: str,
    sizes=SIZES,
    n_starts: int = N_STARTS,
    max_iter: int = 200,
    variant: str = "ideal",
    cache: Path | None = None,
    seed: int = 0,
) -> list[Record]:
    """Run every cell of one study.

    Parameters
    ----------
    study
        The study name, a key of :data:`STUDIES`.
    sizes
        The discretizations to sweep.
    n_starts
        The number of starting points per cell.
    max_iter
        The iteration budget of each run.
    variant
        The physical variant.
    cache
        A directory holding one JSON file per completed run, which makes
        the sweep resumable: cached runs are read back instead of redone.
    seed
        The sampling seed.

    Returns
    -------
    list
        Every record of the study.

    Raises
    ------
    ValueError
        If the study name is unknown.
    """
    if study not in STUDIES:
        raise ValueError(f"Unknown study {study!r}; use one of {sorted(STUDIES)}.")
    guesses = sample_guesses(n_starts, seed=seed)
    records = []
    for level in STUDIES[study]:
        for size in sizes:
            for index, guess in enumerate(guesses):
                record = _cached_or_run(
                    study, level, size, guess, index, max_iter, variant, cache
                )
                records.append(record)
    return records


def _cached_or_run(study, level, size, guess, index, max_iter, variant, cache):
    """Return a cached record when available, otherwise run the cell."""
    path = None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        stem = f"{study}_{level.name}_{size}_{index}_{variant}".replace(" ", "-")
        path = cache / f"{stem}.json"
        if path.exists():
            return Record(**json.loads(path.read_text(encoding="utf-8")))
    record = run_one(
        study=study,
        level=level.level,
        size=size,
        guess=guess,
        seed=index,
        algo_name=level.algo_name,
        derivative=level.derivative,
        variant=variant,
        max_iter=max_iter,
    )
    # The level name is what the figure shows, not the transcription key.
    record.level = level.name
    if path is not None:
        path.write_text(json.dumps(record.as_dict()), encoding="utf-8")
    return record


def save_records(records, path: Path) -> None:
    """Write records to a JSON file.

    Parameters
    ----------
    records
        The records.
    path
        The destination file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([record.as_dict() for record in records], indent=1),
        encoding="utf-8",
    )


def load_records(path: Path) -> list[Record]:
    """Read records back from a JSON file.

    Parameters
    ----------
    path
        The source file.

    Returns
    -------
    list
        The records.
    """
    return [Record(**entry) for entry in json.loads(path.read_text(encoding="utf-8"))]
