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
"""Regenerating the committed baselines.

The baselines are what a fresh run is diffed against, so how they were made
has to be written down rather than remembered. This is that recipe, and
running it reproduces them.

They are deliberately smaller than a full sweep. A baseline exists to catch a
regression, and a regression shows up in a handful of starting points as
readily as in twenty-five; a baseline that takes ninety minutes to refresh
never gets refreshed.

Usage::

    python benchmarks/bless.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from gemseo_tempo.benchmarking.studies import run_study
from gemseo_tempo.benchmarking.suites import ALBATROSS
from gemseo_tempo.benchmarking.suites import BRACHISTOCHRONE
from gemseo_tempo.benchmarking.suites import SAIL

RECIPES = (
    (BRACHISTOCHRONE, "transcription", (10, 30), 5),
    (SAIL, "transcription", (25,), 5),
    (ALBATROSS, "optimizer", (8,), 3),
)
"""The suite, study, sizes and starting points of each baseline.

One study per suite, chosen for what it exercises rather than for coverage:
the transcription study is the one whose answer would move if a transcription
changed, and the albatross gets the optimizer study because it has only one
transcription to offer.
"""

MAX_ITER = 250
"""The iteration budget, the same one the baselines were blessed with."""


def main(argv=None) -> int:
    """Run every recipe and write the baselines.

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
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "baselines"
    )
    arguments = parser.parse_args(argv)
    logging.disable(logging.CRITICAL)
    arguments.output.mkdir(parents=True, exist_ok=True)

    for suite, study, sizes, starts in RECIPES:
        records = run_study(
            study, sizes=sizes, n_starts=starts, max_iter=MAX_ITER, suite=suite
        )
        path = arguments.output / f"{suite.name}.json"
        path.write_text(
            json.dumps([record.as_dict() for record in records], indent=1),
            encoding="utf-8",
        )
        solved = sum(record.status == "converged" for record in records)
        print(f"{suite.name:>16}: {len(records)} runs, {solved} solved -> {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
