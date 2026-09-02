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
"""Run the brachistochrone benchmark studies.

Each study varies one factor away from the baseline of Hermite-Simpson
collocation, SLSQP and dense automatic differentiation, sweeping the
discretization and repeating over many starting points.

Examples
--------
A quick check of the machinery, about a minute::

    python examples/benchmark/run_studies.py --smoke

The full sweep, which takes hours and is resumable thanks to the cache::

    python examples/benchmark/run_studies.py --study all
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from gemseo_tempo.benchmarking.plots import plot_study
from gemseo_tempo.benchmarking.profiles import profile_table
from gemseo_tempo.benchmarking.runner import CONVERGED
from gemseo_tempo.benchmarking.studies import SIZES
from gemseo_tempo.benchmarking.studies import STUDIES
from gemseo_tempo.benchmarking.studies import run_study
from gemseo_tempo.benchmarking.studies import save_records

logging.disable(logging.CRITICAL)


def parse_arguments(argv=None):
    """Parse the command line.

    Parameters
    ----------
    argv
        The arguments; ``sys.argv`` when omitted.

    Returns
    -------
    Namespace
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="all", choices=[*sorted(STUDIES), "all"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--starts", type=int, default=25)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--variant", default="ideal")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Run the requested studies and write results and figures.

    Parameters
    ----------
    argv
        The command-line arguments.

    Returns
    -------
    int
        The process exit status.
    """
    arguments = parse_arguments(argv)
    sizes = (3, 10) if arguments.smoke else SIZES
    starts = 2 if arguments.smoke else arguments.starts
    names = sorted(STUDIES) if arguments.study == "all" else [arguments.study]
    if arguments.smoke:
        names = names[:1]

    output = arguments.output
    for name in names:
        started = time.perf_counter()
        records = run_study(
            name,
            sizes=sizes,
            n_starts=starts,
            max_iter=arguments.max_iter,
            variant=arguments.variant,
            cache=output / "cache",
        )
        elapsed = time.perf_counter() - started
        save_records(records, output / f"{name}.json")
        figure = plot_study(records, name, output / f"{name}.png", sizes=sizes)
        solved = sum(record.status == CONVERGED for record in records)
        print(
            f"{name}: {len(records)} runs, {solved} solved, {elapsed:.1f} s -> {figure}"
        )
        for level, fractions in profile_table(records, (50, 100, 200)).items():
            shares = ", ".join(f"{value:.0%}" for value in fractions)
            print(f"    {level:>20}  solved within 50/100/200 calls: {shares}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
