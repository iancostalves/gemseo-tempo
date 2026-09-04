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
"""Diffing a benchmark run against the committed baselines.

A performance regression should be caught rather than noticed. The baselines
under ``benchmarks/baselines`` hold what each cell did when it was last
blessed, and this compares a fresh run against them.

Two quantities are compared per cell, and only two, because they are the ones
that mean something across machines:

- the **solve rate**, how many starting points reached the target, which is a
  property of the formulation and should not move at all;
- the **median calls** to get there, which is a property of the optimizer and
  moves a little with floating-point noise.

Wall time is deliberately **not** compared. It says more about what else the
machine was doing than about the code, and a benchmark that cries wolf gets
ignored.

Usage::

    python benchmarks/compare.py --records results/ --baselines benchmarks/baselines
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from gemseo_tempo.benchmarking.runner import CONVERGED

SOLVE_RATE_TOLERANCE = 0.1
"""How far the solve rate may fall before it counts as a regression."""

CALLS_TOLERANCE = 0.25
"""How far the median call count may rise before it counts as a regression."""


@dataclass
class CellComparison:
    """One cell of a run, against its baseline."""

    key: str
    """The cell, as suite, study, level and size."""

    baseline_rate: float
    """The blessed solve rate."""

    current_rate: float
    """The solve rate now."""

    baseline_calls: float
    """The blessed median calls of the solved runs."""

    current_calls: float
    """The median calls now."""

    @property
    def rate_regressed(self) -> bool:
        """Return whether fewer starting points solved than before.

        Returns
        -------
        bool
            Whether the drop exceeds the tolerance.
        """
        return self.current_rate < self.baseline_rate - SOLVE_RATE_TOLERANCE

    @property
    def calls_regressed(self) -> bool:
        """Return whether the solved runs now cost noticeably more.

        Returns
        -------
        bool
            Whether the rise exceeds the tolerance. A cell that solved
            nothing before cannot regress on cost.
        """
        if self.baseline_calls <= 0.0:
            return False
        return self.current_calls > self.baseline_calls * (1.0 + CALLS_TOLERANCE)

    @property
    def regressed(self) -> bool:
        """Return whether this cell regressed at all.

        Returns
        -------
        bool
            Whether either quantity moved the wrong way.
        """
        return self.rate_regressed or self.calls_regressed


def _median(values) -> float:
    """Return the median of a list, or zero when it is empty.

    Parameters
    ----------
    values
        The values.

    Returns
    -------
    float
        The median.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float(0.5 * (ordered[middle - 1] + ordered[middle]))


def summarize(records) -> dict:
    """Reduce records to one summary per cell.

    Parameters
    ----------
    records
        The records, as dictionaries or :class:`.Record` instances.

    Returns
    -------
    dict
        ``{cell: {"rate": float, "calls": float, "runs": int}}``.
    """
    cells: dict[str, list] = {}
    for record in records:
        data = record if isinstance(record, dict) else record.as_dict()
        suite = (data.get("extra") or {}).get("suite", "brachistochrone")
        key = f"{suite}/{data['study']}/{data['level']}/{data['size']}"
        cells.setdefault(key, []).append(data)

    summary = {}
    for key, group in cells.items():
        solved = [row for row in group if row["status"] == CONVERGED]
        summary[key] = {
            "rate": len(solved) / len(group),
            "calls": _median([row["calls"] for row in solved]),
            "runs": len(group),
        }
    return summary


def compare(current, baseline) -> list[CellComparison]:
    """Compare two summaries, cell by cell.

    Cells missing from either side are skipped rather than reported: a run
    that covers different ground than the baseline is a different run, not a
    regression.

    Parameters
    ----------
    current
        The summary of the fresh run.
    baseline
        The blessed summary.

    Returns
    -------
    list
        One comparison per shared cell, in sorted order.
    """
    return [
        CellComparison(
            key=key,
            baseline_rate=baseline[key]["rate"],
            current_rate=current[key]["rate"],
            baseline_calls=baseline[key]["calls"],
            current_calls=current[key]["calls"],
        )
        for key in sorted(set(current) & set(baseline))
    ]


def load_records(directory: Path) -> list[dict]:
    """Read every record JSON file under a directory.

    Parameters
    ----------
    directory
        The directory holding the records.

    Returns
    -------
    list
        The records as dictionaries.
    """
    records = []
    for path in sorted(directory.rglob("*.json")):
        content = json.loads(path.read_text(encoding="utf-8"))
        records.extend(content if isinstance(content, list) else [content])
    return records


def report(comparisons) -> str:
    """Render the comparison as a table.

    Parameters
    ----------
    comparisons
        The comparisons.

    Returns
    -------
    str
        One line per cell, regressions marked.
    """
    lines = [f"{'cell':<46}{'rate':>16}{'median calls':>18}  "]
    for item in comparisons:
        mark = "REGRESSED" if item.regressed else ""
        lines.append(
            f"{item.key:<46}"
            f"{item.baseline_rate:>7.2f} ->{item.current_rate:>6.2f}"
            f"{item.baseline_calls:>9.0f} ->{item.current_calls:>7.0f}  {mark}"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    """Compare a run against the baselines and report.

    Parameters
    ----------
    argv
        The command-line arguments.

    Returns
    -------
    int
        Zero when nothing regressed, one otherwise, so a workflow fails on
        a regression rather than reporting it quietly.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument(
        "--baselines", type=Path, default=Path(__file__).parent / "baselines"
    )
    arguments = parser.parse_args(argv)

    current = summarize(load_records(arguments.records))
    baseline = summarize(load_records(arguments.baselines))
    comparisons = compare(current, baseline)
    print(report(comparisons))

    shared = len(comparisons)
    regressed = [item for item in comparisons if item.regressed]
    print(f"\n{shared} cells compared, {len(regressed)} regressed")
    if not shared:
        print("nothing in common with the baselines; not treating that as a pass")
        return 1
    return 1 if regressed else 0


if __name__ == "__main__":
    sys.exit(main())
