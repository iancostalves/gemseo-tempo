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
"""Diffing a benchmark run against its baseline."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

from compare import compare  # noqa: E402
from compare import load_records  # noqa: E402
from compare import main  # noqa: E402
from compare import report  # noqa: E402
from compare import summarize  # noqa: E402

logging.disable(logging.CRITICAL)


def _record(status, calls, suite="sail", size=25):
    """Return one record as the runner would write it.

    Parameters
    ----------
    status
        The run status.
    calls
        The discipline calls used.
    suite
        The suite name.
    size
        The discretization.

    Returns
    -------
    dict
        The record.
    """
    return {
        "study": "transcription",
        "level": "collocation",
        "size": size,
        "status": status,
        "calls": calls,
        "extra": {"suite": suite},
    }


BASELINE = [
    _record("converged", 40),
    _record("converged", 44),
    _record("converged", 42),
    _record("inaccurate", 99),
]


def test_a_run_matches_its_own_baseline():
    """Nothing regresses against itself, which is the first thing to check."""
    summary = summarize(BASELINE)
    assert not any(item.regressed for item in compare(summary, summary))


def test_the_summary_counts_only_solved_runs_in_the_cost():
    """Median calls describe the runs that got there, not the ones that did not."""
    summary = summarize(BASELINE)["sail/transcription/collocation/25"]
    assert summary["runs"] == 4
    assert summary["rate"] == 0.75
    # The 99-call failure is excluded, so the median is of 40, 42 and 44.
    assert summary["calls"] == 42


def test_fewer_solves_is_a_regression():
    worse = [dict(row) for row in BASELINE]
    worse[0]["status"] = "inaccurate"
    worse[1]["status"] = "inaccurate"
    result = compare(summarize(worse), summarize(BASELINE))[0]
    assert result.rate_regressed
    assert result.regressed
    assert "REGRESSED" in report([result])


def test_a_much_costlier_solve_is_a_regression():
    worse = [dict(row) for row in BASELINE]
    for row in worse:
        if row["status"] == "converged":
            row["calls"] = int(row["calls"] * 2)
    result = compare(summarize(worse), summarize(BASELINE))[0]
    assert result.calls_regressed
    assert not result.rate_regressed


def test_ordinary_noise_is_not_a_regression():
    """A benchmark that cries wolf gets ignored, so the tolerance is real."""
    noisy = [dict(row) for row in BASELINE]
    for row in noisy:
        if row["status"] == "converged":
            row["calls"] += 4
    assert not compare(summarize(noisy), summarize(BASELINE))[0].regressed


def test_a_cell_that_never_solved_cannot_regress_on_cost():
    empty = [_record("inaccurate", 10), _record("inaccurate", 10)]
    result = compare(summarize(empty), summarize(empty))[0]
    assert result.baseline_calls == 0.0
    assert not result.calls_regressed


def test_cells_only_in_one_run_are_skipped():
    """A run covering different ground is a different run, not a regression."""
    other = [_record("converged", 40, suite="albatross")]
    assert compare(summarize(other), summarize(BASELINE)) == []


def test_the_command_reports_and_fails_on_a_regression(tmp_path):
    baseline = tmp_path / "baselines"
    current = tmp_path / "current"
    baseline.mkdir()
    current.mkdir()
    (baseline / "a.json").write_text(json.dumps(BASELINE), encoding="utf-8")

    (current / "a.json").write_text(json.dumps(BASELINE), encoding="utf-8")
    assert main(["--records", str(current), "--baselines", str(baseline)]) == 0
    assert len(load_records(current)) == len(BASELINE)

    worse = [dict(row) for row in BASELINE]
    worse[0]["status"] = "inaccurate"
    worse[1]["status"] = "inaccurate"
    (current / "a.json").write_text(json.dumps(worse), encoding="utf-8")
    assert main(["--records", str(current), "--baselines", str(baseline)]) == 1


def test_nothing_in_common_is_not_a_pass(tmp_path):
    """An empty comparison must not read as success.

    A workflow that points at the wrong directory should fail loudly rather
    than report that nothing regressed.
    """
    baseline = tmp_path / "baselines"
    current = tmp_path / "current"
    baseline.mkdir()
    current.mkdir()
    (baseline / "a.json").write_text(json.dumps(BASELINE), encoding="utf-8")
    (current / "a.json").write_text(
        json.dumps([_record("converged", 40, suite="albatross")]), encoding="utf-8"
    )
    assert main(["--records", str(current), "--baselines", str(baseline)]) == 1
