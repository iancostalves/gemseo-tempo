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
"""Tests of the benchmark harness.

The sweep itself is far too slow to be a test, so these exercise the
machinery on the cheapest possible configurations.
"""

from __future__ import annotations

import logging

import numpy
import pytest

from gemseo_tempo.benchmarking.plots import plot_study
from gemseo_tempo.benchmarking.profiles import data_profile
from gemseo_tempo.benchmarking.profiles import profile_table
from gemseo_tempo.benchmarking.runner import BUDGET
from gemseo_tempo.benchmarking.runner import CONVERGED
from gemseo_tempo.benchmarking.runner import FAILED
from gemseo_tempo.benchmarking.runner import INACCURATE
from gemseo_tempo.benchmarking.runner import Record
from gemseo_tempo.benchmarking.runner import classify
from gemseo_tempo.benchmarking.runner import run_one
from gemseo_tempo.benchmarking.sampling import sample_guesses
from gemseo_tempo.benchmarking.studies import STUDIES
from gemseo_tempo.benchmarking.studies import load_records
from gemseo_tempo.benchmarking.studies import run_study
from gemseo_tempo.benchmarking.studies import save_records
from gemseo_tempo.benchmarking.transcriptions import build_transcription
from gemseo_tempo.problems.brachistochrone import Guess
from gemseo_tempo.transcriptions import HermiteSimpson
from gemseo_tempo.transcriptions import IndirectShooting
from gemseo_tempo.transcriptions import SingleShooting

logging.disable(logging.CRITICAL)


# -------------------------------------------------------------- sampling


def test_sampling_is_reproducible_and_shared():
    first = sample_guesses(6, seed=3)
    second = sample_guesses(6, seed=3)
    assert len(first) == 6
    assert [g.t_f for g in first] == [g.t_f for g in second]
    assert [g.gamma_level for g in first] == [g.gamma_level for g in second]
    # A different seed gives a different sample.
    assert [g.t_f for g in sample_guesses(6, seed=4)] != [g.t_f for g in first]


def test_sampled_guesses_are_within_bounds():
    for guess in sample_guesses(12, seed=1):
        assert 0.0 <= guess.gamma_level <= numpy.pi / 2
        assert 0.5 <= guess.t_f <= 2.0
        assert set(guess.costates) == {"x", "y", "v"}


# ------------------------------------------------------------ transcription


def test_build_transcription_levels():
    assert isinstance(build_transcription("shooting", 5)[0], SingleShooting)
    assert isinstance(build_transcription("collocation", 5)[0], HermiteSimpson)
    sparse = build_transcription("collocation-sparse", 5)[0]
    assert sparse.sparse_jacobian
    assert isinstance(build_transcription("indirect", 5)[0], IndirectShooting)
    with pytest.raises(ValueError, match="Unknown level"):
        build_transcription("telepathy", 5)


# ---------------------------------------------------------------- runner


def test_runner_records_a_solved_run():
    record = run_one("t", "shooting", 5, Guess(), seed=0, max_iter=100)
    assert record.status == CONVERGED
    assert record.calls > 0
    assert record.error < 1e-2
    assert record.n_design_variables > 0
    assert record.seconds > 0.0
    assert record.as_dict()["level"] == "shooting"


def test_runner_reports_budget_exhaustion_without_hanging():
    """A hopeless budget must be recorded, not raised and not hung."""
    record = run_one("t", "shooting", 5, Guess(t_f=1.9), seed=0, max_iter=2)
    assert record.status in {BUDGET, INACCURATE}
    assert record.status != FAILED


def test_runner_reports_failure_as_a_status():
    record = run_one("t", "telepathy", 5, Guess(), seed=0, max_iter=2)
    assert record.status == FAILED
    assert record.extra["error"] == "ValueError"
    assert numpy.isnan(record.objective)


# ---------------------------------------------------------------- studies


def test_studies_are_one_factor_at_a_time():
    assert set(STUDIES) == {"transcription", "derivative", "optimizer"}
    # Every study keeps the other two factors at the baseline.
    for level in STUDIES["transcription"]:
        assert level.algo_name == "SLSQP"
        assert level.derivative == "auto"
    for level in STUDIES["derivative"]:
        assert level.algo_name == "SLSQP"
        assert "collocation" in level.level
    for level in STUDIES["optimizer"]:
        assert level.derivative == "auto"
        # Only constrained solvers: the others cannot run this problem.
        assert level.algo_name in {"SLSQP", "NLOPT_SLSQP", "NLOPT_COBYLA"}


def test_unknown_study_raises():
    with pytest.raises(ValueError, match="Unknown study"):
        run_study("vibes")


def test_run_study_uses_the_cache(tmp_path):
    """The sweep must be resumable: a cached cell is not recomputed."""
    cache = tmp_path / "cache"
    first = run_study("transcription", sizes=(5,), n_starts=1, max_iter=60, cache=cache)
    written = sorted(cache.glob("*.json"))
    assert len(written) == len(first) == len(STUDIES["transcription"])
    # A second pass reads the records back instead of solving again.
    stamps = {path: path.stat().st_mtime_ns for path in written}
    second = run_study(
        "transcription", sizes=(5,), n_starts=1, max_iter=60, cache=cache
    )
    assert {p: p.stat().st_mtime_ns for p in written} == stamps
    assert [r.level for r in second] == [r.level for r in first]
    assert [r.calls for r in second] == [r.calls for r in first]


def test_records_round_trip(tmp_path):
    records = [
        Record("s", "a", 3, 0, 10, 1.0, 0.8, 0.001, CONVERGED),
        Record("s", "a", 3, 1, 20, 2.0, 0.9, 0.1, INACCURATE),
    ]
    path = tmp_path / "out" / "records.json"
    save_records(records, path)
    assert [r.as_dict() for r in load_records(path)] == [r.as_dict() for r in records]


# --------------------------------------------------------------- reporting


def test_data_profile_counts_only_solved_runs():
    records = [
        Record("s", "a", 3, 0, 10, 1.0, 0.8, 0.0, CONVERGED),
        Record("s", "a", 3, 1, 80, 1.0, 0.8, 0.0, CONVERGED),
        Record("s", "a", 3, 2, 5, 1.0, 9.0, 9.0, BUDGET),
    ]
    profile = data_profile(records, "a", (10, 100))
    # Within 10 calls only the first is solved; within 100, two of three.
    assert profile[0] == pytest.approx(1 / 3)
    assert profile[1] == pytest.approx(2 / 3)
    assert data_profile(records, "absent", (10,))[0] == 0.0
    assert set(profile_table(records, (10,))) == {"a"}


def test_plot_study_writes_a_figure(tmp_path):
    records = [
        Record("s", level, size, seed, 10 + seed, 1.0, 0.82, 0.0, CONVERGED)
        for level in ("a", "b")
        for size in (3, 10)
        for seed in range(3)
    ]
    # One unsolved run, so the success panel is not trivially full.
    records.append(Record("s", "a", 3, 9, 200, 1.0, 3.0, 3.0, BUDGET))
    path = plot_study(records, "demo", tmp_path / "figures" / "demo.png")
    assert path.exists()
    assert path.stat().st_size > 5000


def test_runner_can_force_finite_differences():
    """The derivative-mode study needs FD on the assembled discipline."""
    record = run_one("t", "shooting", 3, Guess(), seed=0, derivative="fd", max_iter=60)
    assert record.status in {CONVERGED, INACCURATE, BUDGET}
    assert record.status != FAILED
    assert record.calls > 0


def test_status_classification():
    """Solved, out of budget and stopped-early are three distinct outcomes."""
    assert classify(1e-4, 10, 100) == CONVERGED
    assert classify(0.5, 100, 100) == BUDGET
    # Stopped early on a wrong answer: neither solved nor out of budget.
    assert classify(0.5, 10, 100) == INACCURATE


def test_unconstrained_solvers_are_rejected_by_this_problem():
    """The terminal equalities rule out L-BFGS-B and friends.

    They are recorded as failures rather than silently compared, which is
    why the optimizer study only lists constrained solvers.
    """
    record = run_one(
        "t", "shooting", 3, Guess(), seed=0, algo_name="L-BFGS-B", max_iter=40
    )
    assert record.status == FAILED
    assert record.extra["error"] == "ValueError"


def test_run_study_without_a_cache():
    records = run_study(
        "transcription", sizes=(3,), n_starts=1, max_iter=40, cache=None
    )
    assert len(records) == len(STUDIES["transcription"])
    assert {r.study for r in records} == {"transcription"}
