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
"""Benchmarking more than one problem."""

from __future__ import annotations

import logging

import numpy
import pytest

from gemseo_tempo.benchmarking.runner import CONVERGED
from gemseo_tempo.benchmarking.runner import FAILED
from gemseo_tempo.benchmarking.runner import run_one
from gemseo_tempo.benchmarking.studies import run_study
from gemseo_tempo.benchmarking.studies import studies_for
from gemseo_tempo.benchmarking.suites import ALBATROSS
from gemseo_tempo.benchmarking.suites import BRACHISTOCHRONE
from gemseo_tempo.benchmarking.suites import SAIL
from gemseo_tempo.benchmarking.suites import SUITES
from gemseo_tempo.benchmarking.suites import get_suite
from gemseo_tempo.benchmarking.suites import sweep_size

logging.disable(logging.CRITICAL)


def test_every_suite_samples_reproducibly():
    """A seed fixes the sample, and a different seed moves it."""
    for suite in SUITES.values():
        first = suite.sample(4, 0)
        assert len(first) == 4
        assert [vars(g) for g in suite.sample(4, 0)] == [vars(g) for g in first]
        assert [vars(g) for g in suite.sample(4, 1)] != [vars(g) for g in first]


@pytest.mark.parametrize("name", sorted(SUITES))
def test_every_suite_builds_and_scores(name):
    """The same runner drives all three problems.

    This is the point of the abstraction: before it, the runner imported the
    brachistochrone and could benchmark nothing else.
    """
    suite = get_suite(name)
    level = suite.levels[1] if len(suite.levels) > 1 else suite.levels[0]
    level = "collocation" if suite.supports("collocation") else level
    record = run_one(
        name,
        level,
        suite.sizes[0],
        suite.sample(1, 0)[0],
        seed=0,
        max_iter=250,
        variant=suite.variants[0],
        suite=suite,
    )
    assert record.extra["suite"] == name
    assert record.status != FAILED
    assert record.n_design_variables > 0
    assert numpy.isfinite(record.objective)
    assert suite.target(suite.variants[0]) > 0.0


def test_a_suite_refuses_a_level_it_does_not_offer():
    """The sail has a path constraint, so indirect shooting is not available.

    Asked for it anyway, the run comes back as a failure carrying the reason
    rather than raising out of the sweep.
    """
    assert not SAIL.supports("indirect")
    record = run_one(
        "sail", "indirect", 25, SAIL.sample(1, 0)[0], seed=0, max_iter=10, suite=SAIL
    )
    assert record.status == FAILED
    assert record.extra["error"] == "ValueError"


def test_studies_are_filtered_to_what_a_suite_can_run():
    """A study needs at least two levels the suite offers to be a comparison."""
    assert "transcription" in studies_for(BRACHISTOCHRONE)
    # The albatross offers only collocation levels, so varying the
    # transcription would compare one thing against itself.
    assert "transcription" not in studies_for(ALBATROSS)
    # The optimizer study uses each suite's own list.
    assert len(studies_for(ALBATROSS)["optimizer"]) == len(ALBATROSS.algorithms)


def test_an_unrunnable_study_says_so():
    with pytest.raises(ValueError, match="cannot run the 'transcription' study"):
        run_study("transcription", suite=ALBATROSS)


def test_unknown_suites_and_studies_are_rejected():
    with pytest.raises(ValueError, match="Unknown suite"):
        get_suite("pelican")
    with pytest.raises(ValueError, match="Unknown study"):
        run_study("vibes", suite=SAIL)


def test_a_sweep_is_the_size_it_says():
    levels = studies_for(SAIL)["transcription"]
    assert sweep_size(SAIL, levels) == len(levels) * len(SAIL.sizes) * SAIL.starts


def test_the_albatross_records_that_it_scores_against_itself():
    """Its profile measures reproducibility, not correctness.

    The brachistochrone is scored against a cycloid and the sail against the
    ideal speed made good. The albatross has no closed form, so its target is
    a converged run, and a reader should know that before trusting the
    profile.
    """
    assert ALBATROSS.scored_against_itself
    assert not BRACHISTOCHRONE.scored_against_itself
    assert not SAIL.scored_against_itself


def test_a_small_sweep_runs_end_to_end(tmp_path):
    """Two starts of one cell, cached, for each of the cheap suites."""
    records = run_study(
        "optimizer",
        sizes=(ALBATROSS.sizes[0],),
        n_starts=2,
        max_iter=200,
        cache=tmp_path,
        suite=ALBATROSS,
    )
    assert len(records) == 2 * len(ALBATROSS.algorithms)
    assert any(record.status == CONVERGED for record in records)
    # Cached files are named by suite, so two problems never collide.
    assert all(path.name.startswith("albatross_") for path in tmp_path.glob("*.json"))
    # Re-running reads the cache back rather than redoing the work.
    again = run_study(
        "optimizer",
        sizes=(ALBATROSS.sizes[0],),
        n_starts=2,
        max_iter=200,
        cache=tmp_path,
        suite=ALBATROSS,
    )
    assert [r.objective for r in again] == [r.objective for r in records]


@pytest.mark.parametrize("name", sorted(SUITES))
def test_every_suite_can_be_built_without_a_formulation(name):
    """Stopping short of the formulation leaves the disciplines uncompiled.

    That is what the timing measurement needs: building the GEMSEO scenario
    is what compiles them, so an eager measurement has to come from a
    scenario that never went through it.
    """
    suite = get_suite(name)
    level = "collocation" if suite.supports("collocation") else suite.levels[0]
    scenario = suite.build_scenario(
        level, suite.sizes[0], suite.sample(1, 0)[0], suite.variants[0]
    )
    assert scenario.problem.design_variables
    assert scenario.disciplines
