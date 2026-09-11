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
"""The stored results, checked against what the code would produce now.

Every figure in the documentation is drawn from results computed once by
``tools/generate_doc_results.py`` and committed, because solving inside a
documentation build would cost an hour of continuous integration and fail
whenever a solver had an off day.

The weakness of that arrangement is drift: the tool is run by hand, so the
stored answers can quietly stop being what the code produces. These tests
are the guard. They do not re-run the optimizations, which is the whole
point; they check the stored results against quantities cheap enough to
recompute on every test run, and against each other.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy
import pytest

from gemseo_tempo.problems.brachistochrone import cycloid_time
from gemseo_tempo.problems.brachistochrone import reference_time
from gemseo_tempo.problems.sail.model import REGISTRY as SAIL_REGISTRY
from gemseo_tempo.problems.sail.problem import LEGS
from gemseo_tempo.problems.sail.problem import straight_run_time
from gemseo_tempo.uncertainty import uncertain_names

DATA = Path(__file__).resolve().parents[1] / "docs" / "_data"
"""Where the stored results live."""

REGENERATE = "regenerate with `python tools/generate_doc_results.py`"
"""What to do about a result that is missing or has drifted."""


def _load(name: str):
    """Load a stored result, failing with the command that would make it."""
    path = DATA / name
    if not path.exists():
        pytest.fail(f"docs/_data/{name} is missing; {REGENERATE}")
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with numpy.load(path, allow_pickle=False) as archive:
        return dict(archive)


# --------------------------------------------------------- brachistochrone


def test_the_stored_optima_are_still_the_right_answers():
    """The one problem with an answer to check against.

    The ideal variant has a closed form, and the other two have stored
    high-accuracy references. A stored optimum that has drifted away from
    them is a figure about to tell a reader something untrue.
    """
    optima = _load("brachistochrone/variants.json")
    assert optima["ideal"]["time"] == pytest.approx(cycloid_time(), rel=1e-4)
    for variant in ("friction", "drag"):
        assert optima[variant]["time"] == pytest.approx(
            reference_time(variant), rel=1e-3
        ), f"the stored {variant} optimum has drifted; {REGENERATE}"
    for record in optima.values():
        assert record["feasible"], REGENERATE


def test_the_stored_cycloid_is_the_analytic_one():
    cycloid = _load("brachistochrone/cycloid.npz")
    assert float(cycloid["time"]) == pytest.approx(cycloid_time(), rel=1e-9)
    # It passes through the target it was built to reach.
    assert float(cycloid["x"][-1]) == pytest.approx(2.0, rel=1e-6)
    assert float(cycloid["y"][-1]) == pytest.approx(2.0, rel=1e-6)


def test_the_stored_trajectories_are_self_consistent():
    """Same length as their own time base, and reaching the target."""
    for variant in ("ideal", "friction", "drag"):
        arrays = _load(f"brachistochrone/variant_{variant}.npz")
        count = len(arrays["ts"])
        for name, series in arrays.items():
            assert len(series) == count, f"{variant}/{name} is a different length"
        assert float(arrays["x"][-1]) == pytest.approx(2.0, abs=1e-3)
        assert float(arrays["y"][-1]) == pytest.approx(2.0, abs=1e-3)


def test_the_convergence_study_refines_towards_the_cycloid():
    study = _load("brachistochrone/convergence.json")
    assert study["exact"] == pytest.approx(cycloid_time(), rel=1e-9)
    for label, errors in study["errors"].items():
        assert len(errors) == len(study["sizes"]), label
        assert all(error >= 0.0 for error in errors), label


def test_ad_beats_finite_differences_and_the_gap_grows():
    """The measurement the transcription guide's argument turns on.

    Not a story that could plausibly reverse: reverse-mode AD costs one
    pass regardless of dimension, central differences cost one extra pass
    per design variable, so growing the design and finding AD still
    faster, and by more, is the whole claim.
    """
    records = _load("brachistochrone/ad_vs_fd.json")
    sizes = [record["n_design"] for record in records]
    assert sizes == sorted(sizes), REGENERATE
    for record in records:
        assert record["ad_seconds"] < record["fd_seconds"], (
            f"AD was not faster at n_design={record['n_design']}; {REGENERATE}"
        )
    ratio = lambda record: record["fd_seconds"] / record["ad_seconds"]  # noqa: E731
    assert ratio(records[-1]) > ratio(records[0]), (
        f"the AD advantage did not grow with dimension; {REGENERATE}"
    )


# ------------------------------------------------------------------- sail


def test_the_stored_sail_baseline_is_still_what_the_polar_says():
    """The straight run is a direct evaluation, so it must still match.

    It is the number the optimized leg is measured against, and unlike the
    optimum it costs nothing to recompute, which makes it the sharpest
    available check that the stored sail results are current.
    """
    downwind = _load("sail/downwind.json")
    assert downwind["straight"] == pytest.approx(
        straight_run_time("downwind"), rel=1e-9
    ), f"the polar has changed since the sail results were stored; {REGENERATE}"
    assert list(downwind["mark"]) == list(LEGS["downwind"])
    # Gybing has to beat sailing straight at the mark, or the page's whole
    # argument is wrong.
    assert downwind["optimized"] < downwind["straight"]


def test_the_stored_track_ends_at_the_mark():
    track = _load("sail/track_downwind.npz")
    mark = LEGS["downwind"]
    assert float(track["x"][-1]) == pytest.approx(mark[0], abs=1.0)
    assert float(track["y"][-1]) == pytest.approx(mark[1], abs=1.0)


def test_the_channel_study_is_monotone():
    """A tighter channel cannot be quicker than a looser one.

    The feasible set of a looser limit contains that of a tighter one, so a
    table that says otherwise is reporting the optimizer rather than the
    physics. This is asserted because it has been false here before.
    """
    study = _load("sail/channel.json")
    widths = study["widths"]
    times = [record["time"] for record in widths]
    assert times == sorted(times), f"the channel study is not monotone; {REGENERATE}"
    for record in widths:
        assert record["across"] <= record["half_width"] + 1.0


def test_the_uncertain_variables_are_the_ones_the_registry_declares():
    """A study of variables the registry no longer has is a stale study."""
    estimators = _load("sail/estimators.json")
    assert estimators["reference"]["n_evaluations"] > 0
    assert uncertain_names(SAIL_REGISTRY) == ["wind_scale", "wind_veer"]


def test_the_estimators_are_scored_against_the_same_reference():
    estimators = _load("sail/estimators.json")
    reference = estimators["reference"]
    for record in estimators["estimates"]:
        # The error each one reports is the one the stored means imply.
        expected = (record["mean"] - reference["mean"]) / reference[
            "standard_deviation"
        ]
        assert record["mean_error"] == pytest.approx(expected, rel=1e-9)
        assert record["cost"] == record["n_evaluations"] + record["n_gradients"]


def test_the_margin_study_reaches_a_probabilistic_design():
    """Including the design the whole argument turns on."""
    study = _load("sail/margin.json")
    labels = [record["name"] for record in study["designs"]]
    assert "deterministic" in labels
    assert any(label.startswith("P(") for label in labels), (
        "no probabilistic design was found; " + REGENERATE
    )
    for record in study["designs"]:
        assert 0.0 <= record["reliability"] <= 1.0


# -------------------------------------------------------------- albatross


def test_the_albatross_cycle_closes():
    """A soaring cycle that does not return to its start is not a cycle.

    Not ``V``: the stored cycle is the ``energy`` mission's, which closes
    on height and attitude only and leaves speed open on purpose so any
    surplus is banked as airspeed (see the mission table in
    docs/problems/albatross.md) — asserting it here would fail on a
    correctly-optimized cycle, not a broken one.

    And ``psi`` closes only modulo a full turn: a loop comes back to the
    same direction of flight having gained ``2*pi``, which is exactly why
    ``PeriodicityConstraint`` takes an ``offsets`` mapping rather than
    demanding plain equality (see "Three things that had to be got right"
    on the same page).
    """
    cycle = _load("albatross/cycle_energy.npz")
    for name in ("h", "gamma"):
        first, last = float(cycle[name][0]), float(cycle[name][-1])
        assert last == pytest.approx(first, abs=1e-2), f"{name} does not close"
    first, last = float(cycle["psi"][0]), float(cycle["psi"][-1])
    residual = (last - first) % (2.0 * numpy.pi)
    assert min(residual, 2.0 * numpy.pi - residual) == pytest.approx(0.0, abs=1e-2), (
        "psi does not close, even modulo a full turn"
    )


def test_the_energy_sweep_crosses_break_even():
    """The finding the sweep exists for: below some wind, the cycle loses.

    The net has to change sign somewhere in the swept range, or the figure
    has nothing to show.
    """
    sweep = _load("albatross/energy_sweep.json")
    nets = [record["air_change"] for record in sweep]
    assert min(nets) < 0.0 < max(nets), f"no break-even in the sweep; {REGENERATE}"
    # And gradient work is what pays for it: zero wind extracts nothing.
    calm = next(record for record in sweep if record["w_ref"] == 0.0)
    assert calm["gradient_work"] == pytest.approx(0.0, abs=1e-6)


def test_the_albatross_uncertainty_study_is_current():
    from gemseo_tempo.problems.albatross.model import REGISTRY

    study = _load("albatross/uncertainty.json")
    assert study["sobol"]["names"] == uncertain_names(REGISTRY)
    assert study["screening"]["ranking"], REGENERATE
    assert study["speedup"]["speedup"] > 1.0
