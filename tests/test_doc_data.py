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

from gemseo_tempo.problems.albatross.model import REGISTRY as ALBATROSS_REGISTRY
from gemseo_tempo.problems.brachistochrone import REGISTRY as BRACHISTOCHRONE_REGISTRY
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


def test_every_transcription_reaches_the_cycloid():
    """The benchmark's reference is the closed form, and every solve meets it.

    Indirect shooting included: it solves the necessary conditions rather
    than the problem, and must land on the same curve.
    """
    study = _load("brachistochrone/transcriptions.json")
    assert study["reference"] == pytest.approx(cycloid_time(), rel=1e-9)
    names = {row["transcription"] for row in study["rows"]}
    assert names == {"shooting", "collocation", "indirect"}
    for row in study["rows"]:
        assert row["met"], row
        assert row["error"] < 1e-5, row


@pytest.mark.parametrize("problem", ["sail", "albatross"])
def test_the_direct_transcriptions_agree_where_there_is_no_closed_form(problem):
    """Shooting and collocation each converge, to answers within a few percent.

    The indirect row is recorded whether or not it converged, since where it
    does not the page says so and why.
    """
    study = _load(f"{problem}/transcriptions.json")
    for name in ("shooting", "collocation"):
        met = [
            row for row in study["rows"] if row["transcription"] == name and row["met"]
        ]
        assert met, f"no {name} solve converged; {REGENERATE}"
        assert max(row["error"] for row in met) < 0.05
    assert [row for row in study["rows"] if row["transcription"] == "indirect"]


def test_ad_beats_finite_differences_in_the_loop_and_the_gap_grows():
    """The measurement the robust design's argument turns on.

    Not a story that could plausibly reverse: a reverse pass costs a few
    evaluations whatever the size of the design, and finite differences
    cost one more per design variable, so refining the wire and finding
    AD still faster, and by more, is the whole claim.
    """
    records = _load("brachistochrone/robust.json")["gradient_cost"]
    sizes = [record["n_design"] for record in records]
    assert sizes == sorted(sizes), REGENERATE
    ratio = lambda record: record["fd_seconds"] / record["ad_seconds"]  # noqa: E731
    assert ratio(records[-1]) > ratio(records[0]), (
        f"the AD advantage did not grow with dimension; {REGENERATE}"
    )
    assert ratio(records[-1]) > 1.0


def test_the_robust_wire_is_the_same_by_either_gradient():
    """The gradient changes what the loop costs, not where it ends."""
    designs = _load("brachistochrone/robust.json")["designs"]
    assert all(design["met"] for design in designs.values()), REGENERATE
    assert designs["robust_fd"]["objective"] == pytest.approx(
        designs["robust"]["objective"], rel=1e-4
    )
    # And the loop runs several times faster with it, compilation apart.
    assert 3.0 * designs["robust"]["seconds"] < designs["robust_fd"]["seconds"]


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


REGISTRIES = {
    "brachistochrone": BRACHISTOCHRONE_REGISTRY,
    "sail": SAIL_REGISTRY,
    "albatross": ALBATROSS_REGISTRY,
}
"""The registry each problem's uncertain variables are read from."""


@pytest.mark.parametrize("problem", list(REGISTRIES))
def test_each_propagation_is_scored_against_its_reference(problem):
    """The errors, costs and variables the tables show are the stored ones.

    A study of variables the registry no longer has is a stale study; an
    error is the difference the stored means imply, in reference deviations;
    and a finite-difference gradient is charged the two evaluations per
    variable it takes, where a reverse pass is charged one.
    """
    study = _load(f"{problem}/propagation.json")
    assert study["names"] == uncertain_names(REGISTRIES[problem]), REGENERATE
    reference = study["reference"]
    dimension = len(study["names"])
    for name, record in study["estimates"].items():
        expected = (record["mean"] - reference["mean"]) / reference[
            "standard_deviation"
        ]
        assert record["mean_error"] == pytest.approx(expected, rel=1e-9)
        per_gradient = 2 * dimension if name.endswith("_fd") else 1
        assert record["model_calls"] == (
            record["n_evaluations"] + per_gradient * record["n_gradients"]
        )


@pytest.mark.parametrize("problem", list(REGISTRIES))
def test_vectorized_sampling_is_the_same_sample_only_faster(problem):
    estimates = _load(f"{problem}/propagation.json")["estimates"]
    vmapped, looped = estimates["sampling_vmapped"], estimates["sampling_looped"]
    assert vmapped["mean"] == pytest.approx(looped["mean"], rel=1e-9)
    assert vmapped["seconds"] < looped["seconds"]


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
    # A wider margin is a tighter channel, which cannot be sailed faster.
    margins = [
        record for record in study["designs"] if not record["name"].startswith("P(")
    ]
    times = [record["objective"] for record in margins]
    assert times == sorted(times), f"the margin study is not monotone; {REGENERATE}"
    # The page says the chance constraint reaches its target and lands where
    # the 50 m margin did.
    chance = study["designs"][-1]
    assert chance["reliability"] >= study["target"]
    fifty = next(r for r in margins if r["name"] == "margin 50 m")
    assert chance["objective"] == pytest.approx(fifty["objective"], abs=0.05)


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


def test_designing_the_wing_with_the_cycle_never_needs_more_wind():
    """Co-design contains both other routes, so it cannot do worse than either.

    Each is solved from the same guess, so a co-design that came out worse
    would mean the solver stopped short, not that sizing first is better.
    """
    strategies = _load("albatross/design_strategies.json")
    for shape in ("loop", "weave"):
        codesign = strategies[f"{shape}_codesign"]["minimum_wind"]
        for route in ("fixed", "sequential"):
            assert codesign <= strategies[f"{shape}_{route}"]["minimum_wind"] + 1e-3


def test_every_mission_met_its_constraints():
    missions = _load("albatross/missions.json")
    for key, mission in missions.items():
        if isinstance(mission, dict):
            assert mission["met"], key


def test_the_upwind_leg_is_slower_than_its_bound():
    """No track can beat the polar's best speed made good held all the way."""
    upwind = _load("sail/upwind.json")
    bound = _load("sail/vmg.json")["upwind"]["ideal_time"]
    assert upwind["met"]
    # One tack costs a few percent; a slower answer is a local optimum.
    assert bound <= upwind["optimized"] < 1.04 * bound


def test_the_re_flown_cycles_stay_on_their_nodes():
    """Collocation never integrates between its nodes; the re-flight does.

    Flying the optimal controls through a tight integrator is how the page
    measures the discretization error, so the stored departures must be
    small against the size of the cycle, and the flown cycle must still
    close, or the figures drawn from it are of a different cycle.
    """
    checks = dict(_load("albatross/dense_check.json"))
    for key, shape in _load("albatross/shapes.json").items():
        if "dense_check" in shape:
            checks[key] = shape["dense_check"]
    for name, check in checks.items():
        for state in ("x", "y", "h"):
            scale = max(check["extent"][state], 10.0)
            assert check["departure"][state] < 0.1 * scale, (name, state)
        assert abs(check["closure"]["h"]) < 1.0, name


@pytest.mark.parametrize("key", ["loop", "weave"])
def test_each_stored_shape_is_the_shape_it_is_named_after(key):
    """An O turns all the way round, and an S turns back.

    Each stored cycle is the best of its starts, and more than one start
    reaches it, so it is not a lucky one-off.
    """
    shape = _load("albatross/shapes.json")[key]
    cycle = _load(f"albatross/cycle_{key}_dense.npz")
    turned = float(cycle["psi"][-1] - cycle["psi"][0])
    assert turned == pytest.approx(2.0 * numpy.pi if key == "loop" else 0.0, abs=0.05)
    assert shape["minimum_wind"] == min(w for w in shape["winds"] if w is not None)
    assert shape["at_best"] >= 2, REGENERATE


@pytest.mark.parametrize("key", ["loop", "weave"])
def test_staying_put_costs_wind(key):
    """The nearer a cycle must end to where it started, the more wind it needs.

    Each step of the continuation towards a hover starts from the one
    before, so a step that needed less wind than the last would mean an
    earlier step stopped short of its optimum.
    """
    steps = _load("albatross/shapes.json")[key]["hover"]
    fractions = [step["fraction"] for step in steps]
    winds = [step["minimum_wind"] for step in steps]
    assert fractions == sorted(fractions, reverse=True)
    assert all(
        later >= 0.999 * earlier
        for earlier, later in zip(winds, winds[1:], strict=False)
    )


def test_hong_et_al_are_replicated_by_the_shape_they_flew():
    """Their heading closes without a turn: their cycle is an S.

    Solved as an S, the minimum wind lands within a few percent of theirs
    and has converged in the discretization.
    """
    study = _load("albatross/hong_2023.json")
    weave = study["weave"]
    assert abs(weave["difference"]) < 0.05, REGENERATE
    coarse, fine = (run["minimum_wind"] for run in weave["runs"])
    assert fine == pytest.approx(coarse, rel=0.01)


def test_the_robust_cycle_trades_its_forecast_energy_for_less_spread():
    """What the albatross page says the robust cycle buys, and with what."""
    study = _load("albatross/robust.json")
    deterministic, robust = (
        study["designs"]["deterministic"],
        study["designs"]["robust"],
    )
    assert robust["met"], REGENERATE
    assert robust["nominal"] < deterministic["nominal"]
    assert robust["std"] < 0.75 * deterministic["std"]
    assert robust["p05"] > deterministic["p05"] + 1.0
    cost = study["gradient_cost"]
    assert cost["ad_seconds"] < cost["fd_seconds"]
