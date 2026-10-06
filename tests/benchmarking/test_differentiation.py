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
"""Tests of the forward, reverse and finite-difference measurements."""

from __future__ import annotations

import jax.numpy as jnp
import matplotlib
import numpy
import pytest

from gemseo_tempo.benchmarking.differentiation import CHECKPOINTED
from gemseo_tempo.benchmarking.differentiation import FD
from gemseo_tempo.benchmarking.differentiation import FORWARD
from gemseo_tempo.benchmarking.differentiation import MODES
from gemseo_tempo.benchmarking.differentiation import REVERSE
from gemseo_tempo.benchmarking.differentiation import as_records
from gemseo_tempo.benchmarking.differentiation import batched_differences
from gemseo_tempo.benchmarking.differentiation import best_step
from gemseo_tempo.benchmarking.differentiation import chain_label
from gemseo_tempo.benchmarking.differentiation import flat_function
from gemseo_tempo.benchmarking.differentiation import measure_modes
from gemseo_tempo.benchmarking.differentiation import plot_compile_times
from gemseo_tempo.benchmarking.differentiation import plot_mode_radar
from gemseo_tempo.benchmarking.differentiation import plot_mode_sweep
from gemseo_tempo.benchmarking.differentiation import plot_stage_modes
from gemseo_tempo.benchmarking.differentiation import radar_scores

matplotlib.use("Agg")

WEIGHTS = jnp.asarray(numpy.random.default_rng(0).normal(size=(3, 5)))


def toy(x):
    """Three outputs of five inputs, with Jacobian W diag(cos x)."""
    return WEIGHTS @ jnp.sin(x)


@pytest.mark.parametrize("jit", [True, False])
def test_every_mode_measures_the_same_jacobian(jit):
    point = jnp.linspace(0.1, 0.5, 5)
    measures = measure_modes(dict.fromkeys(MODES, toy), point, jit=jit, repeats=1)
    assert [m.mode for m in measures] == list(MODES)
    by_mode = {m.mode: m for m in measures}
    for measure in measures:
        assert (measure.n_inputs, measure.n_outputs) == (5, 3)
        assert measure.jit is jit
        assert measure.seconds > 0.0
        assert measure.memory > 0
        assert (measure.compile_jacobian > 0.0) is jit
    assert by_mode[FORWARD].error < 1e-14
    assert by_mode[REVERSE].error < 1e-14
    assert by_mode[CHECKPOINTED].error < 1e-14
    assert 1e-12 < by_mode[FD].error < 1e-6
    assert by_mode[FD].variant in ("batched", "sequential")
    assert by_mode[FD].step in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9)


def test_a_mode_left_out_is_not_measured_and_has_no_reference():
    measures = measure_modes({FD: toy}, jnp.ones(5), repeats=1)
    assert [m.mode for m in measures] == [FD]
    assert numpy.isnan(measures[0].error)


def test_the_best_mode_on_each_axis_sits_at_the_rim():
    records = [
        {
            "mode": FD,
            "compile_function": 1.0,
            "compile_jacobian": 1.0,
            "seconds": 10.0,
            "error": 1e-6,
            "memory": 100,
        },
        {
            "mode": FORWARD,
            "compile_function": 1.0,
            "compile_jacobian": 3.0,
            "seconds": 1.0,
            "error": 0.0,
            "memory": 400,
        },
        {
            "mode": REVERSE,
            "compile_function": 1.0,
            "compile_jacobian": 3.0,
            "seconds": 2.0,
            "error": 0.0,
            "memory": 800,
        },
    ]
    scores = radar_scores(records)
    # Compilation, execution, precision, memory.
    assert scores[FD] == pytest.approx([1.0, 0.2, 0.2, 1.0])
    assert scores[FORWARD][1] == pytest.approx(1.0)
    assert scores[FORWARD][2] == scores[REVERSE][2] == pytest.approx(1.0)
    assert scores[REVERSE][3] == pytest.approx(0.2)


def test_the_figures_draw_the_sweep_the_compilation_and_the_radar():
    records = []
    for size in (2, 4, 8):
        point = jnp.linspace(0.1, 0.5, size)

        def function(x):
            return jnp.sin(x) * jnp.sum(x)

        for jit in (True, False):
            records += as_records(
                measure_modes(dict.fromkeys(MODES, function), point, jit, repeats=1)
            )
    sweep = plot_mode_sweep(records, "n_inputs")
    assert len(sweep.lines) == 8
    assert sweep.get_xscale() == "log"
    compiled = plot_compile_times(records, "n_outputs")
    assert len(compiled.lines) == 4
    radar = plot_mode_radar([r for r in records if r["jit"] and r["n_inputs"] == 8])
    assert radar.name == "polar"
    assert len(radar.lines) == 4


def test_a_sweep_with_a_mode_missing_skips_it():
    records = as_records(measure_modes({REVERSE: toy}, jnp.ones(5), repeats=1))
    axes = plot_mode_sweep(records, "n_outputs")
    assert len(axes.lines) == 1
    assert len(plot_compile_times(records, "n_inputs").lines) == 2


def test_a_problem_becomes_a_function_of_one_vector():
    from gemseo_tempo.problems.brachistochrone import create_wire_scenario

    problem = create_wire_scenario(size=4).problem
    function, point = flat_function(
        problem, ["control.gamma", "t_f"], ["descent_time", "constraint.final.x"]
    )
    assert point.shape == (6,)
    value = function(point)
    assert value.shape == (2,)
    expected = problem.jax_function(dict(problem.input_defaults))["descent_time"]
    assert float(value[0]) == pytest.approx(float(jnp.ravel(expected)[0]))


def test_the_figures_draw_on_given_axes():
    import matplotlib.pyplot as plt

    records = as_records(
        measure_modes(dict.fromkeys(MODES, toy), jnp.ones(5), repeats=1)
    )
    figure = plt.figure()
    flat = figure.add_subplot(1, 3, 1)
    timing = figure.add_subplot(1, 3, 2)
    polar = figure.add_subplot(1, 3, 3, projection="polar")
    assert plot_mode_sweep(records, "n_inputs", axes=flat) is flat
    assert plot_compile_times(records, "n_inputs", axes=timing) is timing
    assert plot_mode_radar(records, axes=polar) is polar


CHAINS = [
    {
        "modes": {"statics": REVERSE, "dynamics": FORWARD, "outputs": REVERSE},
        "seconds": 0.4,
        "memory": 5e4,
    },
    {
        "modes": {"statics": FORWARD, "dynamics": REVERSE, "outputs": REVERSE},
        "seconds": 2.6,
        "memory": 7e5,
    },
]


def test_a_chain_is_labelled_by_the_initial_of_each_stage_mode():
    assert chain_label(CHAINS[0]["modes"]) == "R·F·R"


@pytest.mark.parametrize("fd", [None, {"seconds": 3.3, "memory": 1.5e4}])
def test_the_chain_figure_draws_one_bar_per_choice(fd):
    axes = plot_stage_modes(CHAINS, fd)
    assert len(axes[0].patches) == len(CHAINS) + (fd is not None)
    labels = [label.get_text() for label in axes[0].get_yticklabels()]
    assert labels[:2] == ["R·F·R", "F·R·R"]
    # The cheapest chain is the green one, in time and in memory.
    assert axes[0].patches[0].get_facecolor() == matplotlib.colors.to_rgba("tab:green")
    assert axes[1].patches[0].get_facecolor() == matplotlib.colors.to_rgba("tab:green")


def test_the_chain_figure_draws_on_given_axes():
    import matplotlib.pyplot as plt

    _, given = plt.subplots(1, 2)
    assert plot_stage_modes(CHAINS, axes=given) is given


def test_batched_differences_are_forward_differences_in_one_call():
    point = jnp.linspace(0.1, 0.5, 5)
    expected = WEIGHTS * jnp.cos(point)
    jacobian = batched_differences(toy)(point, 1e-7)
    assert jacobian.shape == (3, 5)
    assert numpy.max(numpy.abs(jacobian - expected)) < 1e-6


def test_the_best_step_is_the_most_precise_one_tried():
    point = jnp.linspace(0.1, 0.5, 5)
    exact = numpy.asarray(WEIGHTS * jnp.cos(point))
    step, error = best_step(batched_differences(toy), point, exact)
    assert step in (1e-7, 1e-8)
    assert error < 1e-7
    # A coarse step alone is what it is.
    coarse, coarse_error = best_step(batched_differences(toy), point, exact, (1e-2,))
    assert coarse == 1e-2
    assert coarse_error > error


def test_finite_differences_alone_fall_back_on_the_given_step():
    measures = measure_modes({FD: toy}, jnp.ones(5), repeats=1, step=1e-5)
    assert measures[0].step == 1e-5
