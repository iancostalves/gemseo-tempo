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
"""Tests of the structure-aware collocation Jacobian."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy
import pytest
from gemseo_jax.jax_discipline import JAXDiscipline

from gemseo_tempo.dynamics import DiffraxIntegrator
from gemseo_tempo.dynamics import PiecewiseLinearControl
from gemseo_tempo.dynamics import StateSpace
from gemseo_tempo.outputs import Lagrange
from gemseo_tempo.scenario import DynamicScenario
from gemseo_tempo.scenario import OptimalControl
from gemseo_tempo.scenario.structured_discipline import StructuredJAXDiscipline
from gemseo_tempo.transcriptions import BoundarySpec
from gemseo_tempo.transcriptions import HermiteSimpson
from gemseo_tempo.transcriptions import SingleShooting

KNOTS = jnp.linspace(0.0, 1.0, 5)
DEFECT_OUTPUTS = ("constraint.defect.x", "constraint.defect.v")


def oscillator():
    """x' = v, v' = -x + u: two states so the blocks are genuinely 2x2."""
    return StateSpace(
        states=["x", "v"],
        controls=["u"],
        rhs=lambda t, s, u, theta: {"x": s["v"], "v": -s["x"] + u["u"]},
    )


def make_oscillator(sparse, num_segments=6, free_final_time=False):
    boundary = BoundarySpec(
        s0={"x": 0.0, "v": 0.0},
        fix_final={"x": True, "v": True},
        s1={"x": 1.0, "v": 0.0},
        free_final_time=free_final_time,
        t_f=1.0 if free_final_time else None,
    )
    return DynamicScenario(
        state_space=oscillator(),
        integrator=DiffraxIntegrator(dt0=0.1),
        controls=(
            PiecewiseLinearControl("u", KNOTS, default_params=0.3 * jnp.ones(5)),
        ),
        functionals=(Lagrange("energy", lambda t, s, u, theta: u["u"] ** 2),),
        boundary=boundary,
        transcription=HermiteSimpson(num_segments=num_segments, sparse_jacobian=sparse),
        t0=0.0,
        t1=1.0,
    )


def dense_defect_jacobian(problem, data):
    """Differentiate only the defect outputs with a dense transform."""
    return jax.jacrev(
        lambda inputs: {
            name: value
            for name, value in problem.jax_function(inputs).items()
            if name.startswith("constraint.defect.")
        }
    )(data)


# ---------------------------------------------------------------- exactness


@pytest.mark.parametrize("free_final_time", [False, True])
def test_blocks_match_the_dense_jacobian(free_final_time):
    """The correctness gate: same derivative, obtained more cheaply."""
    scenario = make_oscillator(True, free_final_time=free_final_time)
    problem = scenario.problem
    data = dict(problem.input_defaults)
    structured = problem.jacobian.function(data)
    dense = dense_defect_jacobian(problem, data)
    for output_name, row in structured.items():
        for input_name, block in row.items():
            expected = numpy.asarray(dense[output_name][input_name])
            actual = numpy.asarray(block)
            assert (
                numpy.max(numpy.abs(actual - expected.reshape(actual.shape))) < 1e-12
            ), (output_name, input_name)


def test_structured_jacobian_covers_the_node_variables():
    problem = make_oscillator(True, free_final_time=True).problem
    assert set(problem.jacobian.output_names) == set(DEFECT_OUTPUTS)
    assert set(problem.jacobian.input_names) == {
        "state.x",
        "state.v",
        "control.u",
        "t_f",
    }


def test_no_structured_jacobian_by_default():
    assert make_oscillator(False).problem.jacobian is None
    assert isinstance(make_oscillator(False).disciplines[0], JAXDiscipline)
    assert not isinstance(
        make_oscillator(False).disciplines[0], StructuredJAXDiscipline
    )


def test_single_segment_with_both_ends_fixed():
    """Every node of a state can be fixed, leaving it no design columns."""
    problem = make_oscillator(True, num_segments=1).problem
    structured = problem.jacobian.function(dict(problem.input_defaults))
    # Both boundary nodes are fixed, so the state contributes no columns.
    assert structured["constraint.defect.x"]["state.x"].shape == (1, 0)
    assert structured["constraint.defect.x"]["control.u"].shape == (1, 2)


# ------------------------------------------------------------- discipline


def test_structured_discipline_matches_the_plain_one():
    """Mixed outputs: defects structurally, the objective densely."""
    sparse = make_oscillator(True)
    plain = make_oscillator(False)
    assert isinstance(sparse.disciplines[0], StructuredJAXDiscipline)
    inputs = ["state.x", "state.v", "control.u"]
    outputs = ["energy", *DEFECT_OUTPUTS]
    for scenario in (sparse, plain):
        discipline = scenario.disciplines[0]
        discipline.add_differentiated_inputs(inputs)
        discipline.add_differentiated_outputs(outputs)
        discipline.linearize()
    for output_name in outputs:
        for input_name in inputs:
            assert (
                numpy.max(
                    numpy.abs(
                        numpy.asarray(
                            sparse.disciplines[0].jac[output_name][input_name]
                        )
                        - numpy.asarray(
                            plain.disciplines[0].jac[output_name][input_name]
                        )
                    )
                )
                < 1e-12
            ), (output_name, input_name)


def test_structured_discipline_with_only_defects_differentiated():
    """No dense output at all: the structured path supplies everything."""
    discipline = make_oscillator(True).disciplines[0]
    discipline.add_differentiated_inputs(["state.x", "state.v", "control.u"])
    discipline.add_differentiated_outputs(list(DEFECT_OUTPUTS))
    discipline.linearize()
    assert set(discipline.jac) == set(DEFECT_OUTPUTS)
    assert discipline.jac["constraint.defect.x"]["control.u"].shape == (6, 7)


def test_free_final_time_column_is_a_column_vector():
    """t_f has size one, so its Jacobian block must be a column."""
    discipline = make_oscillator(True, free_final_time=True).disciplines[0]
    discipline.add_differentiated_inputs(["state.x", "state.v", "control.u", "t_f"])
    discipline.add_differentiated_outputs(list(DEFECT_OUTPUTS))
    discipline.linearize()
    assert discipline.jac["constraint.defect.x"]["t_f"].shape == (6, 1)


def test_check_jacobian_on_the_structured_discipline():
    discipline = make_oscillator(True).disciplines[0]
    assert discipline.check_jacobian(threshold=1e-4, step=1e-7)


def test_uncovered_input_falls_back_to_the_dense_path():
    """A design constant is outside the declared structure."""
    space = StateSpace(
        states=["x"],
        controls=["u"],
        parameters=["a"],
        rhs=lambda t, s, u, theta: {"x": -theta["a"] * s["x"] + u["u"]},
    )

    def build(sparse):
        return DynamicScenario(
            state_space=space,
            integrator=DiffraxIntegrator(dt0=0.1),
            controls=(
                PiecewiseLinearControl("u", KNOTS, default_params=0.2 * jnp.ones(5)),
            ),
            functionals=(Lagrange("energy", lambda t, s, u, theta: u["u"] ** 2),),
            boundary=BoundarySpec(s0={"x": 0.0}, fix_final={"x": True}, s1={"x": 1.0}),
            transcription=HermiteSimpson(num_segments=5, sparse_jacobian=sparse),
            constants={"a": 0.5},
        )

    sparse, plain = build(True), build(False)
    assert not sparse.disciplines[0]._uses_structure(["state.x", "constant.a"])
    inputs = ["state.x", "control.u", "constant.a"]
    outputs = ["energy", "constraint.defect.x"]
    for scenario in (sparse, plain):
        discipline = scenario.disciplines[0]
        discipline.add_differentiated_inputs(inputs)
        discipline.add_differentiated_outputs(outputs)
        discipline.linearize()
    for output_name in outputs:
        for input_name in inputs:
            assert (
                numpy.max(
                    numpy.abs(
                        numpy.asarray(
                            sparse.disciplines[0].jac[output_name][input_name]
                        )
                        - numpy.asarray(
                            plain.disciplines[0].jac[output_name][input_name]
                        )
                    )
                )
                < 1e-12
            ), (output_name, input_name)


# ------------------------------------------------------------ optimization


def test_same_optimum_with_and_without_sparsity():
    """Only the derivative changes; the solution must not."""
    results = {}
    for label, sparse in (("sparse", True), ("dense", False)):
        space = StateSpace(
            states=["x"], controls=["u"], rhs=lambda t, s, u, theta: {"x": u["u"]}
        )
        scenario = DynamicScenario(
            state_space=space,
            integrator=DiffraxIntegrator(dt0=0.1),
            controls=(
                PiecewiseLinearControl("u", KNOTS, default_params=0.2 * jnp.ones(5)),
            ),
            functionals=(Lagrange("energy", lambda t, s, u, theta: u["u"] ** 2),),
            boundary=BoundarySpec(s0={"x": 0.0}, fix_final={"x": True}, s1={"x": 1.0}),
            transcription=HermiteSimpson(num_segments=8, sparse_jacobian=sparse),
        )
        gemseo_scenario = OptimalControl(
            scenario, bounds={"control.u": (-3.0, 3.0)}
        ).create_gemseo_scenario()
        gemseo_scenario.execute(algo_name="SLSQP", max_iter=100)
        results[label] = float(gemseo_scenario.optimization_result.f_opt)
    assert results["sparse"] == pytest.approx(1.0, abs=1e-8)
    assert results["sparse"] == pytest.approx(results["dense"], abs=1e-8)


# ---------------------------------------------------------------- scaling


def test_blocks_are_faster_than_the_dense_transform():
    """The point of the exercise, guarded with a generous margin."""
    scenario = make_oscillator(True, num_segments=100)
    problem = scenario.problem
    data = dict(problem.input_defaults)
    structured = jax.jit(problem.jacobian.function)
    dense = jax.jit(lambda inputs: dense_defect_jacobian(problem, inputs))

    def timed(function):
        jax.block_until_ready(function(data))
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            jax.block_until_ready(function(data))
            samples.append(time.perf_counter() - start)
        return min(samples)

    # Roughly 15x on this grid; a bare inequality cannot flake on a busy box.
    assert timed(structured) < timed(dense)


# ------------------------------------------------------------ shooting note


def test_shooting_jacobian_agrees_across_modes():
    """Single shooting has no exploitable sparsity, only a mode choice.

    The ODE couples every control parameter to the whole trajectory, so its
    Jacobian is structurally dense; what remains useful is picking the
    differentiation mode, which must not change the answer.
    """
    space = StateSpace(
        states=["x"], controls=["u"], rhs=lambda t, s, u, theta: {"x": u["u"]}
    )

    def build(mode, adjoint):
        return DynamicScenario(
            state_space=space,
            integrator=DiffraxIntegrator(
                rtol=1e-11, atol=1e-11, n_save=11, adjoint=adjoint
            ),
            controls=(
                PiecewiseLinearControl("u", KNOTS, default_params=0.2 * jnp.ones(5)),
            ),
            functionals=(Lagrange("energy", lambda t, s, u, theta: u["u"] ** 2),),
            boundary=BoundarySpec(s0={"x": 0.0}),
            transcription=SingleShooting(),
            differentiation_method=mode,
        )

    reverse = build("reverse", "RecursiveCheckpoint")
    forward = build("forward", "Direct")
    for scenario in (reverse, forward):
        discipline = scenario.disciplines[0]
        discipline.add_differentiated_inputs(["control.u"])
        discipline.add_differentiated_outputs(["energy"])
        discipline.linearize()
    assert (
        numpy.max(
            numpy.abs(
                numpy.asarray(reverse.disciplines[0].jac["energy"]["control.u"])
                - numpy.asarray(forward.disciplines[0].jac["energy"]["control.u"])
            )
        )
        < 1e-9
    )
    # Single shooting exposes no structure, so it supplies no Jacobian.
    assert reverse.problem.jacobian is None


def test_structured_jacobian_with_a_statics_chain():
    """The statics chain still feeds the parameters of the blocks."""
    from gemseo_tempo.statics import JAXModel
    from gemseo_tempo.statics import StaticsChain

    gain = JAXModel(
        function=lambda inputs: {"c": 2.0 * inputs["k"]},
        input_names=["k"],
        output_names=["c"],
        default_inputs={"k": 0.5},
        name="gain",
    )
    space = StateSpace(
        states=["x"],
        controls=["u"],
        parameters=["c"],
        rhs=lambda t, s, u, theta: {"x": -theta["c"] * s["x"] + u["u"]},
    )
    problem = HermiteSimpson(num_segments=5, sparse_jacobian=True).transcribe(
        StaticsChain([gain], ["k"]),
        space,
        None,
        (PiecewiseLinearControl("u", KNOTS, default_params=0.2 * jnp.ones(5)),),
        (Lagrange("energy", lambda t, s, u, theta: u["u"] ** 2),),
        (),
        BoundarySpec(s0={"x": 0.0}, fix_final={"x": True}, s1={"x": 1.0}),
        constants={"k": 0.5},
    )
    data = dict(problem.input_defaults)
    structured = problem.jacobian.function(data)
    dense = dense_defect_jacobian(problem, data)
    for output_name, row in structured.items():
        for input_name, block in row.items():
            actual = numpy.asarray(block)
            expected = numpy.asarray(dense[output_name][input_name])
            assert (
                numpy.max(numpy.abs(actual - expected.reshape(actual.shape))) < 1e-12
            ), (output_name, input_name)
