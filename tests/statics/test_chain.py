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
"""Tests of gemseo_tempo.statics.chain."""

from __future__ import annotations

import itertools

import jax
import pytest

from gemseo_tempo.statics import JAXModel
from gemseo_tempo.statics import StaticsChain

CONSTANTS = ["c1", "c2"]


def make_models():
    """Three models: A(c1)->a, B(a,c2)->b, C(b,u)->d with u non-constant."""
    model_a = JAXModel(
        function=lambda inputs: {"a": 2.0 * inputs["c1"]},
        input_names=["c1"],
        output_names=["a"],
        default_inputs={"c1": 1.0},
        name="A",
    )
    model_b = JAXModel(
        function=lambda inputs: {"b": inputs["a"] + inputs["c2"]},
        input_names=["a", "c2"],
        output_names=["b"],
        default_inputs={"a": 2.0, "c2": 3.0},
        name="B",
    )
    model_c = JAXModel(
        function=lambda inputs: {"d": inputs["b"] * inputs["u"]},
        input_names=["b", "u"],
        output_names=["d"],
        default_inputs={"b": 5.0, "u": 1.0},
        name="C",
    )
    return model_a, model_b, model_c


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_split_permutation_invariance(order):
    """The fixed-point split classifies all 6 declaration orders identically."""
    models = make_models()
    permuted = [models[index] for index in order]
    statics, dynamics_side = StaticsChain.split(permuted, CONSTANTS)
    assert {model.discipline.name for model in statics} == {"A", "B"}
    assert [model.discipline.name for model in dynamics_side] == ["C"]


def test_split_out_of_order_defeats_single_pass():
    """B declared before A: a single pass would misclassify B."""
    model_a, model_b, _ = make_models()
    statics, dynamics_side = StaticsChain.split([model_b, model_a], CONSTANTS)
    assert {model.discipline.name for model in statics} == {"A", "B"}
    assert dynamics_side == []


def test_chain_matches_manual_composition():
    model_a, model_b, _ = make_models()
    chain = StaticsChain([model_b, model_a], CONSTANTS)
    outputs = chain({"c1": 5.0, "c2": 3.0})
    # b = 2 * c1 + c2
    assert outputs["a"] == 10.0
    assert outputs["b"] == 13.0


def test_chain_uses_defaults_for_missing_inputs():
    model_a, model_b, _ = make_models()
    chain = StaticsChain([model_a, model_b], CONSTANTS)
    outputs = chain({"c1": 5.0})  # c2 falls back to its default, 3.0
    assert outputs["b"] == 13.0


def test_chain_rejects_non_constant_fed_models():
    model_a, model_b, model_c = make_models()
    with pytest.raises(ValueError, match=r"\['C'\] are not constant-fed"):
        StaticsChain([model_a, model_b, model_c], CONSTANTS)


def test_chain_is_jittable():
    model_a, model_b, _ = make_models()
    chain = StaticsChain([model_a, model_b], CONSTANTS)
    traces = []

    @jax.jit
    def run(inputs):
        traces.append(1)
        return chain(inputs)

    first = run({"c1": 5.0, "c2": 3.0})
    second = run({"c1": 6.0, "c2": 3.0})
    assert len(traces) == 1
    assert first["b"] == 13.0
    assert second["b"] == 15.0


def test_chain_is_gradable_end_to_end():
    model_a, model_b, _ = make_models()
    chain = StaticsChain([model_a, model_b], CONSTANTS)

    def objective(c1):
        return chain({"c1": c1, "c2": 3.0})["b"]

    # b = 2 * c1 + c2, so db/dc1 = 2.
    assert jax.grad(objective)(5.0) == 2.0
