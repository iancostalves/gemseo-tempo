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
"""Tests of gemseo_tempo.statics.model."""

from __future__ import annotations

import pytest

from gemseo_tempo.metadata import Variable
from gemseo_tempo.metadata import VariableRegistry
from gemseo_tempo.statics import AutoModel
from gemseo_tempo.statics import JAXModel


def make_doubler(variables=None):
    return JAXModel(
        function=lambda inputs: {"a": 2.0 * inputs["c1"]},
        input_names=["c1"],
        output_names=["a"],
        default_inputs={"c1": 1.0},
        name="doubler",
        variables=variables,
    )


class Wing(AutoModel):
    def _jax_func(self, span=30.0, area=120.0):
        aspect_ratio = span**2 / area
        return aspect_ratio  # noqa: RET504


def test_jax_model_execution():
    model = make_doubler()
    assert model.discipline.name == "doubler"
    output = model.discipline.jax_out_func({"c1": 3.0})
    assert output["a"] == 6.0
    assert model.variables is None


def test_auto_model_grammar_detection():
    model = Wing("wing")
    assert set(model.discipline.input_grammar.names) == {"span", "area"}
    assert set(model.discipline.output_grammar.names) == {"aspect_ratio"}
    output = model.discipline.jax_out_func({"span": 30.0, "area": 120.0})
    assert output["aspect_ratio"] == pytest.approx(7.5)


def test_variables_registry_accepted():
    registry = VariableRegistry(
        [
            Variable(name="c1", full_name="Constant one"),
            Variable(name="a", full_name="Output a"),
        ]
    )
    model = make_doubler(variables=registry)
    assert model.variables is registry


def test_variables_registry_unknown_name_raises():
    registry = VariableRegistry([Variable(name="typo", full_name="Bad name")])
    with pytest.raises(ValueError, match=r"\['typo'\].*neither inputs nor outputs"):
        make_doubler(variables=registry)


def test_auto_model_with_registry():
    registry = VariableRegistry(
        [
            Variable(name="span", full_name="Wing span", unit="m"),
            Variable(name="aspect_ratio", full_name="Aspect ratio"),
        ]
    )
    model = Wing("wing", variables=registry)
    assert model.variables["span"].unit == "m"
