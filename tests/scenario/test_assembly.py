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
"""Monolithic vs split assembly equivalence tests."""

from __future__ import annotations

import jax.numpy as jnp
import numpy
import pytest
from gemseo_jax.jax_chain import JAXChain

from gemseo_tempo.scenario import DynamicScenario
from gemseo_tempo.transcriptions import SingleShooting


def make_scenario(lqr_pieces, assembly):
    return DynamicScenario(assembly=assembly, **lqr_pieces)


@pytest.fixture
def monolithic(lqr_pieces):
    return make_scenario(lqr_pieces, "monolithic")


@pytest.fixture
def split(lqr_pieces):
    return make_scenario(lqr_pieces, "split")


def test_discipline_counts(monolithic, split):
    assert len(monolithic.disciplines) == 1
    assert [d.name for d in split.disciplines] == ["statics", "dynamics", "outputs"]


def test_monolithic_vs_split_outputs(monolithic, split):
    """Identical objective, constraints, and trajectories to 1e-10."""
    mono = monolithic.disciplines[0]
    mono.execute()
    chain = JAXChain(split.disciplines)
    chain.execute()
    for name in monolithic.problem.output_names:
        mono_value = numpy.asarray(mono.local_data[name])
        split_value = numpy.asarray(chain.local_data[name])
        assert numpy.max(numpy.abs(mono_value - split_value)) < 1e-10, name


def test_monolithic_vs_split_jacobians(monolithic, split):
    """Identical d(cost)/d(control.u) and d(cost)/d(constant.k) to 1e-10."""
    mono = monolithic.disciplines[0]
    mono.add_differentiated_inputs(["control.u", "constant.k"])
    mono.add_differentiated_outputs(["cost", "terminal"])
    mono.linearize()
    # Reverse mode: the chain re-traces the composed function, and AUTO
    # would pick forward AD, incompatible with the Recursive adjoint.
    chain = JAXChain(
        split.disciplines,
        differentiation_method=JAXChain.DifferentiationMethod.REVERSE,
    )
    chain.add_differentiated_inputs(["control.u", "constant.k"])
    chain.add_differentiated_outputs(["cost", "terminal"])
    chain.linearize()
    for output in ("cost", "terminal"):
        for input_ in ("control.u", "constant.k"):
            difference = numpy.max(
                numpy.abs(
                    numpy.asarray(mono.jac[output][input_])
                    - numpy.asarray(chain.jac[output][input_])
                )
            )
            assert difference < 1e-10, (output, input_)


def test_split_check_jacobian(split):
    """check_jacobian green on each split discipline."""
    for discipline in split.disciplines:
        assert discipline.check_jacobian(threshold=1e-3, step=1e-7), discipline.name


def test_unknown_assembly(lqr_pieces):
    with pytest.raises(ValueError, match="Unknown assembly"):
        make_scenario(lqr_pieces, "modular")


def test_forward_mode_with_direct_adjoint(lqr_pieces):
    """Forward AD works when the integrator uses the Direct adjoint, and
    matches the reverse-mode jacobian."""
    from gemseo_tempo.dynamics import DiffraxIntegrator

    forward_pieces = dict(lqr_pieces)
    forward_pieces["integrator"] = DiffraxIntegrator(
        rtol=1e-12, atol=1e-12, n_save=21, adjoint="Direct"
    )
    forward = DynamicScenario(
        assembly="monolithic",
        differentiation_method="forward",
        **forward_pieces,
    )
    reverse = make_scenario(lqr_pieces, "monolithic")
    forward_disc = forward.disciplines[0]
    reverse_disc = reverse.disciplines[0]
    for discipline in (forward_disc, reverse_disc):
        discipline.add_differentiated_inputs(["control.u", "constant.k"])
        discipline.add_differentiated_outputs(["cost"])
        discipline.linearize()
    forward_jac = numpy.asarray(forward_disc.jac["cost"]["control.u"])
    reverse_jac = numpy.asarray(reverse_disc.jac["cost"]["control.u"])
    # Relative, and not tight. The two runs use different adjoints, Direct
    # against RecursiveCheckpoint, so they do not take the same steps, and
    # the solver's error control bounds the solution rather than its
    # derivative. An absolute 1e-9 held on one platform and failed on
    # another by 3.6e-8, which said more about the step sequence than about
    # the derivative. What this is really checking is that forward mode
    # computes the same jacobian at all, and any real error there would be
    # of order one rather than of order 1e-8.
    difference = numpy.max(numpy.abs(forward_jac - reverse_jac))
    assert difference / numpy.max(numpy.abs(reverse_jac)) < 1e-6


def test_forward_mode_rejects_recursive_adjoint(lqr_pieces):
    """Forward AD with the default Recursive adjoint is rejected up front."""
    with pytest.raises(ValueError, match="does not support 'forward'"):
        DynamicScenario(
            assembly="monolithic",
            differentiation_method="forward",
            **lqr_pieces,
        )


def test_split_requires_stage_pieces(lqr_pieces):
    class NoSplitShooting(SingleShooting):
        def transcribe(self, *args, **kwargs):
            problem = super().transcribe(*args, **kwargs)
            problem.dynamics = None
            return problem

    with pytest.raises(ValueError, match="does not support the split"):
        DynamicScenario(assembly="split", transcription=NoSplitShooting(), **lqr_pieces)


def test_split_without_statics(lqr_pieces):
    """No statics chain: the split assembly has two disciplines."""
    pieces = dict(lqr_pieces)
    pieces["statics"] = None
    pieces["constants"] = {"c": 1.0}
    scenario = DynamicScenario(assembly="split", **pieces)
    assert [d.name for d in scenario.disciplines] == ["dynamics", "outputs"]
    chain = JAXChain(scenario.disciplines)
    chain.execute()
    assert float(chain.local_data["cost"][0]) > 0.0


def test_lagrange_augmentation_matches_trapezoid(lqr_pieces):
    """The quadrature state matches fine-grid trapezoid to 1e-6."""
    from gemseo_tempo.dynamics import DiffraxIntegrator

    pieces = dict(lqr_pieces)
    pieces["integrator"] = DiffraxIntegrator(rtol=1e-12, atol=1e-12, n_save=2001)
    scenario = DynamicScenario(assembly="monolithic", **pieces)
    discipline = scenario.disciplines[0]
    discipline.execute()
    trapezoid = jnp.trapezoid(
        jnp.asarray(discipline.local_data["trajectory.y"]),
        x=jnp.asarray(discipline.local_data["times"]),
    )
    assert float(discipline.local_data["cost"][0]) == pytest.approx(
        float(trapezoid), abs=1e-6
    )
