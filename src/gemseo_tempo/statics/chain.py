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
"""The chain of constant-fed (statics) models.

A statics model is one whose every input is a declared constant or an
output of another statics model: it can run once per optimizer iteration,
ahead of the dynamics. :meth:`.StaticsChain.split` generalizes the NOADS
single-pass detection with a fixed-point iteration, since a single pass
misclassifies models declared out of dependency order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo_jax.jax_chain import JAXChain

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo_jax.jax_discipline import DataType

    from gemseo_tempo.statics.model import Model


class StaticsChain:
    """A jittable chain over constant-fed models.

    The chain composes into jitted programs: :meth:`__call__` delegates to
    the underlying :class:`gemseo_jax.jax_chain.JAXChain` JAX function, so
    it can be traced, differentiated, and compiled as part of a larger
    program (e.g. the monolithic scenario assembly).
    """

    chain: JAXChain
    """The underlying chain discipline, with couplings resolved by GEMSEO."""

    models: list[Model]
    """The chained models."""

    def __init__(
        self,
        models: Sequence[Model],
        constant_inputs: Sequence[str],
    ) -> None:
        """Build the chain, checking that every model is constant-fed.

        Parameters
        ----------
        models
            The models to chain, in any declaration order.
        constant_inputs
            The names of the declared constants.

        Raises
        ------
        ValueError
            If a model is not constant-fed; use :meth:`split` first to
            partition a mixed collection.
        """
        statics, dynamics_side = self.split(models, constant_inputs)
        if dynamics_side:
            names = sorted(model.discipline.name for model in dynamics_side)
            raise ValueError(
                f"Models {names} are not constant-fed: their inputs are"
                " neither declared constants nor outputs of statics models."
                " Use StaticsChain.split() to partition them out."
            )
        self.models = statics
        self.chain = JAXChain(
            [model.discipline for model in statics], name="statics_chain"
        )

    def __call__(self, inputs: DataType) -> DataType:
        """Evaluate the chain on named inputs.

        Missing inputs fall back to the discipline defaults, so the chain
        can be evaluated with only the inputs of interest (e.g. the design
        variables) inside a jitted program.

        Parameters
        ----------
        inputs
            The input arrays, keyed by name.

        Returns
        -------
        DataType
            All chain outputs, keyed by name.
        """
        data = {**self.chain.default_input_data, **inputs}
        return self.chain.jax_out_func(
            {name: data[name] for name in self.chain.input_grammar.names}
        )

    @classmethod
    def split(
        cls,
        models: Sequence[Model],
        constant_inputs: Sequence[str],
    ) -> tuple[list[Model], list[Model]]:
        """Partition models into (statics, dynamics-side).

        Fixed-point iteration on the constant-fed property: a model joins
        the statics side once all its inputs are declared constants or
        outputs of models already on the statics side, and the pass repeats
        until no model moves. The NOADS single pass misclassifies models
        declared out of dependency order; the fixed point is independent of
        declaration order.

        Parameters
        ----------
        models
            The models to partition, in any declaration order.
        constant_inputs
            The names of the declared constants.

        Returns
        -------
        tuple
            ``(statics, dynamics_side)``, each preserving declaration order.
        """
        constants = set(constant_inputs)
        statics: list[Model] = []
        static_outputs: set[str] = set()
        remaining = list(models)
        changed = True
        while changed:
            changed = False
            still_remaining = []
            for model in remaining:
                if all(
                    name in constants or name in static_outputs
                    for name in model.discipline.input_grammar.names
                ):
                    statics.append(model)
                    static_outputs.update(model.discipline.output_grammar.names)
                    changed = True
                else:
                    still_remaining.append(model)
            remaining = still_remaining
        return statics, remaining
