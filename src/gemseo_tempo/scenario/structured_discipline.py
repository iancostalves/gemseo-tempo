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
"""A discipline that uses a transcription's structure-aware Jacobian.

A dense automatic transform differentiates every output row by row, which
is wasteful when the transcription already knows that most of those rows
touch only a handful of inputs. Collocation defects are the motivating
case: segment ``k`` involves only nodes ``k`` and ``k+1``, so the whole
defect Jacobian is a band of small blocks.

This discipline splits the work. Outputs covered by the
:class:`.StructuredJacobian` are taken from it; every other differentiated
output still goes through an automatic transform, but only for those
outputs, which are the cheap ones (a scalar objective costs a single
reverse pass). The result is assembled into the arrays GEMSEO expects, so
nothing downstream has to know that anything special happened.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
from gemseo_jax.jax_discipline import JAXDiscipline
from numpy import asarray
from numpy import atleast_2d

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable

    from gemseo_tempo.transcriptions.base import StructuredJacobian


class StructuredJAXDiscipline(JAXDiscipline):
    """A JAX discipline with a transcription-supplied sparse Jacobian."""

    def __init__(
        self, *args, structured_jacobian: StructuredJacobian, **kwargs
    ) -> None:
        """Wrap the usual discipline arguments around a structured Jacobian.

        Parameters
        ----------
        *args
            The positional arguments of
            :class:`~gemseo_jax.jax_discipline.JAXDiscipline`.
        structured_jacobian
            The structure-aware Jacobian of part of the outputs.
        **kwargs
            The keyword arguments of the base discipline.
        """
        self._structured_jacobian = structured_jacobian
        # This discipline computes its Jacobian itself, so the compilation
        # that JAXDiscipline.compile_jit applies to the generic jacobian
        # function never reaches it. Compile the structured function here,
        # and cache a compiled transform per set of differentiated names.
        self._compiled_structure = jax.jit(structured_jacobian.function)
        self._compiled_dense: dict[tuple, Callable] = {}
        super().__init__(*args, **kwargs)

    def _uses_structure(self, input_names: Iterable[str]) -> bool:
        """Return whether the structured Jacobian covers the given inputs.

        Parameters
        ----------
        input_names
            The differentiated input names.

        Returns
        -------
        bool
            Whether every one of them is covered.
        """
        return set(input_names).issubset(self._structured_jacobian.input_names)

    def _dense_jacobian(self, output_names, input_names, data):
        """Differentiate a subset of the outputs with an automatic transform.

        Parameters
        ----------
        output_names
            The outputs to differentiate.
        input_names
            The inputs to differentiate with respect to.
        data
            The full input data.

        Returns
        -------
        dict
            ``{output: {input: array}}`` for the requested subset.
        """
        key = (tuple(output_names), tuple(input_names))
        transform = self._compiled_dense.get(key)
        if transform is None:

            def subset(differentiated, fixed):
                outputs = self.jax_out_func({**fixed, **differentiated})
                return {name: outputs[name] for name in output_names}

            transform = jax.jit(jax.jacrev(subset))
            self._compiled_dense[key] = transform
        return transform({name: data[name] for name in input_names}, data)

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        """Assemble the Jacobian from the structured and dense parts."""
        data = {name: self.io.data[name] for name in self.input_grammar.names}
        differentiated_inputs = list(self._differentiated_input_names)
        differentiated_outputs = list(self._differentiated_output_names)
        covered = set(self._structured_jacobian.output_names)

        if self._uses_structure(differentiated_inputs):
            structured_outputs = [
                name for name in differentiated_outputs if name in covered
            ]
        else:
            # An input outside the declared structure (a design constant, say)
            # would need derivatives this Jacobian does not provide, so the
            # dense transform handles the whole call rather than guessing.
            structured_outputs = []
        dense_outputs = [
            name for name in differentiated_outputs if name not in structured_outputs
        ]

        jacobian = {}
        if structured_outputs:
            structured = self._compiled_structure(data)
            for name in structured_outputs:
                jacobian[name] = {
                    input_name: structured[name][input_name]
                    for input_name in differentiated_inputs
                }
        if dense_outputs:
            jacobian.update(
                self._dense_jacobian(dense_outputs, differentiated_inputs, data)
            )

        sizes = {
            name: 1 if isinstance(value, float) else asarray(value).size
            for name, value in self.io.data.items()
        }
        self.jac = {
            output_name: {
                input_name: (
                    atleast_2d(asarray(block)).reshape((-1, 1))
                    if sizes[input_name] == 1
                    else atleast_2d(asarray(block))
                )
                for input_name, block in inner.items()
            }
            for output_name, inner in jacobian.items()
        }
        self._has_jacobian = True
