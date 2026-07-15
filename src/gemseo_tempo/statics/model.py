# Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
# Copyright 2025 IRT Saint Exupéry, https://www.irt-saintexupery.com
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
"""Model wrappers around gemseo-jax disciplines.

Ported from NOADS ``noads/core/model.py`` (LGPL-3.0) and extended with an
optional :class:`.VariableRegistry` carrying per-variable metadata.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from gemseo_jax.auto_jax_discipline import AutoJAXDiscipline
from gemseo_jax.jax_discipline import JAXDiscipline

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from gemseo_jax.jax_discipline import DataType
    from gemseo_jax.jax_discipline import NumberLike

    from gemseo_tempo.metadata.registry import VariableRegistry


class Model:
    """Base class of all gemseo-tempo models, wrapping a gemseo-jax discipline.

    A model holds a single :class:`gemseo_jax.jax_discipline.JAXDiscipline`,
    i.e. a JAX-traceable function with named inputs and outputs that GEMSEO
    can execute, couple with other disciplines, differentiate with automatic
    differentiation, and compile just-in-time. A model optionally carries a
    :class:`.VariableRegistry` describing its variables; registry names are
    validated against the discipline grammar at construction.
    """

    discipline: JAXDiscipline
    """The wrapped discipline."""

    variables: VariableRegistry | None
    """The metadata of the model variables, if any."""

    def __init__(
        self,
        discipline: JAXDiscipline,
        variables: VariableRegistry | None = None,
    ) -> None:
        """Wrap an existing discipline.

        Parameters
        ----------
        discipline
            An instantiated gemseo-jax discipline to wrap.
        variables
            The metadata of the model variables; every registered name must
            be an input or an output of the discipline.

        Raises
        ------
        ValueError
            If a registered variable name is neither an input nor an output
            of the discipline.
        """
        self.discipline = discipline
        self.variables = variables
        if variables is not None:
            grammar_names = set(discipline.input_grammar.names).union(
                discipline.output_grammar.names
            )
            unknown = sorted(set(variables) - grammar_names)
            if unknown:
                raise ValueError(
                    f"Variables {unknown} of model {discipline.name!r} are"
                    " neither inputs nor outputs of its discipline."
                )


class JAXModel(Model):
    """Model built from a dict-based JAX function.

    The function maps a dictionary of named input arrays to a dictionary of
    named output arrays. Input and output names are given explicitly, which
    allows functions whose signature cannot be introspected (e.g. closures
    or partial applications). For plain functions with named arguments,
    prefer :class:`AutoModel`, which detects names automatically.
    """

    def __init__(
        self,
        function: Callable[[DataType], DataType],
        input_names: Sequence[str],
        output_names: Sequence[str],
        default_inputs: Mapping[str, NumberLike],
        name: str,
        variables: VariableRegistry | None = None,
    ) -> None:
        """Build the discipline from the function and names.

        Parameters
        ----------
        function
            The JAX function, mapping a dict of input arrays keyed by
            ``input_names`` to a dict of output arrays keyed by
            ``output_names``.
        input_names
            The names of the inputs.
        output_names
            The names of the outputs.
        default_inputs
            The default value of each input, keyed by name; also determines
            the shapes used for tracing and compilation.
        name
            The name of the discipline.
        variables
            The metadata of the model variables.
        """
        discipline = JAXDiscipline(
            function=function,
            input_names=input_names,
            output_names=output_names,
            default_inputs=default_inputs,
            name=name,
        )
        super().__init__(discipline, variables)


class AutoModel(Model):
    """Model whose inputs and outputs are auto-detected from a JAX function.

    Subclasses implement :meth:`_jax_func` with named arguments; the
    argument names become the discipline input names and the returned
    variables the output names (via
    :class:`gemseo_jax.auto_jax_discipline.AutoJAXDiscipline`). This is the
    most common base class for gemseo-tempo models, as it removes the
    boilerplate of declaring grammars by hand.
    """

    discipline: AutoJAXDiscipline
    """The wrapped discipline."""

    def __init__(
        self,
        name: str,
        variables: VariableRegistry | None = None,
    ) -> None:
        """Create the discipline from the subclass JAX function.

        Parameters
        ----------
        name
            The name of the discipline.
        variables
            The metadata of the model variables.
        """
        discipline = AutoJAXDiscipline(name=name, function=self._jax_func)
        super().__init__(discipline, variables)

    @abstractmethod
    def _jax_func(self, *args: Sequence[NumberLike]) -> tuple[NumberLike]:
        """Compute the outputs; wrapped by the discipline.

        Argument names define the discipline inputs and their default
        values; the names of the returned variables define the outputs. The
        function must be JAX-traceable (pure, no Python side effects on
        traced values).
        """
