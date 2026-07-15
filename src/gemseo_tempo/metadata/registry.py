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
"""The :class:`.VariableRegistry` mapping names to variables."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING
from typing import Any

import pint
import yaml

from gemseo_tempo.metadata.units import UREG
from gemseo_tempo.metadata.variable import Variable

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator
    from pathlib import Path


class VariableRegistry(Mapping[str, Variable]):
    """A read-mostly mapping from grammar names to :class:`.Variable`.

    Registries travel with models, drive the generated documentation
    (Sprint 13), and build uncertain spaces (Sprint 12). Registration
    validates units eagerly so errors surface at declaration time, not
    deep inside a scenario assembly.
    """

    def __init__(self, variables: Iterable[Variable] = ()) -> None:
        """Initialize the registry, optionally from an iterable of variables.

        Parameters
        ----------
        variables
            The initial variables.
        """
        self._variables: dict[str, Variable] = {}
        self.register_many(variables)

    def __getitem__(self, name: str) -> Variable:
        return self._variables[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._variables)

    def __len__(self) -> int:
        return len(self._variables)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({sorted(self._variables)})"

    def register(self, var: Variable, *, overwrite: bool = False) -> None:
        """Register a variable.

        Parameters
        ----------
        var
            The variable.
        overwrite
            Whether to silently replace an existing variable of the same
            name.

        Raises
        ------
        ValueError
            If the name is already registered and ``overwrite`` is false,
            or if the unit is not pint-parsable.
        """
        if var.name in self._variables and not overwrite:
            raise ValueError(
                f"Variable {var.name!r} is already registered;"
                " pass overwrite=True to replace it."
            )
        try:
            UREG.Unit(var.unit)
        except pint.errors.UndefinedUnitError as error:
            raise ValueError(
                f"Variable {var.name!r} has an unknown unit {var.unit!r}."
            ) from error
        self._variables[var.name] = var

    def register_many(self, variables: Iterable[Variable]) -> None:
        """Register several variables.

        Parameters
        ----------
        variables
            The variables.
        """
        for var in variables:
            self.register(var)

    @classmethod
    def from_yaml(cls, path: str | Path) -> VariableRegistry:
        """Load a registry from a YAML file written by :meth:`to_yaml`.

        Parameters
        ----------
        path
            The path to the YAML file.

        Returns
        -------
        VariableRegistry
            The registry.
        """
        with open(path, encoding="utf-8") as stream:
            records = yaml.safe_load(stream)
        return cls(Variable.from_record(record) for record in records)

    def to_yaml(self, path: str | Path) -> None:
        """Write the registry to a YAML file.

        Parameters
        ----------
        path
            The path to the YAML file.
        """
        with open(path, "w", encoding="utf-8") as stream:
            yaml.safe_dump(self.to_records(), stream, sort_keys=False)

    def subset(self, names: Iterable[str]) -> VariableRegistry:
        """Return a new registry restricted to the given names.

        Parameters
        ----------
        names
            The names to keep.

        Returns
        -------
        VariableRegistry
            The restricted registry.

        Raises
        ------
        KeyError
            If a name is not registered.
        """
        return self.__class__(self._variables[name] for name in names)

    def filter(self, tag: str) -> VariableRegistry:
        """Return a new registry with the variables carrying a tag.

        Parameters
        ----------
        tag
            The tag to filter on.

        Returns
        -------
        VariableRegistry
            The filtered registry.
        """
        return self.__class__(
            var for var in self._variables.values() if tag in var.tags
        )

    def to_records(self) -> list[dict[str, Any]]:
        """Return the variables as YAML-serializable records.

        Returns
        -------
        list
            One record per variable, in registration order.
        """
        return [var.to_record() for var in self._variables.values()]
