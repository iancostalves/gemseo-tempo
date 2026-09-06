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
"""Turning declared distributions into a space to sample.

:class:`~gemseo_tempo.metadata.variable.Variable` has carried a
``distribution`` field since the metadata layer was written, and until now
nothing has read it. This is where it is read: a variable that declares a
distribution can be made uncertain, and one that does not cannot.

That refusal is deliberate. A variable with no declared distribution is one
nobody has thought about, and supplying a default would be a guess wearing a
distribution's clothes. Sampling it would produce numbers that look like
uncertainty quantification without being it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.algos.parameter_space import ParameterSpace

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo_tempo.metadata.registry import VariableRegistry

DISTRIBUTION_KEY = "name"
"""The key naming the distribution class inside a ``distribution`` mapping."""


def make_parameter_space(
    registry: VariableRegistry,
    names: Sequence[str],
    prefix: str = "",
) -> ParameterSpace:
    """Build a parameter space from the distributions a registry declares.

    Parameters
    ----------
    registry
        The variable metadata.
    names
        The raw variable names to make uncertain.
    prefix
        A grammar prefix to prepend to each name in the space, so a
        ``constant.`` input can be sampled under the name the discipline
        knows it by.

    Returns
    -------
    ParameterSpace
        The space, one random variable per name.

    Raises
    ------
    KeyError
        If a name is not in the registry.
    ValueError
        If a variable declares no distribution, or the declaration does not
        name one.
    """
    space = ParameterSpace()
    for name in names:
        if name not in registry:
            msg = f"Variable {name!r} is not in the registry."
            raise KeyError(msg)
        declared = registry[name].distribution
        if not declared:
            msg = (
                f"Variable {name!r} declares no distribution, so it cannot be"
                " made uncertain. Give it one in the registry rather than"
                " letting a default stand in for a decision nobody made."
            )
            raise ValueError(msg)
        parameters = dict(declared)
        distribution = parameters.pop(DISTRIBUTION_KEY, "")
        if not distribution:
            msg = (
                f"The distribution of {name!r} has no {DISTRIBUTION_KEY!r};"
                " it should name a GEMSEO distribution class."
            )
            raise ValueError(msg)
        space.add_random_variable(prefix + name, distribution, **parameters)
    return space


def uncertain_names(registry: VariableRegistry) -> list[str]:
    """Return every variable of a registry that declares a distribution.

    Parameters
    ----------
    registry
        The variable metadata.

    Returns
    -------
    list
        The names, in registry order.
    """
    return [name for name in registry if registry[name].distribution]


def nominal_values(registry: VariableRegistry, names: Sequence[str]) -> dict:
    """Return the mean of each named distribution.

    The point a Taylor expansion is taken about, and the value a
    deterministic design would use for the same parameter.

    Parameters
    ----------
    registry
        The variable metadata.
    names
        The variable names.

    Returns
    -------
    dict
        The mean of each, by name.
    """
    space = make_parameter_space(registry, names)
    return {name: float(space.distributions[name].mean[0]) for name in names}


__all__ = [
    "DISTRIBUTION_KEY",
    "make_parameter_space",
    "nominal_values",
    "uncertain_names",
]
