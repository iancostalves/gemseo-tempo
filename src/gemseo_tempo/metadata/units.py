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
"""Unit conversion at discipline boundaries.

Inside jitted code all values are canonical pint base units. Conversion
happens here, at the boundaries, with affine factors precomputed eagerly
as Python floats, so converted expressions stay jit-safe: pint objects
never meet JAX tracers.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING
from typing import Any

import pint

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

UREG = pint.UnitRegistry()
"""The unit registry shared by all of gemseo-tempo."""


@functools.cache
def _affine_coefficients(unit: str) -> tuple[float, float]:
    """Return the affine map from ``unit`` to its pint base units.

    Parameters
    ----------
    unit
        A pint-parsable unit.

    Returns
    -------
    tuple
        ``(factor, offset)`` such that ``base = value * factor + offset``.
        The offset is zero for all multiplicative units and nonzero only
        for offset units such as ``degC``.
    """
    offset = float(UREG.Quantity(0.0, unit).to_base_units().magnitude)
    at_one = float(UREG.Quantity(1.0, unit).to_base_units().magnitude)
    return at_one - offset, offset


def base_unit(unit: str) -> str:
    """Return the pint base units of a unit.

    Parameters
    ----------
    unit
        A pint-parsable unit.

    Returns
    -------
    str
        The base units, e.g. ``"meter / second"`` for ``"km/h"``.
    """
    return str(UREG.Quantity(1.0, unit).to_base_units().units)


def to_base(value: Any, unit: str) -> Any:
    """Convert a value from ``unit`` to base units.

    Safe to call on JAX tracers: the conversion coefficients are Python
    floats computed eagerly (and cached) per unit string.

    Parameters
    ----------
    value
        The value in ``unit``; a number or an array.
    unit
        A pint-parsable unit.

    Returns
    -------
    Any
        The value in base units.
    """
    factor, offset = _affine_coefficients(unit)
    return value * factor + offset


def from_base(value: Any, unit: str) -> Any:
    """Convert a value from base units to ``unit``.

    Parameters
    ----------
    value
        The value in the base units of ``unit``; a number or an array.
    unit
        A pint-parsable unit.

    Returns
    -------
    Any
        The value in ``unit``.
    """
    factor, offset = _affine_coefficients(unit)
    return (value - offset) / factor


def convert(value: Any, src: str, dst: str) -> Any:
    """Convert a value between two commensurable units.

    Parameters
    ----------
    value
        The value in ``src``; a number or an array.
    src
        The source unit.
    dst
        The destination unit.

    Returns
    -------
    Any
        The value in ``dst``.

    Raises
    ------
    pint.DimensionalityError
        If the two units have different dimensionalities.
    """
    src_unit = UREG.Unit(src)
    dst_unit = UREG.Unit(dst)
    if src_unit.dimensionality != dst_unit.dimensionality:
        raise pint.DimensionalityError(
            src_unit, dst_unit, src_unit.dimensionality, dst_unit.dimensionality
        )
    return from_base(to_base(value, src), dst)


def make_unit_shell(
    input_units: Mapping[str, str],
    output_units: Mapping[str, str],
    func: Callable[[dict[str, Any]], dict[str, Any]],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Wrap a dict-to-dict function with boundary unit conversion.

    The wrapped function operates in base units. The shell accepts inputs
    expressed in ``input_units``, converts them to base units, calls the
    function, and converts the outputs named in ``output_units`` from
    base units back. Keys absent from the unit mappings pass through
    untouched. All conversion factors are precomputed as Python floats at
    wrap time, so the shell stays jit-safe.

    Parameters
    ----------
    input_units
        The unit of each converted input, keyed by input name.
    output_units
        The desired unit of each converted output, keyed by output name.
    func
        The dict-to-dict function operating in base units.

    Returns
    -------
    Callable
        The wrapped function, dict in ``input_units`` to dict in
        ``output_units``.
    """
    input_coefficients = {
        name: _affine_coefficients(unit) for name, unit in input_units.items()
    }
    output_coefficients = {
        name: _affine_coefficients(unit) for name, unit in output_units.items()
    }

    def shell(inputs: dict[str, Any]) -> dict[str, Any]:
        """Convert inputs to base units, call ``func``, convert outputs back."""
        base_inputs = {
            name: (
                value * input_coefficients[name][0] + input_coefficients[name][1]
                if name in input_coefficients
                else value
            )
            for name, value in inputs.items()
        }
        outputs = func(base_inputs)
        return {
            name: (
                (value - output_coefficients[name][1]) / output_coefficients[name][0]
                if name in output_coefficients
                else value
            )
            for name, value in outputs.items()
        }

    return shell
