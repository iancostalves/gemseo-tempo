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
"""The :class:`.Variable` metadata record."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass(frozen=True, slots=True)
class Variable:
    """Metadata attached to a named quantity of a model.

    The ``name`` is the grammar key under which the quantity travels
    between disciplines. Everything else is metadata: it drives the
    generated documentation (Sprint 13), the design-space bounds, and the
    uncertain-space construction (Sprint 12), but never enters jitted code.
    """

    name: str
    """The grammar key, e.g. ``"c_l"``."""

    full_name: str
    """The human-readable name, e.g. ``"Lift coefficient"``."""

    description: str = ""
    """A free-text description."""

    unit: str = "dimensionless"
    """The pint-parsable unit of the canonical value."""

    lower_bound: float | None = None
    """The lower bound in ``unit``, if any."""

    upper_bound: float | None = None
    """The upper bound in ``unit``, if any."""

    tags: frozenset[str] = frozenset()
    """Free-form tags, e.g. ``{"state"}``, used for filtering."""

    distribution: dict[str, Any] | None = field(default=None)
    """An optional probability distribution consumed by the UQ layer.

    For example ``{"name": "OTNormalDistribution", "mu": 0.03,
    "sigma": 0.005}``, interpreted by :mod:`gemseo_tempo.uncertainty`.
    """

    def to_record(self) -> dict[str, Any]:
        """Return a YAML-serializable record of this variable.

        Returns
        -------
        dict
            The fields of this variable, with ``tags`` as a sorted list.
        """
        return {
            "name": self.name,
            "full_name": self.full_name,
            "description": self.description,
            "unit": self.unit,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "tags": sorted(self.tags),
            "distribution": self.distribution,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Variable:
        """Build a variable from a record as produced by :meth:`to_record`.

        Parameters
        ----------
        record
            The record; ``tags`` may be any iterable of strings.

        Returns
        -------
        Variable
            The variable.
        """
        data = dict(record)
        data["tags"] = frozenset(data.get("tags") or ())
        return cls(**data)
