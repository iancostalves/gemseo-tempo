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
"""Variable metadata: names, descriptions, units, bounds, and distributions.

Units are metadata only. Inside jitted code all values are canonical
(pint base) units; conversion happens at discipline boundaries, never
inside traced code.
"""

from __future__ import annotations

from gemseo_tempo.metadata.registry import VariableRegistry
from gemseo_tempo.metadata.units import UREG
from gemseo_tempo.metadata.units import base_unit
from gemseo_tempo.metadata.units import convert
from gemseo_tempo.metadata.units import from_base
from gemseo_tempo.metadata.units import make_unit_shell
from gemseo_tempo.metadata.units import to_base
from gemseo_tempo.metadata.variable import Variable

__all__ = [
    "UREG",
    "Variable",
    "VariableRegistry",
    "base_unit",
    "convert",
    "from_base",
    "make_unit_shell",
    "to_base",
]
