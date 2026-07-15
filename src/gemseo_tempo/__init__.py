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
"""GEMSEO plugin for multidisciplinary optimization of dynamic systems."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

import jax

# Dynamic systems need double precision: single precision accumulates error over
# long integration horizons and degrades gradient checks.
jax.config.update("jax_enable_x64", True)

try:
    __version__ = version("gemseo-tempo")
except PackageNotFoundError:  # pragma: no cover - only hit on non-installed source
    __version__ = "0.0.0"
