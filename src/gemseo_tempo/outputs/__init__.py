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
"""The outputs stage: functionals and constraints over trajectories."""

from __future__ import annotations

from gemseo_tempo.outputs.constraints import ControlGroupConstraint
from gemseo_tempo.outputs.constraints import PathConstraint
from gemseo_tempo.outputs.constraints import PeriodicityConstraint
from gemseo_tempo.outputs.constraints import TerminalConstraint
from gemseo_tempo.outputs.functionals import FinalRate
from gemseo_tempo.outputs.functionals import Lagrange
from gemseo_tempo.outputs.functionals import Mayer
from gemseo_tempo.outputs.functionals import TimeIntegral

__all__ = [
    "ControlGroupConstraint",
    "FinalRate",
    "Lagrange",
    "Mayer",
    "PathConstraint",
    "PeriodicityConstraint",
    "TerminalConstraint",
    "TimeIntegral",
]
