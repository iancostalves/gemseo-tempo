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
"""The brachistochrone problem in three physical variants."""

from __future__ import annotations

from gemseo_tempo.problems.brachistochrone.model import GRAVITY
from gemseo_tempo.problems.brachistochrone.model import REGISTRY
from gemseo_tempo.problems.brachistochrone.model import TARGET
from gemseo_tempo.problems.brachistochrone.model import VARIANTS
from gemseo_tempo.problems.brachistochrone.model import create_state_space
from gemseo_tempo.problems.brachistochrone.model import variant_constants
from gemseo_tempo.problems.brachistochrone.problem import Guess
from gemseo_tempo.problems.brachistochrone.problem import create_optimization
from gemseo_tempo.problems.brachistochrone.problem import create_scenario
from gemseo_tempo.problems.brachistochrone.problem import optimal_path_angle
from gemseo_tempo.problems.brachistochrone.problem import simulate_guess
from gemseo_tempo.problems.brachistochrone.reference import cycloid_time
from gemseo_tempo.problems.brachistochrone.reference import reference_time

__all__ = [
    "GRAVITY",
    "REGISTRY",
    "TARGET",
    "VARIANTS",
    "Guess",
    "create_optimization",
    "create_scenario",
    "create_state_space",
    "cycloid_time",
    "optimal_path_angle",
    "reference_time",
    "simulate_guess",
    "variant_constants",
]
