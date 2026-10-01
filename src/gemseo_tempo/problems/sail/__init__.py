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
"""Minimum-time sailing through a spatially varying wind field."""

from __future__ import annotations

from gemseo_tempo.problems.sail.model import REGISTRY
from gemseo_tempo.problems.sail.model import channel_constraint
from gemseo_tempo.problems.sail.model import create_state_space
from gemseo_tempo.problems.sail.model import no_go_constraint
from gemseo_tempo.problems.sail.model import true_wind_angle
from gemseo_tempo.problems.sail.polar import NO_GO_ANGLE
from gemseo_tempo.problems.sail.polar import best_vmg_angle
from gemseo_tempo.problems.sail.polar import boat_speed
from gemseo_tempo.problems.sail.polar import downwind_vmg
from gemseo_tempo.problems.sail.polar import plot_polar
from gemseo_tempo.problems.sail.polar import speed_shape
from gemseo_tempo.problems.sail.problem import CORRIDOR_HALF_WIDTH
from gemseo_tempo.problems.sail.problem import LEGS
from gemseo_tempo.problems.sail.problem import Guess
from gemseo_tempo.problems.sail.problem import create_optimization
from gemseo_tempo.problems.sail.problem import create_scenario
from gemseo_tempo.problems.sail.problem import default_guess
from gemseo_tempo.problems.sail.problem import simulate_guess
from gemseo_tempo.problems.sail.problem import straight_run_time
from gemseo_tempo.problems.sail.wind import WIND_FIELDS
from gemseo_tempo.problems.sail.wind import WindField
from gemseo_tempo.problems.sail.wind import create_wind

__all__ = [
    "CORRIDOR_HALF_WIDTH",
    "LEGS",
    "NO_GO_ANGLE",
    "REGISTRY",
    "WIND_FIELDS",
    "Guess",
    "WindField",
    "best_vmg_angle",
    "boat_speed",
    "channel_constraint",
    "create_optimization",
    "create_scenario",
    "create_state_space",
    "create_wind",
    "default_guess",
    "downwind_vmg",
    "no_go_constraint",
    "plot_polar",
    "simulate_guess",
    "speed_shape",
    "straight_run_time",
    "true_wind_angle",
]
