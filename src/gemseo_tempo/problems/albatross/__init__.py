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
"""Dynamic soaring of an albatross in a wind that changes with height."""

from __future__ import annotations

from gemseo_tempo.problems.albatross.model import MIN_HEIGHT
from gemseo_tempo.problems.albatross.model import REFERENCE_HEIGHT
from gemseo_tempo.problems.albatross.model import ROUGHNESS
from gemseo_tempo.problems.albatross.model import create_state_space
from gemseo_tempo.problems.albatross.model import wind_gradient
from gemseo_tempo.problems.albatross.model import wind_speed
from gemseo_tempo.problems.albatross.plots import energy_budget
from gemseo_tempo.problems.albatross.plots import plot_controls
from gemseo_tempo.problems.albatross.plots import plot_cycle_shapes
from gemseo_tempo.problems.albatross.plots import plot_energy
from gemseo_tempo.problems.albatross.plots import plot_energy_analysis
from gemseo_tempo.problems.albatross.plots import plot_frames
from gemseo_tempo.problems.albatross.plots import plot_maneuver_pattern
from gemseo_tempo.problems.albatross.plots import plot_pareto
from gemseo_tempo.problems.albatross.plots import plot_polar
from gemseo_tempo.problems.albatross.plots import plot_scenario
from gemseo_tempo.problems.albatross.plots import plot_section_trade
from gemseo_tempo.problems.albatross.plots import plot_trajectory
from gemseo_tempo.problems.albatross.plots import plot_wind_profile
from gemseo_tempo.problems.albatross.plots import schematic_cycle
from gemseo_tempo.problems.albatross.problem import NOMINAL_WIND
from gemseo_tempo.problems.albatross.problem import REGISTRY
from gemseo_tempo.problems.albatross.problem import TURN
from gemseo_tempo.problems.albatross.problem import Guess
from gemseo_tempo.problems.albatross.problem import create_codesign_optimization
from gemseo_tempo.problems.albatross.problem import create_codesign_scenario
from gemseo_tempo.problems.albatross.problem import create_energy_optimization
from gemseo_tempo.problems.albatross.problem import create_energy_scenario
from gemseo_tempo.problems.albatross.problem import gliding_baseline
from gemseo_tempo.problems.albatross.problem import optimal_trajectory
from gemseo_tempo.problems.albatross.problem import soaring_cycle
from gemseo_tempo.problems.albatross.problem import solve_pareto_front
from gemseo_tempo.problems.albatross.statics import DESIGN_DEFAULTS
from gemseo_tempo.problems.albatross.statics import AirfoilSection
from gemseo_tempo.problems.albatross.statics import FlightEnvelope
from gemseo_tempo.problems.albatross.statics import StructuralMass
from gemseo_tempo.problems.albatross.statics import WingGeometry
from gemseo_tempo.problems.albatross.statics import create_statics

__all__ = [
    "DESIGN_DEFAULTS",
    "MIN_HEIGHT",
    "NOMINAL_WIND",
    "REFERENCE_HEIGHT",
    "REGISTRY",
    "ROUGHNESS",
    "TURN",
    "AirfoilSection",
    "FlightEnvelope",
    "Guess",
    "StructuralMass",
    "WingGeometry",
    "create_codesign_optimization",
    "create_codesign_scenario",
    "create_energy_optimization",
    "create_energy_scenario",
    "create_state_space",
    "create_statics",
    "energy_budget",
    "gliding_baseline",
    "optimal_trajectory",
    "plot_controls",
    "plot_cycle_shapes",
    "plot_energy",
    "plot_energy_analysis",
    "plot_frames",
    "plot_maneuver_pattern",
    "plot_pareto",
    "plot_polar",
    "plot_scenario",
    "plot_section_trade",
    "plot_wind_profile",
    "plot_trajectory",
    "schematic_cycle",
    "soaring_cycle",
    "solve_pareto_front",
    "wind_gradient",
    "wind_speed",
]
