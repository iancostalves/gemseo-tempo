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
"""The dynamics stage: state spaces, controls, and integrators."""

from __future__ import annotations

from gemseo_tempo.dynamics.controls import AbstractControl
from gemseo_tempo.dynamics.controls import CubicControl
from gemseo_tempo.dynamics.controls import DelayedControl
from gemseo_tempo.dynamics.controls import ParametricControl
from gemseo_tempo.dynamics.controls import PiecewiseLinearControl
from gemseo_tempo.dynamics.fields import SpatialField
from gemseo_tempo.dynamics.integrators import FORWARD
from gemseo_tempo.dynamics.integrators import REVERSE
from gemseo_tempo.dynamics.integrators import AbstractIntegrator
from gemseo_tempo.dynamics.integrators import AdjointMethod
from gemseo_tempo.dynamics.integrators import DiffraxIntegrator
from gemseo_tempo.dynamics.integrators import QuasiStaticIntegrator
from gemseo_tempo.dynamics.integrators import Trajectory
from gemseo_tempo.dynamics.integrators import check_adjoint_compatibility
from gemseo_tempo.dynamics.state_space import StateSpace
from gemseo_tempo.dynamics.tables import smooth_abs
from gemseo_tempo.dynamics.tables import smooth_clip
from gemseo_tempo.dynamics.tables import smooth_max
from gemseo_tempo.dynamics.tables import smooth_min
from gemseo_tempo.dynamics.tables import smooth_table
from gemseo_tempo.dynamics.time_scaling import normalize_time

__all__ = [
    "FORWARD",
    "REVERSE",
    "AbstractControl",
    "AbstractIntegrator",
    "AdjointMethod",
    "CubicControl",
    "DelayedControl",
    "DiffraxIntegrator",
    "ParametricControl",
    "PiecewiseLinearControl",
    "QuasiStaticIntegrator",
    "SpatialField",
    "StateSpace",
    "Trajectory",
    "check_adjoint_compatibility",
    "normalize_time",
    "smooth_abs",
    "smooth_clip",
    "smooth_max",
    "smooth_min",
    "smooth_table",
]
