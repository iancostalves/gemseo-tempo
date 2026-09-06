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
"""Uncertain parameters, sensitivity, and design under uncertainty."""

from __future__ import annotations

from gemseo_tempo.uncertainty.batch import batch_evaluator
from gemseo_tempo.uncertainty.batch import batch_jacobian
from gemseo_tempo.uncertainty.batch import freeze_design
from gemseo_tempo.uncertainty.batch import looped_evaluator
from gemseo_tempo.uncertainty.batch import measure_speedup
from gemseo_tempo.uncertainty.spaces import make_parameter_space
from gemseo_tempo.uncertainty.spaces import nominal_values
from gemseo_tempo.uncertainty.spaces import uncertain_names

__all__ = [
    "batch_evaluator",
    "batch_jacobian",
    "freeze_design",
    "looped_evaluator",
    "make_parameter_space",
    "measure_speedup",
    "nominal_values",
    "uncertain_names",
]
