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
"""Transcriptions: the finite-dimensional problems the optimizer sees."""

from __future__ import annotations

from gemseo_tempo.transcriptions.base import AbstractTranscription
from gemseo_tempo.transcriptions.base import BoundarySpec
from gemseo_tempo.transcriptions.base import StagePiece
from gemseo_tempo.transcriptions.base import TranscribedProblem
from gemseo_tempo.transcriptions.base import VariableSpec
from gemseo_tempo.transcriptions.collocation import HermiteSimpson
from gemseo_tempo.transcriptions.indirect import IndirectShooting
from gemseo_tempo.transcriptions.single_shooting import SingleShooting

__all__ = [
    "AbstractTranscription",
    "BoundarySpec",
    "HermiteSimpson",
    "IndirectShooting",
    "SingleShooting",
    "StagePiece",
    "TranscribedProblem",
    "VariableSpec",
]
