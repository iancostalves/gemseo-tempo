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
"""The statics stage: models running once per iteration, ahead of the dynamics."""

from __future__ import annotations

from gemseo_tempo.statics.chain import StaticsChain
from gemseo_tempo.statics.model import AutoModel
from gemseo_tempo.statics.model import JAXModel
from gemseo_tempo.statics.model import Model

__all__ = [
    "AutoModel",
    "JAXModel",
    "Model",
    "StaticsChain",
]
