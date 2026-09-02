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
"""One-factor-at-a-time benchmark studies of the transcriptions."""

from __future__ import annotations

from gemseo_tempo.benchmarking.profiles import data_profile
from gemseo_tempo.benchmarking.profiles import profile_table
from gemseo_tempo.benchmarking.runner import Record
from gemseo_tempo.benchmarking.runner import run_one
from gemseo_tempo.benchmarking.sampling import sample_guesses
from gemseo_tempo.benchmarking.studies import BASELINE
from gemseo_tempo.benchmarking.studies import SIZES
from gemseo_tempo.benchmarking.studies import STUDIES
from gemseo_tempo.benchmarking.studies import load_records
from gemseo_tempo.benchmarking.studies import run_study
from gemseo_tempo.benchmarking.studies import save_records

__all__ = [
    "BASELINE",
    "SIZES",
    "STUDIES",
    "Record",
    "data_profile",
    "load_records",
    "profile_table",
    "run_one",
    "run_study",
    "sample_guesses",
    "save_records",
]
