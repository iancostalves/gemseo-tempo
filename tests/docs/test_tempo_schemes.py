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
"""The transcription scheme: pure geometry, drawn live, nothing to go stale."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "docs" / "_ext"))

from tempo_schemes import plot_transcription_scheme  # noqa: E402


def test_the_scheme_draws_six_panels():
    figure = plot_transcription_scheme()
    assert len(figure.axes) == 6


def test_every_panel_is_labelled():
    figure = plot_transcription_scheme()
    titles = [axis.get_title() for axis in figure.axes]
    assert any("Single shooting" in title for title in titles)
    assert any("Collocation" in title for title in titles)
    assert sum("control" in title for title in titles) == 2
    assert sum("state" in title for title in titles) == 2
    assert sum("dataflow" in title for title in titles) == 2


def test_the_dataflow_panels_carry_three_boxes_each():
    figure = plot_transcription_scheme()
    dataflow_axes = [axis for axis in figure.axes if "dataflow" in axis.get_title()]
    assert len(dataflow_axes) == 2
    for axis in dataflow_axes:
        assert len(axis.patches) == 3 + 3  # three boxes, three arrows
