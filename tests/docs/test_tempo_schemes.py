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


def test_the_scheme_draws_nine_panels():
    figure = plot_transcription_scheme()
    assert len(figure.axes) == 9


def test_every_panel_is_labelled():
    figure = plot_transcription_scheme()
    titles = [axis.get_title() for axis in figure.axes]
    for method in ("Single shooting", "Multiple shooting", "Collocation"):
        assert sum(title.startswith(method) for title in titles) == 3, method
    assert sum("control" in title for title in titles) == 3
    assert sum("state" in title for title in titles) == 3
    assert sum("dataflow" in title for title in titles) == 3


def test_the_n2_panels_carry_three_blocks_each():
    figure = plot_transcription_scheme()
    dataflow_axes = [axis for axis in figure.axes if "dataflow" in axis.get_title()]
    assert len(dataflow_axes) == 3
    for axis in dataflow_axes:
        # Three blocks, plus one arrowhead patch per elbow; the elbows'
        # straight runs are lines rather than patches.
        expected = 3 + (3 if axis.get_title().startswith("Single") else 4)
        assert len(axis.patches) == expected, axis.get_title()


def test_only_a_defect_carrying_method_reports_back_to_the_optimizer():
    """The one structural difference the N2 diagram exists to show.

    Single shooting reaches the optimizer only through the objective;
    multiple shooting and collocation hand it defects directly, which is
    the extra off-diagonal cell and the reason their problems are larger
    and sparser.
    """
    figure = plot_transcription_scheme()
    labels = {
        axis.get_title(): {text.get_text() for text in axis.texts}
        for axis in figure.axes
        if "dataflow" in axis.get_title()
    }
    assert "delta_2" in labels["Multiple shooting — dataflow"]
    assert "zeta_c" in labels["Collocation — dataflow"]
    single = labels["Single shooting — dataflow"]
    assert not {"delta_2", "zeta_c"} & single
    for panel in labels.values():
        assert {"Opt", "Traj", "Obj", "f_obj", "s(t)"} <= panel
