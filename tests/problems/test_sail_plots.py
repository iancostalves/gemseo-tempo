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
"""The sail's own figures, and the angle-convention scheme."""

from __future__ import annotations

import numpy
from matplotlib import pyplot

from gemseo_tempo.problems.sail.plots import plot_angle_convention
from gemseo_tempo.problems.sail.plots import plot_channel_trade
from gemseo_tempo.problems.sail.plots import plot_reliability
from gemseo_tempo.problems.sail.plots import plot_tracks
from gemseo_tempo.problems.sail.plots import plot_wind_field
from gemseo_tempo.problems.sail.wind import create_wind


def test_the_figures_draw():
    wind = create_wind("uniform")
    assert plot_wind_field(wind) is not None
    assert plot_angle_convention() is not None
    tracks = {"downwind": (numpy.array([0.0, 100.0]), numpy.array([0.0, 10.0]))}
    assert plot_tracks(tracks) is not None
    assert plot_tracks(tracks, mark=(800.0, 0.0), channel=150.0) is not None
    assert (
        plot_channel_trade(
            [250.0, 150.0, 80.0], [160.0, 161.0, 162.0], [190.0, 150.0, 80.0]
        )
        is not None
    )
    assert (
        plot_reliability(
            {"deterministic": (160.0, 0.5), "P(inside) >= 95%": (161.0, 0.98)}
        )
        is not None
    )


def test_the_figures_accept_existing_axes():
    wind = create_wind("uniform")
    _, axes = pyplot.subplots()
    assert plot_wind_field(wind, axes) is axes
    _, axes = pyplot.subplots()
    assert plot_angle_convention(axes) is axes
    _, axes = pyplot.subplots()
    tracks = {"downwind": (numpy.array([0.0, 1.0]), numpy.array([0.0, 1.0]))}
    assert plot_tracks(tracks, axes) is axes
    _, axes = pyplot.subplots()
    assert plot_channel_trade([150.0], [160.0], [150.0], axes) is axes
    _, axes = pyplot.subplots()
    assert plot_reliability({"deterministic": (160.0, 0.5)}, axes) is axes


def test_the_angle_convention_marks_the_no_go_cone():
    """Pure geometry: the no-go wedge and both example headings are there."""
    from gemseo_tempo.problems.sail.polar import NO_GO_ANGLE

    axes = plot_angle_convention()
    texts = " ".join(text.get_text() for text in axes.texts)
    assert "no-go" in texts
    assert f"{NO_GO_ANGLE:.0f}" in texts
    assert len(axes.patches) >= 1
