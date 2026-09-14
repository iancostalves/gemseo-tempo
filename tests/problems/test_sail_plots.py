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
import pytest
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


def test_the_points_of_sail_are_all_named():
    """The vocabulary figure has to carry the vocabulary."""
    from gemseo_tempo.problems.sail.plots import plot_points_of_sail
    from gemseo_tempo.problems.sail.polar import NO_GO_ANGLE

    axes = plot_points_of_sail()
    texts = " ".join(text.get_text() for text in axes.texts)
    for name in ("close-hauled", "beam reach", "broad reach", "running"):
        assert name in texts, name
    assert "port tack" in texts
    assert "starboard tack" in texts
    assert "no-go zone" in texts
    assert f"{NO_GO_ANGLE:.0f}" in texts


def test_the_wind_triangle_closes():
    """Drawn as vectors, so the arithmetic is the figure's own claim.

    The apparent wind is the true wind minus the boat's velocity, and the
    figure is only worth showing if the three arrows actually form that
    triangle.
    """
    from gemseo_tempo.problems.sail.aero import apparent_wind
    from gemseo_tempo.problems.sail.plots import plot_wind_triangle

    axes = plot_wind_triangle(boat=5.0, twa_degrees=100.0, tws=10.0)
    texts = " ".join(text.get_text() for text in axes.texts)
    assert "apparent wind" in texts
    assert "true wind" in texts

    speed, angle = apparent_wind(5.0, 100.0, 10.0)
    # The triangle the figure draws, checked against the model that
    # computes it: |v_A| from the components, and the angle it comes from.
    along = -10.0 * numpy.cos(numpy.radians(100.0)) - 5.0
    across = -10.0 * numpy.sin(numpy.radians(100.0))
    assert float(speed) == pytest.approx(float(numpy.hypot(along, across)), rel=1e-9)
    assert abs(float(angle)) <= 180.0


def test_the_sail_coefficients_show_where_the_polar_comes_from():
    """Lift peaking at forty-five degrees, and a cone the drive implies."""
    from gemseo_tempo.problems.sail.plots import plot_sail_coefficients

    axes = plot_sail_coefficients()
    assert len(axes) == 2
    titles = [axis.get_title() for axis in axes]
    assert any("Lift and drag" in title for title in titles)
    assert any("push the boat" in title for title in titles)
    for axis in axes:
        assert axis.lines


def test_the_new_figures_accept_existing_axes():
    from gemseo_tempo.problems.sail.plots import plot_points_of_sail
    from gemseo_tempo.problems.sail.plots import plot_sail_coefficients
    from gemseo_tempo.problems.sail.plots import plot_wind_triangle

    _, axes = pyplot.subplots()
    assert plot_points_of_sail(axes) is axes
    _, axes = pyplot.subplots()
    assert plot_wind_triangle(axes=axes) is axes
    _, pair = pyplot.subplots(1, 2)
    assert plot_sail_coefficients(pair) is pair
