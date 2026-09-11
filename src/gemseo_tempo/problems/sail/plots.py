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
"""Figures of the sailing problem.

The polar itself is drawn by
:func:`~gemseo_tempo.problems.sail.polar.plot_polar`, which predates this
module and stays where it is. What is here is everything about a *course*:
the wind blowing over it, the track the boat takes across it, and what a
channel constraint costs.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy  # noqa: E402
from matplotlib import pyplot  # noqa: E402

from gemseo_tempo.problems.sail.wind import EXTENT  # noqa: E402


def plot_angle_convention(axes=None):
    """Draw the heading, true wind angle, and the no-go cone.

    Pure geometry, no data: an arrow for the wind (blowing *towards*, by
    the field's own convention), an arrow for a heading inside the no-go
    cone and one outside it, and the angle each subtends against the
    "from" bearing the wind is named by.

    Parameters
    ----------
    axes
        The axis to draw on; a new one when omitted.

    Returns
    -------
    Axes
        The axis drawn on.
    """
    from gemseo_tempo.problems.sail.polar import NO_GO_ANGLE

    if axes is None:
        _, axes = pyplot.subplots(figsize=(4.8, 4.8))
    axes.plot([0.0], [0.0], marker="o", markersize=8.0, color="black")

    # The wind blows towards the east; it is named for the opposite,
    # westward, bearing.
    axes.annotate(
        "",
        xy=(1.0, 0.0),
        xytext=(0.0, 0.0),
        arrowprops={"arrowstyle": "-|>", "color": "tab:blue", "linewidth": 1.8},
    )
    axes.text(
        1.05, 0.0, "wind\n(blows towards)", color="tab:blue", fontsize=9, va="center"
    )

    cone = numpy.radians(NO_GO_ANGLE)
    for angle, label, color in (
        (numpy.pi - cone / 2, "inside the no-go cone", "tab:red"),
        (numpy.pi / 2, "a beam reach", "tab:green"),
    ):
        axes.annotate(
            "",
            xy=(0.8 * numpy.cos(angle), 0.8 * numpy.sin(angle)),
            xytext=(0.0, 0.0),
            arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 1.6},
        )
        axes.text(
            0.85 * numpy.cos(angle),
            0.85 * numpy.sin(angle),
            label,
            color=color,
            fontsize=8.5,
            ha="center" if abs(numpy.cos(angle)) < 0.3 else "left",
        )

    wedge_angles = numpy.linspace(numpy.pi - cone, numpy.pi + cone, 40)
    axes.fill(
        numpy.concatenate([[0.0], 0.45 * numpy.cos(wedge_angles)]),
        numpy.concatenate([[0.0], 0.45 * numpy.sin(wedge_angles)]),
        color="tab:red",
        alpha=0.15,
    )
    axes.text(-0.5, 0.05, f"+/-{NO_GO_ANGLE:.0f} deg", fontsize=8, color="tab:red")

    axes.set_xlim(-1.3, 1.5)
    axes.set_ylim(-1.1, 1.1)
    axes.set_aspect("equal")
    axes.set_title("True wind angle and the no-go cone")
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)
    return axes


def plot_wind_field(wind, axes=None, samples: int = 19, scale: float = 260.0):
    """Draw a wind field as arrows over the sailing area.

    Parameters
    ----------
    wind
        The :class:`~gemseo_tempo.problems.sail.wind.WindField`.
    axes
        The axis to draw on; a new one when omitted.
    samples
        The number of arrows along each axis.
    scale
        The matplotlib quiver scale; larger draws shorter arrows.

    Returns
    -------
    Axes
        The axis drawn on.
    """
    if axes is None:
        _, axes = pyplot.subplots(figsize=(5.2, 5.0))
    axis = numpy.linspace(-EXTENT, EXTENT, samples)
    grid_x, grid_y = numpy.meshgrid(axis, axis, indexing="ij")
    east, north = wind.velocity(grid_x, grid_y)
    east = numpy.asarray(east)
    north = numpy.asarray(north)
    speed = numpy.hypot(east, north)
    axes.quiver(grid_x, grid_y, east, north, speed, cmap="viridis", scale=scale)
    axes.set_xlabel("east (m)")
    axes.set_ylabel("north (m)")
    axes.set_aspect("equal")
    axes.set_title("The breeze over the course")
    return axes


def plot_tracks(tracks, axes=None, mark=None, channel=None):
    """Draw one or more tracks across the course.

    Parameters
    ----------
    tracks
        A mapping of label to ``(x, y)`` arrays.
    axes
        The axis to draw on; a new one when omitted.
    mark
        The ``(x, y)`` of the mark being sailed to.
    channel
        The half-width of the channel, drawn as two dashed lines when
        given.

    Returns
    -------
    Axes
        The axis drawn on.
    """
    if axes is None:
        _, axes = pyplot.subplots(figsize=(6.8, 4.2))
    for label, (x, y) in tracks.items():
        axes.plot(
            numpy.asarray(x), numpy.asarray(y), marker="o", markersize=2.5, label=label
        )
    if channel is not None:
        for sign in (1.0, -1.0):
            axes.axhline(
                sign * channel,
                color="tab:red",
                linestyle="--",
                linewidth=1.0,
            )
        axes.axhline(
            channel,
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
            label=f"channel +/-{channel:.0f} m",
        )
    if mark is not None:
        axes.plot(
            [mark[0]],
            [mark[1]],
            marker="*",
            markersize=14.0,
            color="black",
            linestyle="none",
            label="mark",
        )
    axes.plot(
        [0.0],
        [0.0],
        marker="s",
        markersize=6.0,
        color="grey",
        linestyle="none",
        label="start",
    )
    axes.set_xlabel("east (m)")
    axes.set_ylabel("north (m)")
    axes.set_title("The track sailed")
    axes.legend()
    axes.grid(alpha=0.3)
    return axes


def plot_channel_trade(half_widths, times, across, axes=None):
    """Draw what tightening the channel costs in time.

    Parameters
    ----------
    half_widths
        The channel half-widths, in m.
    times
        The arrival time attained at each, in s.
    across
        How far the boat actually strayed at each, in m.
    axes
        The axis to draw on; a new one when omitted.

    Returns
    -------
    Axes
        The axis drawn on.
    """
    if axes is None:
        _, axes = pyplot.subplots(figsize=(6.4, 3.6))
    half_widths = numpy.asarray(half_widths, dtype=float)
    axes.plot(
        half_widths,
        numpy.asarray(times, dtype=float),
        marker="o",
        color="tab:blue",
        label="arrival time",
    )
    axes.set_xlabel("channel half-width (m)")
    axes.set_ylabel("time to the mark (s)", color="tab:blue")
    axes.tick_params(axis="y", labelcolor="tab:blue")

    strayed = axes.twinx()
    strayed.plot(
        half_widths,
        numpy.asarray(across, dtype=float),
        marker="s",
        color="tab:red",
        linestyle="--",
        label="furthest across",
    )
    # The line the constraint would trace if it bound exactly everywhere.
    strayed.plot(half_widths, half_widths, color="tab:red", linewidth=0.8, alpha=0.4)
    strayed.set_ylabel("furthest across (m)", color="tab:red")
    strayed.tick_params(axis="y", labelcolor="tab:red")
    axes.set_title("What the channel costs")
    axes.grid(alpha=0.3)
    return axes


def plot_reliability(assessments, axes=None):
    """Draw objective against reliability, one point per design.

    The figure the safety-margin argument turns on: a design that is both
    faster and more reliable sits below and to the right of the others.

    Parameters
    ----------
    assessments
        A mapping of label to ``(objective, reliability)``.
    axes
        The axis to draw on; a new one when omitted.

    Returns
    -------
    Axes
        The axis drawn on.
    """
    if axes is None:
        _, axes = pyplot.subplots(figsize=(6.4, 4.0))
    for label, (objective, reliability) in assessments.items():
        marker = "*" if label.startswith("P(") else "o"
        size = 16.0 if marker == "*" else 8.0
        axes.plot(
            [100.0 * reliability],
            [objective],
            marker=marker,
            markersize=size,
            linestyle="none",
            label=label,
        )
    axes.set_xlabel("reliability on an independent sample (%)")
    axes.set_ylabel("time to the mark (s)")
    axes.set_title("What a safety margin costs")
    axes.legend()
    axes.grid(alpha=0.3)
    return axes


__all__ = [
    "plot_angle_convention",
    "plot_channel_trade",
    "plot_reliability",
    "plot_tracks",
    "plot_wind_field",
]
