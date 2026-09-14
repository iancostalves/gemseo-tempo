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


def plot_points_of_sail(axes=None):
    """Draw the points of sail: what each wind angle is called, and the no-go zone.

    The sailor's own picture of the problem, and the vocabulary the rest of
    this page uses. The boat cannot advance inside the **no-go zone**; on
    either side of it, close-hauled, beam reach, broad reach and running
    name the angles it does sail at, and which side the wind enters on is
    the tack it is on. Crossing the wind from one tack to the other is a
    *tack* going upwind and a *gybe* going downwind.

    Adapted from the points-of-sail diagram of Ciccarella, Dalang and
    Vinckenbosch, *An Optimal Control Problem Arising in Sailboat
    Trajectory Optimization* (arXiv:2404.03773), which sets out the same
    vocabulary for the same kind of problem.

    Pure geometry, no data.

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
        _, axes = pyplot.subplots(figsize=(6.4, 6.0))

    # The wind comes from the top and blows down the page, which is how a
    # points-of-sail diagram is always drawn.
    for offset in (-0.3, 0.0, 0.3):
        axes.annotate(
            "",
            xy=(offset, 1.04),
            xytext=(offset, 1.42),
            arrowprops={"arrowstyle": "-|>", "color": "tab:cyan", "linewidth": 1.6},
        )
    axes.text(0.0, 1.48, "wind", color="tab:cyan", ha="center", fontsize=10)

    cone = numpy.radians(NO_GO_ANGLE)
    wedge = numpy.linspace(numpy.pi / 2 - cone, numpy.pi / 2 + cone, 60)
    axes.fill(
        numpy.concatenate([[0.0], numpy.cos(wedge)]),
        numpy.concatenate([[0.0], numpy.sin(wedge)]),
        color="tab:red",
        alpha=0.12,
    )
    axes.text(
        0.0,
        0.55,
        f"no-go zone\n+/-{NO_GO_ANGLE:.0f} deg",
        color="tab:red",
        ha="center",
        fontsize=9,
    )

    # Each point of sail, on both tacks, named once. The angle is measured
    # from dead upwind, which is the convention the model's TWA uses.
    for twa, name in (
        (45.0, "close-hauled"),
        (90.0, "beam reach"),
        (135.0, "broad reach"),
        (180.0, "running"),
    ):
        for side in (1.0, -1.0):
            bearing = numpy.pi / 2 - side * numpy.radians(twa)
            tip = (numpy.cos(bearing), numpy.sin(bearing))
            axes.annotate(
                "",
                xy=tip,
                xytext=(0.0, 0.0),
                arrowprops={"arrowstyle": "-|>", "color": "tab:blue"},
            )
            if side > 0.0:
                axes.text(
                    1.1 * tip[0] + (0.0 if twa == 180.0 else 0.04),
                    1.1 * tip[1] - (0.12 if twa == 180.0 else 0.0),
                    name,
                    fontsize=9,
                    ha="center" if twa == 180.0 else "left",
                    va="center",
                )
            if twa == 180.0:
                break

    # Which side the wind enters on names the tack, so the labels go out to
    # the sides rather than on top of the points of sail themselves.
    axes.text(1.5, 0.28, "starboard tack", fontsize=9, color="tab:green", ha="right")
    axes.text(-1.5, 0.28, "port tack", fontsize=9, color="tab:green", ha="left")
    axes.plot([0.0], [0.0], marker="o", color="black", markersize=7)
    axes.set_xlim(-1.75, 1.75)
    axes.set_ylim(-1.35, 1.6)
    axes.set_aspect("equal")
    axes.set_title("The points of sail")
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)
    return axes


def plot_wind_triangle(boat=5.0, twa_degrees=100.0, tws=10.0, axes=None):
    """Draw the velocity triangle: true wind, boat velocity, apparent wind.

    What the sail actually feels. The water is static, so the boat's
    velocity over the ground is also its velocity through it, and the
    apparent wind is the true wind minus the boat's own motion. An aircraft
    in a wind gradient closes the same triangle from the other side, where
    it is written ``v_K = v_W + v_A`` — see
    :func:`~gemseo_tempo.problems.albatross.plots.plot_cycle_schematic`.

    Adapted from the wind triangle of Hong, Liu, Holzapfel and Sachs,
    *Dynamic Soaring Under Different Atmospheric Stability Conditions*,
    Journal of Guidance, Control, and Dynamics 46(5), 2023.

    Parameters
    ----------
    boat
        The boat's speed through the water, in m/s.
    twa_degrees
        The true wind angle, in degrees.
    tws
        The true wind speed, in m/s.
    axes
        The axis to draw on; a new one when omitted.

    Returns
    -------
    Axes
        The axis drawn on.
    """
    if axes is None:
        _, axes = pyplot.subplots(figsize=(6.8, 4.8))

    # Resolved in the boat's own frame, with the heading along +x.
    twa = numpy.radians(twa_degrees)
    boat_vector = numpy.array([boat, 0.0])
    true_wind = numpy.array([-tws * numpy.cos(twa), -tws * numpy.sin(twa)])
    apparent = true_wind - boat_vector

    def arrow(start, vector, colour, label, offset):
        """Draw one labelled vector of the triangle."""
        axes.annotate(
            "",
            xy=start + vector,
            xytext=start,
            arrowprops={"arrowstyle": "-|>", "color": colour, "linewidth": 2.0},
        )
        middle = start + 0.5 * vector
        axes.text(
            middle[0] + offset[0],
            middle[1] + offset[1],
            label,
            color=colour,
            fontsize=9.5,
        )

    origin = numpy.zeros(2)
    arrow(origin, boat_vector, "tab:blue", "boat  $v_B$", (-0.6, 0.35))
    arrow(origin, true_wind, "tab:cyan", "true wind  $v_W$", (0.15, 0.35))
    arrow(true_wind, -boat_vector, "tab:green", "$-\\,v_B$", (-0.2, 0.35))
    arrow(origin, apparent, "tab:red", "apparent wind  $v_A$", (-1.8, -0.75))

    axes.plot([0.0], [0.0], marker="o", color="black", markersize=6)
    # Annotations do not move the data limits, so the extent is set from
    # the vectors themselves; without this the axis comes out empty.
    corners = numpy.stack([origin, boat_vector, true_wind, apparent])
    margin = 0.35 * float(numpy.max(numpy.abs(corners)))
    # Wider to the left and below, where the labels sit.
    axes.set_xlim(corners[:, 0].min() - 1.6 * margin, corners[:, 0].max() + margin)
    axes.set_ylim(corners[:, 1].min() - 0.6 * margin, corners[:, 1].max() + margin)
    axes.set_aspect("equal")
    axes.set_title("The velocity triangle, with the water static")
    axes.set_xlabel("along the heading (m/s)")
    axes.set_ylabel("across the heading (m/s)")
    axes.grid(alpha=0.3)
    return axes


def plot_sail_coefficients(axes=None):
    """Draw the sail's lift and drag, and what is left to push the boat.

    The surrogate behind the polar's shape: a flat plate's lift, greatest
    where the sail sits at forty-five degrees to the apparent wind, and a
    drag that grows first with the lift and then with the plate held across
    the flow. Close to the wind the sail is a wing working at a few
    degrees; running, it is a parachute at ninety.

    These curves are **not** what the dynamics evaluate. The trajectory
    reads a fitted polar instead, for the reason given in
    :func:`~gemseo_tempo.problems.sail.polar.boat_speed`; this is where
    that polar's shape comes from, and where its no-go cone comes from —
    the right-hand panel crosses zero at the angle below which the sail
    cannot pull the boat forward at all.

    Parameters
    ----------
    axes
        Two axes to draw on; a new figure when omitted.

    Returns
    -------
    ndarray
        The axes drawn on.
    """
    from gemseo_tempo.problems.sail import aero

    if axes is None:
        _, axes = pyplot.subplots(1, 2, figsize=(9.6, 3.8))
    axes = numpy.atleast_1d(axes)

    alpha = numpy.linspace(0.0, 90.0, 181)
    axes[0].plot(
        alpha,
        numpy.asarray(aero.lift_coefficient(alpha)),
        color="tab:blue",
        label="C_L",
    )
    axes[0].plot(
        alpha, numpy.asarray(aero.drag_coefficient(alpha)), color="tab:red", label="C_D"
    )
    axes[0].set_xlabel("angle of attack (deg)")
    axes[0].set_ylabel("coefficient")
    axes[0].set_title("Lift and drag of the sail")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    awa = numpy.linspace(0.0, 180.0, 361)
    axes[1].axhline(0.0, color="0.6", linewidth=0.8)
    axes[1].plot(
        awa, numpy.asarray(aero.driving_force(0.0, awa, 10.0)), color="tab:purple"
    )
    axes[1].set_xlabel("apparent wind angle (deg)")
    axes[1].set_ylabel("driving force (N)")
    axes[1].set_title("What is left to push the boat forward")
    axes[1].grid(alpha=0.3)
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
    "plot_points_of_sail",
    "plot_sail_coefficients",
    "plot_wind_triangle",
    "plot_channel_trade",
    "plot_reliability",
    "plot_tracks",
    "plot_wind_field",
]
