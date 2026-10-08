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
"""Figures of a soaring cycle.

Each view answers a different question. :func:`plot_scenario` sets the
scene: the terrestrial frame, the wind profile and one cycle.
:func:`plot_frames` takes one instant of it apart, in the frames the
equations are written in. :func:`plot_trajectory` shows where the bird went
through the boundary layer and where it gained or lost energy,
:func:`plot_maneuver_pattern` what several cycles add up to, and
:func:`plot_energy_analysis` why the cycle pays for itself.
:func:`plot_controls` and :func:`plot_energy` are the plain time histories.

The composition of the three-dimensional views follows the figures of
Barnes (2004) as reproduced by Koessler (2019) and the scenario sketch of
Hong et al. (2023): a sea surface, the path drawn as a wing-span ribbon so
that the bank is visible, and the wind profile drawn in the scene rather
than beside it.

Every function takes either a :class:`.Trajectory` or a mapping of arrays
keyed by name with the times under ``"ts"``, which is what the stored
results under ``docs/_data`` load as.
"""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib

matplotlib.use("Agg")
import numpy  # noqa: E402
from matplotlib import pyplot  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
"""Blue, orange and aqua: the first three slots of a categorical palette
validated as a set for colour-vision deficiency, in that fixed order."""

INK = "#0b0b0b"
"""Primary text and outlines."""

MUTED = "#8a8985"
"""Axes, frames, shadows and other recessive marks."""

SEA = "#cde2fb"
"""The sea surface, the lightest step of the blue ramp."""

WIND_MAP = LinearSegmentedColormap.from_list(
    "wind", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
"""Wind speed: one hue, light to dark."""

VIEW = (26.0, -115.0)
"""The elevation and azimuth the three-dimensional views are seen from."""

SCENARIO_VIEW = (20.0, -30.0)
"""The view of the schematic scenario, as in Hong et al. (2023), Fig. 1:
from downwind and a little behind, so that the snake runs across the page
and the wind blows towards the reader."""

RATE_MAP = LinearSegmentedColormap.from_list(
    "energy_rate", ["#e34948", "#f0efec", "#2a78d6"]
)
"""Energy rate: red losing, grey neutral, blue gaining."""


def _series(trajectory, name):
    """Return one timeseries of a trajectory as a numpy array.

    Parameters
    ----------
    trajectory
        The trajectory.
    name
        The state, control or output name.

    Returns
    -------
    ndarray
        The series.

    Raises
    ------
    KeyError
        If no series of that name exists.
    """
    groups = (
        (trajectory,)
        if isinstance(trajectory, Mapping)
        else (trajectory.states, trajectory.controls, trajectory.outputs)
    )
    for group in groups:
        if name in group:
            return numpy.asarray(group[name])
    raise KeyError(f"No series named {name!r} in the trajectory.")


def _times(trajectory):
    """Return the save times of a trajectory or of a mapping of arrays.

    Parameters
    ----------
    trajectory
        The trajectory, or a mapping with the times under ``"ts"``.

    Returns
    -------
    ndarray
        The times.
    """
    if isinstance(trajectory, Mapping):
        return numpy.asarray(trajectory["ts"])
    return numpy.asarray(trajectory.ts)


def _scalar(cycle, name, default):
    """Return a scalar stored alongside a cycle, or a default.

    Parameters
    ----------
    cycle
        The trajectory, or a mapping of arrays.
    name
        The name of the scalar.
    default
        The value when the cycle does not carry one.

    Returns
    -------
    float
        The value.
    """
    if isinstance(cycle, Mapping) and name in cycle:
        return float(numpy.ravel(cycle[name])[0])
    return float(default)


def _unit_vectors(gamma, psi, mu):
    """Return the velocity, lift and span directions in the terrestrial frame.

    With x downwind, y crosswind and h up, the air-relative velocity points
    along ``(cos gamma cos psi, cos gamma sin psi, sin gamma)``. Bank rolls
    the lift about it, towards the left of the path for a positive angle,
    which is the direction the heading then turns (``dpsi/dt`` is
    proportional to ``L sin mu``). The span is perpendicular to both.

    Parameters
    ----------
    gamma
        The flight path angle.
    psi
        The heading.
    mu
        The bank angle.

    Returns
    -------
    tuple
        Three ``(n, 3)`` arrays: velocity, lift and span (right to left).
    """
    gamma, psi, mu = (numpy.asarray(value, dtype=float) for value in (gamma, psi, mu))
    velocity = numpy.stack(
        [
            numpy.cos(gamma) * numpy.cos(psi),
            numpy.cos(gamma) * numpy.sin(psi),
            numpy.sin(gamma),
        ],
        axis=-1,
    )
    left = numpy.stack([-numpy.sin(psi), numpy.cos(psi), 0.0 * psi], axis=-1)
    up = numpy.stack(
        [
            -numpy.sin(gamma) * numpy.cos(psi),
            -numpy.sin(gamma) * numpy.sin(psi),
            numpy.cos(gamma),
        ],
        axis=-1,
    )
    lift = numpy.cos(mu)[..., None] * up + numpy.sin(mu)[..., None] * left
    span = numpy.cos(mu)[..., None] * left - numpy.sin(mu)[..., None] * up
    return velocity, lift, span


def _phases(cycle) -> dict[str, int]:
    """Locate the four phases of a cycle.

    The windward climb and the leeward descent are where the wind gradient
    adds most energy while climbing into the wind and descending with it;
    the turns are the highest and the lowest points.

    Parameters
    ----------
    cycle
        The trajectory, or a mapping of arrays.

    Returns
    -------
    dict
        The index of each phase, keyed by its name.
    """
    gamma = _series(cycle, "gamma")
    along = numpy.cos(_series(cycle, "psi"))
    height = _series(cycle, "h")
    gain = _series(cycle, "gradient_power")

    def strongest(mask):
        candidates = numpy.flatnonzero(mask)
        if candidates.size == 0:
            return int(numpy.argmax(gain))
        return int(candidates[numpy.argmax(gain[candidates])])

    return {
        "windward climb": strongest((gamma > 0.0) & (along < 0.0)),
        "upper turn": int(numpy.argmax(height)),
        "leeward descent": strongest((gamma < 0.0) & (along > 0.0)),
        "lower turn": int(numpy.argmin(height)),
    }


SHAPES = ("O", "S")
"""The two cycle shapes the dynamic-soaring literature draws."""


def schematic_cycle(shape: str = "O", points: int = 241) -> dict:
    """Return one idealized cycle of either shape the literature draws.

    Geometry only: no dynamics are integrated and nothing is optimized. The
    shapes are those of the sketches in Barnes (2004), reproduced by
    Koessler (2019), and of Hong et al. (2023), Fig. 1, built so that each
    climbs while heading into the wind and descends while heading with it,
    which is what the energy equation asks for.

    - ``"O"``, a loop: the heading gains a full turn, the bird climbs on the
      upwind half and descends on the downwind half, and the wind carries
      the loop downwind.
    - ``"S"``, a snake: the heading swings either side of crosswind, the bird
      climbs while turned into the wind and descends while turned away from
      it, and the pattern travels across the wind.

    Parameters
    ----------
    shape
        ``"O"`` or ``"S"``.
    points
        The number of samples over the cycle.

    Returns
    -------
    dict
        The positions, attitude angles and a phase parameter under ``"ts"``,
        in the form the other figures of this module take.

    Raises
    ------
    ValueError
        If the shape is neither ``"O"`` nor ``"S"``.
    """
    if shape not in SHAPES:
        msg = f"Unknown shape {shape!r}; use one of {list(SHAPES)}."
        raise ValueError(msg)
    phase = numpy.linspace(0.0, 2.0 * numpy.pi, points)
    radius, climb, mean_height, drift = 20.0, 10.0, 13.0, 25.0
    if shape == "O":
        heading = phase
        height = mean_height - climb * numpy.sin(phase)
        rise = -climb * numpy.cos(phase)
        bank = numpy.full(points, 0.6)
    else:
        heading = 0.5 * numpy.pi + 1.4 * numpy.sin(phase)
        height = mean_height - climb * numpy.cos(phase)
        rise = climb * numpy.sin(phase)
        bank = 0.7 * numpy.cos(phase)
    # Constant airspeed: the path advances by the radius per radian.
    step = radius * numpy.diff(phase)
    east = numpy.concatenate(
        [[0.0], numpy.cumsum(step * numpy.cos(0.5 * (heading[1:] + heading[:-1])))]
    )
    north = numpy.concatenate(
        [[0.0], numpy.cumsum(step * numpy.sin(0.5 * (heading[1:] + heading[:-1])))]
    )
    east = east + drift * phase / (2.0 * numpy.pi)
    gamma = numpy.arctan(rise / radius)
    return {
        "ts": phase,
        "x": east,
        "y": north - north[0],
        "h": height,
        "gamma": gamma,
        "psi": heading,
        "mu": bank,
        # The sign and the height dependence of the wind-gradient term,
        # which is all the phases are located by.
        "gradient_power": -numpy.sin(gamma) * numpy.cos(heading) / height,
    }


def _sea(axes, x_range, y_range):
    """Draw the sea surface as a plane at zero height, behind everything else.

    Matplotlib sorts 3D artists by depth, which puts a translucent plane in
    front of whatever is above it half the time. Drawing order is fixed
    instead: the sea first, then the rest in the order it is added.
    """
    axes.computed_zorder = False
    east, north = numpy.meshgrid(x_range, y_range)
    axes.plot_surface(
        east,
        north,
        0.0 * east,
        color=SEA,
        alpha=0.6,
        linewidth=0,
        shade=False,
        zorder=0,
    )


def _keep_whole(figure):
    """Stop a tight bounding box from cropping 3D axis labels.

    A tight bounding box is computed from each axis's own extent, which for
    a 3D axis leaves its labels out, so saving with ``bbox_inches="tight"``
    cuts them off. An invisible frame over the whole figure keeps it all.
    """
    figure.add_artist(
        pyplot.Rectangle(
            (0.0, 0.0), 1.0, 1.0, transform=figure.transFigure, fill=False, linewidth=0
        )
    )


def _clean_3d(axes):
    """Make a 3D axis recessive: light panes, muted ticks."""
    for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1.0)
    axes.tick_params(colors=MUTED, labelsize=8)


def plot_trajectory(trajectory, axes=None):
    """Draw the cycle inside the boundary layer, coloured by its energy rate.

    The wall behind the path is shaded by the wind speed at each height,
    read off the cycle itself, so the reader sees which part of the
    boundary layer each leg crosses. The path is coloured by the net rate
    of change of the air-relative energy, drag and wind gradient together:
    red where the bird is losing energy, blue where the wind is paying.
    Its shadow on the sea and a few drop lines give the depth a flat
    picture otherwise loses.

    Parameters
    ----------
    trajectory
        The cycle to draw, a trajectory or a mapping of arrays.
    axes
        The 3D axis to draw on; a new one when omitted.

    Returns
    -------
    Axes
        The axis drawn on.
    """
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    if axes is None:
        figure = pyplot.figure(figsize=(7.6, 5.6))
        axes = figure.add_subplot(projection="3d")
    east = _series(trajectory, "x")
    north = _series(trajectory, "y")
    height = _series(trajectory, "h")
    rate = _series(trajectory, "drag_power") + _series(trajectory, "gradient_power")
    wind = _series(trajectory, "wind")

    margin_x = 0.12 * max(numpy.ptp(east), 1.0)
    margin_y = 0.12 * max(numpy.ptp(north), 1.0)
    x_range = numpy.array([east.min() - margin_x, east.max() + margin_x])
    y_range = numpy.array([north.min() - margin_y, north.max() + margin_y])
    top = 1.1 * max(height.max(), 1.0)
    _sea(axes, x_range, y_range)

    # The boundary layer, on the back wall: the wind speed at each height,
    # interpolated from what the cycle itself flew through.
    order = numpy.argsort(height)
    levels = numpy.linspace(0.0, top, 40)
    speeds = numpy.interp(levels, height[order], wind[order])
    wall_x, wall_h = numpy.meshgrid(x_range, levels)
    shade = WIND_MAP((speeds - speeds.min()) / max(numpy.ptp(speeds), 1e-9))[
        :, None, :
    ].repeat(2, axis=1)
    axes.plot_surface(
        wall_x,
        0.0 * wall_x + y_range[1],
        wall_h,
        facecolors=shade,
        alpha=0.55,
        linewidth=0,
        shade=False,
    )
    axes.text(
        x_range[0], y_range[1], top, "wind speed W(h), darker is faster", fontsize=8
    )

    points = numpy.stack([east, north, height], axis=1)
    segments = numpy.stack([points[:-1], points[1:]], axis=1)
    limit = max(numpy.max(numpy.abs(rate)), 1e-9)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit)
    # A dark underlay, so that the neutral middle of the colour map, where
    # the cycle neither gains nor loses, still reads against the page.
    axes.plot(east, north, height, color=INK, linewidth=4.0, solid_capstyle="round")
    path = Line3DCollection(
        segments, cmap=RATE_MAP, norm=norm, linewidths=2.6, capstyle="round"
    )
    path.set_array(0.5 * (rate[:-1] + rate[1:]))
    axes.add_collection3d(path)

    axes.plot(east, north, 0.0 * height, color=MUTED, linewidth=1.0, linestyle="--")
    for index in numpy.linspace(0, len(east) - 1, 13).astype(int)[1:-1]:
        axes.plot(
            [east[index]] * 2,
            [north[index]] * 2,
            [0.0, height[index]],
            color=MUTED,
            linewidth=0.6,
        )

    colorbar = axes.figure.colorbar(path, ax=axes, shrink=0.6, pad=0.1)
    colorbar.set_label("net energy rate de/dt (m/s)")
    axes.set_xlim(*x_range)
    axes.set_ylim(*y_range)
    axes.set_zlim(0.0, top)
    axes.set_box_aspect((numpy.ptp(x_range), numpy.ptp(y_range), 1.6 * top))
    axes.set_xlabel("x, downwind (m)")
    axes.set_ylabel("y, crosswind (m)")
    axes.set_zlabel("h (m)")
    axes.view_init(elev=VIEW[0], azim=VIEW[1])
    _clean_3d(axes)
    axes.set_title("The cycle inside the boundary layer")
    return axes


def plot_controls(trajectory, axes=None):
    """Draw the lift coefficient and bank angle over the cycle.

    Parameters
    ----------
    trajectory
        The cycle to draw.
    axes
        The axis to draw on; a new one when omitted.

    Returns
    -------
    Axes
        The axis drawn on.
    """
    if axes is None:
        _, axes = pyplot.subplots(figsize=(7.0, 3.5))
    times = _times(trajectory)
    axes.plot(times, _series(trajectory, "cl"), color="tab:blue", label="lift coeff")
    axes.set_xlabel("time (s)")
    axes.set_ylabel("lift coefficient", color="tab:blue")
    axes.tick_params(axis="y", labelcolor="tab:blue")

    banked = axes.twinx()
    banked.plot(
        times,
        numpy.degrees(_series(trajectory, "mu")),
        color="tab:red",
        label="bank",
    )
    banked.set_ylabel("bank angle (deg)", color="tab:red")
    banked.tick_params(axis="y", labelcolor="tab:red")
    axes.set_title("Controls over the cycle")
    return axes


def plot_energy(trajectory, axes=None):
    """Draw the height and airspeed trade over the cycle.

    Parameters
    ----------
    trajectory
        The cycle to draw.
    axes
        The axis to draw on; a new one when omitted.

    Returns
    -------
    Axes
        The axis drawn on.
    """
    if axes is None:
        _, axes = pyplot.subplots(figsize=(7.0, 3.5))
    times = _times(trajectory)
    axes.plot(times, _series(trajectory, "h"), color="tab:green", label="height")
    axes.set_xlabel("time (s)")
    axes.set_ylabel("height (m)", color="tab:green")
    axes.tick_params(axis="y", labelcolor="tab:green")

    speed = axes.twinx()
    speed.plot(times, _series(trajectory, "V"), color="tab:purple")
    speed.set_ylabel("airspeed (m/s)", color="tab:purple")
    speed.tick_params(axis="y", labelcolor="tab:purple")
    axes.set_title("Height traded against airspeed")
    return axes


def plot_energy_analysis(trajectory, axes=None):
    """Draw the energy of the cycle in both frames, and the budget that links them.

    This is the figure that explains why dynamic soaring works at all, and
    it needs three panels because the story needs two frames and a
    reconciliation.

    The **air-relative** specific energy, ``h + V^2 / (2 g)``, is the one the
    aerodynamics see: drag acts on the airspeed, and the drag polar knows
    nothing about the ground. The **inertial** specific energy,
    ``h + V_g^2 / (2 g)``, is the one an observer on the water sees, and the
    two differ by the wind the bird is carried along with. Because the wind
    is stronger higher up, the gap between the two curves opens and closes
    as the bird climbs and dives, which is the whole trick made visible: the
    bird is repeatedly trading between two currencies at a favourable rate.

    The third panel is the audit. Differentiating the air-relative energy
    along the equations of motion leaves exactly two terms, drag and the
    wind gradient, and the panel plots both against the finite difference of
    the energy itself. If those do not lie on top of each other, the model
    is generating energy from a sign error rather than from the wind.

    Parameters
    ----------
    trajectory
        The cycle to draw.
    axes
        Three axes to draw on; a new figure when omitted.

    Returns
    -------
    ndarray
        The three axes drawn on.
    """
    if axes is None:
        _, axes = pyplot.subplots(3, 1, figsize=(7.5, 9.0), sharex=True)
    axes = numpy.atleast_1d(axes)
    times = _times(trajectory)
    air = _series(trajectory, "specific_energy")
    ground = _series(trajectory, "ground_energy")

    axes[0].plot(times, air, color=SERIES[0], label="air-relative")
    axes[0].plot(times, ground, color=SERIES[1], label="inertial")
    axes[0].fill_between(times, air, ground, color=SERIES[1], alpha=0.12)
    axes[0].set_ylabel("specific energy (m)")
    axes[0].set_title("Energy in both frames; the gap is the wind carried along")
    axes[0].legend(loc="best")

    # The two frames swing by very different amounts, and that difference
    # is the point, so they share one scale: turning through the wind moves
    # the ground speed far more than the airspeed, and an energy that
    # oscillates with the heading is not one the bird can spend.
    axes[1].axhline(0.0, color="0.6", linewidth=0.8)
    axes[1].plot(times, air - air[0], color=SERIES[0], label="air-relative")
    axes[1].plot(times, ground - ground[0], color=SERIES[1], label="inertial")
    axes[1].set_ylabel("change from the start (m)")
    axes[1].legend(loc="best", fontsize="small")
    axes[1].set_title("Change from the start, on one scale")

    drag = _series(trajectory, "drag_power")
    gradient = _series(trajectory, "gradient_power")
    measured = numpy.gradient(air, times)
    axes[2].axhline(0.0, color="0.6", linewidth=0.8)
    axes[2].plot(times, drag, color=RATE_MAP(0.0), label="drag, always a loss")
    axes[2].plot(times, gradient, color=RATE_MAP(1.0), label="wind gradient")
    axes[2].plot(times, drag + gradient, color="k", linewidth=1.6, label="sum")
    axes[2].plot(
        times,
        measured,
        color="k",
        linestyle=":",
        linewidth=2.0,
        label="measured de/dt",
    )
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("rate (m/s)")
    axes[2].set_title("The budget: the sum must match the measured rate")
    axes[2].legend(loc="best", fontsize="small")
    return axes


def energy_budget(trajectory) -> dict[str, float]:
    """Return the energy accounting of a cycle, integrated over it.

    Parameters
    ----------
    trajectory
        The cycle.

    Returns
    -------
    dict
        The change in each frame, and the drag and wind-gradient work, all
        in m of specific energy.
    """
    times = _times(trajectory)
    air = _series(trajectory, "specific_energy")
    ground = _series(trajectory, "ground_energy")
    return {
        "air_change": float(air[-1] - air[0]),
        "ground_change": float(ground[-1] - ground[0]),
        "drag_work": float(numpy.trapezoid(_series(trajectory, "drag_power"), times)),
        "gradient_work": float(
            numpy.trapezoid(_series(trajectory, "gradient_power"), times)
        ),
    }


PHASES = (
    ("Windward climb", numpy.pi, "climbing", "tab:green"),
    ("Upper turn", 1.5 * numpy.pi, "level, highest", "tab:blue"),
    ("Leeward descent", 2.0 * numpy.pi, "descending", "tab:red"),
    ("Lower turn", 2.5 * numpy.pi, "level, lowest", "tab:orange"),
)
"""The four legs of one loop: name, central phase, height, colour.

The phases follow the cycle
:func:`~gemseo_tempo.problems.albatross.problem.soaring_cycle` builds,
where the heading winds through a full turn and the climb angle lags it by
a quarter cycle. With the wind blowing towards ``+x``, that puts
the climb where the bird is headed into the wind and the descent where it
is headed with it, which is the whole manoeuvre.
"""


def _cycle_track(phase):
    """Return the air-relative ground track and height of a loop.

    Integrating the kinematics with the heading winding through a full
    turn gives a closed loop in the air frame; the height follows the
    climb angle a quarter cycle behind it.

    Parameters
    ----------
    phase
        The cycle phase, in radians.

    Returns
    -------
    tuple
        The crosswind-plane coordinates and the height, each normalized.
    """
    return numpy.sin(phase), -numpy.cos(phase), -numpy.sin(phase)


def _phase_panel(axes, name, centre, height_label, colour, label_heading=False):
    """Draw one leg of the cycle: the track from above, and the height.

    Parameters
    ----------
    axes
        The axis to draw on.
    name
        The leg name, used as the panel title.
    centre
        The phase at the middle of this leg, in radians.
    height_label
        What the height is doing on this leg.
    colour
        The colour of the highlighted arc.
    label_heading
        Whether to name the heading arrow, which only the first panel
        needs to do.
    """
    whole = numpy.linspace(0.0, 2.0 * numpy.pi, 240)
    track_x, track_y, _ = _cycle_track(whole)
    axes.plot(track_x, track_y, color="0.8", linewidth=1.2)

    leg = numpy.linspace(centre - numpy.pi / 4, centre + numpy.pi / 4, 60)
    leg_x, leg_y, _ = _cycle_track(leg)
    axes.plot(leg_x, leg_y, color=colour, linewidth=2.6)

    # The heading is the phase itself, so the arrow is the tangent.
    point_x, point_y, height = _cycle_track(centre)
    tip = (point_x + 0.5 * numpy.cos(centre), point_y + 0.5 * numpy.sin(centre))
    axes.annotate(
        "",
        xy=tip,
        xytext=(point_x, point_y),
        arrowprops={"arrowstyle": "-|>", "color": colour, "linewidth": 2.0},
    )
    if label_heading:
        axes.text(
            tip[0],
            tip[1] + 0.18,
            "heading",
            color=colour,
            fontsize=8,
            ha="center",
        )

    # The wind, blowing towards +x, against which the heading is read.
    for offset in (1.42, 1.62):
        axes.annotate(
            "",
            xy=(0.45, offset),
            xytext=(-0.75, offset),
            arrowprops={"arrowstyle": "-|>", "color": "tab:cyan", "alpha": 0.8},
        )
    axes.text(0.52, 1.44, "wind", color="tab:cyan", fontsize=8, va="center")

    # A height bar on the right: where this leg sits in the gradient, and
    # which way it is going. The wind arrows grow with height, which is
    # the gradient the whole manoeuvre lives off.
    bar_x = 1.95
    axes.plot([bar_x, bar_x], [-1.05, 1.05], color="0.7", linewidth=1.0)
    for level in numpy.linspace(-0.9, 0.9, 5):
        axes.annotate(
            "",
            xy=(bar_x + 0.12 + 0.30 * (level + 1.0), level),
            xytext=(bar_x + 0.12, level),
            arrowprops={"arrowstyle": "-|>", "color": "tab:cyan", "alpha": 0.45},
        )
    axes.text(bar_x, 1.2, "height", fontsize=8, color="0.35", ha="center")
    axes.plot([bar_x], [height], marker="o", color=colour, markersize=8, zorder=5)
    if "level" not in height_label:
        # Start clear of the marker, so the arrow reads as the direction
        # of travel rather than as part of the marker.
        rising = height_label == "climbing"
        start = height + (0.16 if rising else -0.16)
        axes.annotate(
            "",
            xy=(bar_x, start + (0.5 if rising else -0.5)),
            xytext=(bar_x, start),
            arrowprops={"arrowstyle": "-|>", "color": colour, "linewidth": 2.0},
        )

    axes.text(
        0.2,
        -1.85,
        height_label,
        color=colour,
        fontsize=9,
        ha="center",
        fontweight="bold",
    )
    axes.set_title(name, fontsize=10)
    axes.set_xlim(-1.5, 2.95)
    axes.set_ylim(-2.1, 1.95)
    axes.set_aspect("equal")
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)


def plot_cycle_schematic(axes=None):
    """Draw one soaring loop, decomposed into its four legs.

    The manoeuvre is three-dimensional, and drawing it as a single
    side-on wave hides the half of it that matters: the bird climbs while
    headed *into* the wind, turns through the top of the gradient,
    descends while headed *with* it, and turns back at the bottom. Each
    panel shows the same loop from above, with one leg highlighted and the
    heading drawn as the tangent, beside a height bar placing that leg in
    the wind gradient.

    Pure geometry, no data: the track is the closed air-relative loop the
    kinematics give when the heading winds through a full turn, which is
    the cycle :func:`~gemseo_tempo.problems.albatross.problem.soaring_cycle`
    builds and the optimizer refines.

    Parameters
    ----------
    axes
        Four axes to draw on; a new figure when omitted.

    Returns
    -------
    ndarray
        The four axes drawn on.
    """
    if axes is None:
        figure, axes = pyplot.subplots(1, 4, figsize=(11.5, 3.8))
        figure.suptitle("One dynamic-soaring loop, leg by leg", fontsize=11)
    axes = numpy.atleast_1d(axes)
    for index, (axis, phase) in enumerate(zip(axes, PHASES, strict=True)):
        _phase_panel(axis, *phase, label_heading=index == 0)
    return axes


PHASE_LABELS = {
    "windward climb": ((-0.08, 0.0, 0.02), "right"),
    "upper turn": ((0.0, 0.0, 0.12), "center"),
    "leeward descent": ((0.08, 0.0, 0.04), "left"),
    "lower turn": ((0.08, 0.0, -0.08), "left"),
}
"""Where each phase is labelled, as a fraction of the cycle's size."""


def _glider(axes, point, velocity, span, size, color=INK):
    """Draw a small glider glyph: a wing along the span, a body along the path."""
    wing = numpy.stack([point - 0.5 * size * span, point + 0.5 * size * span])
    body = numpy.stack([point - 0.25 * size * velocity, point + 0.35 * size * velocity])
    for piece, width in ((wing, 2.6), (body, 1.8)):
        axes.plot(*piece.T, color=color, linewidth=width, solid_capstyle="round")


def plot_scenario(shape: str = "S", count: int = 2, w_ref=None, axes=None):
    """Draw the scenario: the terrestrial frame, the wind profile, a cycle.

    A schematic after the opening figures of Hong et al. (2023) and
    Koessler (2019), both of which draw an S: the sea, the frame the
    positions are integrated in, the logarithmic wind profile drawn as
    arrows on a vertical plane, and an idealized cycle,
    :func:`schematic_cycle`, laid end to end, with the bird drawn at the
    four phases that make it up. Nothing here is a result: the cycle is
    geometry. Only the wind arrows are computed, from the same wind model
    the dynamics use, so that they are to scale with each other.

    Parameters
    ----------
    shape
        ``"S"`` or ``"O"``; see :func:`schematic_cycle`.
    count
        The number of consecutive cycles drawn; the phases are named on the
        first.
    w_ref
        The reference wind speed the arrows are drawn for; the nominal wind
        when omitted.
    axes
        The 3D axis to draw on; a new one when omitted.

    Returns
    -------
    Axes
        The axis drawn on.
    """
    from gemseo_tempo.problems.albatross.model import wind_speed
    from gemseo_tempo.problems.albatross.problem import NOMINAL_WIND

    if axes is None:
        figure = pyplot.figure(figsize=(8.0, 5.8))
        axes = figure.add_subplot(projection="3d")
    w_ref = NOMINAL_WIND if w_ref is None else w_ref
    cycle = schematic_cycle(shape)
    lap = numpy.stack([_series(cycle, name) for name in ("x", "y", "h")], axis=1)
    shift = lap[-1] - lap[0]
    points = numpy.concatenate([lap + number * shift for number in range(count)])
    east, north, height = points.T
    extent = max(numpy.ptp(east), numpy.ptp(north), 1.0)
    x_range = numpy.array([east.min() - 0.15 * extent, east.max() + 0.15 * extent])
    y_range = numpy.array([north.min() - 0.2 * extent, north.max() + 0.2 * extent])
    top = 1.15 * max(height.max(), 1.0)
    _sea(axes, x_range, y_range)

    # The terrestrial frame, on the sea in front of the second cycle.
    origin = numpy.array([x_range[1] - 0.1 * extent, y_range[1] - 0.3 * extent, 0.0])
    for direction, label in (
        ((0.2 * extent, 0.0, 0.0), "x"),
        ((0.0, 0.2 * extent, 0.0), "y"),
        ((0.0, 0.0, 0.45 * top), "h"),
    ):
        axes.quiver(
            *origin, *direction, color=INK, linewidth=1.2, arrow_length_ratio=0.15
        )
        axes.text(*(origin + 1.12 * numpy.asarray(direction)), label, fontsize=8.5)

    # The wind profile, as arrows on the back plane, to scale with each other.
    levels = numpy.linspace(1.0, 0.95 * top, 7)
    speeds = numpy.array([float(wind_speed(level, w_ref)) for level in levels])
    reach = 0.2 * extent * speeds / speeds.max()
    base = x_range[0]
    for level, arrow_length in zip(levels, reach, strict=True):
        axes.quiver(
            base,
            y_range[1],
            level,
            arrow_length,
            0.0,
            0.0,
            color=SERIES[0],
            linewidth=1.4,
            arrow_length_ratio=0.18,
        )
    profile = numpy.linspace(0.6, 0.95 * top, 60)
    axes.plot(
        [base, base], [y_range[1]] * 2, [0.0, 0.95 * top], color=MUTED, linewidth=0.8
    )
    axes.plot(
        base
        + 0.2
        * extent
        * numpy.array([float(wind_speed(level, w_ref)) for level in profile])
        / speeds.max(),
        0.0 * profile + y_range[1],
        profile,
        color=SERIES[0],
        linewidth=1.0,
    )
    axes.text(
        base,
        y_range[1],
        1.04 * top,
        f"wind W(h), {w_ref:.1f} m/s at 10 m",
        fontsize=8.5,
        ha="center",
    )

    # The cycles, their shadow, and the bird at each phase of every cycle.
    axes.plot(east, north, height, color=SERIES[1], linewidth=2.2)
    axes.plot(east, north, 0.0 * height, color=MUTED, linewidth=0.9, linestyle="--")
    velocity, _, span = _unit_vectors(
        _series(cycle, "gamma"), _series(cycle, "psi"), _series(cycle, "mu")
    )
    for number in range(count):
        for name, index in _phases(cycle).items():
            point = lap[index] + number * shift
            axes.plot(*numpy.stack([point * (1, 1, 0), point]).T, color=MUTED, lw=0.6)
            _glider(axes, point, velocity[index], span[index], 0.06 * extent)
            if number == 0:
                offset, align = PHASE_LABELS[name]
                axes.text(
                    *(point + numpy.asarray(offset) * (extent, extent, top)),
                    name,
                    fontsize=8.5,
                    ha=align,
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.75,
                        "edgecolor": "none",
                        "pad": 1,
                    },
                )

    # Where the first cycle starts, and where the same state recurs.
    axes.scatter(*lap[[0, -1]].T, color=SERIES[1], s=22, depthshade=False)

    axes.set_xlim(*x_range)
    axes.set_ylim(*y_range)
    axes.set_zlim(0.0, top * 1.1)
    # With the axes hidden there is nothing to leave room for, so zoom in.
    # Heights are stretched, as every sketch of the manoeuvre does.
    axes.set_box_aspect((numpy.ptp(x_range), numpy.ptp(y_range), 2.5 * top), zoom=1.1)
    axes.view_init(elev=SCENARIO_VIEW[0], azim=SCENARIO_VIEW[1])
    axes.set_axis_off()
    axes.set_title(
        f"Schematic: the frame, the wind profile and {count} {shape} cycles"
        if count > 1
        else f"Schematic: the frame, the wind profile and an {shape} cycle"
    )
    return axes


def plot_frames(cycle, w_ref=None, axes=None):
    """Take one instant of a cycle apart, frame by frame.

    Three panels, at the windward climb, where the wind pays most:

    - the wind triangle seen from above, ground velocity = air-relative
      velocity + wind, which is the relation between the terrestrial frame
      and the wind-fixed one;
    - the forces in the vertical plane of the air-relative velocity, per
      unit weight: lift and drag are defined against the airspeed, and the
      wind-fixed frame is accelerating, so a fictitious force ``-m dW/dt``
      along x appears there, which is where the energy comes from (Koessler
      2019, Fig. 6);
    - the whole cycle traced in both frames: the ground track, and the
      track in the wind-fixed frame, the ground track less the wind the bird
      has been carried by.

    Parameters
    ----------
    cycle
        The cycle, a trajectory or a mapping of arrays; the wind-fixed
        track needs ``"x_air"`` when the cycle is a mapping that carries it,
        and is integrated from ``"wind"`` otherwise.
    w_ref
        The reference wind speed; read from ``cycle["w_ref"]`` when the
        cycle carries it, the nominal wind otherwise.
    axes
        Three axes to draw on; a new figure when omitted.

    Returns
    -------
    ndarray
        The three axes drawn on.
    """
    from gemseo_tempo.problems.albatross.model import wind_gradient
    from gemseo_tempo.problems.albatross.problem import NOMINAL_WIND
    from gemseo_tempo.problems.albatross.statics import GRAVITY

    if axes is None:
        figure = pyplot.figure(figsize=(8.4, 7.8), layout="constrained")
        grid = figure.add_gridspec(
            2, 2, height_ratios=(1.0, 1.1), width_ratios=(1.0, 1.1)
        )
        axes = numpy.array(
            [
                figure.add_subplot(grid[0, 0]),
                figure.add_subplot(grid[0, 1]),
                figure.add_subplot(grid[1, :]),
            ]
        )
    axes = numpy.atleast_1d(axes)
    w_ref = _scalar(cycle, "w_ref", NOMINAL_WIND) if w_ref is None else w_ref
    times = _times(cycle)
    index = _phases(cycle)["windward climb"]
    speed = _series(cycle, "V")[index]
    gamma = _series(cycle, "gamma")[index]
    psi = _series(cycle, "psi")[index]
    height = _series(cycle, "h")[index]
    wind = _series(cycle, "wind")[index]

    def arrow(axis, tail, tip, color, label, where=1.0, width=2.0, size=1.0):
        """Draw a vector and label it beside the point a fraction along it."""
        tail = numpy.asarray(tail, dtype=float)
        tip = numpy.asarray(tip, dtype=float)
        axis.annotate(
            "",
            xy=tip,
            xytext=tail,
            arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": width},
        )
        direction = tip - tail
        # Push the label off the vector, to its left-hand side.
        normal = numpy.array([-direction[1], direction[0]])
        normal /= max(numpy.linalg.norm(normal), 1e-12)
        anchor = tail + where * direction + 0.09 * size * normal
        axis.text(*anchor, label, fontsize=8.5, color=INK, ha="center", va="center")

    def frame(axis, points):
        """Fit the limits to the vectors drawn, with room for the labels."""
        points = numpy.asarray(points)
        low, high = points.min(axis=0), points.max(axis=0)
        pad = 0.3 * max(numpy.ptp(points, axis=0).max(), 1e-9)
        axis.set_xlim(low[0] - pad, high[0] + pad)
        axis.set_ylim(low[1] - pad, high[1] + pad)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])

    # (a) The wind triangle, horizontal components only.
    air = speed * numpy.cos(gamma) * numpy.array([numpy.cos(psi), numpy.sin(psi)])
    ground = air + (wind, 0.0)
    triangle = axes[0]
    size = max(numpy.linalg.norm(air), numpy.linalg.norm(ground), wind)
    arrow(
        triangle, (0.0, 0.0), air, SERIES[0], "air-relative,\nV cos γ", 0.5, size=size
    )
    arrow(triangle, air, ground, SERIES[2], "wind W(h)", 0.5, size=size)
    arrow(triangle, ground, ground, SERIES[1], "", size=size)
    arrow(triangle, (0.0, 0.0), ground, SERIES[1], "ground", 0.55, size=-size)
    triangle.axhline(0.0, color=MUTED, linewidth=0.6, zorder=0)
    triangle.text(0.0, 0.0, "  x →", color=MUTED, fontsize=8, va="bottom")
    frame(triangle, [(0.0, 0.0), air, ground])
    triangle.set_title("(a) The wind triangle, from above", fontsize=10)

    # (b) Forces per unit weight in the vertical plane of the airspeed.
    load = _series(cycle, "load_factor")[index]
    drag = -_series(cycle, "drag_power")[index] / speed
    mu = _series(cycle, "mu")[index]
    wind_rate = float(wind_gradient(height, w_ref)) * speed * numpy.sin(gamma)
    # Only the part of the x axis along the heading lies in this plane.
    fictitious = -wind_rate / GRAVITY * numpy.cos(psi)
    along = numpy.array([numpy.cos(gamma), numpy.sin(gamma)])
    normal = numpy.array([-numpy.sin(gamma), numpy.cos(gamma)])
    lift = load * numpy.cos(mu) * normal
    forces = axes[1]
    origin = numpy.zeros(2)
    arrow(forces, origin, 0.9 * along, MUTED, "airspeed", 1.15, 1.2, size=0.0)
    arrow(
        forces,
        origin,
        lift,
        SERIES[0],
        f"lift L cos μ\n{load * numpy.cos(mu):.2f} mg",
        0.6,
        size=-6.0,
    )
    arrow(forces, origin, (0.0, -1.0), INK, "weight\n1 mg", 1.25, size=0.0)
    # The two small forces decide the energy, so they are labelled off to
    # the side, with a leader to the arrow.
    small = (
        (-drag * along, RATE_MAP(0.0), f"drag {drag:.2f} mg", (-0.3, -0.45), "right"),
        (
            numpy.array([fictitious, 0.0]),
            SERIES[2],
            f"−m dW/dt, {fictitious:+.2f} mg\n(the frame accelerates)",
            (0.3, -0.45),
            "left",
        ),
    )
    for tip, color, label, place, align in small:
        arrow(forces, origin, tip, color, "", size=0.0, width=2.4)
        forces.annotate(
            label,
            xy=tip,
            xytext=place,
            fontsize=8.5,
            ha=align,
            arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6},
        )
    frame(
        forces,
        [origin, lift, (0.0, -1.0), 0.9 * along, (-1.6, -0.5), (1.9, -0.5)],
    )
    forces.set_xlabel("along the heading →", fontsize=8.5, color=MUTED)
    forces.set_title("(b) Forces on the air path, per unit weight", fontsize=10)

    # (c) The cycle in the terrestrial and in the wind-fixed frame.
    east = _series(cycle, "x")
    north = _series(cycle, "y")
    if isinstance(cycle, Mapping) and "x_air" in cycle:
        east_air = numpy.asarray(cycle["x_air"])
    else:
        carried = numpy.concatenate(
            [
                [0.0],
                numpy.cumsum(
                    0.5
                    * (_series(cycle, "wind")[1:] + _series(cycle, "wind")[:-1])
                    * numpy.diff(times)
                ),
            ]
        )
        east_air = east - carried
    tracks = axes[2]
    tracks.plot(
        east, north, color=SERIES[1], linewidth=2.0, label="terrestrial: over the sea"
    )
    tracks.plot(
        east_air,
        north,
        color=SERIES[0],
        linewidth=2.0,
        label="wind-fixed: through the air",
    )
    tracks.plot(east[0], north[0], marker="o", color=INK, markersize=6)
    tracks.annotate(
        "start",
        (east[0], north[0]),
        textcoords="offset points",
        xytext=(6, -12),
        fontsize=8.5,
    )
    tracks.plot(east[index], north[index], marker="o", color=SERIES[2], markersize=7)
    tracks.plot(
        east_air[index], north[index], marker="o", color=SERIES[2], markersize=7
    )
    tracks.set_aspect("equal")
    tracks.set_xlabel("x, downwind (m)")
    tracks.set_ylabel("y, crosswind (m)")
    tracks.legend(loc="best", fontsize="small")
    tracks.set_title(
        "(c) One cycle in both frames; the dots mark the instant above", fontsize=10
    )
    return axes


def plot_maneuver_pattern(cycles, count: int = 3, axes=None):
    """Draw several consecutive cycles as a wing-span ribbon, one panel each.

    After Barnes (2004) as reproduced by Koessler (2019), Figs. 2-4. A
    periodic cycle repeats with a shift, so the optimum is laid end to end
    ``count`` times; the ribbon follows the wing span, which makes the bank
    visible, and is widened by a stated factor so that it reads at the
    scale of the pattern. Its shadow falls on the sea.

    Parameters
    ----------
    cycles
        The cycles to draw, keyed by panel title.
    count
        The number of consecutive cycles.
    axes
        One 3D axis per cycle; a new figure when omitted.

    Returns
    -------
    list
        The axes drawn on.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from gemseo_tempo.problems.albatross.statics import DESIGN_DEFAULTS

    if axes is None:
        figure = pyplot.figure(figsize=(6.4, 4.4 * len(cycles)), layout="none")
        figure.subplots_adjust(left=0.08, right=0.92, bottom=0.1, top=0.95)
        _keep_whole(figure)
        axes = [
            figure.add_subplot(len(cycles), 1, row + 1, projection="3d")
            for row in range(len(cycles))
        ]
    for axis, color, (title, cycle) in zip(axes, SERIES, cycles.items(), strict=False):
        east = _series(cycle, "x")
        north = _series(cycle, "y")
        height = _series(cycle, "h")
        _, _, span = _unit_vectors(
            _series(cycle, "gamma"), _series(cycle, "psi"), _series(cycle, "mu")
        )
        points = numpy.stack([east, north, height], axis=1)
        shift = points[-1] - points[0]
        shift[2] = 0.0
        extent = max(numpy.ptp(east), numpy.ptp(north), 1.0)
        wing_span = _scalar(cycle, "wing_span", DESIGN_DEFAULTS["wing_span"])
        widening = 0.12 * extent / wing_span
        half = 0.5 * wing_span * widening

        laps = [points + lap * shift for lap in range(count)]
        every = numpy.concatenate(laps)
        margin = 0.1 * extent
        x_range = numpy.array([every[:, 0].min() - margin, every[:, 0].max() + margin])
        y_range = numpy.array([every[:, 1].min() - margin, every[:, 1].max() + margin])
        _sea(axis, x_range, y_range)
        for lap in laps:
            left = lap + half * span
            right = lap - half * span
            quads = [
                [left[k], left[k + 1], right[k + 1], right[k]]
                for k in range(len(lap) - 1)
            ]
            axis.add_collection3d(
                Poly3DCollection(
                    quads, facecolor=color, edgecolor=color, linewidths=0.2, alpha=0.9
                )
            )
            axis.plot(*lap[:, :2].T, 0.0 * lap[:, 2], color=MUTED, linewidth=0.8)
        axis.quiver(
            x_range[0] + margin,
            y_range[0] + margin,
            0.0,
            0.25 * extent,
            0.0,
            0.0,
            color=INK,
            linewidth=1.0,
            arrow_length_ratio=0.2,
        )
        axis.text(
            x_range[0] + margin,
            y_range[0] + margin,
            0.05 * extent,
            "wind",
            fontsize=8.5,
        )
        axis.set_xlim(*x_range)
        axis.set_ylim(*y_range)
        top = 1.1 * height.max()
        axis.set_zlim(0.0, top)
        # A pattern is far longer than it is high: stretch the heights so
        # that the climbs read, and say by how much.
        stretch = max(1.5, 0.3 * max(numpy.ptp(x_range), numpy.ptp(y_range)) / top)
        axis.set_box_aspect(
            (numpy.ptp(x_range), numpy.ptp(y_range), stretch * top), zoom=1.25
        )
        # Seen from beside the direction the pattern travels, a little
        # behind it, so that several cycles span the width of the panel.
        travel = numpy.degrees(numpy.arctan2(shift[1], shift[0]))
        axis.view_init(elev=30.0, azim=travel - 105.0)
        _clean_3d(axis)
        axis.set_xlabel("x, downwind (m)", fontsize=8)
        axis.set_ylabel("y, crosswind (m)", fontsize=8)
        axis.set_zlabel("h (m)", fontsize=8)
        for ticks in (axis.xaxis, axis.yaxis):
            ticks.set_major_locator(MaxNLocator(3))
        axis.set_zticks([0.0, 10.0 * numpy.floor(height.max() / 10.0)])
        axis.set_title(
            f"{title}: {count} cycles\n"
            f"(ribbon {widening:.1f}× the span, heights ×{stretch:.1f})",
            fontsize=10,
        )
    return axes


def plot_cycle_shapes(count: int = 3, axes=None):
    """Draw the O and the S cycle as the literature sketches them.

    Schematic, after the maneuver figures of Barnes (2004) as reproduced by
    Koessler (2019), Figs. 2-4: :func:`schematic_cycle` of each shape, laid
    end to end and drawn as a wing-span ribbon over the sea by
    :func:`plot_maneuver_pattern`. Nothing here is optimized.

    Parameters
    ----------
    count
        The number of consecutive cycles.
    axes
        Two 3D axes to draw on; a new figure when omitted.

    Returns
    -------
    list
        The axes drawn on.
    """
    axes = plot_maneuver_pattern(
        {
            "O, a loop (schematic)": schematic_cycle("O"),
            "S, a snake (schematic)": schematic_cycle("S"),
        },
        count,
        axes,
    )
    # The snake travels across the wind: look down on it so that its
    # heading reversals show.
    axes[1].view_init(elev=45.0, azim=-70.0)
    return axes


def plot_wind_profile(reference_winds=(6.0, 12.0, 20.0), axes=None):
    """Draw the wind profile and its gradient against height.

    This is the figure that explains why the bird flies low. The wind is
    slowed to nothing at the water and recovers over the first few metres,
    so the gradient it lives off is steepest exactly where flying is most
    dangerous, and has largely died away by the height a cycle tops out at.

    Parameters
    ----------
    reference_winds
        The wind speeds at the reference height to draw, in m/s.
    axes
        Two axes to draw on; a new figure when omitted.

    Returns
    -------
    ndarray
        The two axes drawn on.
    """
    from gemseo_tempo.problems.albatross.model import MIN_HEIGHT
    from gemseo_tempo.problems.albatross.model import REFERENCE_HEIGHT
    from gemseo_tempo.problems.albatross.model import wind_gradient
    from gemseo_tempo.problems.albatross.model import wind_speed

    if axes is None:
        _, axes = pyplot.subplots(1, 2, figsize=(9.0, 4.2), sharey=True)
    axes = numpy.atleast_1d(axes)
    heights = numpy.linspace(0.2, 45.0, 400)
    for wind in reference_winds:
        label = f"{wind:g} m/s at {REFERENCE_HEIGHT:g} m"
        axes[0].plot(
            [float(wind_speed(h, wind)) for h in heights], heights, label=label
        )
        axes[1].plot(
            [float(wind_gradient(h, wind)) for h in heights], heights, label=label
        )
    for axis in axes:
        axis.axhline(MIN_HEIGHT, color="tab:red", linestyle=":", linewidth=1.0)
        axis.axhline(REFERENCE_HEIGHT, color="0.5", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("wind speed (m/s)")
    axes[0].set_ylabel("height (m)")
    axes[0].set_title("The profile")
    axes[0].legend(loc="lower right", fontsize="small")
    axes[1].set_xlabel("gradient dW/dh (1/s)")
    axes[1].set_xscale("log")
    axes[1].set_title("What the bird actually lives off")
    return axes


def plot_polar(design=None, axes=None):
    """Draw the drag polar and the glide ratio of a wing.

    Parameters
    ----------
    design
        Overrides of the default design.
    axes
        Two axes to draw on; a new figure when omitted.

    Returns
    -------
    ndarray
        The two axes drawn on.
    """
    from gemseo_tempo.problems.albatross.problem import _design_values
    from gemseo_tempo.problems.albatross.statics import create_statics

    if axes is None:
        _, axes = pyplot.subplots(1, 2, figsize=(9.0, 4.0))
    axes = numpy.atleast_1d(axes)
    statics = create_statics()(_design_values(design))
    profile = float(statics["cd0"])
    induced = float(statics["induced_factor"])
    lift_limit = float(statics["cl_max"])

    lift = numpy.linspace(0.0, lift_limit, 300)
    drag = profile + induced * lift**2
    ratio = numpy.divide(lift, drag, out=numpy.zeros_like(lift), where=drag > 0.0)
    best = int(numpy.argmax(ratio))

    axes[0].plot(drag, lift, color="tab:blue")
    axes[0].set_xlabel("drag coefficient")
    axes[0].set_ylabel("lift coefficient")
    axes[0].set_title("Drag polar")

    axes[1].plot(lift, ratio, color="tab:blue")
    axes[1].axvline(
        lift[best],
        color="tab:green",
        linestyle="--",
        linewidth=1.0,
        label=f"best glide {ratio[best]:.1f} at CL {lift[best]:.2f}",
    )
    axes[1].axvline(
        lift_limit,
        color="tab:red",
        linestyle=":",
        linewidth=1.4,
        label=f"stall at CL {lift_limit:.2f}",
    )
    axes[1].set_xlabel("lift coefficient")
    axes[1].set_ylabel("lift over drag")
    axes[1].set_title("Glide ratio, and where the cycle is forced to sit")
    axes[1].legend(loc="lower right", fontsize="small")
    return axes


def plot_section_trade(axes=None):
    """Draw the cruise-against-manoeuvre trade the wing section makes.

    A thin, clean section glides further; a thick, cambered one holds more
    lift and turns tighter. The two cannot be had at once, and which end of
    the trade wins turns out to depend entirely on what the bird is being
    asked to do.

    Parameters
    ----------
    axes
        The axis to draw on; a new one when omitted.

    Returns
    -------
    Axes
        The axis drawn on.
    """
    from gemseo_tempo.problems.albatross.statics import DESIGN_DEFAULTS
    from gemseo_tempo.problems.albatross.statics import create_statics

    if axes is None:
        _, axes = pyplot.subplots(figsize=(7.0, 4.0))
    chain = create_statics()
    sections = numpy.linspace(0.0, 1.0, 41)
    ratios, lifts = [], []
    for section in sections:
        outputs = chain({**DESIGN_DEFAULTS, "section": float(section)})
        profile = float(outputs["cd0"])
        induced = float(outputs["induced_factor"])
        ratios.append(1.0 / (2.0 * numpy.sqrt(profile * induced)))
        lifts.append(float(outputs["cl_max"]))

    axes.plot(sections, ratios, color="tab:blue")
    axes.set_xlabel("section: thin and clean (0) to thick and cambered (1)")
    axes.set_ylabel("best glide ratio", color="tab:blue")
    axes.tick_params(axis="y", labelcolor="tab:blue")
    lift_axis = axes.twinx()
    lift_axis.plot(sections, lifts, color="tab:red")
    lift_axis.set_ylabel("maximum lift coefficient", color="tab:red")
    lift_axis.tick_params(axis="y", labelcolor="tab:red")
    axes.set_title("Cruise against manoeuvre: the two cannot be had at once")
    return axes


def plot_pareto(points, wings=None, axes=None):
    """Draw the trade between the weakest wind and the fastest crossing.

    Parameters
    ----------
    points
        Rows of ``[w_ref, crossing time]``, as
        :func:`.solve_pareto_front` returns them.
    wings
        Optional ``{label: (w_ref, time)}`` of particular designs to mark on
        the curve. The points must be in the **same** two objectives as the
        front, that is the wind each design needs to cover the same distance
        and the time it then takes. A minimum wind measured without a
        distance requirement is a different quantity and will appear to
        dominate the front, which is a plotting error rather than a result.
    axes
        The axis to draw on; a new one when omitted.

    Returns
    -------
    Axes
        The axis drawn on.
    """
    if axes is None:
        _, axes = pyplot.subplots(figsize=(7.0, 4.5))
    points = numpy.asarray(points)
    axes.plot(points[:, 0], points[:, 1], "o-", color="tab:blue", label="Pareto front")
    for label, (wind, time) in (wings or {}).items():
        axes.plot(wind, time, "*", markersize=14, label=label)
    axes.set_xlabel("reference wind the bird needs (m/s)")
    axes.set_ylabel("time to cross (s)")
    axes.set_title("Weaker wind or a faster crossing, not both")
    axes.legend(loc="best", fontsize="small")
    return axes
