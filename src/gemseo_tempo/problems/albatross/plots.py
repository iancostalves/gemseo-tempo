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

Four views, each answering a different question. :func:`plot_trajectory`
shows where the bird went, :func:`plot_controls` how it was flown,
:func:`plot_energy` whether the cycle paid for itself, and
:func:`plot_energy_analysis` why.

The last one is the one worth reading carefully, and it is described where
it is defined.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy  # noqa: E402
from matplotlib import pyplot  # noqa: E402


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
    for group in (trajectory.states, trajectory.controls, trajectory.outputs):
        if name in group:
            return numpy.asarray(group[name])
    raise KeyError(f"No series named {name!r} in the trajectory.")


def plot_trajectory(trajectory, axes=None):
    """Draw the flight path in three dimensions, coloured by energy.

    Parameters
    ----------
    trajectory
        The cycle to draw.
    axes
        The 3D axis to draw on; a new one when omitted.

    Returns
    -------
    Axes
        The axis drawn on.
    """
    if axes is None:
        figure = pyplot.figure(figsize=(7.0, 5.5))
        axes = figure.add_subplot(projection="3d")
    east = _series(trajectory, "x")
    north = _series(trajectory, "y")
    height = _series(trajectory, "h")
    energy = _series(trajectory, "specific_energy")

    points = numpy.stack([east, north, height], axis=1)
    for start, end, value in zip(points[:-1], points[1:], energy[:-1], strict=True):
        axes.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=pyplot.cm.viridis(
                (value - energy.min()) / max(numpy.ptp(energy), 1e-9)
            ),
            linewidth=2.0,
        )
    axes.set_xlabel("downwind x (m)")
    axes.set_ylabel("crosswind y (m)")
    axes.set_zlabel("height (m)")
    axes.set_title("Soaring cycle, coloured by specific energy")
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
    times = numpy.asarray(trajectory.ts)
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
    times = numpy.asarray(trajectory.ts)
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
    times = numpy.asarray(trajectory.ts)
    air = _series(trajectory, "specific_energy")
    ground = _series(trajectory, "ground_energy")

    axes[0].plot(times, air, color="tab:blue", label="air-relative")
    axes[0].plot(times, ground, color="tab:orange", label="inertial")
    axes[0].fill_between(times, air, ground, color="tab:orange", alpha=0.12)
    axes[0].set_ylabel("specific energy (m)")
    axes[0].set_title("Energy in both frames; the gap is the wind carried along")
    axes[0].legend(loc="best")

    # The two frames swing by wildly different amounts: turning through the
    # wind moves the ground speed far more than the airspeed, so plotted on
    # one scale the air-relative curve would be a flat line. Each gets its
    # own axis, coloured to match its curve.
    axes[1].axhline(0.0, color="0.6", linewidth=0.8)
    axes[1].plot(times, air - air[0], color="tab:blue")
    axes[1].set_ylabel("air-relative change (m)", color="tab:blue")
    axes[1].tick_params(axis="y", labelcolor="tab:blue")
    inertial = axes[1].twinx()
    inertial.plot(times, ground - ground[0], color="tab:orange")
    inertial.set_ylabel("inertial change (m)", color="tab:orange")
    inertial.tick_params(axis="y", labelcolor="tab:orange")
    axes[1].set_title("Change from the start, on each frame's own scale")

    drag = _series(trajectory, "drag_power")
    gradient = _series(trajectory, "gradient_power")
    measured = numpy.gradient(air, times)
    axes[2].axhline(0.0, color="0.6", linewidth=0.8)
    axes[2].plot(times, drag, color="tab:red", label="drag, always a loss")
    axes[2].plot(times, gradient, color="tab:green", label="wind gradient")
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
    times = numpy.asarray(trajectory.ts)
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
