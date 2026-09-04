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
