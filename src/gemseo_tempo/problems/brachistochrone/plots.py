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
"""Figures of the brachistochrone.

The introductory problem, so the figures are the ones that make a
capability visible rather than the ones that make a result: what the three
variants do to the shape of the descent, what a control law looks like,
and how the error of a transcription falls as the grid is refined.

Every function takes an ``axes`` and returns it, so a caller can compose
them, and draws on a fresh one when given none.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy  # noqa: E402
from matplotlib import pyplot  # noqa: E402

VARIANT_COLOURS = {
    "ideal": "tab:blue",
    "friction": "tab:orange",
    "drag": "tab:green",
}
"""One colour per variant, so the same physics is the same colour."""


def plot_kinematics(axes=None):
    """Draw the bead-on-a-wire geometry the state space describes.

    Pure geometry: no trajectory, no data, nothing to solve. A short
    section of wire, the bead on it, and the path angle `gamma` and speed
    `v` annotated the way they appear in the equations of motion.

    Parameters
    ----------
    axes
        The axis to draw on; a new one when omitted.

    Returns
    -------
    Axes
        The axis drawn on.
    """
    if axes is None:
        _, axes = pyplot.subplots(figsize=(5.4, 4.2))
    wire_x = numpy.linspace(0.0, 2.0, 200)
    wire_y = 0.9 * (1.0 - numpy.cos(1.6 * wire_x))
    axes.plot(wire_x, wire_y, color="black", linewidth=1.5)

    bead_x = 1.1
    bead_y = numpy.interp(bead_x, wire_x, wire_y)
    slope = numpy.interp(bead_x, wire_x[:-1], numpy.diff(wire_y) / numpy.diff(wire_x))
    gamma = numpy.arctan(slope)
    axes.plot([bead_x], [bead_y], marker="o", markersize=14.0, color="tab:blue")

    length = 0.45
    axes.annotate(
        "",
        xy=(bead_x + length * numpy.cos(gamma), bead_y + length * numpy.sin(gamma)),
        xytext=(bead_x, bead_y),
        arrowprops={"arrowstyle": "-|>", "color": "tab:red", "linewidth": 1.6},
    )
    axes.text(
        bead_x + length * numpy.cos(gamma) + 0.03,
        bead_y + length * numpy.sin(gamma),
        "v",
        color="tab:red",
        fontsize=11,
    )
    axes.annotate(
        "",
        xy=(bead_x + 0.5, bead_y),
        xytext=(bead_x, bead_y),
        arrowprops={"arrowstyle": "-", "color": "grey", "linewidth": 1.0},
    )
    arc = numpy.linspace(0.0, gamma, 30)
    axes.plot(
        bead_x + 0.28 * numpy.cos(arc),
        bead_y + 0.28 * numpy.sin(arc),
        color="grey",
        linewidth=1.0,
    )
    axes.text(
        bead_x + 0.32,
        bead_y + 0.06,
        "gamma",
        fontsize=10,
        color="grey",
    )
    axes.annotate(
        "",
        xy=(bead_x, bead_y + 0.35),
        xytext=(bead_x, bead_y),
        arrowprops={"arrowstyle": "-|>", "color": "black", "linewidth": 1.2},
    )
    axes.text(bead_x + 0.03, bead_y + 0.35, "g", fontsize=10)

    axes.invert_yaxis()
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m), downward")
    axes.set_title("The bead, the wire, and the path angle")
    axes.set_aspect("equal")
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)
    return axes


def plot_descent(paths, axes=None, cycloid=None):
    """Draw the descent paths, with depth increasing downwards.

    Parameters
    ----------
    paths
        A mapping of label to ``(x, y)`` arrays.
    axes
        The axis to draw on; a new one when omitted.
    cycloid
        The analytic ``(x, y)`` of the ideal solution, drawn dashed
        underneath as the answer the numerics are trying to reproduce.

    Returns
    -------
    Axes
        The axis drawn on.
    """
    if axes is None:
        _, axes = pyplot.subplots(figsize=(6.4, 4.0))
    if cycloid is not None:
        axes.plot(
            numpy.asarray(cycloid[0]),
            numpy.asarray(cycloid[1]),
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="cycloid (exact)",
        )
    for label, (x, y) in paths.items():
        axes.plot(
            numpy.asarray(x),
            numpy.asarray(y),
            color=VARIANT_COLOURS.get(label),
            marker="o",
            markersize=3.0,
            label=label,
        )
    axes.invert_yaxis()
    axes.set_xlabel("horizontal position x (m)")
    axes.set_ylabel("depth y (m)")
    axes.set_title("The quickest way down")
    axes.legend()
    axes.grid(alpha=0.3)
    return axes


def plot_control_law(times, laws, axes=None):
    """Draw one or more path-angle laws against time.

    Parameters
    ----------
    times
        The save times, shared by every law.
    laws
        A mapping of label to path angle in radians.
    axes
        The axis to draw on; a new one when omitted.

    Returns
    -------
    Axes
        The axis drawn on.
    """
    if axes is None:
        _, axes = pyplot.subplots(figsize=(6.4, 3.2))
    for label, angles in laws.items():
        axes.plot(
            numpy.asarray(times), numpy.degrees(numpy.asarray(angles)), label=label
        )
    axes.set_xlabel("time (s)")
    axes.set_ylabel("path angle (deg)")
    axes.set_title("How the bead is steered")
    axes.legend()
    axes.grid(alpha=0.3)
    return axes


def plot_convergence(sizes, errors, axes=None):
    """Draw the error against the discretization, on log axes.

    Parameters
    ----------
    sizes
        The segment counts.
    errors
        A mapping of label to relative error at each size.
    axes
        The axis to draw on; a new one when omitted.

    Returns
    -------
    Axes
        The axis drawn on.
    """
    if axes is None:
        _, axes = pyplot.subplots(figsize=(6.4, 3.6))
    sizes = numpy.asarray(sizes, dtype=float)
    for label, values in errors.items():
        values = numpy.asarray(values, dtype=float)
        # A converged transcription reaches the reference exactly, and a
        # zero cannot be drawn on a log axis; floor it at the smallest
        # error worth distinguishing rather than dropping the point.
        axes.plot(sizes, numpy.maximum(values, 1e-12), marker="o", label=label)
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("segments")
    axes.set_ylabel("relative error in the descent time")
    axes.set_title("Refining the grid")
    axes.legend()
    axes.grid(alpha=0.3, which="both")
    return axes


def plot_gradient_cost(n_design, ad_seconds, fd_seconds, axes=None):
    """Draw the cost of one gradient, by AD and by finite differences.

    The figure the transcription guide's AD-changes-the-comparison
    argument turns on: AD flat against the design dimension, finite
    differences growing with it, both measured on the same objective at
    the same points.

    Parameters
    ----------
    n_design
        The design dimension at each measurement.
    ad_seconds
        The time for one reverse-mode gradient, at each dimension.
    fd_seconds
        The time for one central-difference gradient, at each dimension.
    axes
        The axis to draw on; a new one when omitted.

    Returns
    -------
    Axes
        The axis drawn on.
    """
    if axes is None:
        _, axes = pyplot.subplots(figsize=(6.4, 4.0))
    n_design = numpy.asarray(n_design, dtype=float)
    axes.plot(
        n_design,
        numpy.asarray(ad_seconds, dtype=float),
        marker="o",
        color="tab:blue",
        label="automatic differentiation",
    )
    axes.plot(
        n_design,
        numpy.asarray(fd_seconds, dtype=float),
        marker="s",
        color="tab:red",
        label="finite differences",
    )
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("design dimension")
    axes.set_ylabel("time for one gradient (s)")
    axes.set_title("The cost of a gradient, AD against finite differences")
    axes.legend()
    axes.grid(alpha=0.3, which="both")
    return axes


def plot_history(iterations, series, axes=None, log=True):
    """Draw one or more quantities against the optimizer iteration.

    Parameters
    ----------
    iterations
        The iteration numbers.
    series
        A mapping of label to value at each iteration.
    axes
        The axis to draw on; a new one when omitted.
    log
        Whether to use a logarithmic vertical axis, which is what a
        constraint violation falling towards zero wants.

    Returns
    -------
    Axes
        The axis drawn on.
    """
    if axes is None:
        _, axes = pyplot.subplots(figsize=(6.4, 3.4))
    for label, values in series.items():
        axes.plot(
            numpy.asarray(iterations), numpy.asarray(values), marker=".", label=label
        )
    if log:
        axes.set_yscale("log")
    axes.set_xlabel("iteration")
    axes.set_title("What the optimizer did")
    axes.legend()
    axes.grid(alpha=0.3, which="both")
    return axes


__all__ = [
    "VARIANT_COLOURS",
    "plot_control_law",
    "plot_convergence",
    "plot_descent",
    "plot_gradient_cost",
    "plot_history",
    "plot_kinematics",
]
