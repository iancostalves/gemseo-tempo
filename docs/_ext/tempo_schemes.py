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
"""Schematic diagrams for the concept pages, drawn rather than imported.

These are pure geometry: no stored data, nothing solved, safe to draw live
in a ``{plot}`` block every time the documentation builds. They exist for
one figure, the transcription comparison in
:doc:`../concepts/transcriptions`, and are matplotlib rather than TikZ
because this project draws its own figures at documentation build time
throughout — see :mod:`tempo_docdata` for why results are precomputed but
a drawing like this one never is.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy  # noqa: E402
from matplotlib import pyplot  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

VARIABLE = "#0050b4"
"""A design variable the optimizer holds directly."""

COMPUTED = "#333333"
"""A value obtained by integration or interpolation, not held directly."""

CONSTRAINED = "#b40000"
"""A constraint location."""


def _block(axes, xy, width, height, text, *, fill="#e8eef7", edge="#333333"):
    """Draw one labelled rounded box.

    Parameters
    ----------
    axes
        The axis to draw on.
    xy
        The lower-left corner.
    width
        The box width.
    height
        The box height.
    text
        The label.
    fill
        The face colour.
    edge
        The edge colour.

    Returns
    -------
    matplotlib.patches.FancyBboxPatch
        The box, for arrows to attach to.
    """
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=fill,
        edgecolor=edge,
        linewidth=1.1,
    )
    axes.add_patch(box)
    axes.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=9,
    )
    return box


def _arrow(axes, start, end, label="", *, color="#666666"):
    """Draw a curved arrow with a small label near its midpoint."""
    arrow = FancyArrowPatch(
        start,
        end,
        connectionstyle="arc3,rad=0.15",
        arrowstyle="-|>",
        mutation_scale=10,
        color=color,
        linewidth=1.0,
    )
    axes.add_patch(arrow)
    if label:
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        axes.text(
            midpoint[0],
            midpoint[1] + 0.03,
            label,
            fontsize=7.5,
            color=color,
            ha="center",
        )


def _dataflow_panel(axes, title, boxes_and_arrows):
    """Draw one Opt/Traj/Obj dataflow diagram.

    Parameters
    ----------
    axes
        The axis to draw on.
    title
        The panel title.
    boxes_and_arrows
        A callable drawing the boxes and arrows onto ``axes``.
    """
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_aspect("equal")
    axes.axis("off")
    axes.set_title(title, fontsize=10, fontweight="bold")
    boxes_and_arrows(axes)


def _triangle_dataflow(axes, middle_label, forward_label, feedback_label):
    """Draw the shared Opt/middle/Objective triangle layout.

    Parameters
    ----------
    axes
        The axis to draw on.
    middle_label
        The label of the middle box (the trajectory or the defects).
    forward_label
        The arrow label from the optimizer to the middle box.
    feedback_label
        The arrow label from the middle box to the objective.
    """
    _block(axes, (0.03, 0.66), 0.32, 0.24, "Optimizer", fill="#dde8fb")
    _block(axes, (0.62, 0.66), 0.35, 0.24, middle_label)
    _block(axes, (0.30, 0.06), 0.35, 0.24, "Objective", fill="#fbe1e1")
    # Optimizer -> middle, along the top.
    _arrow(axes, (0.35, 0.80), (0.62, 0.80), forward_label)
    # middle -> objective, down the right side.
    _arrow(axes, (0.75, 0.66), (0.58, 0.30), feedback_label)
    # objective -> optimizer, back along the left side, closing the loop.
    _arrow(axes, (0.32, 0.18), (0.12, 0.66), "f_obj")


def _shooting_dataflow(axes):
    _triangle_dataflow(axes, "Trajectory\n(integrator)", "u(t_c)", "s(t), u(t)")


def _collocation_dataflow(axes):
    _triangle_dataflow(axes, "Defects\n(algebraic)", "s(t_c), u(t_c)", "zeta_c")


def _control_panel(axes, title, node_style):
    """Draw a control-trajectory panel with nodes styled per transcription."""
    times = numpy.linspace(0.0, 4.0, 200)
    control = 0.5 + 0.6 * times - 0.25 * times**2 + 0.04 * times**3
    axes.plot(times, control, color="#222222", linewidth=1.3)
    node_times = numpy.array([0.0, 1.0, 2.0, 3.0, 4.0])
    node_values = 0.5 + 0.6 * node_times - 0.25 * node_times**2 + 0.04 * node_times**3
    if node_style == "variable":
        axes.scatter(node_times, node_values, color=VARIABLE, zorder=5, s=28)
    axes.set_title(title, fontsize=10, fontweight="bold")
    axes.set_xlabel("t", fontsize=8)
    axes.set_ylabel("u(t)", fontsize=8)
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)


def _state_panel(axes, title, style):
    """Draw a state-trajectory panel: smooth (shooting) or node-held (collocation)."""
    times = numpy.linspace(0.0, 4.0, 200)
    state = 0.4 + 0.55 * times - 0.28 * times**2 + 0.055 * times**3
    axes.plot(times, state, color="#222222", linewidth=1.3)
    node_times = numpy.array([0.0, 1.3, 2.6, 3.9])
    node_values = 0.4 + 0.55 * node_times - 0.28 * node_times**2 + 0.055 * node_times**3
    if style == "collocation":
        axes.scatter(node_times, node_values, color=VARIABLE, zorder=5, s=28)
        collocation_times = numpy.array([0.65, 2.0, 3.25])
        collocation_values = (
            0.4
            + 0.55 * collocation_times
            - 0.28 * collocation_times**2
            + 0.055 * collocation_times**3
        )
        axes.scatter(
            collocation_times,
            collocation_values,
            facecolor="white",
            edgecolor=CONSTRAINED,
            zorder=6,
            s=26,
            linewidth=1.2,
        )
    else:
        axes.scatter(node_times[:1], node_values[:1], color=VARIABLE, zorder=5, s=28)
        axes.scatter(node_times[-1:], node_values[-1:], color=COMPUTED, zorder=5, s=22)
    axes.set_title(title, fontsize=10, fontweight="bold")
    axes.set_xlabel("t", fontsize=8)
    axes.set_ylabel("s(t)", fontsize=8)
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)


def plot_transcription_scheme(axes=None):
    """Draw the transcription comparison: discretization, then dataflow.

    Two columns, single shooting and collocation. The top row shows how
    the control is held (every node a design variable for both, since
    both parameterize the control at nodes); the middle row shows how the
    *state* is held — a single variable at the start and a computed value
    at the end for shooting, every node a variable tied down by defects at
    the collocation points for collocation; the bottom row is the dataflow
    between the optimizer, the trajectory or defects, and the objective.
    Filled blue points are decision variables the optimizer holds
    directly; open red points are constraint locations; the plain black
    point is a value obtained by integration rather than held.

    Parameters
    ----------
    axes
        Unused; this figure always makes its own 2x3 grid, since a single
        axis cannot hold six independent panels. Present for interface
        consistency with the rest of the project's plot functions.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    del axes
    figure, grid = pyplot.subplots(3, 2, figsize=(7.2, 8.0))

    _control_panel(grid[0][0], "Single shooting — control", "variable")
    _control_panel(grid[0][1], "Collocation — control", "variable")
    _state_panel(grid[1][0], "Single shooting — state", "shooting")
    _state_panel(grid[1][1], "Collocation — state", "collocation")
    _dataflow_panel(grid[2][0], "Single shooting — dataflow", _shooting_dataflow)
    _dataflow_panel(grid[2][1], "Collocation — dataflow", _collocation_dataflow)

    handles = [
        pyplot.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=VARIABLE,
            label="decision variable",
        ),
        pyplot.Line2D(
            [], [], marker="o", linestyle="none", color=COMPUTED, label="computed value"
        ),
        pyplot.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=CONSTRAINED,
            label="constraint location",
        ),
    ]
    figure.legend(
        handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    return figure


__all__ = ["plot_transcription_scheme"]
