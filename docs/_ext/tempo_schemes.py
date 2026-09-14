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


def _elbow(axes, start, end, label="", *, color="#666666", label_offset=(0.0, 0.03)):
    """Draw one N2 elbow arrow: straight out, square turn, straight in.

    The square turn is the whole point. An N2 chart is read as a matrix, so
    data leaves a block sideways along its row and enters the next one
    vertically down its column; a curved arrow drawn corner to corner says
    nothing about which row and column it travelled.

    Parameters
    ----------
    axes
        The axis to draw on.
    start
        Where the arrow leaves, on the source block's east or west face.
    end
        Where the arrow arrives, on the target block's north or south face.
    label
        The data carried, drawn at the corner.
    color
        The arrow colour.
    label_offset
        Where to put the label relative to the corner.
    """
    corner = (end[0], start[1])
    axes.plot(
        [start[0], corner[0]],
        [start[1], corner[1]],
        color=color,
        linewidth=1.0,
        solid_capstyle="butt",
    )
    arrow = FancyArrowPatch(
        corner,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        color=color,
        linewidth=1.0,
        shrinkA=0.0,
        shrinkB=0.0,
    )
    axes.add_patch(arrow)
    if label:
        axes.text(
            corner[0] + label_offset[0],
            corner[1] + label_offset[1],
            label,
            fontsize=7.5,
            color=color,
            ha="left" if label_offset[0] >= 0.0 else "right",
            va="bottom" if label_offset[1] >= 0.0 else "top",
        )


# The three blocks sit on a descending diagonal, so that every arrow runs
# along one block's row and down another's column, which is what makes the
# picture an N2 chart rather than three boxes and some arrows.
_BOX = (0.27, 0.17)
"""The width and height of an N2 block."""

_OPT = (0.02, 0.72)
_TRAJ = (0.36, 0.40)
_OBJ = (0.70, 0.08)
"""The lower-left corners of the optimizer, trajectory and objective blocks."""


def _centre(corner):
    """Return the centre of a block from its lower-left corner."""
    return (corner[0] + _BOX[0] / 2, corner[1] + _BOX[1] / 2)


def _n2_panel(axes, title, middle_label, forward_label, defect_label=""):
    """Draw one Opt/Traj/Obj N2 diagram.

    Reproduces the layout of the ISDC paper's transcription figure: the
    three components on the diagonal, the forward path along the rows, and
    the feedback the optimizer receives coming back along them. What
    distinguishes the transcriptions is exactly what the off-diagonal
    cells carry, so the one structural difference between the panels is
    the extra ``Traj -> Opt`` arrow a defect-carrying method needs.

    Parameters
    ----------
    axes
        The axis to draw on.
    title
        The panel title.
    middle_label
        The label of the middle block.
    forward_label
        What the optimizer hands the middle block.
    defect_label
        What the middle block hands straight back to the optimizer; empty
        for single shooting, which has no defects to report.
    """
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_aspect("equal")
    axes.axis("off")
    axes.set_title(title, fontsize=9.5, fontweight="bold")

    _block(axes, _OPT, *_BOX, "Opt", fill="#dde8fb", edge=VARIABLE)
    _block(axes, _TRAJ, *_BOX, middle_label, fill="#ededed", edge=COMPUTED)
    _block(axes, _OBJ, *_BOX, "Obj", fill="#fbe1e1", edge=CONSTRAINED)

    opt, traj, obj = _centre(_OPT), _centre(_TRAJ), _centre(_OBJ)
    east = _BOX[0] / 2
    half = _BOX[1] / 2

    # Forward: out of a block's east face, down into the next block's north.
    _elbow(
        axes,
        (opt[0] + east, opt[1]),
        (traj[0], traj[1] + half),
        forward_label,
        color=VARIABLE,
        label_offset=(0.02, 0.02),
    )
    _elbow(
        axes,
        (traj[0] + east, traj[1]),
        (obj[0], obj[1] + half),
        "s(t)",
        color=COMPUTED,
        label_offset=(0.02, 0.02),
    )
    # Feedback: out of a west face, back up into the optimizer's south.
    _elbow(
        axes,
        (obj[0] - east, obj[1]),
        (opt[0] + 0.05, opt[1] - half),
        "f_obj",
        color=CONSTRAINED,
        label_offset=(0.02, -0.02),
    )
    if defect_label:
        # The defects go straight back to the optimizer as constraints,
        # never through the objective: that extra cell is the whole
        # difference between shooting and a defect-based transcription.
        _elbow(
            axes,
            (traj[0] - east, traj[1]),
            (opt[0] - 0.05, opt[1] - half),
            defect_label,
            color=CONSTRAINED,
            label_offset=(-0.02, 0.02),
        )


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
    """Draw a state-trajectory panel, per transcription.

    Parameters
    ----------
    axes
        The axis to draw on.
    title
        The panel title.
    style
        ``"shooting"`` for one variable at the start and a computed value
        at the end, ``"multiple"`` for one variable per segment start with
        the continuity constraints between them, ``"collocation"`` for
        every node a variable with defects at the collocation points.
    """
    times = numpy.linspace(0.0, 4.0, 200)
    state = 0.4 + 0.55 * times - 0.28 * times**2 + 0.055 * times**3
    axes.plot(times, state, color="#222222", linewidth=1.3)
    node_times = numpy.array([0.0, 1.3, 2.6, 3.9])
    node_values = 0.4 + 0.55 * node_times - 0.28 * node_times**2 + 0.055 * node_times**3
    if style == "multiple":
        # One decision variable at each segment start, and a continuity
        # constraint where the integrated segment has to meet the next. At
        # an interior junction the same point is both, so the ring is drawn
        # unfilled and larger, around the variable rather than over it.
        axes.scatter(node_times[:-1], node_values[:-1], color=VARIABLE, zorder=5, s=28)
        axes.scatter(
            node_times[1:],
            node_values[1:],
            facecolor="none",
            edgecolor=CONSTRAINED,
            zorder=6,
            s=80,
            linewidth=1.2,
        )
    elif style == "collocation":
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

    Three columns, in the order they grow: single shooting, multiple
    shooting, collocation. The top row shows how the control is held
    (every node a design variable for all three, since all three
    parameterize the control at nodes); the middle row shows how the
    *state* is held — a single variable at the start and a computed value
    at the end for single shooting, one per segment for multiple shooting,
    every node a variable tied down by defects at the collocation points
    for collocation; the bottom row is the N2 diagram of what the
    optimizer, the trajectory and the objective pass each other. Filled
    blue points are decision variables the optimizer holds directly; open
    red points are constraint locations; the plain black point is a value
    obtained by integration rather than held.

    Multiple shooting is drawn because it is the step between the other
    two and reading them without it hides why collocation looks the way it
    does; this library does not implement it, for the reasons in
    :doc:`../concepts/transcriptions`.

    Parameters
    ----------
    axes
        Unused; this figure always makes its own 3x3 grid, since a single
        axis cannot hold nine independent panels. Present for interface
        consistency with the rest of the project's plot functions.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    del axes
    figure, grid = pyplot.subplots(3, 3, figsize=(10.2, 8.4))

    columns = (
        ("Single shooting", "shooting", "Traj", "u(t_c)", ""),
        ("Multiple shooting", "multiple", "Traj", "u(t_c), s_2", "delta_2"),
        ("Collocation", "collocation", "Traj", "u(t_c), s(t_c)", "zeta_c"),
    )
    for column, (name, style, middle, forward, defect) in enumerate(columns):
        _control_panel(grid[0][column], f"{name} — control", "variable")
        _state_panel(grid[1][column], f"{name} — state", style)
        _n2_panel(grid[2][column], f"{name} — dataflow", middle, forward, defect)
        if style == "multiple":
            grid[2][column].text(
                0.5,
                -0.04,
                "not implemented in this library",
                transform=grid[2][column].transAxes,
                ha="center",
                fontsize=8,
                style="italic",
                color="#666666",
            )

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
