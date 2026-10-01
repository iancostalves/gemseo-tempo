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
"""Figures for the benchmark studies.

The cost panel is a row of box plots, one per level, of the number of
discipline calls against the discretization. A second row reports the
share of starting points that actually reached the reference, because a
distribution of costs means little without knowing how many of those runs
solved the problem at all.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gemseo_tempo.benchmarking.runner import CONVERGED  # noqa: E402


def _group(records, level, size):
    """Return the records of one cell."""
    return [r for r in records if r.level == level and r.size == size]


def plot_study(
    records, study: str, path: Path, sizes=None, problem: str = "Brachistochrone"
):
    """Draw the cost and success figure of one study.

    Parameters
    ----------
    records
        The records of the study.
    study
        The study name, used in the title.
    path
        The destination image file.
    sizes
        The discretizations, in order; inferred when omitted.
    problem
        The problem name, used in the title; the brachistochrone when
        omitted, since that is where this figure originated.

    Returns
    -------
    Path
        The path written.
    """
    levels = list(dict.fromkeys(record.level for record in records))
    sizes = sizes or sorted({record.size for record in records})
    figure, axes = plt.subplots(
        2,
        len(levels),
        figsize=(3.1 * len(levels), 5.6),
        sharey="row",
        squeeze=False,
    )
    for column, level in enumerate(levels):
        cost_axis = axes[0][column]
        rate_axis = axes[1][column]
        data = []
        rates = []
        for size in sizes:
            cell = _group(records, level, size)
            solved = [r for r in cell if r.status == CONVERGED]
            data.append([r.calls for r in solved] or [0])
            rates.append(100.0 * len(solved) / len(cell) if cell else 0.0)
        cost_axis.boxplot(data, tick_labels=[str(s) for s in sizes])
        cost_axis.set_title(level)
        cost_axis.grid(axis="y", alpha=0.3)
        rate_axis.bar([str(s) for s in sizes], rates, color="tab:blue", alpha=0.8)
        rate_axis.set_ylim(0, 105)
        rate_axis.grid(axis="y", alpha=0.3)
        if column == 0:
            cost_axis.set_ylabel("Discipline calls (solved runs)")
            rate_axis.set_ylabel("Starting points solved [%]")
    figure.supxlabel("Discretization (segments, control intervals or save points)")
    figure.suptitle(f"{problem}: effect of the {study}")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


TRANSCRIPTION_COLORS = {
    "shooting": "#2a78d6",
    "collocation": "#eb6834",
    "indirect": "#1baf7a",
}
"""One colour per transcription, the same on every page."""


def plot_transcriptions(rows, axes=None):
    """Draw the error and the cost of each transcription against the grid.

    The direct transcriptions are lines over the discretization. The
    indirect one has no grid to refine, its accuracy being the
    integrator's, so it is a single level drawn across the whole range.
    A solve that did not meet its constraints is left out: its answer is
    not an answer to the problem.

    Parameters
    ----------
    rows
        The benchmark rows: ``transcription``, ``size`` (``None`` for the
        indirect one), ``error``, ``seconds`` and, optionally, ``met``.
    axes
        Two axes, for the error and for the time; a new figure when
        omitted.

    Returns
    -------
    list
        The axes drawn on.
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    sizes = sorted({row["size"] for row in rows if row["size"] is not None})
    for name, color in TRANSCRIPTION_COLORS.items():
        chosen = [
            row for row in rows if row["transcription"] == name and row.get("met", True)
        ]
        if not chosen:
            continue
        for axis, key in zip(axes, ("error", "seconds"), strict=True):
            if chosen[0]["size"] is None:
                axis.hlines(
                    chosen[0][key],
                    sizes[0],
                    sizes[-1],
                    colors=color,
                    linestyles="--",
                    label=f"{name} (no grid)",
                )
                continue
            chosen = sorted(chosen, key=lambda row: row["size"])
            axis.plot(
                [row["size"] for row in chosen],
                [row[key] for row in chosen],
                marker="o",
                color=color,
                label=name,
            )
    for axis, label in zip(
        axes, ("relative error of the optimum", "wall time (s)"), strict=True
    ):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("segments or control intervals")
        axis.set_ylabel(label)
        axis.grid(True, which="major", alpha=0.3)
    axes[0].legend(fontsize=8)
    return axes
