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
"""The brachistochrone's figures: given an axis, drawn on it; else a new one."""

from __future__ import annotations

import numpy
from matplotlib import pyplot

from gemseo_tempo.problems.brachistochrone.plots import plot_control_law
from gemseo_tempo.problems.brachistochrone.plots import plot_convergence
from gemseo_tempo.problems.brachistochrone.plots import plot_descent
from gemseo_tempo.problems.brachistochrone.plots import plot_gradient_cost
from gemseo_tempo.problems.brachistochrone.plots import plot_history
from gemseo_tempo.problems.brachistochrone.plots import plot_kinematics


def test_the_figures_draw():
    assert plot_kinematics() is not None
    assert (
        plot_descent({"drag": (numpy.array([0.0, 1.0]), numpy.array([0.0, 1.0]))})
        is not None
    )
    assert (
        plot_descent(
            {"ideal": (numpy.array([0.0, 1.0]), numpy.array([0.0, 1.0]))},
            cycloid=(numpy.array([0.0, 1.0]), numpy.array([0.0, 0.9])),
        )
        is not None
    )
    assert (
        plot_control_law(numpy.linspace(0.0, 1.0, 5), {"guess": numpy.zeros(5)})
        is not None
    )
    assert plot_convergence([5, 10, 20], {"shooting": [0.1, 0.01, 0.0]}) is not None
    assert (
        plot_history([0, 1, 2], {"objective": [1.0, 0.5, 0.25]}, log=False) is not None
    )
    assert plot_gradient_cost([40, 400], [1e-4, 1e-4], [0.07, 0.4]) is not None


def test_the_figures_accept_existing_axes():
    _, axes = pyplot.subplots()
    assert plot_kinematics(axes) is axes
    _, axes = pyplot.subplots()
    assert (
        plot_descent({"drag": (numpy.array([0.0, 1.0]), numpy.array([0.0, 1.0]))}, axes)
        is axes
    )
    _, axes = pyplot.subplots()
    assert plot_control_law([0.0, 1.0], {"guess": [0.0, 0.1]}, axes) is axes
    _, axes = pyplot.subplots()
    assert plot_convergence([5, 10], {"shooting": [0.1, 0.0]}, axes) is axes
    _, axes = pyplot.subplots()
    assert plot_history([0, 1], {"objective": [1.0, 0.5]}, axes) is axes
    _, axes = pyplot.subplots()
    assert plot_gradient_cost([40, 400], [1e-4, 1e-4], [0.07, 0.4], axes) is axes


def test_the_kinematics_diagram_is_pure_geometry():
    """No trajectory needed: it draws the wire, the bead, gamma and v."""
    axes = plot_kinematics()
    assert len(axes.lines) >= 1
    texts = " ".join(text.get_text() for text in axes.texts)
    assert "gamma" in texts
    assert "v" in texts


def test_convergence_floors_a_zero_error_for_the_log_axis():
    """A converged transcription reaches the reference exactly.

    Zero cannot be drawn on a log axis, so it has to be floored rather
    than dropped, and this is the case that exercises that floor.
    """
    axes = plot_convergence([10, 20], {"shooting": [0.0, 0.0]})
    line = axes.lines[0]
    assert numpy.all(numpy.asarray(line.get_ydata()) > 0.0)
