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
"""Starting points for the benchmark studies.

Random node values would produce trajectories that are not merely poor but
meaningless, so the starting points are drawn over the three physical
guess parameters instead: the path angle at the start, its increase over
the descent, and the guessed descent time. The same sample is shared by
every level of a study, so a comparison is never confounded by one level
having been handed easier starting points.
"""

from __future__ import annotations

import numpy
from scipy.stats import qmc

from gemseo_tempo.problems.brachistochrone.problem import HALF_PI
from gemseo_tempo.problems.brachistochrone.problem import Guess

BOUNDS = {
    "gamma_level": (0.15, 1.2),
    "gamma_slope": (-0.3, 1.2),
    "t_f": (0.5, 2.0),
}
"""The sampled range of each guess parameter."""

COSTATE_BOUNDS = (-1.5, 1.5)
"""The sampled range of each initial costate of the indirect method."""


def sample_guesses(count: int, seed: int = 0) -> list[Guess]:
    """Draw starting points by Latin hypercube sampling.

    Parameters
    ----------
    count
        The number of starting points.
    seed
        The seed, which makes the sample reproducible.

    Returns
    -------
    list
        The starting points, each also carrying initial costates so that
        the indirect transcription can use the same sample.
    """
    names = list(BOUNDS)
    sampler = qmc.LatinHypercube(d=len(names) + 3, seed=seed)
    unit = sampler.random(count)
    lower = numpy.array([BOUNDS[name][0] for name in names])
    upper = numpy.array([BOUNDS[name][1] for name in names])
    scaled = lower + unit[:, : len(names)] * (upper - lower)
    costates = COSTATE_BOUNDS[0] + unit[:, len(names) :] * (
        COSTATE_BOUNDS[1] - COSTATE_BOUNDS[0]
    )
    guesses = []
    for row, costate_row in zip(scaled, costates, strict=True):
        values = dict(zip(names, row, strict=True))
        values["gamma_level"] = float(numpy.clip(values["gamma_level"], 0.0, HALF_PI))
        guesses.append(
            Guess(
                gamma_level=values["gamma_level"],
                gamma_slope=float(values["gamma_slope"]),
                t_f=float(values["t_f"]),
                costates=dict(
                    zip(("x", "y", "v"), map(float, costate_row), strict=True)
                ),
            )
        )
    return guesses


SAIL_BOUNDS = {
    "psi_level": (0.33, 0.66),
    "psi_slope": (-1.30, -0.68),
    "t_f": (148.0, 190.0),
}
"""The sampled range of each sail guess parameter, in rad and s.

Centred on the polar-derived starting point rather than spread over
everything the bounds allow: a downwind leg started pointing upwind is not a
harder instance of the problem, it is a different one. Wider ranges were
tried first and every start converged to the straight run instead of gybing,
which measures the strength of that local optimum rather than the optimizer.

Half the sample is mirrored onto the other tack, since gybing north and
gybing south are equally good and the problem is symmetric between them.
"""

ALBATROSS_BOUNDS = {
    "period": (8.5, 11.0),
    "climb_amplitude": (0.22, 0.32),
    "airspeed": (26.0, 34.0),
    "mean_height": (14.0, 21.0),
}
"""The sampled range of each albatross guess parameter, in s, rad, m/s and m.

Bounded by what the inverse-dynamics guess can actually fly. A short period
at high speed and a steep climb asks for more than 90 degrees of bank, which
is inverted flight and outside the control bounds, so the scenario refuses to
build before the optimizer ever sees it.
"""


def _scaled(bounds, count, seed):
    """Draw a Latin hypercube over named bounds.

    Parameters
    ----------
    bounds
        The ``{name: (lower, upper)}`` ranges.
    count
        The number of points.
    seed
        The seed, which makes the sample reproducible.

    Returns
    -------
    list
        One ``{name: value}`` dictionary per point.
    """
    names = list(bounds)
    unit = qmc.LatinHypercube(d=len(names), seed=seed).random(count)
    lower = numpy.array([bounds[name][0] for name in names])
    upper = numpy.array([bounds[name][1] for name in names])
    return [
        dict(zip(names, row, strict=True)) for row in lower + unit * (upper - lower)
    ]


def sample_sail_guesses(count: int, seed: int = 0):
    """Draw starting points for the sail problem.

    Parameters
    ----------
    count
        The number of starting points.
    seed
        The seed.

    Returns
    -------
    list
        The starting points.
    """
    from gemseo_tempo.problems.sail.problem import Guess as SailGuess

    guesses = []
    for index, values in enumerate(_scaled(SAIL_BOUNDS, count, seed)):
        if index % 2:
            # The other tack: the mark is dead downwind, so gybing either way
            # is the same problem reflected.
            values = {
                "psi_level": -values["psi_level"],
                "psi_slope": -values["psi_slope"],
                "t_f": values["t_f"],
            }
        guesses.append(SailGuess(**values))
    return guesses


def sample_albatross_guesses(count: int, seed: int = 0):
    """Draw starting points for the albatross problem.

    Parameters
    ----------
    count
        The number of starting points.
    seed
        The seed.

    Returns
    -------
    list
        The starting points.
    """
    from gemseo_tempo.problems.albatross.problem import Guess as BirdGuess

    return [BirdGuess(**values) for values in _scaled(ALBATROSS_BOUNDS, count, seed)]
