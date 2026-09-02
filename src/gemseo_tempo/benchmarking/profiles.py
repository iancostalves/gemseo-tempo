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
"""Data profiles over the benchmark records.

A data profile answers "what fraction of the starting points does this
level solve within a given budget of discipline calls", which complements
the box plots: the box plots show what a solved run costs, the profile
shows how often the level solves anything at all.
"""

from __future__ import annotations

import numpy

from gemseo_tempo.benchmarking.runner import CONVERGED


def data_profile(records, level: str, budgets):
    """Return the fraction of starting points solved within each budget.

    Parameters
    ----------
    records
        The records of one study.
    level
        The level to profile.
    budgets
        The increasing budgets, in discipline calls.

    Returns
    -------
    ndarray
        The solved fraction at each budget, between 0 and 1.
    """
    cell = [record for record in records if record.level == level]
    if not cell:
        return numpy.zeros(len(budgets))
    calls = numpy.array(
        [record.calls if record.status == CONVERGED else numpy.inf for record in cell]
    )
    return numpy.array([float(numpy.mean(calls <= budget)) for budget in budgets])


def profile_table(records, budgets):
    """Return the data profile of every level.

    Parameters
    ----------
    records
        The records of one study.
    budgets
        The increasing budgets, in discipline calls.

    Returns
    -------
    dict
        The solved fractions, keyed by level.
    """
    levels = dict.fromkeys(record.level for record in records)
    return {level: data_profile(records, level, budgets) for level in levels}
