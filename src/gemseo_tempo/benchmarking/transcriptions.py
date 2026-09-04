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
"""Turning a level name into a transcription.

Kept apart from both the runner and the suites, because both need it and
neither should import the other.
"""

from __future__ import annotations

from gemseo_tempo.transcriptions.collocation import HermiteSimpson
from gemseo_tempo.transcriptions.indirect import IndirectShooting
from gemseo_tempo.transcriptions.single_shooting import SingleShooting


def build_transcription(level: str, size: int):
    """Return the transcription and control settings of a level.

    Parameters
    ----------
    level
        The level name.
    size
        The discretization.

    Returns
    -------
    tuple
        The transcription and the effective size to build the problem with.

    Raises
    ------
    ValueError
        If the level is unknown.
    """
    if level == "shooting":
        return SingleShooting(), size
    if level == "collocation":
        return HermiteSimpson(num_segments=size), size
    if level == "collocation-sparse":
        return HermiteSimpson(num_segments=size, sparse_jacobian=True), size
    if level == "indirect":
        # Only the brachistochrone offers an indirect transcription here: it
        # needs a closed-form optimal control, and a path constraint would
        # change the necessary conditions anyway.
        from gemseo_tempo.problems.brachistochrone.problem import optimal_path_angle

        # The indirect method has no path discretization; the abscissa is
        # its integrator save resolution instead.
        return IndirectShooting(u_star=optimal_path_angle), size
    raise ValueError(f"Unknown level {level!r}.")


__all__ = ["build_transcription"]
