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
"""Reading the stored results that the documentation's figures are drawn from.

Every figure in this documentation is drawn at build time, so the code a
reader sees is the code that made the picture. What the code is *not*
allowed to do is solve anything: an optimizer in a documentation build is
an hour of CI time and a page that fails when a solver has an off day.

So the rule is a division of labour. Anything that needs an optimizer, or a
sample of trajectories, is computed once by ``tools/generate_doc_results.py``
and stored under ``docs/_data``; the figure only loads it. Anything that is
a direct function evaluation, a polar curve or a wind profile, is computed
live, because it costs milliseconds and cannot go stale.

A figure that takes noticeable time to draw is a figure breaking that rule.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy

DATA = Path(__file__).resolve().parents[1] / "_data"
"""The directory holding the stored results, ``docs/_data``."""


def path(name: str) -> Path:
    """Return the full path of a stored result.

    Parameters
    ----------
    name
        The path relative to ``docs/_data``, such as
        ``"brachistochrone/variants.npz"``.

    Returns
    -------
    pathlib.Path
        The full path.

    Raises
    ------
    FileNotFoundError
        If nothing has been stored under that name, with the command that
        would produce it.
    """
    full = DATA / name
    if not full.exists():
        msg = (
            f"{name!r} is not in docs/_data; regenerate the stored results"
            " with `python tools/generate_doc_results.py`."
        )
        raise FileNotFoundError(msg)
    return full


def load(name: str) -> Any:
    """Load a stored result, by extension.

    Parameters
    ----------
    name
        The path relative to ``docs/_data``.

    Returns
    -------
    Any
        A mapping of arrays for ``.npz``, the decoded object for ``.json``,
        and the path itself for anything else, which is what GEMSEO's
        post-processing wants of a stored history.
    """
    full = path(name)
    if full.suffix == ".npz":
        with numpy.load(full, allow_pickle=False) as archive:
            return dict(archive)
    if full.suffix == ".json":
        return json.loads(full.read_text(encoding="utf-8"))
    return full


__all__ = ["DATA", "load", "path"]
