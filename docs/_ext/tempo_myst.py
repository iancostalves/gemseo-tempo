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
"""Sphinx extension for the gemseo-tempo variable grammar.

Placeholder registered from Sprint 0 on. The ``{var}`` role, ``{variables}``
table directive, and ``{variable-glossary}`` directive land in Sprint 13.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sphinx.application import Sphinx


def setup(app: Sphinx) -> dict[str, object]:
    """Register the extension with Sphinx.

    Parameters
    ----------
    app
        The Sphinx application.

    Returns
    -------
    dict
        Extension metadata.
    """
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
