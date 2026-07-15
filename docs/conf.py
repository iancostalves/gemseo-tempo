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
"""Sphinx configuration for gemseo-tempo."""

import sys
from pathlib import Path

from gemseo_tempo import __version__

# Local Sphinx extensions (tempo_myst provides the variable grammar).
sys.path.insert(0, str(Path(__file__).parent / "_ext"))

project = "gemseo-tempo"
copyright = "2026, ISAE-SUPAERO"
author = "Ian Costa-Alves"
version = __version__
release = __version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "tempo_myst",
]

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = f"gemseo-tempo {__version__}"

# numpy-style docstrings via napoleon.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
