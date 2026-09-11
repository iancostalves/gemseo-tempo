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
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "matplotlib.sphinxext.plot_directive",
    "sphinxcontrib.bibtex",
    "tempo_myst",
]

# Numbered [N] citations and a generated References page, matching the
# convention of the companion NOADS documentation
# (https://iancostalves.github.io/noads/).
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"

# Figures are drawn at build time from results stored under docs/_data, so
# the code a reader sees is the code that made the picture. Note that the
# directive has to be written inside an ``eval-rst`` block: it calls
# ``insert_input`` on the state machine, which MyST's mock does not
# implement, so the Markdown-native ``{plot}`` form fails to parse.
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [("png", 140)]
plot_rcparams = {"figure.autolayout": True, "savefig.bbox": "tight"}

autosummary_generate = True
autodoc_default_options = {"members": True, "undoc-members": False}
# Instance attributes documented by a following string literal would
# otherwise be reported twice, once from the class and once from __init__.
autodoc_typehints = "description"

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
