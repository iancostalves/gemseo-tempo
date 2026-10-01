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

One more rule, learned the hard way: a page with more than one ``.. plot::``
directive must not write each one's code inline in its own ``{eval-rst}``
fence. MyST parses each fence through a fresh nested document, and
``plot_directive`` names an inline block's output file from a counter it
stores as a document attribute — so every fence's first (and only) block
resets to the same counter value and silently overwrites the previous
figure with the next one, with no warning from the build. The fix used
throughout this documentation is ``docs/_ext/gallery/``: one small script
per figure, referenced as ``.. plot:: _ext/gallery/<name>.py`` (a path,
resolved relative to ``docs/``, regardless of which page it is used from),
which names its own output file from the script path instead of the
counter and is immune to this.

The same stored results reach the prose, not only the figures. Two pieces
of grammar read them:

``{docvalue}`file:key:format```
    A number quoted from a stored result, such as
    ``{docvalue}`albatross/codesign.json:minimum_wind:.2f```.

``{formulation} file``
    The problem the optimizer is handed, from a stored
    ``ProblemDescription``: objective, design variables and constraints.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from typing import ClassVar

import numpy
from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util.docutils import SphinxDirective
from sphinx.util.docutils import SphinxRole

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
    return read(path(name))


def read(full: Path) -> Any:
    """Read a stored result from its full path, by extension.

    Parameters
    ----------
    full
        The path of the stored result.

    Returns
    -------
    Any
        A mapping of arrays for ``.npz``, the decoded object for ``.json``,
        and the path itself for anything else.
    """
    if full.suffix == ".npz":
        with numpy.load(full, allow_pickle=False) as archive:
            return dict(archive)
    if full.suffix == ".json":
        return json.loads(full.read_text(encoding="utf-8"))
    return full


# ------------------------------------------------ numbers quoted in prose


def lookup(record: Any, key: str) -> Any:
    """Follow a dotted key through nested mappings and sequences.

    Parameters
    ----------
    record
        The stored result, as :func:`read` returns it.
    key
        The dotted path, such as ``"budget.drag_work"``; an integer part
        indexes a list or an array.

    Returns
    -------
    Any
        The value, with a one-element array reduced to its scalar.

    Raises
    ------
    KeyError
        If a part of the key names nothing.
    """
    value = record
    for part in key.split("."):
        if isinstance(value, Mapping):
            if part not in value:
                raise KeyError(f"no {part!r} in {sorted(value)}")
            value = value[part]
        elif isinstance(value, (list, tuple, numpy.ndarray)):
            value = value[int(part)]
        else:
            raise KeyError(f"cannot index {type(value).__name__} with {part!r}")
    if isinstance(value, numpy.ndarray) and value.size == 1:
        return value.item()
    return value


def split_target(text: str) -> tuple[str, str, str]:
    """Split a ``{docvalue}`` target into file, key and format.

    Parameters
    ----------
    text
        The role content, ``file:key`` or ``file:key:format``. The format
        may end in ``*factor`` to scale the value first, so a time stored
        in seconds can be quoted in milliseconds with ``.3f*1e3``.

    Returns
    -------
    tuple
        The file relative to the data directory, the dotted key, and the
        format specification (empty for ``str``).

    Raises
    ------
    ValueError
        If the file or the key is missing.
    """
    source, _, rest = text.strip().partition(":")
    key, _, spec = rest.partition(":")
    if not source or not key:
        msg = f"expected 'file:key[:format]', got {text!r}"
        raise ValueError(msg)
    return source, key, spec


def render(value: Any, spec: str) -> str:
    """Format a stored value, scaling it first if the format asks.

    Parameters
    ----------
    value
        The stored value.
    spec
        A format specification, optionally ending in ``*factor``.

    Returns
    -------
    str
        The formatted value.
    """
    if "*" in spec:
        spec, factor = spec.rsplit("*", 1)
        value = value * float(factor)
    return format(value, spec)


class DocValueRole(SphinxRole):
    """The ``{docvalue}`` role: a number quoted from a stored result.

    ``{docvalue}`albatross/codesign.json:minimum_wind:.2f``` renders the
    stored minimum wind to two decimals. The number on the page is read
    from the same file ``tests/test_doc_data.py`` checks, so prose cannot
    quote a value the stored results no longer hold. A missing file or key
    is a build error.
    """

    def run(self) -> tuple[list[nodes.Node], list[nodes.system_message]]:
        """Resolve the target and render the value.

        Returns
        -------
        tuple
            The rendered value, or a problematic node and its error.
        """
        try:
            source, key, spec = split_target(self.text)
            full = Path(self.config.tempo_data_dir) / source
            value = render(lookup(read(full), key), spec)
        except (OSError, KeyError, IndexError, ValueError, TypeError) as error:
            message = self.inliner.reporter.error(
                f"{{docvalue}} {self.text!r}: {error}", line=self.lineno
            )
            problem = self.inliner.problematic(self.rawtext, self.rawtext, message)
            return [problem], [message]
        self.env.note_dependency(str(full))
        return [nodes.inline(self.rawtext, value, classes=["tempo-docvalue"])], []


# ------------------------------------------ the problem the optimizer sees


def _bound(value: float | None, infinite: str) -> str:
    """Render one side of an interval, ``None`` standing for infinity."""
    return infinite if value is None else f"{value:g}"


def _table(headers: list[str], rows: list[list[nodes.Node]]) -> nodes.table:
    """Build a docutils table from header labels and rows of cell nodes."""
    table = nodes.table(classes=["tempo-formulation"])
    group = nodes.tgroup(cols=len(headers))
    table += group
    for _ in headers:
        group += nodes.colspec(colwidth=1)
    head = nodes.thead()
    group += head
    head += nodes.row(
        "", *(nodes.entry("", nodes.paragraph(text=label)) for label in headers)
    )
    body = nodes.tbody()
    group += body
    for cells in rows:
        body += nodes.row(
            "", *(nodes.entry("", nodes.paragraph("", "", cell)) for cell in cells)
        )
    return table


SENSES = {"eq": "= 0", "ineq": "≤ 0"}
"""How each constraint sense reads, in GEMSEO's convention."""


class FormulationDirective(SphinxDirective):
    """The ``{formulation}`` directive: the problem handed to the optimizer.

    Renders a :class:`~gemseo_tempo.scenario.description.ProblemDescription`
    stored by ``tools/generate_doc_results.py``: the objective and its
    sense, then the design variables with their sizes and bounds, then the
    constraints. Written by hand, a problem statement goes stale as soon as
    a bound or a discretization changes; stored, it is regenerated with the
    results it describes.
    """

    required_arguments = 1
    option_spec: ClassVar[dict] = {"caption": directives.unchanged}

    def run(self) -> list[nodes.Node]:
        """Read the stored description and render it.

        Returns
        -------
        list
            A summary paragraph and the two tables.
        """
        full = Path(self.config.tempo_data_dir) / self.arguments[0]
        try:
            record = read(full)
            variables = record["design_variables"]
            constraints = record["constraints"]
            objective = record["objective"]
            maximize = record["maximize"]
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise self.error(
                f"{{formulation}} {self.arguments[0]!r}: {error}"
            ) from error
        self.env.note_dependency(str(full))

        design_size = sum(variable["size"] for variable in variables)
        constraint_size = sum(constraint["size"] for constraint in constraints)
        summary = nodes.paragraph()
        summary += nodes.strong(text="maximize " if maximize else "minimize ")
        summary += nodes.literal(text=objective)
        summary += nodes.Text(
            f" over {design_size} scalar design variables in {len(variables)}"
            f" groups, subject to {constraint_size} scalar constraints in"
            f" {len(constraints)} groups."
        )
        if "caption" in self.options:
            summary.insert(0, nodes.Text(self.options["caption"] + " "))

        design = _table(
            ["design variable", "size", "bounds"],
            [
                [
                    nodes.literal(text=variable["name"]),
                    nodes.Text(str(variable["size"])),
                    nodes.literal(
                        text=f"[{_bound(variable['lower_bound'], '-inf')},"
                        f" {_bound(variable['upper_bound'], 'inf')}]"
                    ),
                ]
                for variable in variables
            ],
        )
        limits = _table(
            ["constraint", "sense", "size"],
            [
                [
                    nodes.literal(text=constraint["name"]),
                    nodes.Text(SENSES.get(constraint["kind"], constraint["kind"])),
                    nodes.Text(str(constraint["size"])),
                ]
                for constraint in constraints
            ],
        )
        return [summary, design, limits] if constraints else [summary, design]


def setup(app) -> dict[str, object]:
    """Register the role and the directive with Sphinx.

    Parameters
    ----------
    app
        The Sphinx application.

    Returns
    -------
    dict
        Extension metadata.
    """
    app.add_config_value("tempo_data_dir", str(DATA), "env")
    app.add_role("docvalue", DocValueRole())
    app.add_directive("formulation", FormulationDirective)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }


__all__ = ["DATA", "load", "lookup", "path", "read", "render", "split_target"]
