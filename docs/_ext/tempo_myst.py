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
"""The variable grammar: documentation generated from the registries.

Every variable of every model already carries its full name, its unit, its
bounds and a description, in a
:class:`~gemseo_tempo.metadata.registry.VariableRegistry` that the code
itself reads. This extension makes the documentation read the same record,
so that a page cannot disagree with the model it describes. A hand-written
table is a copy, and a copy of a number is a number that will be wrong
later.

Three pieces of grammar, all in the ``tempo`` domain:

``{var}`c_l```
    An inline mention. Renders the full name, the symbol and the unit, with
    the description as hover text, linked to the glossary entry.
    ``{var}`c_l <symbol>``` selects a shorter style; the styles are
    :data:`STYLES`.

``{variables} module:REGISTRY``
    A generated table, filtered by tag and with the columns chosen per
    call.

``{variable-glossary} package``
    Every variable of every registry under a package, grouped by the model
    it belongs to, alphabetical, and anchored so the role above can link to
    it.

``{variable-context} module:REGISTRY``
    Declares which model the bare names on a page refer to.

Variables are scoped by model, and that is not tidiness either. Writing the
grammar made the clash visible: ``x`` is a horizontal position, an eastward
position and a downwind position in the three problems; ``y`` is a depth, a
northward position and a crosswind one; and ``mu`` is a Coulomb friction
coefficient in the brachistochrone and a bank angle on the albatross. A
single global glossary would have had to pick one of each and be wrong
about the rest. So an entry belongs to a model, a page says which model it
is about, and a name that is unique across all of them still resolves
without being told.

A name that appears in no registry is a build error by default, which is
the point of the whole arrangement: the documentation is checked against
the code rather than merely written alongside it. Set
``tempo_unknown_variable = "warn"`` to downgrade it.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.domains import Domain
from sphinx.domains import ObjType
from sphinx.errors import ExtensionError
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import make_refnode

from gemseo_tempo.metadata.registry import VariableRegistry
from gemseo_tempo.metadata.units import UREG

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sphinx.application import Sphinx
    from sphinx.builders import Builder
    from sphinx.environment import BuildEnvironment

    from gemseo_tempo.metadata.variable import Variable

logger = logging.getLogger(__name__)

STYLES = ("full", "name", "symbol", "unit")
"""The rendering styles of the ``{var}`` role.

``full``
    ``Lift coefficient (c_l, –)``, the default.
``name``
    ``Lift coefficient``.
``symbol``
    ``c_l``, the grammar key.
``unit``
    ``–``.
"""

DEFAULT_COLUMNS = ("name", "full_name", "unit", "bounds")
"""The columns of a ``{variables}`` table when none are asked for."""

COLUMN_TITLES = {
    "name": "Symbol",
    "full_name": "Name",
    "description": "Description",
    "unit": "Unit",
    "bounds": "Bounds",
    "lower_bound": "Lower",
    "upper_bound": "Upper",
    "tags": "Tags",
    "distribution": "Distribution",
}
"""The heading of each available column."""


def format_unit(unit: str, dimensionless: str, unit_format: str) -> str:
    """Render a unit for display.

    Parameters
    ----------
    unit
        The pint-parsable unit of the variable.
    dimensionless
        What to show when the variable has no unit.
    unit_format
        A pint format specification, such as ``"~P"`` for short pretty.

    Returns
    -------
    str
        The rendered unit.
    """
    if not unit or unit == "dimensionless":
        return dimensionless
    try:
        rendered = format(UREG.Unit(unit), unit_format)
    except Exception:  # noqa: BLE001 - a bad unit is the registry's problem
        return unit
    return rendered or dimensionless


def format_bounds(variable: Variable) -> str:
    """Render the bounds of a variable as an interval.

    Parameters
    ----------
    variable
        The variable.

    Returns
    -------
    str
        ``"[a, b]"``, ``">= a"``, ``"<= b"``, or an empty string when the
        variable is unbounded.
    """
    lower, upper = variable.lower_bound, variable.upper_bound
    if lower is None and upper is None:
        return ""
    if lower is None:
        return f"<= {upper:g}"
    if upper is None:
        return f">= {lower:g}"
    return f"[{lower:g}, {upper:g}]"


def render(variable: Variable, style: str, dimensionless: str, unit_format: str) -> str:
    """Render a variable in one of the styles of the ``{var}`` role.

    Parameters
    ----------
    variable
        The variable.
    style
        One of :data:`STYLES`.
    dimensionless
        What to show for a variable with no unit.
    unit_format
        A pint format specification.

    Returns
    -------
    str
        The rendered text.
    """
    unit = format_unit(variable.unit, dimensionless, unit_format)
    if style == "symbol":
        return variable.name
    if style == "name":
        return variable.full_name
    if style == "unit":
        return unit
    return f"{variable.full_name} ({variable.name}, {unit})"


def load_registry(target: str) -> VariableRegistry:
    """Import a registry named as ``"module:attribute"``.

    Parameters
    ----------
    target
        The import target, e.g.
        ``"gemseo_tempo.problems.sail.model:REGISTRY"``.

    Returns
    -------
    VariableRegistry
        The registry.

    Raises
    ------
    ExtensionError
        If the target is malformed, cannot be imported, or does not name a
        registry.
    """
    if ":" not in target:
        msg = f'{target!r} is not a registry target; write it as "module:ATTRIBUTE".'
        raise ExtensionError(msg)
    module_name, _, attribute = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        msg = f"Cannot import {module_name!r} for the registry {target!r}: {error}"
        raise ExtensionError(msg) from error
    registry = getattr(module, attribute, None)
    if not isinstance(registry, VariableRegistry):
        msg = f"{target!r} is {type(registry).__name__}, not a VariableRegistry."
        raise ExtensionError(msg)
    return registry


def walk_registries(package_name: str) -> Iterator[tuple[str, VariableRegistry]]:
    """Yield every module-level registry under a package.

    Parameters
    ----------
    package_name
        The package to walk, e.g. ``"gemseo_tempo.problems"``.

    Yields
    ------
    tuple
        The ``"module:ATTRIBUTE"`` target and the registry.

    Raises
    ------
    ExtensionError
        If the package cannot be imported.
    """
    try:
        package = importlib.import_module(package_name)
    except ImportError as error:
        msg = f"Cannot import {package_name!r} to collect its registries: {error}"
        raise ExtensionError(msg) from error

    module_names = [package_name]
    if hasattr(package, "__path__"):
        module_names.extend(
            info.name
            for info in pkgutil.walk_packages(package.__path__, f"{package_name}.")
        )
    seen: set[int] = set()
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception as error:  # noqa: BLE001 - report and carry on
            logger.warning(
                "tempo: skipping %s while collecting registries: %s",
                module_name,
                error,
                type="tempo",
                subtype="import",
            )
            continue
        for attribute in sorted(vars(module)):
            value = getattr(module, attribute)
            # A registry re-exported from a package facade is the same
            # object under a second name; yielding it twice would make the
            # conflict check compare a record with itself.
            if isinstance(value, VariableRegistry) and id(value) not in seen:
                seen.add(id(value))
                yield f"{module_name}:{attribute}", value


def registry_label(target: str) -> str:
    """Return the model a registry belongs to.

    Parameters
    ----------
    target
        The ``"module:ATTRIBUTE"`` target of the registry.

    Returns
    -------
    str
        A short label: the problem name for anything under
        ``gemseo_tempo.problems``, and the defining module otherwise.

    Every registry of one problem shares a label and they are merged,
    because a page about the albatross should resolve the name of a wing
    variable as readily as the name of a state, and both are the albatross.
    Merging is safe only if the registries agree wherever they overlap,
    which :func:`variable_index` checks and reports.
    """
    module_name, _, attribute = target.partition(":")
    parts = module_name.split(".")
    if "problems" in parts and len(parts) > parts.index("problems") + 1:
        return parts[parts.index("problems") + 1]
    label = parts[-1]
    return label if attribute == "REGISTRY" else f"{label}.{attribute.lower()}"


def variable_index(env: BuildEnvironment) -> dict[str, dict[str, Variable]]:
    """Return every declared variable, by model and then by name.

    The index is what makes an unknown name a build error rather than a
    typo that reaches the reader. It is built once and cached on the
    environment.

    Parameters
    ----------
    env
        The build environment.

    Returns
    -------
    dict
        ``label -> {name: variable}``.
    """
    cached = getattr(env, "_tempo_variable_index", None)
    if cached is not None:
        return cached

    index: dict[str, dict[str, Variable]] = {}
    for package_name in env.config.tempo_variable_packages:
        for target, registry in walk_registries(package_name):
            label = registry_label(target)
            scope = index.setdefault(label, {})
            for name, variable in registry.items():
                existing = scope.get(name)
                if existing is None:
                    scope[name] = variable
                elif existing.to_record() != variable.to_record():
                    # Within one model a name has to mean one thing.
                    logger.warning(
                        "tempo: %r means two different things inside %r;"
                        " the first declaration is the one documented.",
                        name,
                        label,
                        type="tempo",
                        subtype="conflict",
                    )
    env._tempo_variable_index = index  # noqa: SLF001
    return index


def resolve_variable(
    env: BuildEnvironment, name: str, context: str = ""
) -> tuple[str, Variable] | None:
    """Find the model a bare variable name refers to.

    A name is looked for in the page's declared model first, then across
    every model. A name that several models declare, with no page context
    to disambiguate it, is reported rather than guessed at.

    Parameters
    ----------
    env
        The build environment.
    name
        The variable name, optionally qualified as ``"model:name"``.
    context
        The label declared by ``{variable-context}`` on this page.

    Returns
    -------
    tuple or None
        The model label and the variable, or ``None`` when no model
        declares the name.
    """
    index = variable_index(env)
    if ":" in name:
        label, _, bare = name.partition(":")
        scope = index.get(label.strip(), {})
        variable = scope.get(bare.strip())
        return (label.strip(), variable) if variable is not None else None

    if context and name in index.get(context, {}):
        return context, index[context][name]

    matches = [
        (label, scope[name]) for label, scope in sorted(index.items()) if name in scope
    ]
    if not matches:
        return None
    if len(matches) > 1:
        distinct = {variable.full_name for _, variable in matches}
        if len(distinct) > 1:
            logger.warning(
                "tempo: %r means different things in %s;"
                " declare the model with {variable-context}"
                ' or qualify it as "model:%s".',
                name,
                ", ".join(label for label, _ in matches),
                name,
                type="tempo",
                subtype="ambiguous",
            )
    return matches[0]


def _context(env: BuildEnvironment) -> str:
    """Return the model declared for the document being read.

    Parameters
    ----------
    env
        The build environment.

    Returns
    -------
    str
        The label, or an empty string when the page declared none.
    """
    return env.temp_data.get("tempo:context", "")


def _unknown(env: BuildEnvironment, name: str, location: Any) -> None:
    """Record and report a variable name that is in no registry.

    Parameters
    ----------
    env
        The build environment.
    name
        The unknown name.
    location
        Where to point the message.
    """
    logger.warning(
        "tempo: no registry declares the variable %r.",
        name,
        location=location,
        type="tempo",
        subtype="unknown",
    )
    if env.config.tempo_unknown_variable == "error":
        unknown = getattr(env, "_tempo_unknown", None)
        if unknown is None:
            unknown = env._tempo_unknown = set()  # noqa: SLF001
        unknown.add(name)


class VariableXRefRole(XRefRole):
    """The ``{var}`` role: a variable mention, rendered from the registry.

    ``{var}`c_l``` renders the full style. The explicit-title form is
    repurposed as a style selector, so ``{var}`c_l <symbol>``` renders the
    symbol alone; docutils parses that as title ``c_l`` and target
    ``symbol``, and this class reads it back the way the grammar means it.
    """

    def run(self) -> tuple[list[nodes.Node], list[nodes.system_message]]:
        """Resolve the role against the ``tempo`` domain.

        The grammar spells the role ``{var}`` rather than ``{tempo:var}``,
        and an unqualified role name leaves the domain empty, so it is
        supplied here.

        Returns
        -------
        tuple
            The nodes and the system messages.
        """
        if ":" not in self.name:
            self.name = "tempo:var"
        return super().run()

    def process_link(
        self,
        env: BuildEnvironment,
        refnode: nodes.Element,
        has_explicit_title: bool,
        title: str,
        target: str,
    ) -> tuple[str, str]:
        """Resolve the variable and choose the text to display.

        Parameters
        ----------
        env
            The build environment.
        refnode
            The reference node being built.
        has_explicit_title
            Whether the role used the ``title <target>`` form, which this
            grammar reads as ``name <style>``.
        title
            The title as docutils parsed it.
        target
            The target as docutils parsed it.

        Returns
        -------
        tuple
            The text to display and the name to link to.
        """
        name, style = (title, target) if has_explicit_title else (target, "full")
        name = name.strip()
        style = style.strip() or "full"
        if style not in STYLES:
            logger.warning(
                "tempo: %r is not a variable style; use one of %s.",
                style,
                ", ".join(STYLES),
                location=refnode,
                type="tempo",
                subtype="style",
            )
            style = "full"

        found = resolve_variable(env, name, _context(env))
        if found is None:
            _unknown(env, name, refnode)
            return name, name

        label, variable = found
        config = env.config
        refnode["reftitle"] = variable.description or variable.full_name
        text = render(
            variable, style, config.tempo_dimensionless, config.tempo_unit_format
        )
        return text, f"{label}/{variable.name}"


class VariablesDirective(SphinxDirective):
    """The ``{variables}`` directive: a table generated from a registry."""

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    option_spec: ClassVar[dict] = {
        "filter": directives.unchanged,
        "columns": directives.unchanged,
        "names": directives.unchanged,
        "unit-format": directives.unchanged,
        "caption": directives.unchanged,
    }

    def run(self) -> list[nodes.Node]:
        """Build the table.

        Returns
        -------
        list
            The table node.

        Raises
        ------
        ExtensionError
            If a column or filter is not understood, or a requested name is
            not in the registry.
        """
        target = self.arguments[0].strip()
        self._label = registry_label(target)
        registry = load_registry(target)
        variables = self._select(registry)
        columns = self._columns()
        unit_format = self.options.get("unit-format", self.config.tempo_unit_format)

        table = nodes.table(classes=["tempo-variables"])
        group = nodes.tgroup(cols=len(columns))
        table += group
        for _ in columns:
            group += nodes.colspec(colwidth=1)

        head = nodes.thead()
        group += head
        head += self._row([nodes.Text(COLUMN_TITLES[key]) for key in columns])

        body = nodes.tbody()
        group += body
        for variable in variables:
            body += self._row(
                [self._cell(variable, key, unit_format) for key in columns]
            )

        caption = self.options.get("caption")
        if caption:
            table.insert(0, nodes.title(text=caption))
        return [table]

    def _select(self, registry: VariableRegistry) -> list[Variable]:
        """Return the variables to show, in registry order.

        Parameters
        ----------
        registry
            The registry.

        Returns
        -------
        list
            The selected variables.

        Raises
        ------
        ExtensionError
            If a filter is malformed or a requested name is absent.
        """
        names = self.options.get("names")
        if names:
            wanted = [name.strip() for name in names.split(",") if name.strip()]
            missing = [name for name in wanted if name not in registry]
            if missing:
                msg = f"The registry has no variable {missing!r}."
                raise ExtensionError(msg)
            return [registry[name] for name in wanted]

        variables = [registry[name] for name in registry]
        expression = self.options.get("filter")
        if not expression:
            return variables
        key, _, value = expression.partition("=")
        if key.strip() != "tags" or not value.strip():
            msg = (
                f"{expression!r} is not a variable filter;"
                ' the only one is "tags=a,b", which keeps variables'
                " carrying any of those tags."
            )
            raise ExtensionError(msg)
        wanted_tags = {tag.strip() for tag in value.split(",") if tag.strip()}
        return [variable for variable in variables if variable.tags & wanted_tags]

    def _columns(self) -> list[str]:
        """Return the columns to render.

        Returns
        -------
        list
            The column keys.

        Raises
        ------
        ExtensionError
            If a column is not one of :data:`COLUMN_TITLES`.
        """
        requested = self.options.get("columns")
        if not requested:
            return list(DEFAULT_COLUMNS)
        columns = [key.strip() for key in requested.split(",") if key.strip()]
        unknown = [key for key in columns if key not in COLUMN_TITLES]
        if unknown:
            msg = (
                f"{unknown!r} are not variable columns;"
                f" the available ones are {sorted(COLUMN_TITLES)}."
            )
            raise ExtensionError(msg)
        return columns

    def _cell(self, variable: Variable, key: str, unit_format: str) -> nodes.Node:
        """Return the content of one cell.

        Parameters
        ----------
        variable
            The variable of the row.
        key
            The column key.
        unit_format
            A pint format specification.

        Returns
        -------
        docutils.nodes.Node
            The cell content, a cross-reference for the symbol column.
        """
        if key == "name":
            return _xref(self._label, variable.name, variable.name, self.env)
        if key == "unit":
            text = format_unit(
                variable.unit, self.config.tempo_dimensionless, unit_format
            )
        elif key == "bounds":
            text = format_bounds(variable)
        elif key == "tags":
            text = ", ".join(sorted(variable.tags))
        elif key == "distribution":
            declared = variable.distribution or {}
            text = declared.get("name", "") if declared else ""
        else:
            text = str(getattr(variable, key, "") or "")
        return nodes.Text(text)

    @staticmethod
    def _row(cells: list[nodes.Node]) -> nodes.row:
        """Wrap cell contents in a table row.

        Parameters
        ----------
        cells
            One node per column.

        Returns
        -------
        docutils.nodes.row
            The row.
        """
        row = nodes.row()
        for content in cells:
            entry = nodes.entry()
            entry += nodes.paragraph("", "", content)
            row += entry
        return row


class VariableContextDirective(SphinxDirective):
    """The ``{variable-context}`` directive: which model this page is about.

    Bare names on the page resolve against this model first, which is what
    lets a page about the albatross write ``{var}`x`` and mean the downwind
    position rather than the brachistochrone's horizontal one.
    """

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    option_spec: ClassVar[dict] = {}

    def run(self) -> list[nodes.Node]:
        """Record the model for the rest of the document.

        Returns
        -------
        list
            No nodes; the directive only sets the context.

        Raises
        ------
        ExtensionError
            If no registry carries the label.
        """
        target = self.arguments[0].strip()
        label = registry_label(target) if ":" in target else target
        if label not in variable_index(self.env):
            known = ", ".join(sorted(variable_index(self.env)))
            msg = f"{label!r} is not a known model; the models are {known}."
            raise ExtensionError(msg)
        self.env.temp_data["tempo:context"] = label
        return []


class VariableGlossaryDirective(SphinxDirective):
    """The ``{variable-glossary}`` directive: every variable, anchored."""

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    option_spec: ClassVar[dict] = {
        "recursive": directives.flag,
        "unit-format": directives.unchanged,
    }

    def run(self) -> list[nodes.Node]:
        """Collect the registries and emit anchored entries, model by model.

        Returns
        -------
        list
            One section per model, each holding a definition list.
        """
        target = self.arguments[0].strip()
        if ":" in target:
            registries = [(target, load_registry(target))]
        else:
            registries = list(walk_registries(target))

        unit_format = self.options.get("unit-format", self.config.tempo_unit_format)
        by_label: dict[str, dict[str, Variable]] = {}
        for registry_target, registry in registries:
            scope = by_label.setdefault(registry_label(registry_target), {})
            for name in registry:
                scope.setdefault(name, registry[name])

        domain = self.env.get_domain("tempo")
        produced: list[nodes.Node] = []
        for label in sorted(by_label):
            section = nodes.section(ids=[f"tempo-model-{nodes.make_id(label)}"])
            section += nodes.title(text=label)
            listing = nodes.definition_list(classes=["tempo-glossary"])
            # Case-insensitive, so that V does not sort ahead of aspect_ratio.
            for name in sorted(by_label[label], key=str.lower):
                variable = by_label[label][name]
                node_id = f"tempo-var-{nodes.make_id(label)}-{nodes.make_id(name)}"
                listing += self._entry(node_id, variable, unit_format)
                domain.note_variable(label, name, node_id, variable)
            section += listing
            produced.append(section)

        self.env.note_reread()
        return produced

    def _entry(
        self, node_id: str, variable: Variable, unit_format: str
    ) -> nodes.definition_list_item:
        """Render one glossary entry.

        Parameters
        ----------
        node_id
            The anchor.
        variable
            The variable.
        unit_format
            A pint format specification.

        Returns
        -------
        docutils.nodes.definition_list_item
            The entry.
        """
        term = nodes.term()
        term["ids"].append(node_id)
        term += nodes.literal(text=variable.name)
        term += nodes.Text(" ")
        term += nodes.emphasis(text=variable.full_name)
        unit = format_unit(variable.unit, self.config.tempo_dimensionless, unit_format)
        term += nodes.Text(f" [{unit}]")

        pieces = [variable.description or "No description."]
        bounds = format_bounds(variable)
        if bounds:
            pieces.append(f"Bounds: {bounds}.")
        if variable.tags:
            pieces.append(f"Tags: {', '.join(sorted(variable.tags))}.")
        definition = nodes.definition()
        definition += nodes.paragraph("", " ".join(pieces))

        entry = nodes.definition_list_item()
        entry += term
        entry += definition
        return entry


def _xref(label: str, name: str, text: str, env: BuildEnvironment) -> nodes.Node:
    """Return a pending cross-reference to a variable's glossary entry.

    Parameters
    ----------
    label
        The model the variable belongs to.
    name
        The variable name to link to.
    text
        The text to display.
    env
        The build environment.

    Returns
    -------
    docutils.nodes.Node
        The reference node.
    """
    from sphinx.addnodes import pending_xref

    node = pending_xref(
        "",
        nodes.literal(text=text),
        refdomain="tempo",
        reftype="var",
        reftarget=f"{label}/{name}",
        refexplicit=True,
        refwarn=False,
    )
    node["refdoc"] = env.docname
    return node


class TempoDomain(Domain):
    """The ``tempo`` domain, holding the variables the glossary anchors."""

    name = "tempo"
    label = "gemseo-tempo"
    object_types: ClassVar[dict[str, ObjType]] = {
        "variable": ObjType("variable", "var")
    }
    roles: ClassVar[dict] = {"var": VariableXRefRole()}
    directives: ClassVar[dict] = {
        "variables": VariablesDirective,
        "variable-glossary": VariableGlossaryDirective,
        "variable-context": VariableContextDirective,
    }
    initial_data: ClassVar[dict] = {"variables": {}}

    @property
    def variables(self) -> dict[str, tuple[str, str, str]]:
        """Return the anchored variables, by ``"model/name"``.

        Returns
        -------
        dict
            ``"model/name" -> (docname, node_id, full_name)``.
        """
        return self.data.setdefault("variables", {})

    def note_variable(
        self, label: str, name: str, node_id: str, variable: Variable
    ) -> None:
        """Record where a variable's glossary entry lives.

        Parameters
        ----------
        label
            The model the variable belongs to.
        name
            The grammar name.
        node_id
            The anchor.
        variable
            The variable, for its human-readable name.
        """
        self.variables[f"{label}/{name}"] = (
            self.env.docname,
            node_id,
            variable.full_name,
        )

    def clear_doc(self, docname: str) -> None:
        """Drop the anchors a document contributed.

        Parameters
        ----------
        docname
            The document being re-read.
        """
        for name, (owner, _, _) in list(self.variables.items()):
            if owner == docname:
                del self.variables[name]

    def merge_domaindata(self, docnames: list[str], otherdata: dict) -> None:
        """Merge the data of a parallel read.

        Parameters
        ----------
        docnames
            The documents read in the other process.
        otherdata
            That process's domain data.
        """
        for name, entry in otherdata.get("variables", {}).items():
            if entry[0] in docnames:
                self.variables[name] = entry

    def resolve_xref(
        self,
        env: BuildEnvironment,
        fromdocname: str,
        builder: Builder,
        typ: str,
        target: str,
        node: nodes.Element,
        contnode: nodes.Element,
    ) -> nodes.Element | None:
        """Resolve a ``{var}`` reference to its glossary entry.

        Parameters
        ----------
        env
            The build environment.
        fromdocname
            The referring document.
        builder
            The builder.
        typ
            The reference type.
        target
            The variable name.
        node
            The pending reference.
        contnode
            The content to wrap.

        Returns
        -------
        docutils.nodes.Element or None
            The resolved reference, or ``None`` when no glossary entry
            exists, which leaves the text rendered but unlinked.
        """
        entry = self.variables.get(target)
        if entry is None:
            return None
        docname, node_id, full_name = entry
        return make_refnode(builder, fromdocname, docname, node_id, contnode, full_name)

    def get_objects(self) -> Iterator[tuple[str, str, str, str, str, int]]:
        """Yield the variables for the general index and the search.

        Yields
        ------
        tuple
            ``(name, dispname, type, docname, anchor, priority)``.
        """
        for key, (docname, node_id, full_name) in self.variables.items():
            label, _, name = key.partition("/")
            yield (
                key,
                f"{name} ({full_name}, {label})",
                "variable",
                docname,
                node_id,
                1,
            )


def _check_unknown(app: Sphinx, exception: Exception | None) -> None:
    """Fail the build if any mentioned variable was in no registry.

    A warning alone would be enough under ``-W``, but the point of the
    grammar is that a page cannot quietly disagree with the code, so the
    default is to fail whether or not warnings are errors.

    Parameters
    ----------
    app
        The Sphinx application.
    exception
        The exception that ended the build, if any.

    Raises
    ------
    ExtensionError
        If names were mentioned that no registry declares.
    """
    if exception is not None:
        return
    unknown = getattr(app.env, "_tempo_unknown", None)
    if unknown and app.config.tempo_unknown_variable == "error":
        listed = ", ".join(sorted(unknown))
        msg = (
            f"No registry declares these variables: {listed}."
            " Fix the name, add it to a registry, or set"
            ' tempo_unknown_variable = "warn".'
        )
        raise ExtensionError(msg)


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
    app.add_config_value("tempo_variable_packages", ["gemseo_tempo"], "env")
    app.add_config_value("tempo_unknown_variable", "error", "env")
    app.add_config_value("tempo_unit_format", "~P", "env")
    app.add_config_value("tempo_dimensionless", "–", "env")
    app.add_domain(TempoDomain)
    # The grammar spells these without a domain prefix, so they are
    # registered at the top level as well as inside the domain.
    app.add_directive("variables", VariablesDirective)
    app.add_directive("variable-glossary", VariableGlossaryDirective)
    app.add_directive("variable-context", VariableContextDirective)
    app.add_role("var", VariableXRefRole())
    app.connect("build-finished", _check_unknown)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
