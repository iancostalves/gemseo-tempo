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
"""The variable grammar, built by Sphinx and read back out of the HTML.

These build a miniature documentation project rather than calling the
directives directly, because what matters is what reaches the page: a role
that resolves to the wrong model, or a table that renders empty, is invisible
to any test that stops at the node tree.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

# The grammar is documentation machinery, and Sphinx is in the doc extra
# rather than the test one. Without it there is nothing here to test.
pytest.importorskip("sphinx")
pytest.importorskip("myst_parser")

sys.path.insert(0, str(Path(__file__).parents[2] / "docs" / "_ext"))

from tempo_myst import format_bounds  # noqa: E402
from tempo_myst import format_unit  # noqa: E402
from tempo_myst import load_registry  # noqa: E402
from tempo_myst import registry_label  # noqa: E402
from tempo_myst import render  # noqa: E402
from tempo_myst import walk_registries  # noqa: E402

from gemseo_tempo.metadata.variable import Variable  # noqa: E402

DASH = "–"
"""The en dash standing in for a dimensionless unit."""


def _text(html: str) -> str:
    """Strip tags and collapse whitespace, for asserting on what is read."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


# ------------------------------------------------------------ the pieces


def test_a_unit_is_rendered_the_way_a_reader_expects():
    assert format_unit("m/s**2", DASH, "~P") == "m/s²"
    assert format_unit("dimensionless", DASH, "~P") == DASH
    assert format_unit("", DASH, "~P") == DASH
    # pint is more capable than one might guess, and does know furlongs.
    assert format_unit("furlong/fortnight", DASH, "~P") == "fur/fortnight"
    # A unit it cannot parse is shown as written rather than swallowed.
    assert format_unit("bogons!!", DASH, "~P") == "bogons!!"


def test_bounds_read_as_intervals():
    def variable(lower, upper):
        return Variable(name="v", full_name="V", lower_bound=lower, upper_bound=upper)

    assert format_bounds(variable(None, None)) == ""
    assert format_bounds(variable(0.5, None)) == ">= 0.5"
    assert format_bounds(variable(None, 40.0)) == "<= 40"
    assert format_bounds(variable(8.0, 80.0)) == "[8, 80]"


def test_the_styles_say_different_amounts():
    variable = Variable(name="c_l", full_name="Lift coefficient")
    assert render(variable, "full", DASH, "~P") == f"Lift coefficient (c_l, {DASH})"
    assert render(variable, "name", DASH, "~P") == "Lift coefficient"
    assert render(variable, "symbol", DASH, "~P") == "c_l"
    assert render(variable, "unit", DASH, "~P") == DASH


def test_a_registry_is_found_by_its_import_target():
    registry = load_registry("gemseo_tempo.problems.sail.model:REGISTRY")
    assert "psi" in registry


@pytest.mark.parametrize(
    "target",
    ["not_a_target", "gemseo_tempo.nowhere:REGISTRY", "gemseo_tempo:__doc__"],
)
def test_a_bad_registry_target_is_refused(target):
    from sphinx.errors import ExtensionError

    with pytest.raises(ExtensionError):
        load_registry(target)


def test_every_registry_of_a_problem_shares_its_label():
    """Because a page about the albatross is about all of it.

    The wing variables live in a different registry from the states, and a
    reader writing ``{var}`wing_area``` on the albatross page does not
    care which.
    """
    assert registry_label("gemseo_tempo.problems.albatross.model:REGISTRY") == (
        "albatross"
    )
    assert (
        registry_label("gemseo_tempo.problems.albatross.problem:DESIGN_REGISTRY")
        == "albatross"
    )
    assert registry_label("gemseo_tempo.metadata.registry:REGISTRY") == "registry"


def test_a_re_exported_registry_is_walked_once():
    """The package facades re-export the same object under a second name."""
    targets = [target for target, _ in walk_registries("gemseo_tempo.problems")]
    registries = [registry for _, registry in walk_registries("gemseo_tempo.problems")]
    assert len(targets) == len({id(registry) for registry in registries})


# ------------------------------------------------------------- the build


@pytest.mark.sphinx("html", testroot="grammar")
def test_the_role_resolves_against_the_page_model(app):
    """``mu`` is a bank angle here and a friction coefficient elsewhere."""
    app.build()
    html = (Path(app.outdir) / "index.html").read_text(encoding="utf-8")
    body = _text(html)
    assert "Airspeed (V, m/s)" in body
    assert "Bank angle (mu, rad)" in body
    assert "Downwind position (x, m)" in body
    # And the qualified form reaches the other model from the same page.
    assert "Coulomb friction coefficient (mu," in body


@pytest.mark.sphinx("html", testroot="grammar")
def test_the_styles_reach_the_page(app):
    app.build()
    body = _text((Path(app.outdir) / "index.html").read_text(encoding="utf-8"))
    # Each role is its own element, so the stripped text is spaced out.
    assert re.search(r"Short forms:\s*V\s*,\s*Airspeed\s*and\s*m/s", body)


@pytest.mark.sphinx("html", testroot="grammar")
def test_every_mention_links_to_its_own_glossary_entry(app):
    """Scoped anchors, so the three x's do not collide."""
    app.build()
    html = (Path(app.outdir) / "index.html").read_text(encoding="utf-8")
    assert 'href="#tempo-var-albatross-x"' in html
    assert 'href="#tempo-var-albatross-mu"' in html
    assert 'href="#tempo-var-brachistochrone-mu"' in html
    # The anchors those point at exist.
    for anchor in ("tempo-var-albatross-x", "tempo-var-brachistochrone-mu"):
        assert f'id="{anchor}"' in html


@pytest.mark.sphinx("html", testroot="grammar")
def test_a_table_is_generated_from_the_registry(app):
    app.build()
    html = (Path(app.outdir) / "index.html").read_text(encoding="utf-8")
    tables = re.findall(r"<tbody>(.*?)</tbody>", html, re.S)
    assert tables, "no table was generated"
    controls = _text(tables[0])
    # The two albatross controls, with their units and the bound the
    # statics chain sets.
    assert "cl" in controls
    assert "Lift coefficient" in controls
    assert "mu" in controls
    assert "Bank angle" in controls
    assert "[-0.2, 1.5]" in controls
    # A state is not a control, so it is not in this table.
    assert "Airspeed" not in controls


@pytest.mark.sphinx("html", testroot="grammar")
def test_named_variables_appear_in_the_order_they_were_asked_for(app):
    app.build()
    html = (Path(app.outdir) / "index.html").read_text(encoding="utf-8")
    tables = re.findall(r"<tbody>(.*?)</tbody>", html, re.S)
    selected = _text(tables[1])
    assert selected.index("psi") < selected.index("t_f")
    assert "Total sailing time" in selected


@pytest.mark.sphinx("html", testroot="grammar")
def test_the_glossary_groups_the_models_and_anchors_every_variable(app):
    app.build()
    html = (Path(app.outdir) / "index.html").read_text(encoding="utf-8")
    models = set(re.findall(r'id="tempo-model-([^"]+)"', html))
    assert {"albatross", "brachistochrone", "sail"} <= models
    # Every variable of every problem is anchored exactly once.
    anchors = re.findall(r'id="(tempo-var-[^"]+)"', html)
    assert len(anchors) == len(set(anchors))
    from docutils.nodes import make_id

    declared = {
        f"tempo-var-{make_id(registry_label(target))}-{make_id(name)}"
        for target, registry in walk_registries("gemseo_tempo.problems")
        for name in registry
    }
    assert declared <= set(anchors)


@pytest.mark.sphinx("html", testroot="grammar")
def test_descriptions_reach_the_reader(app):
    """The glossary is only worth generating if the registry says something."""
    app.build()
    body = _text((Path(app.outdir) / "index.html").read_text(encoding="utf-8"))
    assert "No description." not in body


@pytest.mark.sphinx("html", testroot="unknown")
def test_a_name_no_registry_declares_fails_the_build(app):
    """The point of the whole arrangement.

    A page that mentions a variable the code does not have is a page that
    has drifted from the code, and finding that out at build time is the
    reason the grammar reads the registry instead of trusting the author.
    """
    from sphinx.errors import ExtensionError

    # force_all, so the role runs even when a previous build of the same
    # root left its doctrees behind and nothing looks out of date.
    with pytest.raises(ExtensionError, match="not_a_variable"):
        app.build(force_all=True)


@pytest.mark.sphinx("html", testroot="unknown-warn")
def test_the_failure_can_be_downgraded_to_a_warning(app, warning):
    app.build(force_all=True)
    assert "not_a_variable" in warning.getvalue()
    # The text still reaches the reader, unlinked.
    body = _text((Path(app.outdir) / "index.html").read_text(encoding="utf-8"))
    assert "not_a_variable" in body
