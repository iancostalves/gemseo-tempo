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
"""Checks that keep the documentation in step with the code.

A convention that the docs are updated alongside the code is worth only as
much as its enforcement, so these tests make it mechanical: a new module
that nobody documented, or a headline number that drifted away from what
the code computes, fails the suite.
"""

from __future__ import annotations

import pkgutil
from pathlib import Path

import pytest

import gemseo_tempo
from gemseo_tempo.problems.brachistochrone import cycloid_time

DOCS = Path(__file__).parent.parent / "docs"
REFERENCE = DOCS / "reference" / "index.md"

UNDOCUMENTED = frozenset(
    {
        # Re-export facades. Documenting a package and the module that defines
        # an object makes every cross-reference to it ambiguous, so each object
        # is documented once, under its defining module.
        "gemseo_tempo.metadata",
        "gemseo_tempo.statics",
        "gemseo_tempo.dynamics",
        "gemseo_tempo.outputs",
        "gemseo_tempo.transcriptions",
        "gemseo_tempo.scenario",
        "gemseo_tempo.problems",
        "gemseo_tempo.problems.brachistochrone",
        "gemseo_tempo.benchmarking",
    }
)
"""Modules deliberately left out of the API reference."""


def public_modules() -> set[str]:
    """Return every importable module of the package.

    Returns
    -------
    set
        The dotted module names, including the subpackages themselves.
    """
    names = {"gemseo_tempo"}
    for info in pkgutil.walk_packages(gemseo_tempo.__path__, prefix="gemseo_tempo."):
        if not any(part.startswith("_") for part in info.name.split(".")[1:]):
            names.add(info.name)
    return names


def test_every_module_is_in_the_api_reference():
    """A module added without a docs entry must fail the suite."""
    listed = REFERENCE.read_text(encoding="utf-8")
    missing = sorted(
        name
        for name in public_modules() - UNDOCUMENTED
        if name != "gemseo_tempo" and name not in listed
    )
    assert not missing, (
        "These modules are missing from docs/reference/index.md: "
        f"{missing}. Add them there, or to UNDOCUMENTED with a reason."
    )


def test_api_reference_lists_no_stale_modules():
    """The reference must not point at modules that no longer exist."""
    listed = REFERENCE.read_text(encoding="utf-8")
    known = public_modules()
    referenced = {
        line.strip()
        for line in listed.splitlines()
        if line.strip().startswith("gemseo_tempo")
    }
    assert not (referenced - known), sorted(referenced - known)


def test_every_documentation_page_is_reachable():
    """Every page must appear in a toctree, or Sphinx warns and -W fails."""
    index = (DOCS / "index.md").read_text(encoding="utf-8")
    pages = [
        path
        for path in DOCS.rglob("*.md")
        if "_build" not in path.parts and "generated" not in path.parts
    ]
    for path in pages:
        if path.name == "index.md" and path.parent == DOCS:
            continue
        entry = path.relative_to(DOCS).with_suffix("").as_posix()
        assert entry in index, f"{entry} is in no toctree of docs/index.md"


def test_quickstart_quotes_the_computed_descent_time():
    """The headline number must not drift away from the code."""
    quickstart = (DOCS / "getting_started" / "quickstart.md").read_text(
        encoding="utf-8"
    )
    assert f"{cycloid_time():.5f}" in quickstart
    landing = (DOCS / "index.md").read_text(encoding="utf-8")
    assert f"{cycloid_time():.6f}" in landing


def test_documented_extras_match_the_packaging():
    """The install page must list the extras that actually exist."""
    pyproject = (DOCS.parent / "pyproject.toml").read_text(encoding="utf-8")
    declared = {
        line.split("=")[0].strip()
        for line in pyproject.splitlines()
        if line.endswith("= [") and not line.startswith(" ")
    }
    extras = declared & {"test", "doc", "calibration", "uq", "benchmark", "ipopt"}
    page = (DOCS / "getting_started" / "installation.md").read_text(encoding="utf-8")
    for extra in extras:
        assert f"`{extra}`" in page, f"extra {extra!r} is undocumented"
    # And nothing that was removed is still advertised.
    for gone in {"benchmark", "ipopt"} - extras:
        assert f"`{gone}`" not in page, f"extra {gone!r} no longer exists"


@pytest.mark.parametrize(
    "page",
    [
        "concepts/architecture.md",
        "concepts/transcriptions.md",
        "concepts/differentiation.md",
        "guides/scenario.md",
    ],
)
def test_key_pages_are_not_empty_stubs(page):
    text = (DOCS / page).read_text(encoding="utf-8")
    assert len(text.splitlines()) > 20, f"{page} looks like a stub"
