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
import re
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
    """Every page must appear in a toctree, or Sphinx warns and -W fails.

    Toctrees nest: a problem's sub-pages are listed by that problem's own
    ``index.md``, which the root then lists. So a page counts as reachable
    if the root names it or if its own directory's index does.
    """
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
        if entry in index:
            continue
        local = path.parent / "index.md"
        assert local.exists() and path.stem in local.read_text(encoding="utf-8"), (
            f"{entry} is in no toctree, neither the root's nor its own section's"
        )


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
    extras = declared & {"test", "doc", "uq", "dev", "calibration", "benchmark"}
    page = (DOCS / "getting_started" / "installation.md").read_text(encoding="utf-8")
    for extra in extras:
        assert f"`{extra}`" in page, f"extra {extra!r} is undocumented"
    # And nothing that was removed is still advertised. Calibration was
    # planned, dropped, and its extra taken out rather than left promising
    # a capability nothing in the library provides.
    for gone in {"benchmark", "calibration"} - extras:
        assert f"`{gone}`" not in page, f"extra {gone!r} no longer exists"


@pytest.mark.parametrize(
    "page",
    [
        "concepts/architecture.md",
        "concepts/transcriptions.md",
        "concepts/differentiation.md",
        "guides/scenario.md",
        "problems/brachistochrone/physics.md",
        "problems/sail/physics.md",
        "problems/albatross/physics.md",
    ],
)
def test_key_pages_are_not_empty_stubs(page):
    text = (DOCS / page).read_text(encoding="utf-8")
    assert len(text.splitlines()) > 20, f"{page} looks like a stub"


PROBLEMS = ("brachistochrone", "sail", "albatross")
"""The problems, in the order the documentation reads them."""

PARTS = ("index", "physics", "optimization", "uncertainty")
"""The pages of a problem, in the order its toctree reads them."""

PROBLEM_PAGES = [
    f"problems/{problem}/{part}.md" for problem in PROBLEMS for part in PARTS
]
"""Every problem page, in the order the documentation reads them."""


@pytest.mark.parametrize("page", PROBLEM_PAGES)
def test_problem_pages_generate_their_variable_tables(page):
    """The sprint's acceptance criterion, made executable.

    A variable table written by hand is a copy of the registry, and a copy
    of a number is a number that will be wrong later. Every problem page
    therefore declares which model it is about, and the page that describes
    the model generates its tables from that model's registry — the other
    two cite it rather than repeating it.
    """
    text = (DOCS / page).read_text(encoding="utf-8")
    assert "```{variable-context}" in text, f"{page} declares no model"
    if page.endswith("physics.md"):
        assert "```{variables}" in text, f"{page} generates no variable table"


@pytest.mark.parametrize("page", PROBLEM_PAGES)
def test_problem_pages_reference_benchmark_pngs_that_exist(page):
    """A pre-rendered benchmark figure that is not on disk breaks the build.

    ``plot_study`` renders these once, outside the documentation build (see
    :mod:`tempo_docdata`), and the page embeds the PNG with a plain
    ``{image}`` directive rather than a ``{plot}`` block. Sphinx only warns
    about a missing one, and ``-W`` catches that; this test says so without
    a full build, and names which page and file are missing.
    """
    text = (DOCS / page).read_text(encoding="utf-8")
    for match in re.finditer(r"```\{image\}\s+(\S+)", text):
        target = match.group(1)
        if "_data/" not in target:
            continue
        path = (DOCS / page).parent / target
        assert path.resolve().exists(), (
            f"{page} embeds {target}, which does not exist; "
            "regenerate with `python tools/generate_doc_results.py`"
        )


SKELETON = (
    "Problem",
    "Deterministic optimal control",
    "Uncertainty propagation",
    "Optimization under uncertainty",
)
"""The sections of a problem, in order: the problem; what the deterministic
optimum is and how each method reaches it; what the uncertainty does to
that optimum, by each way of propagating it; the design that accounts for
the uncertainty. They are spread over the problem's pages, with the
physics between the problem and the optimum."""


@pytest.mark.parametrize("problem", PROBLEMS)
def test_problem_pages_follow_the_paper_skeleton(problem):
    """A problem reads as a paper, and can be cited as one.

    Read in the order of its toctree, its pages carry the skeleton's
    sections in order; every displayed equation carries a label, so that
    the text can refer to it by number, from this page or another; and the
    problem cites its sources, each page closing on the references it cites.
    """
    folder = DOCS / "problems" / problem
    index = (folder / "index.md").read_text(encoding="utf-8")
    toctree = re.search(r"```\{toctree\}\n(?::.*\n)*\n((?:\S+\n)+)```", index)
    assert toctree, f"{problem}/index.md has no toctree"
    assert tuple(toctree.group(1).split()) == PARTS[1:], (
        f"{problem} reads its pages as {toctree.group(1).split()}"
    )

    headings = []
    cited = False
    for part in PARTS:
        page = f"{problem}/{part}.md"
        text = (folder / f"{part}.md").read_text(encoding="utf-8")
        headings += re.findall(r"^## (.+)$", text, flags=re.MULTILINE)
        unlabelled = re.findall(r"```\{math\}\n(?!:label:)", text)
        assert not unlabelled, f"{page} has {len(unlabelled)} unlabelled equations"
        assert "{cite}" not in text, f"{page} cites outside its own references"
        if "{footcite" in text:
            cited = True
            assert text.rstrip().endswith(
                "## References\n\n```{footbibliography}\n```"
            ), f"{page} cites, but does not close on its references"
        else:
            assert "## References" not in text, (
                f"{page} lists references it never cites"
            )
    sections = tuple(heading for heading in headings if heading in SKELETON)
    assert sections == SKELETON, f"{problem} has sections {sections}"
    assert cited, f"{problem} cites nothing"


def test_every_variable_the_pages_mention_is_declared():
    """A mention of a name no registry has fails the documentation build.

    The build enforces this, so this test exists to say what the rule is
    and to catch it without a full Sphinx run.
    """
    import re
    import sys

    # The extension is built on docutils, which only the doc extras install.
    pytest.importorskip("docutils")
    sys.path.insert(0, str(DOCS / "_ext"))
    from tempo_myst import registry_label
    from tempo_myst import walk_registries

    declared: dict[str, set[str]] = {}
    for target, registry in walk_registries("gemseo_tempo"):
        declared.setdefault(registry_label(target), set()).update(registry)
    every = set().union(*declared.values())

    pattern = re.compile(r"\{var\}`([^`<]+?)(?:\s*<[^`>]*>)?`")
    for path in DOCS.rglob("*.md"):
        if "_build" in path.parts:
            continue
        for mention in pattern.findall(path.read_text(encoding="utf-8")):
            name = mention.strip()
            if ":" in name:
                label, _, bare = name.partition(":")
                assert bare in declared.get(label, set()), (
                    f"{path.name} mentions {name!r}, which {label!r} has not"
                )
            else:
                assert name in every, f"{path.name} mentions undeclared {name!r}"
