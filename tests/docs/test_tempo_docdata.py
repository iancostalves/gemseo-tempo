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
"""The stored-result grammar, built by Sphinx and read back out of the HTML.

A number quoted in prose and a problem statement are both copies of a
stored result, and a copy can only be trusted if it is made at build time
from the file the tests check. These build a miniature project to see what
reaches the page, and a second one to see a stale quotation fail the build.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy
import pytest

pytest.importorskip("sphinx")
pytest.importorskip("myst_parser")

sys.path.insert(0, str(Path(__file__).parents[2] / "docs" / "_ext"))

from tempo_docdata import lookup  # noqa: E402
from tempo_docdata import read  # noqa: E402
from tempo_docdata import render  # noqa: E402
from tempo_docdata import split_target  # noqa: E402


def _text(html: str) -> str:
    """Strip tags and collapse whitespace, for asserting on what is read."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


# ------------------------------------------------------------ the pieces


def test_a_target_splits_into_file_key_and_format():
    assert split_target("a/b.json:x.y:.2f") == ("a/b.json", "x.y", ".2f")
    assert split_target("a.json:x") == ("a.json", "x", "")


@pytest.mark.parametrize("target", ["a.json", ":x", "a.json:"])
def test_a_target_without_a_key_is_refused(target):
    with pytest.raises(ValueError, match="file:key"):
        split_target(target)


def test_a_value_can_be_scaled_before_it_is_formatted():
    assert render(1.81e-5, ".3f*1e3") == "0.018"
    assert render(0.06446, ".1f*1e3") == "64.5"
    assert render(2.5, "") == "2.5"


def test_a_dotted_key_walks_mappings_and_lists():
    record = {"a": {"b": [1.0, {"c": 2.5}]}}
    assert lookup(record, "a.b.1.c") == 2.5
    assert lookup(record, "a.b.0") == 1.0


def test_a_missing_key_names_what_was_there():
    with pytest.raises(KeyError, match="'b'"):
        lookup({"a": 1.0, "b": 2.0}, "c")
    with pytest.raises(KeyError, match="cannot index"):
        lookup({"a": 1.0}, "a.b")


def test_a_stored_array_reduces_to_its_scalar(tmp_path):
    full = tmp_path / "arrays.npz"
    numpy.savez(full, time=numpy.asarray(0.8243), xs=numpy.arange(3.0))
    record = read(full)
    assert lookup(record, "time") == pytest.approx(0.8243)
    assert isinstance(lookup(record, "time"), float)
    assert lookup(record, "xs.2") == 2.0
    other = tmp_path / "history.h5"
    other.touch()
    assert read(other) == other


# ------------------------------------------------------------ on the page


@pytest.mark.sphinx("html", testroot="paper")
def test_quoted_values_reach_the_page_formatted(app):
    app.build(force_all=True)
    body = _text((Path(app.outdir) / "index.html").read_text(encoding="utf-8"))
    assert "minimum wind is 8.63 m/s" in body
    assert "drag work -38.5 m" in body
    assert "nets +0.13 m" in body
    assert "Unformatted: 18.0" in body
    assert "scaled: 38546 mm" in body


@pytest.mark.sphinx("html", testroot="paper")
def test_the_formulation_is_rendered_from_the_stored_description(app):
    app.build(force_all=True)
    body = _text((Path(app.outdir) / "index.html").read_text(encoding="utf-8"))
    assert "At twenty segments: minimize time over 22 scalar design variables" in body
    assert "subject to 61 scalar constraints in 2 groups" in body
    assert "control.gamma 21 [0, 1.5708]" in body
    assert "t_f 1 [0.05, inf]" in body
    assert "constraint.final.x = 0 1" in body
    assert "constraint.lift_limit ≤ 0 60" in body


@pytest.mark.sphinx("html", testroot="paper-missing")
def test_a_value_no_stored_result_holds_fails_the_build(app, warning):
    app.build(force_all=True)
    log = warning.getvalue()
    assert "study.json:not_a_key:.2f" in log
    assert "not_stored.json" in log
