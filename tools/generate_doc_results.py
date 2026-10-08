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
"""Computing everything the documentation's figures are drawn from.

The documentation draws its figures at build time, so a reader sees the
code that made the picture. What that code must not do is solve anything:
an optimizer in a documentation build is an hour of continuous integration
and a page that breaks whenever a solver has an off day, which this
library has seen happen.

So every optimization happens here, once, and the answers are stored under
``docs/_data``. The figures load them. Direct function evaluations, a polar
curve or a wind profile, are left to the page, because they cost
milliseconds and cannot go stale.

This is the same arrangement ``tools/generate_brachistochrone_references.py``
uses for the reference descent times, and it has the same weakness: run by
hand, the stored answers can drift from the code that would now produce
them. ``tests/test_doc_data.py`` is the guard, checking the stored results
against quantities cheap enough to recompute on every test run.

Usage::

    python tools/generate_doc_results.py                 # everything
    python tools/generate_doc_results.py --only sail     # one problem
    python tools/generate_doc_results.py --smoke         # tiny, for tests
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from doc_results.albatross import generate_albatross
from doc_results.brachistochrone import generate_brachistochrone
from doc_results.common import set_data_path
from doc_results.sail import generate_sail

logging.disable(logging.CRITICAL)

PROBLEMS = ("brachistochrone", "sail", "albatross")
"""The problems, in the order the documentation introduces them."""


def main(argv=None) -> int:
    """Generate the stored results.

    Parameters
    ----------
    argv
        The command-line arguments.

    Returns
    -------
    int
        Zero.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=PROBLEMS, default=list(PROBLEMS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)

    if arguments.output is not None:
        set_data_path(arguments.output)

    generators = {
        "brachistochrone": generate_brachistochrone,
        "sail": generate_sail,
        "albatross": generate_albatross,
    }
    for problem in arguments.only:
        started = time.perf_counter()
        generators[problem](smoke=arguments.smoke)
        print(f"  ({time.perf_counter() - started:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
