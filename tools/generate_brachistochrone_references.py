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
"""Regenerate the stored brachistochrone reference times.

The friction and drag variants have no closed form we are prepared to
assert, so their reference descent times are computed here at a fine
discretization with two independent transcriptions and stored in
``gemseo_tempo.problems.brachistochrone.reference.STORED_REFERENCES``.
Agreement between the two transcriptions is the accuracy check.

Run with::

    python tools/generate_brachistochrone_references.py
"""

from __future__ import annotations

import logging

from gemseo_tempo.problems.brachistochrone import GRAVITY
from gemseo_tempo.problems.brachistochrone import TARGET
from gemseo_tempo.problems.brachistochrone import create_optimization
from gemseo_tempo.problems.brachistochrone import cycloid_time
from gemseo_tempo.transcriptions import HermiteSimpson
from gemseo_tempo.transcriptions import SingleShooting

logging.disable(logging.CRITICAL)

FINE_SIZE = 200
"""The discretization used for the references."""


def solve(variant: str, transcription, size: int) -> float:
    """Solve one variant and return the descent time."""
    _, gemseo_scenario = create_optimization(
        transcription=transcription, variant=variant, size=size
    )
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=400)
    return float(gemseo_scenario.optimization_result.f_opt)


def main() -> None:
    """Print the reference table."""
    print(f"ideal (closed form)   t_f = {cycloid_time():.6f} s")
    print()
    print("STORED_REFERENCES = {")
    for variant in ("friction", "drag"):
        collocation = solve(variant, HermiteSimpson(num_segments=FINE_SIZE), FINE_SIZE)
        shooting = solve(variant, SingleShooting(), FINE_SIZE)
        spread = abs(collocation - shooting)
        print(
            f'    ("{variant}", {TARGET[0]}, {TARGET[1]}, {GRAVITY}):'
            f" {collocation:.6f},"
            f"  # shooting {shooting:.6f}, spread {spread:.2e}"
        )
    print("}")


if __name__ == "__main__":
    main()
