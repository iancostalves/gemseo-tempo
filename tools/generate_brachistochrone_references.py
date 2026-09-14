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

from gemseo_tempo.dynamics.integrators import DiffraxIntegrator
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
    """Solve one variant and return the descent time.

    Shooting integrates with a **fixed** step here, which matters only
    since the friction law started reading the control's turning rate. The
    rate of a piecewise-linear control is piecewise constant, so at two
    hundred nodes the right-hand side has two hundred jumps in it, and an
    adaptive controller spends the solve chasing them: left adaptive, the
    two transcriptions disagreed by 43% at this discretization while
    agreeing to 0.02% at twenty. A fixed step crosses the jumps the same
    way every time, which is the same lesson the albatross's Jacobian
    checks learned.
    """
    _, gemseo_scenario = create_optimization(
        transcription=transcription,
        variant=variant,
        size=size,
        integrator=DiffraxIntegrator(dt0=1.0 / (20 * size), n_save=size + 1),
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
