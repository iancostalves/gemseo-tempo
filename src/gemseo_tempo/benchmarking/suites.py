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
"""The problems a benchmark run can be pointed at.

The runner used to import the brachistochrone directly, which meant it could
only ever benchmark that one problem. A :class:`Suite` is the small amount of
knowledge the runner actually needs about a problem: how to build it at a
given transcription, discretization and starting point, what to score the
answer against, how to draw starting points, and which levels it can be asked
for at all.

Three suites are defined, and they are deliberately not the same size. Each
cell is a full optimization, so the cost lives in the starts and the
discretizations rather than in the problem:

- The **brachistochrone** is cheap and has a closed-form answer, so it gets
  the full sweep and carries the weight of the comparison.
- The **sail** is a few times more expensive and scored against the ideal
  gybe time, so it gets fewer starts.
- The **albatross** is the richest problem and the most expensive, so it gets
  the smallest sweep of all. It also has no closed form, so it is scored
  against its own converged optimum, which measures reproducibility rather
  than correctness. That is worth knowing when reading its profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence


@dataclass(frozen=True)
class Suite:
    """One benchmark problem, and how to run it."""

    name: str
    """The suite name, recorded with every result."""

    build: Callable
    """``(level, size, start, variant) -> (scenario, gemseo_scenario)``."""

    build_scenario: Callable
    """``(level, size, start, variant) -> scenario``, without a formulation.

    Building the GEMSEO scenario is what compiles the disciplines, so
    anything that wants to see them uncompiled has to stop short of it.
    """

    target: Callable
    """``(variant) -> float``, the objective a run is scored against."""

    sample: Callable
    """``(count, seed) -> list`` of starting points."""

    levels: tuple[str, ...]
    """The transcription levels this problem can be asked for."""

    sizes: tuple[int, ...]
    """The discretizations swept."""

    starts: int
    """The number of starting points per cell."""

    variants: tuple[str, ...] = ("default",)
    """The physical variants swept, if any."""

    algorithms: tuple[str, ...] = ("SLSQP", "NLOPT_SLSQP", "NLOPT_COBYLA")
    """The optimizers compared when the study varies the optimizer."""

    accuracy: float = 1e-2
    """The relative error below which a run counts as converged.

    Per suite, because the problems do not share a floor. The
    brachistochrone is scored against an exact cycloid and reaches it. The
    sail is scored against an ideal gybe that no smooth heading can fly, so
    even a perfect run lands a fraction of a percent above it.
    """

    scored_against_itself: bool = False
    """Whether the target is a converged run rather than a known answer.

    When true the data profile measures reproducibility: whether a run
    returns to the same optimum from a different start, not whether that
    optimum is right.
    """

    extra: dict = field(default_factory=dict)
    """Anything a suite needs to carry that the runner does not interpret."""

    def supports(self, level: str) -> bool:
        """Return whether this problem can be run at a level.

        Parameters
        ----------
        level
            The level name.

        Returns
        -------
        bool
            Whether the suite declares it.
        """
        return level in self.levels


def _brachistochrone_build(level, size, start, variant):
    """Build the brachistochrone at one level.

    Parameters
    ----------
    level
        The transcription level.
    size
        The discretization.
    start
        The starting point.
    variant
        The physical variant.

    Returns
    -------
    tuple
        The dynamic scenario and the GEMSEO scenario.
    """
    from gemseo_tempo.benchmarking.transcriptions import build_transcription
    from gemseo_tempo.problems.brachistochrone.problem import create_optimization

    transcription, effective = build_transcription(level, size)
    return create_optimization(
        transcription=transcription, variant=variant, size=effective, guess=start
    )


def _sail_build(level, size, start, variant):
    """Build the downwind sail leg at one level.

    Parameters
    ----------
    level
        The transcription level.
    size
        The discretization.
    start
        The starting point.
    variant
        Unused; the sail has one physical variant.

    Returns
    -------
    tuple
        The dynamic scenario and the GEMSEO scenario.
    """
    from gemseo_tempo.benchmarking.transcriptions import build_transcription
    from gemseo_tempo.problems.sail.problem import create_optimization

    transcription, effective = build_transcription(level, size)
    return create_optimization(
        transcription=transcription, leg="downwind", size=effective, guess=start
    )


def _albatross_build(level, size, start, variant):
    """Build the albatross co-design cycle at one level.

    Parameters
    ----------
    level
        The transcription level.
    size
        The discretization.
    start
        The starting cycle.
    variant
        Unused; the albatross has one physical variant.

    Returns
    -------
    tuple
        The dynamic scenario and the GEMSEO scenario.
    """
    from gemseo_tempo.benchmarking.transcriptions import build_transcription
    from gemseo_tempo.problems.albatross.problem import create_codesign_optimization

    transcription, effective = build_transcription(level, size)
    return create_codesign_optimization(
        transcription=transcription, size=effective, guess=start
    )


def _sail_scenario(level, size, start, variant):
    """Build the sail without a formulation.

    Parameters
    ----------
    level
        The transcription level.
    size
        The discretization.
    start
        The starting point.
    variant
        The physical variant.

    Returns
    -------
    DynamicScenario
        The scenario, its disciplines not yet compiled.
    """
    from gemseo_tempo.benchmarking.transcriptions import build_transcription
    from gemseo_tempo.problems.sail.problem import create_scenario

    transcription, effective = build_transcription(level, size)
    return create_scenario(
        transcription=transcription, leg="downwind", size=effective, guess=start
    )


def _sail_target(variant):
    """Return the ideal gybe time of the downwind leg.

    Parameters
    ----------
    variant
        Unused.

    Returns
    -------
    float
        The time a two-legged gybe at the best speed made good would take,
        in s. No trajectory achieves it, since a smooth heading cannot turn
        instantly, so a converged run sits a fraction of a percent above.
    """
    import jax.numpy as jnp

    from gemseo_tempo.problems.sail.polar import best_vmg_angle
    from gemseo_tempo.problems.sail.polar import boat_speed
    from gemseo_tempo.problems.sail.problem import LEG_LENGTH
    from gemseo_tempo.problems.sail.wind import create_wind

    east, north = create_wind("uniform").velocity(0.0, 0.0)
    speed = float(jnp.hypot(east, north))
    angle = best_vmg_angle("downwind")
    made_good = float(boat_speed(angle, speed)) * abs(
        float(jnp.cos(jnp.radians(angle)))
    )
    return LEG_LENGTH / made_good


_ALBATROSS_REFERENCE = 8.649
"""The converged minimum wind of the albatross co-design, in m/s.

Measured at sixteen collocation segments from the seeded start. The albatross
has no closed form, so this is a recorded result rather than a known answer,
and a run is scored on returning to it.
"""


def _albatross_scenario(level, size, start, variant):
    """Build the albatross without a formulation.

    Parameters
    ----------
    level
        The transcription level.
    size
        The discretization.
    start
        The starting point.
    variant
        The physical variant.

    Returns
    -------
    DynamicScenario
        The scenario, its disciplines not yet compiled.
    """
    from gemseo_tempo.benchmarking.transcriptions import build_transcription
    from gemseo_tempo.problems.albatross.problem import create_codesign_scenario

    transcription, effective = build_transcription(level, size)
    return create_codesign_scenario(
        transcription=transcription, size=effective, guess=start
    )


def _albatross_target(variant):
    """Return the reference minimum wind.

    Parameters
    ----------
    variant
        Unused.

    Returns
    -------
    float
        The reference, in m/s.
    """
    return _ALBATROSS_REFERENCE


def _brachistochrone_scenario(level, size, start, variant):
    """Build the brachistochrone without a formulation.

    Parameters
    ----------
    level
        The transcription level.
    size
        The discretization.
    start
        The starting point.
    variant
        The physical variant.

    Returns
    -------
    DynamicScenario
        The scenario, its disciplines not yet compiled.
    """
    from gemseo_tempo.benchmarking.transcriptions import build_transcription
    from gemseo_tempo.problems.brachistochrone.problem import create_scenario

    transcription, effective = build_transcription(level, size)
    return create_scenario(
        transcription=transcription, variant=variant, size=effective, guess=start
    )


def _brachistochrone_target(variant):
    """Return the analytic descent time of a variant.

    Parameters
    ----------
    variant
        The physical variant.

    Returns
    -------
    float
        The reference time, in s.
    """
    from gemseo_tempo.problems.brachistochrone.reference import reference_time

    return reference_time(variant)


def _sample(module_function):
    """Return a sampler that imports lazily.

    Parameters
    ----------
    module_function
        The name of the sampling function.

    Returns
    -------
    Callable
        The sampler.
    """

    def sample(count, seed=0):
        from gemseo_tempo.benchmarking import sampling

        return getattr(sampling, module_function)(count, seed)

    return sample


BRACHISTOCHRONE = Suite(
    name="brachistochrone",
    build=_brachistochrone_build,
    build_scenario=_brachistochrone_scenario,
    target=_brachistochrone_target,
    sample=_sample("sample_guesses"),
    levels=("shooting", "collocation", "collocation-sparse", "indirect"),
    sizes=(3, 10, 30, 100),
    starts=25,
    variants=("ideal", "friction", "drag"),
)
"""The cheap problem with a closed-form answer, swept in full."""

SAIL = Suite(
    name="sail",
    build=_sail_build,
    build_scenario=_sail_scenario,
    target=_sail_target,
    sample=_sample("sample_sail_guesses"),
    # No indirect: a path constraint changes the necessary conditions, and
    # IndirectShooting refuses such a problem rather than solving it wrongly.
    levels=("shooting", "collocation", "collocation-sparse"),
    # Ten segments cannot reach the ideal: its discretization floor is 1.9%,
    # which is above any sensible accuracy, so that cell would report failure
    # for a run that had converged perfectly well.
    sizes=(25, 40),
    starts=10,
    # The question this problem asks is whether the optimizer gybes at all,
    # and the answer is bimodal. Over ten starts, nine land between 0.25%
    # and 3.2% of the ideal and one lands at 12.7%, which is the straight
    # run. Nothing falls in between, so the threshold sits in the gap and
    # separates the two populations rather than being tuned to a pass rate.
    accuracy=4e-2,
)
"""The downwind gybe, scored against the ideal speed made good."""

ALBATROSS = Suite(
    name="albatross",
    build=_albatross_build,
    build_scenario=_albatross_scenario,
    target=_albatross_target,
    sample=_sample("sample_albatross_guesses"),
    levels=("collocation", "collocation-sparse"),
    sizes=(8, 16),
    starts=5,
    algorithms=("SLSQP", "NLOPT_SLSQP"),
    accuracy=2e-2,
    scored_against_itself=True,
)
"""The co-design cycle: richest, most expensive, smallest sweep."""

SUITES = {suite.name: suite for suite in (BRACHISTOCHRONE, SAIL, ALBATROSS)}
"""Every suite, by name."""


def get_suite(name: str) -> Suite:
    """Return a suite by name.

    Parameters
    ----------
    name
        The suite name.

    Returns
    -------
    Suite
        The suite.

    Raises
    ------
    ValueError
        If no suite has that name.
    """
    if name not in SUITES:
        raise ValueError(f"Unknown suite {name!r}; use one of {sorted(SUITES)}.")
    return SUITES[name]


def sweep_size(suite: Suite, levels: Sequence[str]) -> int:
    """Return how many runs a full sweep of a suite would take.

    Parameters
    ----------
    suite
        The suite.
    levels
        The levels of the study.

    Returns
    -------
    int
        The number of cells times the starting points.
    """
    return len(levels) * len(suite.sizes) * suite.starts


__all__ = [
    "ALBATROSS",
    "BRACHISTOCHRONE",
    "SAIL",
    "SUITES",
    "Suite",
    "get_suite",
    "sweep_size",
]
