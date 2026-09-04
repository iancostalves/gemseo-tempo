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
"""What compiling actually costs, and when it pays for itself.

Sprint 9 found that disciplines were never JIT-compiled and reported the fix
as a single speedup number. That number conflates two costs which scale
differently: compilation grows with the size of the program, while the saving
is per evaluation. Quoting their ratio hides the only question a user has,
which is whether compiling is worth it for the problem in front of them.

So three times are measured rather than one, and from them comes the
**crossover**: the number of evaluations after which compiling has repaid
itself,

.. code-block::

    crossover = compile_seconds / (eager_seconds - compiled_seconds)

A problem that converges in fewer evaluations than that never repays its
compilation. A discretization sweep is exactly where the crossover becomes
visible, because both terms move with size and they need not move together.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from dataclasses import dataclass

import numpy


@dataclass
class Timing:
    """How long one problem takes to compile and to evaluate."""

    suite: str
    """The suite name."""

    level: str
    """The transcription level."""

    size: int
    """The discretization."""

    compile_seconds: float
    """Time to compile, measured once, on the first call after compiling."""

    compiled_seconds: float
    """Time of one evaluation once compiled."""

    eager_seconds: float
    """Time of one evaluation without compiling."""

    n_design_variables: int
    """The size of the finite-dimensional problem."""

    @property
    def speedup(self) -> float:
        """Return how much faster a compiled evaluation is.

        Returns
        -------
        float
            The ratio of eager to compiled evaluation time.
        """
        return self.eager_seconds / self.compiled_seconds

    @property
    def crossover(self) -> float:
        """Return the evaluations needed to repay compilation.

        Returns
        -------
        float
            The number of evaluations, or infinity when compiling never
            pays because it does not make evaluation faster.
        """
        saving = self.eager_seconds - self.compiled_seconds
        if saving <= 0.0:
            return float("inf")
        return self.compile_seconds / saving

    def as_dict(self) -> dict:
        """Return the timing as a plain dictionary.

        Returns
        -------
        dict
            The fields, plus the two derived quantities.
        """
        data = asdict(self)
        data["speedup"] = self.speedup
        data["crossover"] = self.crossover
        return data


def _perturbed(inputs, step: int) -> dict:
    """Return the inputs nudged, so the call is not served from the cache.

    A GEMSEO discipline caches on its inputs, so executing twice with the
    same data times a dictionary lookup rather than the computation. Every
    timed call therefore gets its own slightly different point, which is
    also more honest: an optimizer never asks the same question twice.

    Parameters
    ----------
    inputs
        The input data.
    step
        The index of the call, which sets the size of the nudge.

    Returns
    -------
    dict
        A copy with one array moved by a negligible amount.
    """
    nudged = dict(inputs)
    for name, value in inputs.items():
        array = numpy.asarray(value, dtype=float)
        if array.size:
            nudged[name] = array + 1e-9 * (step + 1)
            break
    return nudged


def _evaluate(discipline, inputs, step: int) -> float:
    """Run one discipline evaluation and return how long it took.

    Parameters
    ----------
    discipline
        The discipline to execute.
    inputs
        The input data.
    step
        The index of the call, which keeps it out of the cache.

    Returns
    -------
    float
        The elapsed time, in s.
    """
    data = _perturbed(inputs, step)
    started = time.perf_counter()
    discipline.execute(data)
    return time.perf_counter() - started


def measure(suite, level: str, size: int, repeats: int = 3) -> Timing:
    """Measure compilation and evaluation of one cell.

    The scenario is built without its formulation, since building the
    GEMSEO scenario is what compiles the disciplines and an eager timing has
    to come from one that has not been through it.

    Parameters
    ----------
    suite
        The suite.
    level
        The transcription level.
    size
        The discretization.
    repeats
        How many evaluations to take the best of. The best rather than the
        mean, since anything above the floor is interference from elsewhere
        on the machine.

    Returns
    -------
    Timing
        The three times and the problem size.
    """
    start = suite.sample(1, seed=0)[0]
    variant = suite.variants[0]

    # One scenario, stopping short of the formulation so its disciplines are
    # still uncompiled. Compiling only swaps in jitted functions, so the same
    # discipline can be timed both ways in turn.
    scenario = suite.build_scenario(level, size, start, variant)
    discipline = scenario.disciplines[0]
    inputs = dict(scenario.problem.input_defaults)
    eager = min(_evaluate(discipline, inputs, step) for step in range(repeats))

    started = time.perf_counter()
    discipline.compile_jit(pre_run=False)
    # Compiling is lazy: the cost lands on the first call, so it belongs in
    # the compilation time rather than in the evaluation that follows.
    discipline.execute(_perturbed(inputs, repeats))
    compile_seconds = time.perf_counter() - started
    compiled = min(
        _evaluate(discipline, inputs, repeats + 1 + step) for step in range(repeats)
    )

    return Timing(
        suite=suite.name,
        level=level,
        size=size,
        compile_seconds=compile_seconds,
        compiled_seconds=compiled,
        eager_seconds=eager,
        n_design_variables=sum(
            spec.size for spec in scenario.problem.design_variables.values()
        ),
    )


def sweep(suite, level: str = "collocation", sizes=None) -> list[Timing]:
    """Measure a suite across its discretizations.

    Parameters
    ----------
    suite
        The suite.
    level
        The transcription level.
    sizes
        The discretizations; the suite's own when omitted.

    Returns
    -------
    list
        One :class:`Timing` per size.
    """
    return [measure(suite, level, size) for size in (sizes or suite.sizes)]


def format_table(timings) -> str:
    """Render timings as a markdown table.

    Parameters
    ----------
    timings
        The timings to render.

    Returns
    -------
    str
        The table, with the crossover in the last column.
    """
    header = (
        "| suite | size | variables | compile | compiled | eager |"
        " speedup | crossover |"
    )
    lines = [header, "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for timing in timings:
        crossover = (
            "never" if timing.crossover == float("inf") else f"{timing.crossover:.0f}"
        )
        lines.append(
            f"| {timing.suite} | {timing.size} |"
            f" {timing.n_design_variables} |"
            f" {timing.compile_seconds:.2f} s |"
            f" {timing.compiled_seconds * 1e3:.1f} ms |"
            f" {timing.eager_seconds * 1e3:.1f} ms |"
            f" {timing.speedup:.1f}x | {crossover} |"
        )
    return "\n".join(lines)


__all__ = ["Timing", "format_table", "measure", "sweep"]
