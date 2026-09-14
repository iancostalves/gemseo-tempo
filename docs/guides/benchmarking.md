# Benchmark studies

The harness compares transcriptions, derivative modes and optimizers on the
brachistochrone, over sampled starting points and a sweep of discretizations.

## One factor at a time

A full factorial would spend most of its budget on combinations nobody asks
about. Instead a single baseline is fixed — collocation, SLSQP, dense
automatic differentiation — and each study varies exactly one factor:

```{list-table}
:header-rows: 1
:widths: 22 40 38

* - Study
  - Levels
  - Held at baseline
* - `transcription`
  - shooting, collocation, indirect
  - SLSQP, dense AD
* - `derivative`
  - finite differences, dense AD, sparse AD
  - collocation, SLSQP
* - `optimizer`
  - SLSQP, NLOPT_SLSQP, COBYLA
  - collocation, dense AD
```

Each cell sweeps 3, 10, 30 and 100 and repeats over 25 starting points.

Indirect shooting has no path discretization — its design space is a handful
of costates — so it is placed on the same abscissa by sweeping its integrator
save resolution instead.

## Running

```bash
python examples/benchmark/run_studies.py --smoke --suite all   # minutes
python examples/benchmark/run_studies.py --suite sail          # one problem
python examples/benchmark/run_studies.py --suite all           # everything
```

The full sweep takes on the order of an hour, with the finite-difference
cells of the derivative study dominating the total. Every run is cached as JSON
keyed by its configuration, so the sweep is **resumable**: interrupt it and
re-run, and finished cells cost nothing.

## Starting points

Randomizing raw node values would produce trajectories that are not merely
poor but meaningless. Starting points are instead drawn by Latin hypercube
over three physical guess parameters — the initial path angle, its increase
over the descent, and the guessed descent time — and turned into a consistent
trajectory by simulation. The same sample is shared by every level of a
study, so no level gets easier starts than another.

## Reading the figures

Each figure has two rows. The top row is the familiar box plot of discipline
calls against discretization, one panel per level. The bottom row is the
share of starting points actually solved.

The second row matters: a cost distribution means little on its own, because
runs that exhaust their budget without solving anything would otherwise sit
in the box plot inflating it. Every run reports a status —
`converged`, `inaccurate`, `budget-exhausted` or `failed` — and only solved
runs enter the cost panel.

{func}`~gemseo_tempo.benchmarking.profiles.data_profile` gives the
complementary view: the fraction of starting points solved within a given
budget of calls.

:::{note}
The optimizer study compares only constrained solvers. Every suite here
carries at least one equality (terminal, or collocation's own defects);
L-BFGS-B, TNC and Nelder-Mead reject that outright rather than solve it
badly, so including them would compare failures rather than methods.
:::

Each problem's own optimizer, transcription and AD benchmarks are on its
page, next to the result they benchmark:
{doc}`../problems/brachistochrone/optimization`,
{doc}`../problems/sail/optimization`,
{doc}`../problems/albatross/optimization`.

## Three problems, not one

The runner used to import the brachistochrone directly, so it could benchmark
nothing else. A `Suite` is now the small amount a benchmark needs to know
about a problem: how to build it at a level, size and starting point, what to
score the answer against, and which levels it can be asked for at all.

The three are deliberately not the same size, because each cell is a full
optimization and the cost lives in the starts:

| suite | studies | sizes | starts | scored against |
| --- | --- | --- | --- | --- |
| brachistochrone | all three | 3, 10, 30, 100 | 25 | the cycloid, exactly |
| sail | transcription, derivative, optimizer | 25, 40 | 10 | the ideal gybe |
| albatross | derivative, optimizer | 8, 16 | 5 | its own converged optimum |

A full sweep is roughly ninety minutes, and resumable: every finished run is
cached as JSON keyed by its cell, so an interrupted sweep costs nothing to
restart.

Three things about that table are worth knowing before reading any profile.

**The albatross has no closed form.** It is scored against a converged run, so
its data profile measures whether the optimizer returns to the same answer
from a different start. That is reproducibility, not correctness.

**The albatross has no transcription study.** It offers only collocation
levels, and a study that varies one thing needs at least two of them, so it is
dropped rather than run against itself.

**The suites do not share an accuracy.** The brachistochrone reaches its
cycloid, so a percent is a fair threshold. The sail is scored against an ideal
gybe that no smooth heading can fly, so even a perfect run lands a fraction of
a percent above it. Its threshold is four percent, and that number is not
tuned to a pass rate: over ten starts, nine land between 0.25% and 3.2% and
one lands at 12.7%, which is the straight run. Nothing falls in between, so
the threshold sits in the gap and separates gybing from not gybing.

## What compiling costs, and when it pays

Sprint 9 found the disciplines were never JIT-compiled and reported the fix
as a single speedup number. That conflates two costs which scale
differently: compilation grows with the program, the saving is per
evaluation. So three times are measured on each problem, and from them the
**crossover**, the evaluations needed before compiling has repaid itself —
see the Optimization section of each problem's own page for its numbers.
The answer is the same everywhere measured so far: compiling pays within
single-digit evaluations, against the dozens or hundreds any optimization
makes.

Two cautions on reading those numbers, wherever they are quoted. They are
for a single discipline evaluation, not for a whole optimization, which
also spends time in gradients and in GEMSEO itself; Sprint 9's end-to-end
figures were smaller for that reason. And the measurement perturbs its
inputs on every call, because a GEMSEO discipline caches on its inputs and
timing the same call twice measures a dictionary lookup. Both are easy
mistakes and both were made first.

## Baselines, and catching a regression

`benchmarks/baselines/` holds what each cell did when it was last blessed, and
`benchmarks/compare.py` diffs a fresh run against it:

```bash
python benchmarks/bless.py                              # regenerate them
python benchmarks/compare.py --records benchmark_results
```

Two quantities are compared, and only two, because they are the ones that mean
something across machines: the **solve rate**, which is a property of the
formulation and should not move, and the **median calls** of the runs that
solved, which is a property of the optimizer and moves a little. Wall time is
deliberately not compared: it says more about what else the machine was doing,
and a benchmark that cries wolf gets ignored. A run with nothing in common
with the baselines fails rather than passing vacuously, so a workflow pointed
at the wrong directory says so.

`bless.py` holds the recipe, so the baselines can be regenerated rather than
remembered. They are smaller than a full sweep on purpose: a regression shows
up in a handful of starting points as readily as in twenty-five, and a
baseline that takes ninety minutes to refresh never gets refreshed.

## Why not gemseo-benchmark

The sprint plan named the gemseo-benchmark plugin. It is not used, and that is
a deliberate choice rather than an oversight. Its latest release, 4.0.1, pins
`sphinx<8`, while the documentation here needs Sphinx 9 for myst-parser. It
was installed once and removed for exactly this reason. Isolating it in its
own environment would work, at the cost of a second environment that cannot
build the docs and a heavy `gemseo[all]` dependency; the native machinery
already produces data profiles, plots, a resumable cache and this page, so it
was extended instead.
