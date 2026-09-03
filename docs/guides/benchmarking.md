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
python examples/benchmark/run_studies.py --smoke        # one study, minutes
python examples/benchmark/run_studies.py --study all    # the full sweep
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

## Some results

A first sweep of the optimizer study reproduces the expected picture:

```{list-table}
:header-rows: 1
:widths: 40 60

* - Level
  - Starting points solved within 200 calls
* - SLSQP
  - 67%
* - NLOPT_SLSQP
  - 58%
* - COBYLA (gradient-free)
  - 8%
```

:::{note}
The optimizer study compares only constrained solvers. This problem carries
terminal equalities, and with collocation also defect equalities; L-BFGS-B,
TNC and Nelder-Mead reject it outright rather than solve it badly, so
including them would compare failures rather than methods.
:::
