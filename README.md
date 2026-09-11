# gemseo-tempo

A [GEMSEO](https://gemseo.org) plugin for multidisciplinary optimization of
dynamic systems, built on [JAX](https://github.com/jax-ml/jax),
[diffrax](https://github.com/patrick-kidger/diffrax), and
[gemseo-jax](https://gitlab.com/gemseo/dev/gemseo-jax).

gemseo-tempo composes three stages, statics, dynamics, and outputs, behind a
transcription that decides the finite-dimensional problem the optimizer sees:

- **Direct single shooting**: control parameters as design variables, the ODE
  solved by a differentiable integrator.
- **Direct collocation** (Hermite-Simpson): states and controls at nodes,
  defect equality constraints.
- **Indirect single shooting**: initial costates as design variables, with
  costate equations derived automatically by JAX.

Both a monolithic assembly (one JAXDiscipline, one reverse-mode pass) and a
split assembly (statics, dynamics, and outputs as coupled JAXDisciplines) are
supported, along with a quasi-static vmapped execution path for
AeroMAPS-style simulation models.

## Status: pre-release

gemseo-tempo is **not released yet**. There is no PyPI package, no tagged
version, and no API stability guarantee: names and signatures may change
without a deprecation period.

What that does *not* mean is that it is untested. Everything documented is
exercised by the test suite, which runs at full branch coverage, and the
numbers quoted in the docs are checked against what the code computes.

Implemented today: the three transcriptions (direct single shooting,
Hermite-Simpson collocation, indirect shooting), the monolithic and split
assemblies, the statics, dynamics and outputs stages, structure-aware
Jacobians, three problems (the brachistochrone in three variants, a sailing
leg, and albatross dynamic soaring), one-factor-at-a-time benchmark studies,
sensitivity analysis and optimization under uncertainty, and a variable
grammar that generates the documentation's tables from the registries the
code itself reads.

Still to come: multiple phases, so a trajectory can change its dynamics
part-way, and mesh refinement rather than a discretization chosen by hand.
Calibration against observed trajectories was planned and dropped.

## Installation

There is no release on PyPI, so install from source:

```bash
git clone https://github.com/iancostalves/gemseo-tempo
cd gemseo-tempo
pip install -e ".[test,doc]"
```

Or install it directly as a dependency, without a working copy:

```bash
pip install "gemseo-tempo @ git+ssh://git@github.com/iancostalves/gemseo-tempo"
```

The repository is private, so this needs your GitHub credentials. SSH is the
simplest route; over HTTPS
(`git+https://github.com/iancostalves/gemseo-tempo`) a git credential helper
will supply them. Do not put a token in the URL — it ends up in shell history
and in build logs.

Python 3.11 to 3.13.

### Extras

| extra | brings in | for |
| --- | --- | --- |
| `test` | `pytest`, `pytest-xdist`, `coverage`, `covdefaults` | running the test suite. Use `coverage run -m pytest`, not `pytest --cov`: pytest-cov's in-process coverage startup breaks jaxlib's MLIR extension import. |
| `doc` | `sphinx`, `sphinx-autobuild`, `myst-parser`, `sphinx-design`, `furo`, `matplotlib`, `sphinxcontrib-bibtex` | building the documentation: Markdown pages, figures drawn at build time, and the numbered `[N]` citations on the [references page](docs/references.md). |
| `uq` | `gemseo-umdo`, `openturns` | the uncertainty quantification and optimization-under-uncertainty machinery (`gemseo_tempo.uncertainty`): parameter spaces, Sobol/Morris sensitivity, polynomial chaos, robust design. |
| `dev` | `nox` | driving the checks (lint, tests, docs) the same way CI and `CONTRIBUTING.md` do, without assuming `nox` is already on the path. |

Combine extras as needed, e.g. `pip install -e ".[test,doc,uq]"` for
everything except the nox-driven workflow.

Versions come from git tags via setuptools-scm. With no tags yet, an install
reports something like `0.1.dev0+d20260705` — that is the expected fallback,
not a broken install.

## A first problem: the brachistochrone

The brachistochrone is the classic minimum-time descent: a bead slides
without friction from one point to another under gravity, and we look for
the path angle `gamma(t)` that gets it there fastest. The states are the
position `(x, y)` (with `y` measured downward) and the speed `v`; the
control is the path angle; the final time is free.

Building the problem is three objects, a boundary spec, and a formulation:

```python
import jax.numpy as jnp
import numpy

from gemseo_tempo.dynamics import DiffraxIntegrator, PiecewiseLinearControl, StateSpace
from gemseo_tempo.outputs import Mayer
from gemseo_tempo.scenario import DynamicScenario, OptimalControl
from gemseo_tempo.transcriptions import BoundarySpec

# 1. The dynamics: ds/dt = rhs(t, states, controls, parameters), all name-keyed.
brachistochrone = StateSpace(
    states=["x", "y", "v"],
    controls=["gamma"],
    parameters=["g"],
    rhs=lambda t, s, u, theta: {
        "x": s["v"] * jnp.cos(u["gamma"]),
        "y": s["v"] * jnp.sin(u["gamma"]),
        "v": theta["g"] * jnp.sin(u["gamma"]),
    },
)

# 2. The control trajectory: a piecewise-linear gamma over normalized time,
#    since the final time is free (the optimizer sees the 21 node values).
tau_nodes = jnp.linspace(0.0, 1.0, 21)
gamma = PiecewiseLinearControl("gamma", tau_nodes, default_params=jnp.full(21, 0.5))

# 3. The boundary: start at rest at the origin, hit (2, 2), minimize the
#    free final time.  fix_final promotes terminal equalities; free_final_time
#    swaps in the tau-normalized dynamics and exposes t_f as a design variable.
boundary = BoundarySpec(
    s0={"x": 0.0, "y": 0.0, "v": 0.0},
    fix_final={"x": True, "y": True},
    s1={"x": 2.0, "y": 2.0},
    free_final_time=True,
    t_f=1.0,
)

# Assemble into GEMSEO disciplines (one reverse-AD JAXDiscipline by default).
scenario = DynamicScenario(
    state_space=brachistochrone,
    integrator=DiffraxIntegrator(rtol=1e-10, atol=1e-10, n_save=60),
    controls=(gamma,),
    functionals=(Mayer("time", lambda t1, s1, theta: t1),),  # objective = final time
    boundary=boundary,
    constants={"g": 9.81},
)

# The formulation picks the design variables (here the control + t_f), builds
# the DesignSpace, and wires the objective and terminal constraints.
# Bounds are keyed by the grammar name of the design variable.
optimal_control = OptimalControl(
    scenario, bounds={"control.gamma": (0.0, numpy.pi / 2), "t_f": (0.1, 5.0)}
)
gemseo_scenario = optimal_control.create_gemseo_scenario()
gemseo_scenario.execute(algo_name="SLSQP", max_iter=200)

t_f = gemseo_scenario.optimization_result.x_opt_as_dict["t_f"][0]
print(f"minimum descent time: {t_f:.5f} s")  # -> 0.82434 s
```

SLSQP recovers the cycloid optimum: the analytic minimum time to `(2, 2)`
under `g = 9.81` is `0.82434 s`, matched here to five figures. Switching to
collocation or indirect shooting is a one-argument change
(`transcription=HermiteSimpson(num_segments=20)`); swapping the monolithic
assembly for the split one is `assembly="split"`.

The same problem is packaged, in three physical variants, as
`gemseo_tempo.problems.brachistochrone`:

```python
from gemseo_tempo.problems.brachistochrone import create_optimization

scenario, gemseo_scenario = create_optimization(variant="drag", size=30)
gemseo_scenario.execute(algo_name="SLSQP", max_iter=200)
```

## Documentation

Build it locally with `nox -s docs-serve`, which serves a live-reloading
preview on <http://127.0.0.1:8000>, or `nox -s docs` for a one-shot build
that treats warnings as errors. `nox` comes with the `dev` extra; the build
itself needs the `doc` extra.

Inside an environment that already has the documentation dependencies, the
same two builds run directly and rather faster, since nox otherwise makes a
fresh environment for each:

```bash
sphinx-autobuild docs docs/_build/html --ignore docs/reference/generated
sphinx-build -b html -W --keep-going docs docs/_build/html
```

`sphinx-autobuild` opens a browser tab on its own; the one-shot
`sphinx-build` does not serve anything, so open `docs/_build/html/index.html`
in a browser afterwards to view it.

Every figure on a problem page is drawn at build time from code the reader
could run themselves — except one kind: anything that needs an optimizer
(a converged trajectory, a benchmark sweep) is solved once, ahead of time,
by `tools/generate_doc_results.py`, and stored under `docs/_data/`, which
is committed so an ordinary build never re-solves anything. Regenerate it
after a change that would move those numbers:

```bash
python tools/generate_doc_results.py            # everything, ~1 h
python tools/generate_doc_results.py --only sail  # one problem
python tools/generate_doc_results.py --smoke     # tiny, for the test suite
```

`tests/test_doc_data.py` guards against the stored numbers drifting away
from what the code would now produce, without re-running the optimizations
themselves on every test run.

## License

LGPL-3.0, see `LICENSE.txt`.
