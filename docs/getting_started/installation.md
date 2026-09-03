# Installation

:::{admonition} This is a pre-release
:class: caution

gemseo-tempo has not been released. There is **no package on PyPI**, no
tagged version, and no API stability guarantee — names and signatures may
change without a deprecation period. `pip install gemseo-tempo` will not
work; install from source as shown below.
:::

## What "pre-release" means here

Unreleased does not mean unfinished or unverified. Every documented behaviour
is exercised by the test suite, which runs at full branch coverage across
Python 3.11 to 3.13, the documentation builds with warnings treated as
errors, and the numbers quoted in these pages are checked against what the
code actually computes.

What you should expect to change is the surface: argument names, defaults and
module layout are still settling, and there is no deprecation cycle to soften
that. Pin a commit if you build on it.

```{list-table}
:header-rows: 1
:widths: 50 50

* - Available today
  - Still to come
* - Direct single shooting, Hermite-Simpson collocation, indirect shooting
  - The sail and albatross example problems
* - Monolithic and split assembly
  - Calibration against observed trajectories
* - Statics, dynamics and outputs stages
  - Sensitivity analysis and optimization under uncertainty
* - Structure-aware Jacobians for collocation defects
  - The documentation's variable grammar, generating tables from registries
* - The brachistochrone in three variants, and the benchmark studies
  -
```

## Installing from source

```bash
git clone https://github.com/iancostalves/gemseo-tempo
cd gemseo-tempo
pip install -e ".[test,doc]"
pre-commit install
```

An editable install is the sensible choice while the project moves: pulling
new commits updates the package with no reinstall.

Python 3.11 to 3.13 are supported. The core install pulls in `gemseo-jax`,
`diffrax`, `equinox` and `pint`, and enables double precision in JAX on
import, because dynamic systems accumulate error over long horizons and
single precision degrades gradient checks.

## Installing without a working copy

To depend on gemseo-tempo from another project, install it straight from the
repository:

```bash
pip install "gemseo-tempo @ git+ssh://git@github.com/iancostalves/gemseo-tempo"
```

Pin a commit while the API is unsettled, by appending `@<sha>` to the URL.

The repository is private, so this needs your GitHub credentials. SSH is the
simplest route. Over HTTPS —
`git+https://github.com/iancostalves/gemseo-tempo` — a git credential helper
supplies them instead.

:::{warning}
Do not embed a personal access token in the URL. It leaks into shell
history, into `pip freeze` output and into CI logs. Use SSH, or a credential
helper that keeps the token out of the URL.
:::

## Version numbers

Versions are derived from git tags by setuptools-scm. Since nothing is tagged
yet, an install reports a development version such as
`0.1.dev0+d20260705`. That is the expected fallback, not a sign of a broken
install; `gemseo_tempo.__version__` will report real version numbers once
releases are tagged.

## Extras

```{list-table}
:header-rows: 1
:widths: 20 80

* - Extra
  - Contents
* - `test`
  - `pytest`, `coverage`, `covdefaults`, `pytest-xdist`.
* - `doc`
  - Sphinx, `myst-parser`, `sphinx-design`, `furo`, `sphinx-autobuild`.
* - `calibration`
  - `gemseo-calibration`, for identifying model constants from data.
* - `uq`
  - `gemseo-umdo` and `openturns`, for sensitivity and robust design.
```

Install several at once with `pip install -e ".[test,doc]"`.

:::{note}
Coverage is collected with `coverage run -m pytest` rather than
`pytest --cov`. Starting coverage in-process breaks the jaxlib MLIR
extension import on some platforms, so the plugin form is avoided.
:::

## Checks

`nox` drives everything: `nox -s tests` across Python 3.11 to 3.13,
`nox -s lint`, `nox -s docs` for a warnings-as-errors build, and
`nox -s docs-serve` for a live-reloading preview on
<http://127.0.0.1:8000>.
