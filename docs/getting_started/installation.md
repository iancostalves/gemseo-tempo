# Installation

```bash
pip install gemseo-tempo
```

Python 3.11 to 3.13 are supported. The core install pulls in `gemseo-jax`,
`diffrax`, `equinox` and `pint`, and enables double precision in JAX on
import, because dynamic systems accumulate error over long horizons and
single precision degrades gradient checks.

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

Install several at once with `pip install "gemseo-tempo[test,doc]"`.

:::{note}
Coverage is collected with `coverage run -m pytest` rather than
`pytest --cov`. Starting coverage in-process breaks the jaxlib MLIR
extension import on some platforms, so the plugin form is avoided.
:::

## From a clone

```bash
git clone https://github.com/iancostalves/gemseo-tempo
cd gemseo-tempo
pip install -e ".[test,doc]"
pre-commit install
```

`nox` drives the checks: `nox -s tests` across Python 3.11 to 3.13,
`nox -s lint`, `nox -s docs` for a warnings-as-errors build, and
`nox -s docs-serve` for a live-reloading preview on
<http://127.0.0.1:8000>.
