# gemseo-tempo

A [GEMSEO](https://gemseo.org) plugin for multidisciplinary optimization of
dynamic systems, built on JAX, diffrax and gemseo-jax.

:::{admonition} Pre-release
:class: caution

gemseo-tempo has not been released: there is no package on PyPI, no tagged
version, and the API may still change without a deprecation period. Install
from source — see {doc}`getting_started/installation`, which also sets out
what is available today and what is still to come.
:::

A dynamic optimization problem is described as three stages — **statics**,
**dynamics** and **outputs** — and a **transcription** that decides what the
optimizer actually sees. Swapping the transcription is one argument; the
model does not change.

```{list-table}
:header-rows: 1
:widths: 25 75

* - Transcription
  - What the optimizer sees
* - {doc}`Direct single shooting <concepts/transcriptions>`
  - Control parameters. The trajectory comes from a differentiable ODE solve.
* - {doc}`Hermite-Simpson collocation <concepts/transcriptions>`
  - States and controls at nodes, tied together by defect equalities. No ODE solve.
* - {doc}`Indirect single shooting <concepts/transcriptions>`
  - Initial costates. The adjoint equations are derived by automatic differentiation.
```

Everything is differentiable end to end, so gradients come from automatic
differentiation rather than finite differences, in forward or reverse mode.

## A first look

```python
from gemseo_tempo.problems.brachistochrone import create_optimization

scenario, gemseo_scenario = create_optimization(size=20)
gemseo_scenario.execute(algo_name="SLSQP", max_iter=200)
print(gemseo_scenario.optimization_result.f_opt)  # 0.824339 s
```

That is the brachistochrone, solved to the exact cycloid time. The
{doc}`quickstart <getting_started/quickstart>` builds the same problem from
first principles instead of importing it.

## Where to go next

- {doc}`getting_started/installation` — install and the optional extras.
- {doc}`getting_started/quickstart` — build a problem from scratch.
- {doc}`concepts/architecture` — how the stages fit together.
- {doc}`concepts/transcriptions` — which transcription to reach for.
- {doc}`concepts/differentiation` — AD modes, adjoints and sparsity.
- {doc}`guides/scenario` — assembly, formulations and initial guesses.
- {doc}`guides/benchmarking` — comparing methods on a real problem.
- {doc}`reference/index` — the full API.

```{toctree}
:maxdepth: 2
:hidden:
:caption: Getting started

getting_started/installation
getting_started/quickstart
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Concepts

concepts/architecture
concepts/transcriptions
concepts/differentiation
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Guides

guides/metadata
guides/statics
guides/dynamics
guides/outputs
guides/scenario
guides/benchmarking
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Problems

problems/brachistochrone
problems/sail
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Reference

reference/index
```

## License

gemseo-tempo is distributed under the GNU Lesser General Public License v3.
