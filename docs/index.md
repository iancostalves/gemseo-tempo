# gemseo-tempo

A GEMSEO plugin for multidisciplinary optimization of dynamic systems, built
on JAX, diffrax, and gemseo-jax.

gemseo-tempo composes three stages, statics, dynamics, and outputs, behind a
transcription that decides the finite-dimensional problem the optimizer sees:
direct single shooting, Hermite-Simpson direct collocation, or indirect
single shooting with AD-derived costate equations.

## Installation

```bash
pip install gemseo-tempo
```

Optional extras: `test`, `doc`, `benchmark`, `calibration`, `uq`.

## License

gemseo-tempo is distributed under the GNU Lesser General Public License v3.
