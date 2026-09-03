# API reference

Every module that defines public API. `tests/test_docs.py` checks this list
against the installed package, so a module added without an entry here fails
the test suite.

The subpackages themselves (`gemseo_tempo.dynamics` and friends) are
re-export facades: importing from them is the convenient path, but each
object is documented once, under the module that defines it.

## Metadata

```{eval-rst}
.. autosummary::
   :toctree: generated

   gemseo_tempo.metadata.variable
   gemseo_tempo.metadata.registry
   gemseo_tempo.metadata.units
```

## Statics

```{eval-rst}
.. autosummary::
   :toctree: generated

   gemseo_tempo.statics.model
   gemseo_tempo.statics.chain
```

## Dynamics

```{eval-rst}
.. autosummary::
   :toctree: generated

   gemseo_tempo.dynamics.state_space
   gemseo_tempo.dynamics.controls
   gemseo_tempo.dynamics.integrators
   gemseo_tempo.dynamics.fields
   gemseo_tempo.dynamics.tables
   gemseo_tempo.dynamics.time_scaling
   gemseo_tempo.dynamics.interpolation
```

## Outputs

```{eval-rst}
.. autosummary::
   :toctree: generated

   gemseo_tempo.outputs.functionals
   gemseo_tempo.outputs.constraints
```

## Transcriptions

```{eval-rst}
.. autosummary::
   :toctree: generated

   gemseo_tempo.transcriptions.base
   gemseo_tempo.transcriptions.single_shooting
   gemseo_tempo.transcriptions.collocation
   gemseo_tempo.transcriptions.indirect
```

## Scenario

```{eval-rst}
.. autosummary::
   :toctree: generated

   gemseo_tempo.scenario.dynamic_scenario
   gemseo_tempo.scenario.formulations
   gemseo_tempo.scenario.structured_discipline
```

## Problems

```{eval-rst}
.. autosummary::
   :toctree: generated

   gemseo_tempo.problems.brachistochrone.model
   gemseo_tempo.problems.brachistochrone.problem
   gemseo_tempo.problems.brachistochrone.reference
   gemseo_tempo.problems.sail.model
   gemseo_tempo.problems.sail.polar
   gemseo_tempo.problems.sail.problem
   gemseo_tempo.problems.sail.wind
```

## Benchmarking

```{eval-rst}
.. autosummary::
   :toctree: generated

   gemseo_tempo.benchmarking.runner
   gemseo_tempo.benchmarking.studies
   gemseo_tempo.benchmarking.sampling
   gemseo_tempo.benchmarking.plots
   gemseo_tempo.benchmarking.profiles
```
