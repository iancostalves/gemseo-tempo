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

   gemseo_tempo.scenario.description
   gemseo_tempo.scenario.dynamic_scenario
   gemseo_tempo.scenario.formulations
   gemseo_tempo.scenario.structured_discipline
   gemseo_tempo.scenario.trajectories
```

## Problems

```{eval-rst}
.. autosummary::
   :toctree: generated

   gemseo_tempo.problems.albatross.model
   gemseo_tempo.problems.albatross.plots
   gemseo_tempo.problems.albatross.problem
   gemseo_tempo.problems.albatross.sequential
   gemseo_tempo.problems.albatross.statics
   gemseo_tempo.problems.brachistochrone.model
   gemseo_tempo.problems.brachistochrone.plots
   gemseo_tempo.problems.brachistochrone.problem
   gemseo_tempo.problems.brachistochrone.reference
   gemseo_tempo.problems.brachistochrone.wire
   gemseo_tempo.problems.sail.aero
   gemseo_tempo.problems.sail.model
   gemseo_tempo.problems.sail.plots
   gemseo_tempo.problems.sail.polar
   gemseo_tempo.problems.sail.problem
   gemseo_tempo.problems.sail.wind
```

## Uncertainty

```{eval-rst}
.. autosummary::
   :toctree: generated

   gemseo_tempo.uncertainty.batch
   gemseo_tempo.uncertainty.estimators
   gemseo_tempo.uncertainty.robust
   gemseo_tempo.uncertainty.sensitivity
   gemseo_tempo.uncertainty.spaces
```

## Benchmarking

```{eval-rst}
.. autosummary::
   :toctree: generated

   gemseo_tempo.benchmarking.runner
   gemseo_tempo.benchmarking.studies
   gemseo_tempo.benchmarking.timing
   gemseo_tempo.benchmarking.differentiation
   gemseo_tempo.benchmarking.transcriptions
   gemseo_tempo.benchmarking.sampling
   gemseo_tempo.benchmarking.suites
   gemseo_tempo.benchmarking.nested
   gemseo_tempo.benchmarking.plots
   gemseo_tempo.benchmarking.profiles
```
