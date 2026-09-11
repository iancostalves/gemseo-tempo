# Architecture

A problem is described in three stages, and a transcription decides what the
optimizer sees.

```text
design vars -> [ STATICS ] -> constants -> [ DYNAMICS ] -> trajectory -> [ OUTPUTS ] -> objective,
               constant-fed              StateSpace + rhs               constraints
               chain                     control params u(t)
                                              |
                                   [ TRANSCRIPTION ] decides the finite-dimensional problem:
                                     shooting:    control parameters, ODE solved by the integrator
                                     collocation: states and controls at nodes, defect constraints
                                     indirect:    initial costates, boundary-condition residuals
```

## Statics

Models whose every input is a declared constant or the output of another
such model. They run once per optimizer iteration, ahead of the dynamics, so
sizing relations and derived coefficients are not recomputed at every time
step. {class}`~gemseo_tempo.statics.chain.StaticsChain` finds them by a
fixed-point sweep, which is independent of the order the models were declared
in. See {doc}`../guides/scenario`.

## Dynamics

A {class}`~gemseo_tempo.dynamics.state_space.StateSpace` gives the state
rates and the instantaneous outputs. Two execution models sit behind one
interface: {class}`~gemseo_tempo.dynamics.integrators.DiffraxIntegrator`
integrates the ODE, and
{class}`~gemseo_tempo.dynamics.integrators.QuasiStaticIntegrator` evaluates a
purely algebraic model vmapped over the time grid, for simulation models that
carry no integrated state. See {doc}`../guides/scenario`.

## Outputs

Functionals turn a trajectory into scalars — terminal costs, running-cost
integrals, rates at the final time — and constraints turn it into vectors:
path constraints sampled along the trajectory, control-group sums, terminal
conditions and periodicity. See {doc}`../guides/scenario`.

## Transcription

The transcription is what makes the same model solvable three different ways.
It produces the design variables, one dictionary-to-dictionary JAX function
that computes every output, the objective name and the named constraints.
Grammar prefixes keep the assembled problem readable: `control.`, `state.`,
`costate.`, `constant.`, `initial.`, `trajectory.` and `constraint.`.

See {doc}`transcriptions` for choosing between them.

## Assembly

{class}`~gemseo_tempo.scenario.dynamic_scenario.DynamicScenario` turns all of
that into GEMSEO disciplines, in one of two ways:

- **monolithic** — the whole workflow is a single JAX function in one
  discipline, differentiated in a single pass;
- **split** — statics, dynamics and outputs become three disciplines chained
  by GEMSEO, which then handles coupling, caching and mixed differentiation.

Both give identical objectives and identical Jacobians; the choice is about
what GEMSEO can see and cache, not about the answer. See
{doc}`../guides/scenario`.
