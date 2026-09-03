# Assembling a scenario

{class}`~gemseo_tempo.scenario.dynamic_scenario.DynamicScenario` transcribes
the problem and builds GEMSEO disciplines from it.

## Monolithic or split

```python
DynamicScenario(..., assembly="monolithic")  # default: one discipline
DynamicScenario(..., assembly="split")       # statics, dynamics, outputs
```

Monolithic wraps the whole workflow in a single JAX function and
differentiates it in one pass. Split exposes three disciplines chained by
GEMSEO, which can then cache stages independently and mix differentiation
modes; the statics discipline runs forward-mode, being a cheap algebraic
chain with more outputs than inputs.

Both produce identical objectives and identical Jacobians, verified to 1e-10
in the test suite. Collocation and indirect shooting support monolithic only.

## Formulations

A formulation selects which design variables the optimizer sees:

- {class}`~gemseo_tempo.scenario.formulations.OptimalControl` — the
  transcription's own variables: control parameters or node values, `t_f`,
  and free initial states.
- {class}`~gemseo_tempo.scenario.formulations.SystemDesign` — named constant
  inputs, holding the control shape fixed.
- {class}`~gemseo_tempo.scenario.formulations.ControlAndDesign` — both at
  once, for co-design.

Bounds come from explicit `bounds`, then the variable registry, then
unbounded.

```python
formulation = OptimalControl(scenario, bounds={"control.u": (-3.0, 3.0)})
gemseo_scenario = formulation.create_gemseo_scenario()
gemseo_scenario.execute(algo_name="SLSQP", max_iter=200)
```

## Initial guesses

This is where dynamic problems most often go wrong, so it is worth being
explicit.

Shooting's design variables are control parameters, so a guess is just a
control law. Collocation, however, carries **node states** of its own, and
they must be consistent with the dynamics. Handing it a linear ramp between
the boundary states is the classic mistake: on the brachistochrone that
leaves `v = 0` for the whole trajectory, `dx/dt = v cos(gamma)` degenerates,
and the optimizer misses by **116% at 30 segments** while exhausting its
iteration budget.

The remedy costs one ODE solve and no optimization: simulate the dynamics
under a guessed control law, then seed the transcription with the resulting
trajectory.

```python
problem.set_initial_guess(trajectory)
```

{meth}`~gemseo_tempo.transcriptions.base.TranscribedProblem.set_initial_guess`
interpolates any {class}`~gemseo_tempo.dynamics.integrators.Trajectory` onto
the node grid. With that in place the same problem converges to the exact
cycloid time at every grid size tested, in 30 to 72 discipline calls. The
packaged {doc}`brachistochrone <../problems/brachistochrone>` does exactly
this, and it is also what makes comparing transcriptions fair, since each one
starts from the same physical trajectory.

Transcriptions that have no node grid, such as single shooting, raise
`NotImplementedError` rather than pretend to accept a guess.
