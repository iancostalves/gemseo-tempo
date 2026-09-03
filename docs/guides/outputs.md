# Functionals and constraints

## Functionals

```{list-table}
:header-rows: 1
:widths: 24 76

* - Functional
  - Meaning
* - {class}`~gemseo_tempo.outputs.functionals.Mayer`
  - Terminal cost `phi(t1, s(t1), theta)`.
* - {class}`~gemseo_tempo.outputs.functionals.Lagrange`
  - Running cost, the integral of `l(t, s, u, theta)`.
* - {class}`~gemseo_tempo.outputs.functionals.TimeIntegral`
  - The time integral of a declared state-space output.
* - {class}`~gemseo_tempo.outputs.functionals.FinalRate`
  - Rate of change of a quantity at the final time.
```

Integral terms are handled differently depending on the transcription, and
the difference is deliberate. Under ODE integration they become an **augmented
quadrature state**, so the value is exact under adaptive stepping with no
resampling error. Under collocation they are evaluated by **Simpson
quadrature on the same grid** as the defects, since there is no solver to
augment. Under quasi-static execution they fall back to trapezoid on the save
grid.

Both `Lagrange` and `TimeIntegral` accept a discount rate `rho`, which
multiplies the integrand by `exp(-rho (t - t0))`.

The first functional is the objective — with one exception. Under indirect
shooting the functionals define the problem that shapes the Hamiltonian,
while the optimizer minimizes the residual norm; the achieved cost is
reported separately as `ocp_cost`.

## Constraints

All constraints follow the GEMSEO convention: equalities are `g = 0` and
inequalities are `g <= 0`.

```{list-table}
:header-rows: 1
:widths: 24 76

* - Constraint
  - Meaning
* - {class}`~gemseo_tempo.outputs.constraints.PathConstraint`
  - `g(t, s, u, theta) <= 0` sampled along the trajectory.
* - {class}`~gemseo_tempo.outputs.constraints.ControlGroupConstraint`
  - The sum of a group of controls stays below one at every node.
* - {class}`~gemseo_tempo.outputs.constraints.TerminalConstraint`
  - A condition on the final state, equality or inequality.
* - {class}`~gemseo_tempo.outputs.constraints.PeriodicityConstraint`
  - `s(t1) - s(t0) = 0` on selected states, for closing a cycle.
```

A path constraint is vector-valued by default, one entry per evaluation
point; collocation evaluates it at nodes *and* midpoints. With
`aggregate="ks"` the values collapse to a single Kreisselmeier-Steinhauser
scalar, a smooth upper bound of the maximum that sharpens as `ks_rho` grows.

Terminal conditions on a state are usually better expressed through
`BoundarySpec.fix_final`, which lets collocation compress the fixed node out
of the design space entirely rather than carry it as a variable tied down by
an equality.

:::{note}
Indirect shooting accepts none of these: state path constraints change the
necessary conditions. Declare terminal conditions through `BoundarySpec`
instead.
:::
