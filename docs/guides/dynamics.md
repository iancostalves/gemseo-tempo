# The dynamics stage

## State space

A {class}`~gemseo_tempo.dynamics.state_space.StateSpace` declares names and
supplies two functions over dictionaries: `rhs` returns the rate of each
state, and `output` returns instantaneous outputs.

```python
from gemseo_tempo.dynamics import StateSpace

space = StateSpace(
    states=["x"], controls=["u"], parameters=["a"], outputs=["energy"],
    rhs=lambda t, s, u, theta: {"x": -theta["a"] * s["x"] + u["u"]},
    output=lambda t, s, u, theta: {"energy": s["x"] ** 2},
)
```

`flatten` and `unflatten` move between named dictionaries and the stacked
array the integrators use. Either function can also be overridden in a
subclass instead of passed in.

## Controls

A control is a trajectory parameterized by a finite vector, evaluated inside
the right-hand side or sampled at nodes:

- {class}`~gemseo_tempo.dynamics.controls.PiecewiseLinearControl` — linear
  between node values, constant beyond the ends.
- {class}`~gemseo_tempo.dynamics.controls.CubicControl` — a C2 spline through
  node values.
- {class}`~gemseo_tempo.dynamics.controls.ParametricControl` — a user
  function of time and parameters, for shaped signals such as a sigmoid.
- {class}`~gemseo_tempo.dynamics.controls.DelayedControl` — wraps another
  control with an order-N delay chain.

A delay chain is `order` first-order lags of time constant
`time_constant / order` each, an Erlang filter. Its stage values are genuine
dynamic states: the integrators append them to the flattened state
automatically, and the right-hand side receives the *delayed* value under the
control's own name.

```python
from gemseo_tempo.dynamics import DelayedControl, PiecewiseLinearControl

step = PiecewiseLinearControl("u", knots, default_params=jnp.ones(6))
delayed = DelayedControl(step, time_constant=1.5, order=3)
```

## Integrators

Two execution models behind one interface.

{class}`~gemseo_tempo.dynamics.integrators.DiffraxIntegrator` integrates the
ODE. Give it either a step size for fixed stepping or tolerances for adaptive
stepping:

```python
from gemseo_tempo.dynamics import DiffraxIntegrator

DiffraxIntegrator(rtol=1e-10, atol=1e-10, n_save=101)   # adaptive
DiffraxIntegrator(dt0=0.05, n_save=101)                 # fixed steps
```

The compiled program is cached per model structure, so repeated calls with
the same state space and controls trace and compile exactly once. The adjoint
choice interacts with the differentiation mode — see
{doc}`../concepts/differentiation`.

{class}`~gemseo_tempo.dynamics.integrators.QuasiStaticIntegrator` is for
models with no integrated state: constants are broadcast, controls sampled on
the grid, delay chains solved by diffrax, and the algebraic model is vmapped
over the whole time vector in one pass. It rejects a state space that
declares states, pointing at the alternative.

Both return a {class}`~gemseo_tempo.dynamics.integrators.Trajectory`, which
carries the save times and the states, controls and outputs on that grid, so
any declared quantity is available as a timeseries whatever the transcription.

## Spatial fields

{class}`~gemseo_tempo.dynamics.fields.SpatialField` interpolates a gridded 2D
or 3D field, for example a wind field. Linear interpolation goes through
`map_coordinates`; cubic is a Catmull-Rom tensor product evaluated as pure
gathers, so both are jit- and grad-safe. Queries outside the grid clamp to the
boundary.

## Smooth tables and saturations

Piecewise-linear lookup tables have kinks that stall gradient-based
optimizers.
{func}`~gemseo_tempo.dynamics.tables.smooth_table` fits a C2 spline through
table points instead, and `smooth_abs`, `smooth_max`, `smooth_min` and
`smooth_clip` provide differentiable saturations that are exact away from the
bound and round over a small band near it.

## Free final time

{func}`~gemseo_tempo.dynamics.time_scaling.normalize_time` reformulates a
state space on a normalized time in `[0, 1]`, scaling the rates by `t_f` and
appending `t_f` to the parameters. Setting `free_final_time=True` on the
boundary specification applies this automatically and exposes `t_f` as a
design variable.
