# Building a scenario

A dynamic optimization problem is assembled from four pieces, in this order,
and this guide walks through them the way a problem is actually built:
declare the variables, define the constants stage, define the dynamics
stage, declare what the optimizer optimizes, then assemble and solve. See
{doc}`../concepts/architecture` for how the pieces fit together and
{doc}`../concepts/transcriptions` for what a transcription does with them.

## Declaring variables

Every variable — a state, a control, a constant, an output — carries
metadata: a grammar name, a readable name, a description, a unit, bounds,
tags and, for the uncertainty layer, a probability distribution.

```python
from gemseo_tempo.metadata import Variable, VariableRegistry

registry = VariableRegistry([
    Variable(
        name="v",
        full_name="Speed",
        description="Speed of the bead along its path.",
        unit="m/s",
        lower_bound=0.0,
        tags=frozenset({"state"}),
    ),
])
```

A registry validates as it goes: registering a duplicate name raises unless
`overwrite=True`, and an unparsable unit raises at registration rather than
somewhere deep inside a scenario. Pass one to a
{class}`~gemseo_tempo.statics.model.Model` or a
{class}`~gemseo_tempo.scenario.dynamic_scenario.DynamicScenario` and the
names are checked against the discipline grammar. Registries subset, filter
by tag, and round-trip through YAML including the distribution field:

```python
states = registry.filter("state")
registry.to_yaml("variables.yaml")
VariableRegistry.from_yaml("variables.yaml")
```

Formulation builders read bounds from the registry when no explicit bounds
are given, so declaring a variable once puts it in the design space
correctly.

Inside jitted code every value is in canonical base units; conversion
happens only at discipline boundaries, never inside traced code, because a
pint quantity must not meet a JAX tracer.

```python
from gemseo_tempo.metadata import convert, make_unit_shell, to_base

to_base(3.6, "km/h")            # 1.0 m/s, cached and precomputed
convert(212.0, "degF", "degC")  # 100.0, conversions are affine
```

{func}`~gemseo_tempo.metadata.units.make_unit_shell` wraps a
dictionary-to-dictionary function so it accepts and returns values in
declared units while computing in base units; keys absent from the mappings
pass through untouched.

## The statics stage

A statics model is one whose every input is a declared constant or the
output of another statics model. Such models can run once per optimizer
iteration, ahead of the dynamics, instead of at every time step — the wing
sizing chain in {doc}`../problems/albatross/index` is the motivating case.

```python
from gemseo_tempo.statics import AutoModel, JAXModel, StaticsChain

class Wing(AutoModel):
    def _jax_func(self, span=30.0, area=120.0):
        aspect_ratio = span**2 / area
        return aspect_ratio
```

{class}`~gemseo_tempo.statics.model.AutoModel` detects inputs and outputs
from the function signature and the returned names, which removes the
grammar boilerplate. {class}`~gemseo_tempo.statics.model.JAXModel` takes an
explicit dictionary function with declared names, for closures and partial
applications whose signature cannot be introspected.

{meth}`~gemseo_tempo.statics.chain.StaticsChain.split` partitions models
into the statics side and the rest by a **fixed-point** sweep: a model
joins the statics side once all of its inputs are constants or outputs of
models already there, and the sweep repeats until nothing moves — the same
answer for every declaration order, not just the one written down.

```python
statics, dynamics_side = StaticsChain.split(models, constant_inputs=["k"])
chain = StaticsChain(statics, ["k"])
outputs = chain({"k": 0.5})
```

The chain delegates to the underlying `JAXChain` function, so it composes
into a larger jitted program and differentiates end to end. Missing inputs
fall back to the discipline defaults, so it can be called with only the
inputs of interest.

## The dynamics stage

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

**Controls** are a trajectory parameterized by a finite vector, evaluated
inside the right-hand side or sampled at nodes:

- {class}`~gemseo_tempo.dynamics.controls.PiecewiseLinearControl` — linear
  between node values, constant beyond the ends.
- {class}`~gemseo_tempo.dynamics.controls.CubicControl` — a C2 spline
  through node values.
- {class}`~gemseo_tempo.dynamics.controls.ParametricControl` — a user
  function of time and parameters, for shaped signals such as a sigmoid.
- {class}`~gemseo_tempo.dynamics.controls.DelayedControl` — wraps another
  control with an order-N delay chain, `order` first-order lags of time
  constant `time_constant / order` each. Its stage values are genuine
  dynamic states: the integrators append them to the flattened state
  automatically, and the right-hand side receives the *delayed* value
  under the control's own name.

```python
from gemseo_tempo.dynamics import DelayedControl, PiecewiseLinearControl

step = PiecewiseLinearControl("u", knots, default_params=jnp.ones(6))
delayed = DelayedControl(step, time_constant=1.5, order=3)
```

**Integrators** are two execution models behind one interface.
{class}`~gemseo_tempo.dynamics.integrators.DiffraxIntegrator` integrates
the ODE, given either a step size for fixed stepping or tolerances for
adaptive stepping:

```python
from gemseo_tempo.dynamics import DiffraxIntegrator

DiffraxIntegrator(rtol=1e-10, atol=1e-10, n_save=101)   # adaptive
DiffraxIntegrator(dt0=0.05, n_save=101)                 # fixed steps
```

The compiled program is cached per model structure, so repeated calls with
the same state space and controls trace and compile exactly once. The
adjoint choice interacts with the differentiation mode — see
{doc}`../concepts/differentiation`.
{class}`~gemseo_tempo.dynamics.integrators.QuasiStaticIntegrator` is for
models with no integrated state: constants are broadcast, controls sampled
on the grid, delay chains solved by diffrax, and the algebraic model is
vmapped over the whole time vector in one pass; it rejects a state space
that declares states, pointing at the alternative. Both return a
{class}`~gemseo_tempo.dynamics.integrators.Trajectory`, carrying the save
times and the states, controls and outputs on that grid, so any declared
quantity is available as a timeseries whatever the transcription.

Two more pieces round out a state space. A
{class}`~gemseo_tempo.dynamics.fields.SpatialField` interpolates a gridded
2D or 3D field, such as a wind field: linear interpolation goes through
`map_coordinates`, cubic is a Catmull-Rom tensor product evaluated as pure
gathers, and both are jit- and grad-safe, clamping to the boundary outside
the grid. Piecewise-linear lookup tables have kinks that stall
gradient-based optimizers, so
{func}`~gemseo_tempo.dynamics.tables.smooth_table` fits a C2 spline through
table points instead, and `smooth_abs`, `smooth_max`, `smooth_min` and
`smooth_clip` provide differentiable saturations, exact away from the bound
and rounded over a small band near it — {doc}`../problems/sail/index`'s polar and
{doc}`../problems/albatross/index`'s wind profile both depend on this.

Finally, {func}`~gemseo_tempo.dynamics.time_scaling.normalize_time`
reformulates a state space on a normalized time in `[0, 1]`, scaling the
rates by `t_f` and appending `t_f` to the parameters. Setting
`free_final_time=True` on the boundary specification applies this
automatically and exposes `t_f` as a design variable.

## Functionals and constraints

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
the difference is deliberate. Under ODE integration they become an
**augmented quadrature state**, exact under adaptive stepping with no
resampling error. Under collocation they are evaluated by **Simpson
quadrature on the same grid** as the defects, since there is no solver to
augment; under quasi-static execution they fall back to trapezoid on the
save grid. Both `Lagrange` and `TimeIntegral` accept a discount rate `rho`,
multiplying the integrand by `exp(-rho (t - t0))`.

The first functional is the objective, with one exception: under indirect
shooting the functionals shape the Hamiltonian while the optimizer
minimizes the residual norm, and the achieved cost is reported separately
as `ocp_cost`.

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

All constraints follow the GEMSEO convention: equalities are `g = 0` and
inequalities are `g <= 0`. A path constraint is vector-valued by default,
one entry per evaluation point (collocation evaluates it at nodes *and*
midpoints); with `aggregate="ks"` the values collapse to a single
Kreisselmeier-Steinhauser scalar, a smooth upper bound of the maximum that
sharpens as `ks_rho` grows. Terminal conditions on a state are usually
better expressed through `BoundarySpec.fix_final`, which lets collocation
compress the fixed node out of the design space entirely rather than carry
it as a variable tied down by an equality.

:::{note}
Indirect shooting accepts none of these: state path constraints change the
necessary conditions. Declare terminal conditions through `BoundarySpec`
instead.
:::

## Assembly and formulations

{class}`~gemseo_tempo.scenario.dynamic_scenario.DynamicScenario`
transcribes the problem and builds GEMSEO disciplines from it.

```python
DynamicScenario(..., assembly="monolithic")  # default: one discipline
DynamicScenario(..., assembly="split")       # statics, dynamics, outputs
```

Monolithic wraps the whole workflow in a single JAX function and
differentiates it in one pass. Split exposes three disciplines chained by
GEMSEO, which can then cache stages independently and mix differentiation
modes; the statics discipline runs forward-mode, being a cheap algebraic
chain with more outputs than inputs. Both produce identical objectives and
identical Jacobians, verified to 1e-10 in the test suite. Collocation and
indirect shooting support monolithic only.

A formulation then selects which design variables the optimizer sees:

- {class}`~gemseo_tempo.scenario.formulations.OptimalControl` — the
  transcription's own variables: control parameters or node values, `t_f`,
  and free initial states.
- {class}`~gemseo_tempo.scenario.formulations.SystemDesign` — named
  constant inputs, holding the control shape fixed.
- {class}`~gemseo_tempo.scenario.formulations.ControlAndDesign` — both at
  once, for co-design.

Bounds come from explicit `bounds`, then the variable registry, then
unbounded.

```python
formulation = OptimalControl(scenario, bounds={"control.u": (-3.0, 3.0)})
gemseo_scenario = formulation.create_gemseo_scenario()
gemseo_scenario.execute(algo_name="SLSQP", max_iter=200)
```

A formulation can also carry `disciplines`, **replacing** the scenario's
own rather than extending them — this is how
{doc}`../guides/uncertainty` wires a probability into the optimizer's own
constraints, and the reason it must replace rather than chain is explained
there.

## Initial guesses

This is where dynamic problems most often go wrong, so it is worth being
explicit.

Shooting's design variables are control parameters, so a guess is just a
control law. Collocation, however, carries **node states** of its own, and
they must be consistent with the dynamics. Handing it a linear ramp between
the boundary states is the classic mistake: on the brachistochrone that
leaves `v = 0` for the whole trajectory, `dx/dt = v cos(gamma)`
degenerates, and the optimizer misses by **116% at 30 segments** while
exhausting its iteration budget.

The remedy costs one ODE solve and no optimization: simulate the dynamics
under a guessed control law, then seed the transcription with the
resulting trajectory.

```python
problem.set_initial_guess(trajectory)
```

{meth}`~gemseo_tempo.transcriptions.base.TranscribedProblem.set_initial_guess`
interpolates any {class}`~gemseo_tempo.dynamics.integrators.Trajectory`
onto the node grid. With that in place the same problem converges to the
exact cycloid time at every grid size tested, in 30 to 72 discipline
calls. The packaged {doc}`brachistochrone <../problems/brachistochrone/index>`
does exactly this, and it is also what makes comparing transcriptions
fair, since each one starts from the same physical trajectory. It matters
beyond that one problem, too: the sail's channel-constrained legs have
several local optima, and which one a solve lands in depends on where it
started, not on the width of the channel — see {doc}`../problems/sail/index`.

Transcriptions that have no node grid, such as single shooting, raise
`NotImplementedError` rather than pretend to accept a guess.

## The documentation reads the same record

The registry declared at the top of this guide is not only for design
spaces and unit conversion. The documentation reads it too, through a
small Sphinx extension (`docs/_ext/tempo_myst.py`), so a page cannot
quietly disagree with the model it describes. A hand-written variable
table is a copy, and a copy of a number is a number that will be wrong
later.

Four pieces of grammar. A page first says which model it is about:

````markdown
```{variable-context} gemseo_tempo.problems.albatross.model:REGISTRY
```
````

Then a mention is a role rather than backticks, and renders from the
record:

| written | rendered |
| --- | --- |
| `` {var}`V` `` | Airspeed (V, m/s) |
| `` {var}`V <name>` `` | Airspeed |
| `` {var}`V <symbol>` `` | V |
| `` {var}`V <unit>` `` | m/s |
| `` {var}`brachistochrone:mu` `` | Coulomb friction coefficient (mu, –) |

Each one links to the variable's entry in the {doc}`generated glossary
<../reference/variables>`, and carries its description as hover text. A
table is generated rather than typed:

````markdown
```{variables} gemseo_tempo.problems.albatross.model:REGISTRY
:filter: tags=state
:columns: name, full_name, unit, bounds
```
````

`:filter:` keeps the variables carrying any of the named tags, `:names:`
selects them explicitly and in order, and `:columns:` chooses from `name`,
`full_name`, `description`, `unit`, `bounds`, `lower_bound`, `upper_bound`,
`tags` and `distribution`.

**A name that no registry declares fails the build.** That is the point
of the arrangement: the documentation is checked against the code rather
than merely written alongside it. Set `tempo_unknown_variable = "warn"` in
`conf.py` to downgrade it while a model is in flux.

Writing the grammar made a clash visible that prose had been hiding: the
same grammar name means different things in different problems.

| name | brachistochrone | sail | albatross |
| --- | --- | --- | --- |
| `x` | horizontal position | eastward position | downwind position |
| `y` | depth | northward position | crosswind position |
| `mu` | Coulomb friction coefficient | | bank angle |
| `t_f` | descent time | time to the mark | |

A single global glossary would have had to pick one of each and be wrong
about the rest. So an entry belongs to a model, `{variable-context}` says
which model a page is about, and a name that is unique across all of them
still resolves without being told. Every registry of one problem shares a
label and they are merged, because a page about the albatross should
resolve the name of a wing variable as readily as the name of a state; the
extension checks that the merged registries agree wherever they overlap.
