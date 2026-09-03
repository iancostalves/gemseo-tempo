# Quickstart: the brachistochrone

The brachistochrone is the classic minimum-time descent. A bead slides
without friction from the origin to a target point under gravity, and the
question is which path angle `gamma(t)` gets it there fastest. The states are
the position `(x, y)`, with `y` measured downward, and the speed `v`.

Building the problem takes four objects.

## 1. The dynamics

A {class}`~gemseo_tempo.dynamics.state_space.StateSpace` is a right-hand side
over dictionaries keyed by variable name. Nothing is positional, so the
equations read like the equations.

```python
import jax.numpy as jnp

from gemseo_tempo.dynamics import StateSpace

brachistochrone = StateSpace(
    states=["x", "y", "v"],
    controls=["gamma"],
    parameters=["g"],
    rhs=lambda t, s, u, theta: {
        "x": s["v"] * jnp.cos(u["gamma"]),
        "y": s["v"] * jnp.sin(u["gamma"]),
        "v": theta["g"] * jnp.sin(u["gamma"]),
    },
)
```

## 2. The control

The final time is free, so the dynamics are integrated over a normalized
time and the control is parameterized on `[0, 1]`. The optimizer will see the
21 node values.

```python
from gemseo_tempo.dynamics import PiecewiseLinearControl

tau = jnp.linspace(0.0, 1.0, 21)
gamma = PiecewiseLinearControl("gamma", tau, default_params=jnp.full(21, 0.5))
```

## 3. The boundary conditions

`fix_final` turns a final state into a terminal equality, and
`free_final_time` promotes `t_f` to a design variable and swaps in the
time-normalized dynamics.

```python
from gemseo_tempo.transcriptions import BoundarySpec

boundary = BoundarySpec(
    s0={"x": 0.0, "y": 0.0, "v": 0.0},
    fix_final={"x": True, "y": True},
    s1={"x": 2.0, "y": 2.0},
    free_final_time=True,
    t_f=1.0,
)
```

## 4. Assemble and solve

The objective is the final time, expressed as a Mayer term. The scenario
becomes GEMSEO disciplines — one reverse-differentiated discipline by
default — and the formulation builds the design space and wires the
objective and constraints.

```python
import numpy

from gemseo_tempo.dynamics import DiffraxIntegrator
from gemseo_tempo.outputs import Mayer
from gemseo_tempo.scenario import DynamicScenario, OptimalControl

scenario = DynamicScenario(
    state_space=brachistochrone,
    integrator=DiffraxIntegrator(rtol=1e-10, atol=1e-10, n_save=60),
    controls=(gamma,),
    functionals=(Mayer("time", lambda t1, s1, theta: t1),),
    boundary=boundary,
    constants={"g": 9.81},
)

optimal_control = OptimalControl(
    scenario, bounds={"control.gamma": (0.0, numpy.pi / 2), "t_f": (0.1, 5.0)}
)
gemseo_scenario = optimal_control.create_gemseo_scenario()
gemseo_scenario.execute(algo_name="SLSQP", max_iter=200)

t_f = gemseo_scenario.optimization_result.x_opt_as_dict["t_f"][0]
print(f"minimum descent time: {t_f:.5f} s")
```

This prints `0.82434 s`, which is the exact cycloid time for this target,
available as
{func}`~gemseo_tempo.problems.brachistochrone.reference.cycloid_time`.

## Changing the transcription

The model, the boundary conditions and the objective are unchanged; only the
transcription argument differs.

```python
from gemseo_tempo.transcriptions import HermiteSimpson

scenario = DynamicScenario(
    ...,
    transcription=HermiteSimpson(num_segments=20),
)
```

:::{warning}
Collocation carries node states of its own, so it needs an initial guess that
satisfies the dynamics. A linear ramp between the boundary states leaves
`v = 0` everywhere, which makes `dx/dt = v cos(gamma)` degenerate: cold
started that way, collocation misses this problem by **116% at 30 segments**.
Seed it from a simulated trajectory instead, as
{doc}`the packaged problem <../problems/brachistochrone>` does, and the error
drops to zero to five decimal places. See
{doc}`../guides/scenario` for the mechanism.
:::

## The packaged version

Everything above is available ready-made, in three physical variants:

```python
from gemseo_tempo.problems.brachistochrone import create_optimization

scenario, gemseo_scenario = create_optimization(variant="drag", size=30)
gemseo_scenario.execute(algo_name="SLSQP", max_iter=200)
```
