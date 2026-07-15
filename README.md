# gemseo-tempo

A [GEMSEO](https://gemseo.org) plugin for multidisciplinary optimization of
dynamic systems, built on [JAX](https://github.com/jax-ml/jax),
[diffrax](https://github.com/patrick-kidger/diffrax), and
[gemseo-jax](https://gitlab.com/gemseo/dev/gemseo-jax).

gemseo-tempo composes three stages, statics, dynamics, and outputs, behind a
transcription that decides the finite-dimensional problem the optimizer sees:

- **Direct single shooting**: control parameters as design variables, the ODE
  solved by a differentiable integrator.
- **Direct collocation** (Hermite-Simpson): states and controls at nodes,
  defect equality constraints.
- **Indirect single shooting**: initial costates as design variables, with
  costate equations derived automatically by JAX.

Both a monolithic assembly (one JAXDiscipline, one reverse-mode pass) and a
split assembly (statics, dynamics, and outputs as coupled JAXDisciplines) are
supported, along with a quasi-static vmapped execution path for
AeroMAPS-style simulation models.

## Installation

```bash
pip install gemseo-tempo
```

Optional extras: `test`, `doc`, `benchmark`, `calibration`, `uq`.

## A first problem: the brachistochrone

The brachistochrone is the classic minimum-time descent: a bead slides
without friction from one point to another under gravity, and we look for
the path angle `gamma(t)` that gets it there fastest. The states are the
position `(x, y)` (with `y` measured downward) and the speed `v`; the
control is the path angle; the final time is free.

Building the problem is three objects, a boundary spec, and a formulation:

```python
import jax.numpy as jnp
import numpy

from gemseo_tempo.dynamics import DiffraxIntegrator, PiecewiseLinearControl, StateSpace
from gemseo_tempo.outputs import Mayer
from gemseo_tempo.scenario import DynamicScenario, OptimalControl
from gemseo_tempo.transcriptions import BoundarySpec

# 1. The dynamics: ds/dt = rhs(t, states, controls, parameters), all name-keyed.
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

# 2. The control trajectory: a piecewise-linear gamma over normalized time,
#    since the final time is free (the optimizer sees the 21 node values).
tau_nodes = jnp.linspace(0.0, 1.0, 21)
gamma = PiecewiseLinearControl("gamma", tau_nodes, default_params=jnp.full(21, 0.5))

# 3. The boundary: start at rest at the origin, hit (2, 2), minimize the
#    free final time.  fix_final promotes terminal equalities; free_final_time
#    swaps in the tau-normalized dynamics and exposes t_f as a design variable.
boundary = BoundarySpec(
    s0={"x": 0.0, "y": 0.0, "v": 0.0},
    fix_final={"x": True, "y": True},
    s1={"x": 2.0, "y": 2.0},
    free_final_time=True,
    t_f=1.0,
)

# Assemble into GEMSEO disciplines (one reverse-AD JAXDiscipline by default).
scenario = DynamicScenario(
    state_space=brachistochrone,
    integrator=DiffraxIntegrator(rtol=1e-10, atol=1e-10, n_save=60),
    controls=(gamma,),
    functionals=(Mayer("time", lambda t1, s1, theta: t1),),  # objective = final time
    boundary=boundary,
    constants={"g": 9.81},
)

# The formulation picks the design variables (here the control + t_f), builds
# the DesignSpace, and wires the objective and terminal constraints.
optimal_control = OptimalControl(
    scenario, bounds={"gamma": (0.0, numpy.pi / 2), "t_f": (0.1, 5.0)}
)
gemseo_scenario = optimal_control.create_gemseo_scenario()
gemseo_scenario.execute(algo_name="SLSQP", max_iter=200)

t_f = gemseo_scenario.optimization_result.x_opt_as_dict["t_f"][0]
print(f"minimum descent time: {t_f:.5f} s")  # -> 0.82434 s
```

SLSQP recovers the cycloid optimum: the analytic minimum time to `(2, 2)`
under `g = 9.81` is `0.82434 s`, matched here to five figures. Switching to
collocation or indirect shooting will be a one-argument change
(`transcription=...`) once those land; swapping the monolithic assembly for
the split one is `assembly="split"`.

> This snippet is the hand-written UX preview. The packaged, tested
> `gemseo_tempo.problems.brachistochrone` (all three physical variants and
> all three transcriptions) arrives in Sprint 7.

## License

LGPL-3.0, see `LICENSE.txt`.
