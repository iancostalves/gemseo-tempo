# Brachistochrone

```{variable-context} gemseo_tempo.problems.brachistochrone.model:REGISTRY
```

Minimum-time descent of a bead under gravity, in three physical variants.
With {var}`y <symbol>` measured downward:

```text
dx/dt = v cos(gamma)
dy/dt = v sin(gamma)
dv/dt = g (sin(gamma) - mu cos(gamma)) - (k/m) v^2
```

Three states and one control, generated from the registry:

```{variables} gemseo_tempo.problems.brachistochrone.model:REGISTRY
:filter: tags=state,control
:columns: name, full_name, unit, bounds
```

The variants differ only in their constants, and those are where the
uncertainty of this problem lives:

```{variables} gemseo_tempo.problems.brachistochrone.model:REGISTRY
:filter: tags=constant
:columns: name, full_name, unit, distribution
```

Setting {var}`mu <symbol>` and `k/m` to zero gives the ideal frictionless
case, a nonzero {var}`mu <symbol>` adds Coulomb friction, and a nonzero
{var}`k_over_m <symbol>` adds quadratic drag.

```python
from gemseo_tempo.problems.brachistochrone import create_optimization

scenario, gemseo_scenario = create_optimization(variant="ideal", size=20)
gemseo_scenario.execute(algo_name="SLSQP", max_iter=200)
```

## References

The ideal variant has a closed form. The optimal path is the cycloid
`x = R(theta - sin theta)`, `y = R(1 - cos theta)`, traversed in
`t = theta sqrt(R/g)`; requiring it to pass through the target fixes
`theta_f`, and `R` follows from the depth. For the default target `(2, 2)`
with `g = 9.81`, that gives **0.824339 s**, available as
{func}`~gemseo_tempo.problems.brachistochrone.reference.cycloid_time`.

The friction and drag variants have no comparably simple closed form, so
their references are high-accuracy numerical values computed at 200 segments
and cross-checked between collocation and shooting, which agree to 4.2e-4 s
and 2.8e-6 s respectively. Regenerate them with
`tools/generate_brachistochrone_references.py`.

## The initial guess

The builders do not hand the transcription a linear ramp. A ramp leaves the
speed at zero throughout, `dx/dt = v cos(gamma)` degenerates, and cold-started
collocation misses the optimum by over a hundred percent. Instead the guess
parameters — the initial path angle, its increase, and the descent time — are
turned into a trajectory by integrating the dynamics once, and that seeds the
transcription.

```{list-table}
:header-rows: 1
:widths: 16 42 42

* - Segments
  - Cold start, linear ramp
  - Seeded from a simulated trajectory
* - 30
  - 116.3% error, budget exhausted
  - **0.000%**, 50 calls
* - 100
  - 13.2% error, budget exhausted
  - **0.000%**, 72 calls
```

Because every transcription is seeded from the same trajectory, comparisons
between them are fair.

## Indirect shooting

Stationarity of the Hamiltonian in `gamma` has a closed form,

```text
tan(gamma) = (lam_y v + lam_v g) / (lam_x v - lam_v g mu)
```

supplied as
{func}`~gemseo_tempo.problems.brachistochrone.problem.optimal_path_angle`,
so no inner Newton solve is needed:

```python
from gemseo_tempo.problems.brachistochrone import optimal_path_angle
from gemseo_tempo.transcriptions import IndirectShooting

create_optimization(transcription=IndirectShooting(u_star=optimal_path_angle))
```
