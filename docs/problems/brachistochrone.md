# Brachistochrone

```{variable-context} gemseo_tempo.problems.brachistochrone.model:REGISTRY
```

Minimum-time descent of a bead under gravity, in three physical variants. It
is the smallest problem in the library and the one this page uses as a
guided tour: every capability shown here reappears, at scale, on the sail
and albatross pages.

```python
from gemseo_tempo.problems.brachistochrone import create_optimization

scenario, gemseo_scenario = create_optimization(variant="ideal", size=20)
gemseo_scenario.execute(algo_name="SLSQP", max_iter=200)
```

## Physics

### Kinematics

```{eval-rst}
.. plot:: _ext/gallery/brachistochrone_kinematics.py
   :include-source: false
```

With {var}`y <symbol>` measured downward, a bead sliding on a wire has a
position, a speed, and the angle its path makes with the horizontal:

```{variables} gemseo_tempo.problems.brachistochrone.model:REGISTRY
:filter: tags=state,control
:columns: name, full_name, unit, bounds
```

### Dynamics

```text
dx/dt = v cos(gamma)
dy/dt = v sin(gamma)
dv/dt = g (sin(gamma) - mu cos(gamma)) - (k/m) v^2
```

The path angle {var}`gamma <symbol>` is the control: steer it, and gravity
does the rest. The variants differ only in their constants, and those are
where the uncertainty of this problem lives:

```{variables} gemseo_tempo.problems.brachistochrone.model:REGISTRY
:filter: tags=constant
:columns: name, full_name, unit, distribution
```

Setting {var}`mu <symbol>` and `k/m` to zero gives the ideal frictionless
case, a nonzero {var}`mu <symbol>` adds Coulomb friction, and a nonzero
{var}`k_over_m <symbol>` adds quadratic drag.

### An example simulation

No optimizer, no constraints: one integration of the dynamics under a
guessed, affine path-angle law, which is what a transcription starts from
before anything is solved.

```{eval-rst}
.. plot:: _ext/gallery/brachistochrone_example_simulation.py
   :include-source: false
```

The guessed trajectory does not reach the target and is not meant to: it
only has to leave the state space, and hand the transcription a
consistent point to refine from. See {doc}`../concepts/transcriptions` for
what happens next, and {ref}`the-initial-guess` below for why this
particular guess, rather than a cold linear ramp, is what the builders
use.

## Optimization

### The quickest way down

```{eval-rst}
.. plot:: _ext/gallery/brachistochrone_main_result.py
   :include-source: false
```

All three variants are solved on the same grid, and each is checked against
either the closed form (ideal) or a high-accuracy stored reference
(friction, drag); see {func}`~gemseo_tempo.problems.brachistochrone.reference.reference_time`.

```{note}
The ideal variant has a closed form. The optimal path is the cycloid
`x = R(theta - sin theta)`, `y = R(1 - cos theta)`, traversed in
`t = theta sqrt(R/g)`; requiring it to pass through the target fixes
`theta_f`, and `R` follows from the depth. For the default target `(2, 2)`
with `g = 9.81`, that gives **0.824339 s**, available as
{func}`~gemseo_tempo.problems.brachistochrone.reference.cycloid_time`. The
friction and drag variants have no comparably simple closed form, so their
references are high-accuracy numerical values, cross-checked between
collocation and shooting. Regenerate them with
`tools/generate_brachistochrone_references.py`.
```

(the-initial-guess)=
### The initial guess

The builders do not hand the transcription a linear ramp. A ramp leaves the
speed at zero throughout, `dx/dt = v cos(gamma)` degenerates, and
cold-started collocation misses the optimum by over a hundred percent.
Instead the guess parameters — the initial path angle, its increase, and
the descent time — are turned into a trajectory by integrating the dynamics
once (above), and that seeds the transcription:

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

Because every transcription is seeded from the same trajectory, the
benchmarks below are a fair comparison between them.

### What compiling costs, and when it pays

Sprint 9 found the disciplines were never JIT-compiled and reported the fix
as a single speedup number, conflating two costs that scale differently:
compilation grows with the program, the saving is per evaluation. Three
times measured separately, and from them the **crossover**, the evaluations
needed before compiling has repaid itself:

| size | variables | compile | compiled | eager | speedup | crossover |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 12 | 0.09 s | 0.6 ms | 11.3 ms | 19x | 8 |
| 10 | 40 | 0.09 s | 0.5 ms | 10.8 ms | 20x | 9 |
| 30 | 120 | 0.09 s | 0.6 ms | 10.4 ms | 17x | 9 |
| 100 | 400 | 0.09 s | 0.6 ms | 11.0 ms | 18x | 8 |

Compiling pays here within eight or nine evaluations, against the dozens or
hundreds any optimization makes. See {doc}`../guides/benchmarking` for the
two easy mistakes ruled out to get this measurement (a GEMSEO discipline
caches on its inputs, so a repeated call at the same point times a
dictionary lookup instead of either mode).

### Optimization benchmarks

Does the choice of optimizer matter, on a problem this well posed? A
one-factor-at-a-time sweep of SLSQP, NLOPT's SLSQP and COBYLA, at
increasing size:

```{image} ../_data/brachistochrone/optimizer_study.png
:alt: Optimizer benchmark on the brachistochrone
```

### Transcription benchmarks

```{eval-rst}
.. plot:: _ext/gallery/brachistochrone_convergence.py
   :include-source: false
```

Collocation and single shooting, both refined against the closed-form
cycloid. Both converge; what distinguishes them, at a fixed error, is the
cost of a gradient, not the accuracy at a fixed size — which is exactly
what the next benchmark measures.

### AD benchmarks

```{eval-rst}
.. plot:: _ext/gallery/brachistochrone_gradient_cost.py
   :include-source: false
```

This is the number {doc}`../concepts/transcriptions` argues from. Reverse-mode
automatic differentiation through the whole trajectory costs about one
extra pass regardless of the design dimension; central finite differences
cost one extra pass *per design variable*. Measured directly on this
problem's own JAX function, bypassing GEMSEO's discipline wrapper entirely
since it caches a linearization until its inputs change and would
otherwise time the cache rather than either derivative:

| design variables | one AD gradient | one FD gradient | ratio |
| --- | --- | --- | --- |
| 40 | 0.018 ms | 64.5 ms | 3560x |
| 400 | 0.021 ms | 205.6 ms | 10058x |
| 1600 | 0.023 ms | 819.8 ms | 35120x |

Growing the grid forty-fold, the finite-difference cost grows with it while
the AD cost does not move: this is exactly the crossover the received
comparison in {doc}`../concepts/transcriptions` assumes away.

### Indirect shooting

Stationarity of the Hamiltonian in {var}`gamma <symbol>` has a closed form,

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
