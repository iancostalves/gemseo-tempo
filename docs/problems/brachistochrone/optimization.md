# Optimization

```{variable-context} gemseo_tempo.problems.brachistochrone.model:REGISTRY
```

## The quickest way down

```{eval-rst}
.. plot:: _ext/gallery/brachistochrone_main_result.py
   :include-source: false
```

All three variants are solved on the same grid, and each is checked against
either the closed form (ideal) or a high-accuracy stored reference
(friction, drag); see
{func}`~gemseo_tempo.problems.brachistochrone.reference.reference_time`.

```{note}
The ideal variant has a closed form. The optimal path is the cycloid
`x = R(theta - sin theta)`, `y = R(1 - cos theta)`, traversed in
`t = theta sqrt(R/g)`; requiring it to pass through the target fixes
`theta_f` through a transcendental equation in the aspect ratio, and `R`
follows from the depth. For the default target `(2, 2)` with `g = 9.81`,
that gives **0.824339 s**, available as
{func}`~gemseo_tempo.problems.brachistochrone.reference.cycloid_time`. The
friction and drag variants have no comparably simple closed form, so their
references are high-accuracy numerical values, cross-checked between
collocation and shooting. Regenerate them with
`tools/generate_brachistochrone_references.py`.
```

(the-initial-guess)=
## The initial guess

The builders do not hand the transcription a linear ramp. A ramp leaves the
speed at zero throughout, {eq}`brachistochrone-kinematics` degenerates, and
cold-started collocation misses the optimum by over a hundred percent.
Instead the guess parameters — the initial path angle, its increase, and
the descent time — are turned into a trajectory by integrating the dynamics
once (the example simulation on {doc}`physics`), and that seeds the
transcription:

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

## What compiling costs, and when it pays

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
hundreds any optimization makes. See {doc}`../../guides/benchmarking` for
the two easy mistakes ruled out to get this measurement (a GEMSEO
discipline caches on its inputs, so a repeated call at the same point times
a dictionary lookup instead of either mode).

## Optimization benchmarks

Does the choice of optimizer matter, on a problem this well posed? A
one-factor-at-a-time sweep of SLSQP, NLOPT's SLSQP and COBYLA, at
increasing size:

```{image} ../../_data/brachistochrone/optimizer_study.png
:alt: Optimizer benchmark on the brachistochrone
```

## Transcription benchmarks

```{eval-rst}
.. plot:: _ext/gallery/brachistochrone_convergence.py
   :include-source: false
```

Collocation and single shooting, both refined against the closed-form
cycloid. Both converge; what distinguishes them, at a fixed error, is the
cost of a gradient, not the accuracy at a fixed size — which is exactly
what the next benchmark measures.

```{note}
One caveat the friction variant added. The turning rate in
{eq}`brachistochrone-normal` is piecewise constant for a piecewise-linear
control, so at two hundred nodes the right-hand side has two hundred jumps
in it. An adaptive integrator spends the solve chasing them: left adaptive,
the two transcriptions disagreed by 43% at that discretization while
agreeing to 0.02% at twenty. `tools/generate_brachistochrone_references.py`
therefore integrates with a fixed step, which crosses the jumps the same
way every time.
```

## AD benchmarks

```{eval-rst}
.. plot:: _ext/gallery/brachistochrone_gradient_cost.py
   :include-source: false
```

This is the number {doc}`../../concepts/transcriptions` argues from.
Reverse-mode automatic differentiation through the whole trajectory costs
about one extra pass regardless of the design dimension; central finite
differences cost one extra pass *per design variable*. Measured directly on
this problem's own JAX function, bypassing GEMSEO's discipline wrapper
entirely since it caches a linearization until its inputs change and would
otherwise time the cache rather than either derivative:

| design variables | one AD gradient | one FD gradient | ratio |
| --- | --- | --- | --- |
| 40 | 0.018 ms | 64.5 ms | 3560x |
| 400 | 0.021 ms | 205.6 ms | 10058x |
| 1600 | 0.023 ms | 819.8 ms | 35120x |

Growing the grid forty-fold, the finite-difference cost grows with it while
the AD cost does not move: this is exactly the crossover the received
comparison in {doc}`../../concepts/transcriptions` assumes away.

## Indirect shooting

Stationarity of the Hamiltonian in {var}`gamma <symbol>` has a closed form,

```{math}
:label: brachistochrone-costate
\tan\gamma = \frac{\lambda_y v + \lambda_v g}{\lambda_x v - \lambda_v g \mu}
```

supplied as
{func}`~gemseo_tempo.problems.brachistochrone.problem.optimal_path_angle`,
so no inner Newton solve is needed:

```python
from gemseo_tempo.problems.brachistochrone import optimal_path_angle
from gemseo_tempo.transcriptions import IndirectShooting

create_optimization(transcription=IndirectShooting(u_star=optimal_path_angle))
```

This route is available on the **frictionless** variants only, and the
reason is the physics rather than the software. {eq}`brachistochrone-costate`
is the stationary point of the classical friction law, with the normal
force taken as {math}`g\cos\gamma` alone. The centripetal term in
{eq}`brachistochrone-normal` puts the control into the Hamiltonian through
its own derivative, where there is no pointwise stationarity to solve for,
so a model carrying it is refused rather than answered wrongly. With
{math}`\mu = 0` the term is multiplied by zero, the two laws coincide, and
the closed form is exact again.
