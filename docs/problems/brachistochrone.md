# Brachistochrone

```{variable-context} gemseo_tempo.problems.brachistochrone.model:REGISTRY
```

The fastest wire between two points for a bead sliding under gravity
{footcite:p}`bernoulli_problema_1696`, with Coulomb friction and quadratic
drag as variants. It is the smallest problem in the library and the only one
with a closed-form answer, so every method is checked against it here first:
the frictionless descent takes
{docvalue}`brachistochrone/transcriptions.json:reference:.6f` s.

```python
from gemseo_tempo.problems.brachistochrone import create_optimization

scenario, gemseo_scenario = create_optimization(variant="ideal", size=20)
gemseo_scenario.execute(algo_name="SLSQP", max_iter=200)
```

## Problem

```{eval-rst}
.. plot:: _ext/gallery/brachistochrone_kinematics.py
   :include-source: false
```

One Earth-fixed frame, with {var}`x <symbol>` towards the target and
{var}`y <symbol>` **downward**, so a descent is positive. The path angle
{var}`gamma <symbol>` is measured below the horizontal and is the control.
The final time is free, so the dynamics run on τ = t / t_f ∈ [0, 1] and
{var}`t_f <symbol>` is a design variable.

**Assumptions.**
1. A point mass on a wire whose shape the path angle traces; no climbing,
   γ ∈ [0, π/2].
2. Uniform gravity {var}`g <symbol>` along +y.
3. The normal force is m g cos γ: the centripetal term of a curved wire is
   neglected, as usual {footcite:p}`ashby_brachistochrone_1975`, so the
   friction variant's reference is numerical, for this model.
4. Drag is quadratic, k v², with constant {var}`k_over_m <symbol>`.
5. The bead starts at rest at the origin.

```{math}
:label: eq-brach-dynamics

\dot x = v\cos\gamma, \qquad
\dot y = v\sin\gamma, \qquad
\dot v = g\,(\sin\gamma - \mu\cos\gamma) - \frac{k}{m}\,v^2 ,
```

```{math}
:label: eq-brach-ocp

\min_{\gamma(\cdot),\,t_f}\; t_f
\quad\text{s.t.}\quad
(x, y, v)(0) = 0, \qquad (x, y)(t_f) = (2, 2)\ \text{m}.
```

Collocated on twenty Hermite–Simpson segments
{footcite:p}`hargraves_direct_1987`, the problem the optimizer sees is:

```{formulation} brachistochrone/formulation.json
```

The variables, from the registry the model reads:

```{variables} gemseo_tempo.problems.brachistochrone.model:REGISTRY
:columns: name, full_name, unit, tags
```

## Deterministic optimal control

:::{admonition} Takeaway: the deterministic problem
:class: tip

- **Transcription:** Hermite–Simpson collocation, the cheapest route to the
  optimum, {docvalue}`brachistochrone/transcriptions.json:rows.7.seconds:.1f` s at twenty segments. Single
  shooting reaches the same answer at about five times the cost; indirect
  shooting, warm-started from collocation, confirms it to
  {docvalue}`brachistochrone/transcriptions.json:rows.10.error:.0e`.
- **Discretization:** ten to twenty segments. Beyond that the error stops
  falling, held at the optimizer's tolerance.
- **Differentiation:** reverse mode through the checkpointing adjoint,
  compiled: three outputs, three backward passes, the fastest and the
  leanest exact mode. Finite differences batched into one call run close
  behind, with eight digits rather than sixteen.
:::

### Variants

One method throughout: collocation on twenty segments, solved by SLSQP
{footcite:p}`kraft1988software` from a guess built by simulating the
dynamics once.

```{eval-rst}
.. plot:: _ext/gallery/brachistochrone_main_result.py
   :include-source: false
```

| variant | descent time | reference |
| --- | --- | --- |
| ideal | {docvalue}`brachistochrone/variants.json:ideal.time:.6f` s | {docvalue}`brachistochrone/variants.json:ideal.reference:.6f` s, the cycloid |
| friction, μ = 0.3 | {docvalue}`brachistochrone/variants.json:friction.time:.6f` s | {docvalue}`brachistochrone/variants.json:friction.reference:.6f` s, numerical |
| drag, k/m = 0.05 | {docvalue}`brachistochrone/variants.json:drag.time:.6f` s | {docvalue}`brachistochrone/variants.json:drag.reference:.6f` s, numerical |

Friction reshapes the path: it scales with m g cos γ, so the optimum dives
more steeply than the cycloid and flattens towards the target. Drag slows
the bead without moving the path by more than a couple of centimetres.

The ideal reference is the cycloid through the target,

```{math}
:label: eq-brach-cycloid

x = R(\theta - \sin\theta), \qquad y = R(1 - \cos\theta), \qquad
t = \theta\sqrt{R/g},
```

with θ_f fixed by (θ_f − sin θ_f)/(1 − cos θ_f) = x_f / y_f. The other two
are high-accuracy numerical values, and the stored optima are checked
against all three on every test run.

### Transcriptions

The ideal variant, solved by each transcription with the same optimizer:
single shooting, where only the control is discretized and an adaptive
integrator carries the accuracy; Hermite–Simpson collocation, where the
states at the nodes are unknowns too; and indirect shooting, which solves
Pontryagin's necessary conditions {footcite:p}`pontrjagin_mathematical_1964`
instead of the problem itself.

```{eval-rst}
.. plot:: _ext/gallery/brachistochrone_transcriptions.py
   :include-source: false
```

| transcription | grid | error | time | calls |
| --- | --- | --- | --- | --- |
| shooting | 20 | {docvalue}`brachistochrone/transcriptions.json:rows.2.error:.1e` | {docvalue}`brachistochrone/transcriptions.json:rows.2.seconds:.1f` s | {docvalue}`brachistochrone/transcriptions.json:rows.2.calls:d` |
| collocation | 20 | {docvalue}`brachistochrone/transcriptions.json:rows.7.error:.1e` | {docvalue}`brachistochrone/transcriptions.json:rows.7.seconds:.1f` s | {docvalue}`brachistochrone/transcriptions.json:rows.7.calls:d` |
| indirect | none | {docvalue}`brachistochrone/transcriptions.json:rows.10.error:.1e` | {docvalue}`brachistochrone/transcriptions.json:rows.10.seconds:.1f` s | {docvalue}`brachistochrone/transcriptions.json:rows.10.calls:d` |

All three reach the cycloid to the optimizer's tolerance, and what separates
them is cost. Shooting is at the integrator's accuracy from the coarsest grid,
since its grid only shapes the control, but every call integrates the
dynamics adaptively. Collocation reaches the same answer from ten segments on
and is the cheapest, which is why it is the default.

Indirect shooting has no grid at all. It solves for the
{docvalue}`brachistochrone/transcriptions.json:rows.10.unknowns:d` unknowns
of Pontryagin's boundary-value problem, the initial costates and the final
time, and is the slowest here, because each call integrates the costates as
well as the states and minimizes the Hamiltonian at every step. It also only
converges from a good guess, which
{func}`~gemseo_tempo.transcriptions.indirect.indirect_from_direct` reads off
the finest collocation solution by one least-squares solve: the seed already
satisfies the necessary conditions to
{docvalue}`brachistochrone/transcriptions.json:rows.10.seed_residual:.0e`.
Nothing about the problem is written by hand: JAX differentiates the
dynamics for the costate equations, and the optimal path angle is found by
minimizing the Hamiltonian numerically.

### Differentiation

The ideal descent by single shooting, its Jacobian taken four ways: by
finite differences; by forward mode; and by reverse mode, once through the
Direct adjoint and once through the checkpointing one the library uses by
default. Three outputs, the time and the two terminal misses, against a
growing number of path angles; compiled (solid) and not (dashed).

Each method is measured at its best. Finite differences are forward
differences at the step, from 10⁻⁴ to 10⁻⁹, that comes closest to the
exact Jacobian, timed both batched into one vectorized call and one
evaluation at a time, the faster counting. Compilation is paid once per
program, and is charged separately, below:

```{eval-rst}
.. plot:: _ext/gallery/brachistochrone_mode_sweep.py
   :include-source: false
```

Batched into one vectorized call, finite differences are not the slow
method they are usually taken for: at {docvalue}`brachistochrone/differentiation.json:inputs.35.n_inputs:d`
inputs they take {docvalue}`brachistochrone/differentiation.json:inputs.35.seconds:.3f` s, against
{docvalue}`brachistochrone/differentiation.json:inputs.36.seconds:.3f` s by forward mode, which also pushes
every input direction through the model in one batch and costs about the
same. Reverse mode needs only three backward passes, one per output, and
through the checkpointing adjoint it is the fastest at all sizes but one:
{docvalue}`brachistochrone/differentiation.json:inputs.38.seconds:.3f` s at the largest. Through the Direct
adjoint it pays to store the whole solve, and overtakes the batched methods
only on the largest grid. Not compiling costs a quarter to a half more, less
than expected, because diffrax compiles its own integration whatever the
caller does: what is left uncompiled is the code around it. Compiling has
its own cost, paid once per program:

```{eval-rst}
.. plot:: _ext/gallery/brachistochrone_compile_times.py
   :include-source: false
```

The function compiles in about a second at every size, its batched
differences in under two, its forward Jacobian in two to four and its
reverse one in five to ten, since the reverse program carries the backward
sweep as well as the forward one.
At twenty intervals, each method on the four costs, the best at the rim:

```{eval-rst}
.. plot:: _ext/gallery/brachistochrone_mode_radar.py
   :include-source: false
```

| method | compilation | one Jacobian | error | memory |
| --- | --- | --- | --- | --- |
| finite differences | {docvalue}`brachistochrone/differentiation.json:inputs.14.compile_jacobian:.1f` s | {docvalue}`brachistochrone/differentiation.json:inputs.14.seconds:.3f` s | {docvalue}`brachistochrone/differentiation.json:inputs.14.error:.0e` | {docvalue}`brachistochrone/differentiation.json:inputs.14.memory:.0f*0.001` kB |
| forward | {docvalue}`brachistochrone/differentiation.json:inputs.15.compile_jacobian:.1f` s | {docvalue}`brachistochrone/differentiation.json:inputs.15.seconds:.3f` s | {docvalue}`brachistochrone/differentiation.json:inputs.15.error:.0e` | {docvalue}`brachistochrone/differentiation.json:inputs.15.memory:.0f*0.001` kB |
| reverse | {docvalue}`brachistochrone/differentiation.json:inputs.16.compile_jacobian:.1f` s | {docvalue}`brachistochrone/differentiation.json:inputs.16.seconds:.3f` s | {docvalue}`brachistochrone/differentiation.json:inputs.16.error:.0e` | {docvalue}`brachistochrone/differentiation.json:inputs.16.memory:.0f*0.001` kB |
| reverse, checkpointed | {docvalue}`brachistochrone/differentiation.json:inputs.17.compile_jacobian:.1f` s | {docvalue}`brachistochrone/differentiation.json:inputs.17.seconds:.3f` s | {docvalue}`brachistochrone/differentiation.json:inputs.17.error:.0e` | {docvalue}`brachistochrone/differentiation.json:inputs.17.memory:.0f*0.001` kB |

Reverse mode through the checkpointing adjoint is the cheapest in time at
almost every size, and the leanest of the exact methods in memory, which is what
checkpointing is for. Batched finite differences come a close second in
time and compile fastest, and pay in precision: eight digits, against the
sixteen of every exact mode.

## Uncertainty propagation

:::{admonition} Takeaway: uncertainty propagation
:class: tip

- **Sampling:** vectorized with `vmap`, never looped, and on a wire, not a
  steering schedule, so that only the time is uncertain.
- **Estimator:** first-order Taylor, two model calls, since the descent time
  is nearly linear in the three constants.
- **Differentiation:** finite differences, batched into one call: with
  three uncertain inputs they lose nothing and are the quicker.
:::

A wire, once built, meets friction and drag nobody measured exactly:
{var}`g <symbol>`, {var}`mu <symbol>` and {var}`k_over_m <symbol>` are
drawn from the distributions the registry declares, gravity to a few parts in
a thousand and the other two uniformly over a range. Their means make a bead
with a little of each variant. The wire's shape does not change with them,
so the bead is integrated along the wire rather than in time. With
ζ = √s, the square root of the arc length, which keeps the release from
rest regular,

```{math}
:label: eq-brach-wire

\frac{dx}{d\zeta} = 2\zeta\cos\gamma, \quad
\frac{dy}{d\zeta} = 2\zeta\sin\gamma, \quad
\frac{dv}{d\zeta} = \frac{2\zeta}{v}\Big(g\,(\sin\gamma - \mu\cos\gamma)
  - \frac{k}{m}\,v^2\Big), \quad
\frac{dt}{d\zeta} = \frac{2\zeta}{v} .
```

The end of the wire is then pure geometry, and only the descent time is
uncertain. Optimized this way for the ideal bead, the wire is the cycloid
again, to a part in a million. Optimized for the mean constants and frozen,
it takes {docvalue}`brachistochrone/propagation.json:reference.mean:.4f` s
on average, give or take
{docvalue}`brachistochrone/propagation.json:reference.standard_deviation:.4f` s.
Each way of estimating that, against 20 000 draws, where AD marks exact
derivatives by reverse-mode automatic differentiation and FD central
differences, every perturbed point in the same vectorized call, at the
step closest to the exact derivative at the mean:

| method | error, mean | error, deviation | model calls | time |
| --- | --- | --- | --- | --- |
| sampling, 256 draws | {docvalue}`brachistochrone/propagation.json:estimates.sampling_vmapped.mean_error:+.0e` σ | {docvalue}`brachistochrone/propagation.json:estimates.sampling_vmapped.deviation_error:+.0e` | {docvalue}`brachistochrone/propagation.json:estimates.sampling_vmapped.model_calls:d` | {docvalue}`brachistochrone/propagation.json:estimates.sampling_vmapped.seconds:.3f` s, looped {docvalue}`brachistochrone/propagation.json:estimates.sampling_looped.seconds:.2f` s |
| Taylor, 1st order, AD | {docvalue}`brachistochrone/propagation.json:estimates.taylor1_ad.mean_error:+.0e` σ | {docvalue}`brachistochrone/propagation.json:estimates.taylor1_ad.deviation_error:+.0e` | {docvalue}`brachistochrone/propagation.json:estimates.taylor1_ad.model_calls:d` | {docvalue}`brachistochrone/propagation.json:estimates.taylor1_ad.seconds:.3f` s |
| Taylor, 1st order, FD | {docvalue}`brachistochrone/propagation.json:estimates.taylor1_fd.mean_error:+.0e` σ | {docvalue}`brachistochrone/propagation.json:estimates.taylor1_fd.deviation_error:+.0e` | {docvalue}`brachistochrone/propagation.json:estimates.taylor1_fd.model_calls:d` | {docvalue}`brachistochrone/propagation.json:estimates.taylor1_fd.seconds:.3f` s |
| Taylor, 2nd order, AD | {docvalue}`brachistochrone/propagation.json:estimates.taylor2_ad.mean_error:+.0e` σ | {docvalue}`brachistochrone/propagation.json:estimates.taylor2_ad.deviation_error:+.0e` | {docvalue}`brachistochrone/propagation.json:estimates.taylor2_ad.model_calls:d` | {docvalue}`brachistochrone/propagation.json:estimates.taylor2_ad.seconds:.3f` s |
| Taylor, 2nd order, FD | {docvalue}`brachistochrone/propagation.json:estimates.taylor2_fd.mean_error:+.0e` σ | {docvalue}`brachistochrone/propagation.json:estimates.taylor2_fd.deviation_error:+.0e` | {docvalue}`brachistochrone/propagation.json:estimates.taylor2_fd.model_calls:d` | {docvalue}`brachistochrone/propagation.json:estimates.taylor2_fd.seconds:.3f` s |
| chaos, 12 points | {docvalue}`brachistochrone/propagation.json:estimates.chaos.mean_error:+.0e` σ | {docvalue}`brachistochrone/propagation.json:estimates.chaos.deviation_error:+.0e` | {docvalue}`brachistochrone/propagation.json:estimates.chaos.model_calls:d` | {docvalue}`brachistochrone/propagation.json:estimates.chaos.seconds:.3f` s |
| chaos + AD gradients | {docvalue}`brachistochrone/propagation.json:estimates.chaos_ad.mean_error:+.0e` σ | {docvalue}`brachistochrone/propagation.json:estimates.chaos_ad.deviation_error:+.0e` | {docvalue}`brachistochrone/propagation.json:estimates.chaos_ad.model_calls:d` | {docvalue}`brachistochrone/propagation.json:estimates.chaos_ad.seconds:.3f` s |
| chaos + FD gradients | {docvalue}`brachistochrone/propagation.json:estimates.chaos_fd.mean_error:+.0e` σ | {docvalue}`brachistochrone/propagation.json:estimates.chaos_fd.deviation_error:+.0e` | {docvalue}`brachistochrone/propagation.json:estimates.chaos_fd.model_calls:d` | {docvalue}`brachistochrone/propagation.json:estimates.chaos_fd.seconds:.3f` s |

Over these ranges the descent time is nearly linear in the three
constants, so every method lands within a few hundredths of a deviation of
the reference, and first-order Taylor, one evaluation and one gradient, is
enough. What separates the methods is cost, in two ways. Vectorizing the
sample with `vmap` runs the same 256 draws in one dispatch, in
{docvalue}`brachistochrone/propagation.json:estimates.sampling_vmapped.seconds:.2f` s against
{docvalue}`brachistochrone/propagation.json:estimates.sampling_looped.seconds:.1f` s in a loop. And with only three uncertain inputs, finite
differences cost six evaluations per gradient and reproduce the AD numbers
exactly; on a model this cheap they are even quicker than a reverse pass,
which costs a few evaluations. Where AD earns its keep is a gradient with
respect to many variables, which is what the next section needs.

## Optimization under uncertainty

:::{admonition} Takeaway: optimization under uncertainty
:class: tip

- **Formulation:** the wire designed for the mean constants; a robust
  objective finds the same wire.
- **Sampling:** one fixed sample of {docvalue}`brachistochrone/robust.json:design_sample:d` draws,
  the same at every iteration, checked afterwards on
  {docvalue}`brachistochrone/robust.json:check_sample:d` fresh ones.
- **Differentiation:** automatic, in the loop, once the design is large:
  an AD gradient costs about the same whatever its size. At twenty
  variables, finite differences batched into one call compile in half the
  time and win a single solve.
:::

Three wires, each flown on
{docvalue}`brachistochrone/robust.json:check_sample:d` fresh draws:

| wire | mean | deviation | 95th percentile |
| --- | --- | --- | --- |
| the cycloid, for the ideal bead | {docvalue}`brachistochrone/robust.json:designs.ideal.mean:.5f` s | {docvalue}`brachistochrone/robust.json:designs.ideal.std:.5f` s | {docvalue}`brachistochrone/robust.json:designs.ideal.p95:.5f` s |
| designed for the mean constants | {docvalue}`brachistochrone/robust.json:designs.nominal.mean:.5f` s | {docvalue}`brachistochrone/robust.json:designs.nominal.std:.5f` s | {docvalue}`brachistochrone/robust.json:designs.nominal.p95:.5f` s |
| minimizing mean + 2σ over {docvalue}`brachistochrone/robust.json:design_sample:d` draws | {docvalue}`brachistochrone/robust.json:designs.robust.mean:.5f` s | {docvalue}`brachistochrone/robust.json:designs.robust.std:.5f` s | {docvalue}`brachistochrone/robust.json:designs.robust.p95:.5f` s |

The three agree to a few parts in ten thousand: the uncertainty moves the
descent time, not the best wire, and even the cycloid, designed for no
friction at all, gives almost nothing away. That is a property of this
problem, and it leaves the method to compare. The robust problem, solved
by SLSQP from the same start with each kind of gradient:

| gradient | iterations | compilation | optimization |
| --- | --- | --- | --- |
| AD, reverse mode | {docvalue}`brachistochrone/robust.json:designs.robust.calls:d` | {docvalue}`brachistochrone/robust.json:designs.robust.compile_seconds:.1f` s | {docvalue}`brachistochrone/robust.json:designs.robust.seconds:.1f` s |
| FD, batched | {docvalue}`brachistochrone/robust.json:designs.robust_fd.calls:d` | {docvalue}`brachistochrone/robust.json:designs.robust_fd.compile_seconds:.1f` s | {docvalue}`brachistochrone/robust.json:designs.robust_fd.seconds:.1f` s |

Both reach the same wire in the same number of iterations. Each iteration
differentiates the mean and deviation of
{docvalue}`brachistochrone/robust.json:design_sample:d` descents with respect to all
{docvalue}`brachistochrone/robust.json:gradient_cost.1.n_design:d` design variables. By AD that is one
reverse pass through the whole batch, through the checkpointing adjoint,
the fastest exact route measured; by finite differences, every perturbed
wire flown in the same vectorized call, at the step closest to the exact
gradient. The loop runs about twice as fast with AD, but AD compiles for
twice as long, and over one solve of twenty iterations, compilation
included, the batched differences come out ahead. What turns it is the
size of the design: an AD gradient costs about the same however many
variables there are, while the batch of differences grows with every one,
to {docvalue}`brachistochrone/robust.json:gradient_cost.3.fd_seconds:.2f` s against
{docvalue}`brachistochrone/robust.json:gradient_cost.3.ad_seconds:.2f` s at
{docvalue}`brachistochrone/robust.json:gradient_cost.3.n_design:d` variables:

```{eval-rst}
.. plot:: _ext/gallery/brachistochrone_gradient_cost.py
   :include-source: false
```


## References

```{footbibliography}
```
