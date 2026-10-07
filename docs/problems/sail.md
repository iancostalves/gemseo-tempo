# Sailing to a mark

```{variable-context} gemseo_tempo.problems.sail.model:REGISTRY
```

Minimum-time sailing to a mark 800 m away, inside a race area that stretches
500 m either side of the direct line. The optimizer is never told to gybe or
to tack, and does both: downwind it arrives in
{docvalue}`sail/downwind.json:optimized:.1f` s, against
{docvalue}`sail/downwind.json:straight:.1f` s sailed straight at the mark and
a bound of {docvalue}`sail/vmg.json:downwind.ideal_time:.1f` s set by the
polar itself.

```python
from gemseo_tempo.problems.sail import create_optimization

scenario, gemseo_scenario = create_optimization(leg="downwind", size=25)
gemseo_scenario.execute(algo_name="SLSQP", max_iter=300)
```

## Problem

```{eval-rst}
.. plot:: _ext/gallery/sail_angle_convention.py
   :include-source: false
```

One Earth-fixed frame, {var}`x <symbol>` east and {var}`y <symbol>` north,
with the start at the origin. The heading {var}`psi <symbol>` is measured
counter-clockwise from east, so a compass course is 90° − ψ. The wind field
gives where the air goes; the true wind angle TWA is measured from the
heading to where it comes **from**, positive on starboard tack.

**Assumptions.**
1. The boat is a point that sails at its polar speed at once: a gybe or a
   tack costs nothing but the geometry of the turn.
2. A steady, uniform wind of 10 m/s blowing east.
3. No current and no leeway: the boat goes where it points.
4. An analytic, separable polar, V = f(|TWA|) g(TWS), symmetric between
   tacks, authored rather than measured {footcite:p}`larsson_principles_2014`.
5. The race area bounds the boat to |y| ≤ 500 m.

```{math}
:label: eq-sail-kinematics

\dot x = V(\mathrm{TWA}, \mathrm{TWS})\cos\psi, \qquad
\dot y = V(\mathrm{TWA}, \mathrm{TWS})\sin\psi,
```

```{math}
:label: eq-sail-twa

\mathrm{TWA} = \operatorname{wrap}\!\left(\psi - \beta_\mathrm{from}\right),
\qquad \beta_\mathrm{from} = \operatorname{atan2}(-w_y, -w_x),
\qquad \mathrm{TWS} = \lVert w \rVert ,
```

```{math}
:label: eq-sail-ocp

\min_{\psi(\cdot),\,t_f}\; t_f
\quad\text{s.t.}\quad
(x, y)(0) = 0, \qquad (x, y)(t_f) = (x_m, y_m), \qquad
|y(t)| \le b ,
```

with b = 500 m for the race area. The polar is tabulated against the
cosine of the wind angle, which keeps it smooth through dead downwind where
this problem spends its time, and it is floored smoothly at zero speed
inside the no-go cone. Solved by single shooting on 25 intervals, the
problem the optimizer sees is:

```{formulation} sail/formulation.json
```

```{variables} gemseo_tempo.problems.sail.model:REGISTRY
:columns: name, full_name, unit, tags
```

## Deterministic optimal control

:::{admonition} Takeaway: the deterministic problem
:class: tip

- **Transcription:** single shooting. Collocation settles in a narrower,
  slower gybe, and indirect shooting cannot represent the gybe at all.
- **Discretization:** twenty-five to fifty control intervals; the upwind leg
  from more than one start, since the polar's guess leads to a slow local
  optimum.
- **Differentiation:** forward mode through the Direct adjoint, compiled,
  or finite differences batched into one call: the two run neck and neck
  at every size, forward with exact derivatives, differences with six or
  seven digits. Reverse mode is the slowest and the heaviest, and the
  checkpointing adjoint is not exact here.
:::

### Variants

One method throughout: single shooting on 25 intervals, solved by SLSQP
{footcite:p}`kraft1988software` from headings read off the polar's best
speed-made-good angle.

```{eval-rst}
.. plot:: _ext/gallery/sail_legs.py
   :include-source: false
```

| leg | optimized | sailed straight | polar bound |
| --- | --- | --- | --- |
| downwind | {docvalue}`sail/downwind.json:optimized:.1f` s | {docvalue}`sail/downwind.json:straight:.1f` s | {docvalue}`sail/vmg.json:downwind.ideal_time:.1f` s |
| upwind | {docvalue}`sail/upwind.json:optimized:.1f` s | impossible | {docvalue}`sail/vmg.json:upwind.ideal_time:.1f` s |

The bound is the speed made good, the part of the boat speed that shortens
the distance to the mark,

```{math}
:label: eq-sail-vmg

\mathrm{VMG} = V(\mathrm{TWA}, \mathrm{TWS})\,\lvert\cos\mathrm{TWA}\rvert ,
```

maximized over the polar and held for the whole leg: no track can beat it.
Downwind, the optimizer gybes once, holding the best angle,
{docvalue}`sail/vmg.json:downwind.angle:.1f`°, to a tenth of a degree on both
gybes, and arrives within a fraction of a second of the bound; the gap is the
distance the turn costs. Upwind the mark cannot be sailed at: the optimizer
tacks once, {docvalue}`sail/upwind.json:across:.0f` m out, at the edge of
the race area, and arrives within 4% of the bound; the gap is the tack. The
start matters here: from the polar's guess, which sweeps the heading slowly
through the eye of the wind, the solver settles on a slow local optimum, so
the leg is also started from one tack at half-way and the faster answer
kept.

Narrowing the race area turns the corridor into a constraint that binds:

```{eval-rst}
.. plot:: _ext/gallery/sail_channel_trade.py
   :include-source: false
```

| half-width | downwind time | furthest across |
| --- | --- | --- |
| 250 m | {docvalue}`sail/channel.json:widths.0.time:.2f` s | {docvalue}`sail/channel.json:widths.0.across:.1f` m |
| 150 m | {docvalue}`sail/channel.json:widths.1.time:.2f` s | {docvalue}`sail/channel.json:widths.1.across:.1f` m |
| 80 m | {docvalue}`sail/channel.json:widths.2.time:.2f` s | {docvalue}`sail/channel.json:widths.2.across:.1f` m |

A narrower channel costs time, monotonically. The constrained leg has
several local optima, so each width is solved from two starts, the
unconstrained optimum and the next looser design, and the better is kept.

### Transcriptions

The downwind leg, by each transcription, with SLSQP:

```{eval-rst}
.. plot:: _ext/gallery/sail_transcriptions.py
   :include-source: false
```

| transcription | grid | time | error | wall time |
| --- | --- | --- | --- | --- |
| shooting | 25 | {docvalue}`sail/transcriptions.json:rows.1.objective:.2f` s | {docvalue}`sail/transcriptions.json:rows.1.error:.1e` | {docvalue}`sail/transcriptions.json:rows.1.seconds:.0f` s |
| collocation | 25 | {docvalue}`sail/transcriptions.json:rows.5.objective:.2f` s | {docvalue}`sail/transcriptions.json:rows.5.error:.1e` | {docvalue}`sail/transcriptions.json:rows.5.seconds:.0f` s |
| indirect | none | not converged | | |

Collocation settles in a narrower gybe than shooting and a couple of
percent slower, and refining its grid does not help: that is a local
optimum, not a discretization error. Shooting integrates the dynamics
instead of treating the positions as unknowns, and reaches the wider one.

Indirect shooting cannot solve this leg at all, and the reason is the
physics. In a uniform wind the costates are constant, and the heading that
minimizes the Hamiltonian is the one that maximizes the speed made good
along them: there are two, the best angle on each gybe, with exactly the
same value. The optimal track is a mix of the two, which the direct methods
approximate by gybing, but a control law that returns one heading sails a
single gybe all the way and misses the mark by hundreds of metres. Started
from the finest collocation solution, the residual of the necessary
conditions stays at {docvalue}`sail/transcriptions.json:rows.8.residual:.0f`.

### Differentiation

The downwind leg by single shooting, its Jacobian three ways: by finite
differences, by forward mode and by reverse mode, both modes through the
Direct adjoint the sail always needs. The channel is checked at every save
point, so the outputs grow with the save grid while the inputs grow with
the control intervals; each is swept with the other held, compiled (solid)
and not (dashed).

Each method is measured at its best. Finite differences are forward
differences at the step, from 10⁻⁴ to 10⁻⁹, that comes closest to the
exact Jacobian, timed both batched into one vectorized call and one
evaluation at a time, the faster counting. Compilation is paid once per
program, and is charged separately, below:

```{eval-rst}
.. plot:: _ext/gallery/sail_mode_sweep.py
   :include-source: false
```

Forward mode and batched finite differences run neck and neck at every
size, each a few percent ahead at some sizes and behind at others: at
{docvalue}`sail/differentiation.json:inputs.27.n_inputs:d` inputs,
{docvalue}`sail/differentiation.json:inputs.28.seconds:.1f` s by forward mode against
{docvalue}`sail/differentiation.json:inputs.27.seconds:.1f` s by differences. Both push every input
direction through the solve in one batch, and neither stores it. Reverse
mode through the Direct adjoint does, to sweep back through it: it is the
slowest at every size, {docvalue}`sail/differentiation.json:inputs.29.seconds:.1f` s at the largest,
and its memory grows with the solve, to
{docvalue}`sail/differentiation.json:inputs.29.memory:.0f*1e-6` MB against
{docvalue}`sail/differentiation.json:inputs.28.memory:.1f*1e-6` MB. As the outputs grow at fixed
inputs, forward mode and finite differences stay flat, their cost set by
the inputs alone, while reverse mode grows with every output, to
{docvalue}`sail/differentiation.json:outputs.20.seconds:.1f` s at
{docvalue}`sail/differentiation.json:outputs.20.n_outputs:d` outputs. Compiling the reverse program
takes longer than the forward one, up to five times as long, and the
batched differences compile fastest of all:

```{eval-rst}
.. plot:: _ext/gallery/sail_compile_times.py
   :include-source: false
```

At twenty-five intervals, each method on the four costs, the best at the
rim:

```{eval-rst}
.. plot:: _ext/gallery/sail_mode_radar.py
   :include-source: false
```

| method | compilation | one Jacobian | error | memory |
| --- | --- | --- | --- | --- |
| finite differences | {docvalue}`sail/differentiation.json:inputs.12.compile_jacobian:.1f` s | {docvalue}`sail/differentiation.json:inputs.12.seconds:.3f` s | {docvalue}`sail/differentiation.json:inputs.12.error:.0e` | {docvalue}`sail/differentiation.json:inputs.12.memory:.0f*0.001` kB |
| forward | {docvalue}`sail/differentiation.json:inputs.13.compile_jacobian:.1f` s | {docvalue}`sail/differentiation.json:inputs.13.seconds:.3f` s | {docvalue}`sail/differentiation.json:inputs.13.error:.0e` | {docvalue}`sail/differentiation.json:inputs.13.memory:.0f*0.001` kB |
| reverse | {docvalue}`sail/differentiation.json:inputs.14.compile_jacobian:.1f` s | {docvalue}`sail/differentiation.json:inputs.14.seconds:.3f` s | {docvalue}`sail/differentiation.json:inputs.14.error:.0e` | {docvalue}`sail/differentiation.json:inputs.14.memory:.0f*0.001` kB |

## Uncertainty propagation

:::{admonition} Takeaway: uncertainty propagation
:class: tip

- **Sampling:** vectorized with `vmap`: the largest saving of all, a
  hundredfold on the same draws.
- **Estimator:** first-order Taylor, two model calls, for the spread;
  gradient-enhanced chaos when the mean is wanted to a thousandth of a
  deviation.
- **Differentiation:** finite differences, batched into one call: with two
  uncertain inputs they are quicker than the reverse passes, and give the
  same answer.
:::

The wind on the day is not the forecast one: its strength
{var}`wind_scale <symbol>` is known to 12% and its direction
{var}`wind_veer <symbol>` to 10°, both normally distributed. The downwind
leg's optimal headings, sailed as planned for their
{docvalue}`sail/propagation.json:objective:.1f` s without correction, then end short of or
past the mark by
{docvalue}`sail/propagation.json:reference.standard_deviation:.0f` m along the course, one
deviation either way. Single shooting is used here, so that a frozen heading
schedule responds to the wind. Each way of estimating that spread, against
20 000 draws, where AD marks exact derivatives by reverse-mode automatic
differentiation and FD central
differences, every perturbed point in the same vectorized call, at the
step closest to the exact derivative at the mean:

| method | error, mean | error, deviation | model calls | time |
| --- | --- | --- | --- | --- |
| sampling, 256 draws | {docvalue}`sail/propagation.json:estimates.sampling_vmapped.mean_error:+.0e` σ | {docvalue}`sail/propagation.json:estimates.sampling_vmapped.deviation_error:+.0e` | {docvalue}`sail/propagation.json:estimates.sampling_vmapped.model_calls:d` | {docvalue}`sail/propagation.json:estimates.sampling_vmapped.seconds:.3f` s, looped {docvalue}`sail/propagation.json:estimates.sampling_looped.seconds:.2f` s |
| Taylor, 1st order, AD | {docvalue}`sail/propagation.json:estimates.taylor1_ad.mean_error:+.0e` σ | {docvalue}`sail/propagation.json:estimates.taylor1_ad.deviation_error:+.0e` | {docvalue}`sail/propagation.json:estimates.taylor1_ad.model_calls:d` | {docvalue}`sail/propagation.json:estimates.taylor1_ad.seconds:.3f` s |
| Taylor, 1st order, FD | {docvalue}`sail/propagation.json:estimates.taylor1_fd.mean_error:+.0e` σ | {docvalue}`sail/propagation.json:estimates.taylor1_fd.deviation_error:+.0e` | {docvalue}`sail/propagation.json:estimates.taylor1_fd.model_calls:d` | {docvalue}`sail/propagation.json:estimates.taylor1_fd.seconds:.3f` s |
| Taylor, 2nd order, AD | {docvalue}`sail/propagation.json:estimates.taylor2_ad.mean_error:+.0e` σ | {docvalue}`sail/propagation.json:estimates.taylor2_ad.deviation_error:+.0e` | {docvalue}`sail/propagation.json:estimates.taylor2_ad.model_calls:d` | {docvalue}`sail/propagation.json:estimates.taylor2_ad.seconds:.3f` s |
| Taylor, 2nd order, FD | {docvalue}`sail/propagation.json:estimates.taylor2_fd.mean_error:+.0e` σ | {docvalue}`sail/propagation.json:estimates.taylor2_fd.deviation_error:+.0e` | {docvalue}`sail/propagation.json:estimates.taylor2_fd.model_calls:d` | {docvalue}`sail/propagation.json:estimates.taylor2_fd.seconds:.3f` s |
| chaos, 12 points | {docvalue}`sail/propagation.json:estimates.chaos.mean_error:+.0e` σ | {docvalue}`sail/propagation.json:estimates.chaos.deviation_error:+.0e` | {docvalue}`sail/propagation.json:estimates.chaos.model_calls:d` | {docvalue}`sail/propagation.json:estimates.chaos.seconds:.3f` s |
| chaos + AD gradients | {docvalue}`sail/propagation.json:estimates.chaos_ad.mean_error:+.0e` σ | {docvalue}`sail/propagation.json:estimates.chaos_ad.deviation_error:+.0e` | {docvalue}`sail/propagation.json:estimates.chaos_ad.model_calls:d` | {docvalue}`sail/propagation.json:estimates.chaos_ad.seconds:.3f` s |
| chaos + FD gradients | {docvalue}`sail/propagation.json:estimates.chaos_fd.mean_error:+.0e` σ | {docvalue}`sail/propagation.json:estimates.chaos_fd.deviation_error:+.0e` | {docvalue}`sail/propagation.json:estimates.chaos_fd.model_calls:d` | {docvalue}`sail/propagation.json:estimates.chaos_fd.seconds:.3f` s |

The miss is close to linear in the two inputs, so first-order Taylor,
one evaluation and one gradient, already matches 20 000 draws to a hundredth
of a deviation; second order adds a little curvature to the mean and nothing
useful. Gradient-enhanced chaos is the most accurate of all, and it needs the
gradient at each of its twelve points: twelve reverse passes by AD, or 48
more evaluations by FD, for the same answer, and with two inputs the
differences, all in one vectorized call, are the quicker. The largest
saving is
vectorization: the 256 draws take
{docvalue}`sail/propagation.json:estimates.sampling_vmapped.seconds:.2f` s in
one `vmap` dispatch, against
{docvalue}`sail/propagation.json:estimates.sampling_looped.seconds:.0f` s in a loop.

## Optimization under uncertainty

:::{admonition} Takeaway: optimization under uncertainty
:class: tip

- **Formulation:** a chance constraint, P(inside) ≥ 95%, solved once,
  rather than a safety margin found by trial and error.
- **Sampling:** one fixed sample of {docvalue}`sail/margin.json:design_sample:d`
  draws, the probability smoothed so that it has a gradient, and the design
  checked on 4000 fresh ones.
- **Differentiation:** automatic, in the loop: one gradient of the
  probability with respect to every heading costs a reverse pass through the
  batch, a quarter of what finite differences cost even batched into one
  call.
:::

With the race area narrowed to
{docvalue}`sail/margin.json:half_width:.0f` m either side, the downwind leg's optimum
runs along its edge, and in the wind that actually blows it stays inside
only {docvalue}`sail/margin.json:designs.0.reliability:.0%` of the time. There are two
ways to fix that. A safety margin tightens the channel by a fixed amount and
solves as before. A chance constraint asks for the probability itself,
P(inside) ≥ 95% over {docvalue}`sail/margin.json:design_sample:d` fixed draws, smoothed
so that it has a gradient. Each design, checked on 4000 fresh draws:

| design | time | P(inside) | furthest across, as planned |
| --- | --- | --- | --- |
| deterministic | {docvalue}`sail/margin.json:designs.0.objective:.2f` s | {docvalue}`sail/margin.json:designs.0.reliability:.1%` | {docvalue}`sail/margin.json:designs.0.across:.0f` m |
| margin 25 m | {docvalue}`sail/margin.json:designs.1.objective:.2f` s | {docvalue}`sail/margin.json:designs.1.reliability:.1%` | {docvalue}`sail/margin.json:designs.1.across:.0f` m |
| margin 50 m | {docvalue}`sail/margin.json:designs.2.objective:.2f` s | {docvalue}`sail/margin.json:designs.2.reliability:.1%` | {docvalue}`sail/margin.json:designs.2.across:.0f` m |
| margin 75 m | {docvalue}`sail/margin.json:designs.3.objective:.2f` s | {docvalue}`sail/margin.json:designs.3.reliability:.1%` | {docvalue}`sail/margin.json:designs.3.across:.0f` m |
| P(inside) ≥ 95% | {docvalue}`sail/margin.json:designs.4.objective:.2f` s | {docvalue}`sail/margin.json:designs.4.reliability:.1%` | {docvalue}`sail/margin.json:designs.4.across:.0f` m |

A margin buys reliability with time, but which margin is right can only be
found by trying them: 25 m is not enough and 75 m is more than asked for.
The chance constraint reaches the reliability asked for in one solve, and
lands on the same design as the 50 m margin, which only hindsight could
have picked.

Its gradient is a derivative of a probability, with respect to all
{docvalue}`sail/margin.json:gradient_cost.n_design:d` design variables, through
{docvalue}`sail/margin.json:design_sample:d` sailed legs. By AD that is one reverse
pass through the batch, {docvalue}`sail/margin.json:gradient_cost.ad_seconds:.1f` s,
against {docvalue}`sail/margin.json:gradient_cost.fd_seconds:.1f` s by finite differences
even with every perturbed design sailed in the same vectorized call. The
trajectory's own rows, the channel at every save point, are taken
separately in forward mode, which the differentiation study above found
the faster on the sail. The solve took {docvalue}`sail/margin.json:designs.4.calls:d` iterations,
{docvalue}`sail/margin.json:designs.4.seconds:.0f` s after
{docvalue}`sail/margin.json:designs.4.compile_seconds:.0f` s of compilation.


## References

```{footbibliography}
```
