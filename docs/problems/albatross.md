# Albatross dynamic soaring

```{variable-context} gemseo_tempo.problems.albatross.model:REGISTRY
```

An albatross crosses oceans without flapping by climbing into the wind and
diving back with it, through air that moves faster higher up
{footcite:p}`rayleigh_soaring_1883,sachs_flying_2012`. Each pass takes a
little energy out of the wind gradient, and over a closed cycle it can pay
for the drag exactly. Flying the real bird's wing, that takes a wind of
{docvalue}`albatross/shapes.json:weave.minimum_wind:.2f` m/s at 10 m; with
the wing designed alongside the cycle,
{docvalue}`albatross/design_strategies.json:weave_codesign.minimum_wind:.2f` m/s.

```python
from gemseo_tempo.problems.albatross import create_codesign_optimization

scenario, gemseo_scenario = create_codesign_optimization(size=20, topology="weave")
gemseo_scenario.execute(algo_name="SLSQP", max_iter=500)
```

## Problem

```{eval-rst}
.. plot:: _ext/gallery/albatross_scenario.py
   :include-source: false
```

A schematic, after the opening figures of Hong et al. and Koessler
{footcite:p}`hong_dynamic_2023,koessler_dynamic_2019`, not a result. The
terrestrial frame is fixed to the sea: {var}`x <symbol>` downwind,
{var}`y <symbol>` across the wind, {var}`h <symbol>` up. The wind blows along
+x with a speed W(h) that depends on height only. The airspeed
{var}`V <symbol>`, flight path angle {var}`gamma <symbol>` and heading
{var}`psi <symbol>` are measured in the wind-fixed frame, which moves with
the local air, and the two frames are linked by the wind triangle:

```{math}
:label: eq-alb-triangle

\mathbf v_\mathrm{ground} = \mathbf v_\mathrm{air} + W(h)\,\mathbf e_x .
```

That frame is not inertial: climbing moves the bird into faster air, so it
accelerates at Ẇ = W′(h) V sin γ, and Newton's law written in it gains a
fictitious force −m Ẇ **e**{sub}`x`. That force is what the bird feeds on
{footcite:p}`koessler_dynamic_2019`. One instant of an optimized cycle, taken
apart frame by frame:

```{eval-rst}
.. plot:: _ext/gallery/albatross_frames.py
   :include-source: false
```

**Two families of cycle.** Barnes sorts the cycles an albatross flies into
two families {footcite:p}`barnes_albatross_2004`: an O, a spiral whose
heading gains a full turn per cycle, and an S, a snake whose heading swings
either side of crosswind and back. Both climb while turned into the wind and
descend while turned away from it:

```{eval-rst}
.. plot:: _ext/gallery/albatross_cycle_shapes.py
   :include-source: false
```

**Assumptions.**
1. A point mass without sideslip, on a flat, non-rotating Earth.
2. A quadratic drag polar whose coefficients and stall limit come from the
   wing, through a statics chain.
3. A steady wind along x in a neutral logarithmic boundary layer, with no
   gusts and no updrafts off the waves.
4. Constant mass and air density; the bird flies no lower than 2 m, at a
   load factor of at most 4.5.
5. The cycle repeats: height, airspeed, climb angle and heading return to
   their initial values, the heading up to a full turn for an O.

```{math}
:label: eq-alb-kinematics

\dot x = V\cos\gamma\cos\psi + W(h), \qquad
\dot y = V\cos\gamma\sin\psi, \qquad
\dot h = V\sin\gamma ,
```

```{math}
:label: eq-alb-dynamics

\begin{aligned}
\dot V &= -\frac{D}{m} - g\sin\gamma - \dot W\cos\gamma\cos\psi, \\
\dot\gamma &= \frac{L\cos\mu - m g\cos\gamma}{m V} + \frac{\dot W\sin\gamma\cos\psi}{V}, \\
\dot\psi &= \frac{L\sin\mu}{m V\cos\gamma} + \frac{\dot W\sin\psi}{V\cos\gamma},
\end{aligned}
```

```{math}
:label: eq-alb-aero

L = \tfrac12\rho V^2 S\,C_L, \qquad
D = \tfrac12\rho V^2 S\,\left(C_{D0} + k\,C_L^2\right),
```

with the lift coefficient {var}`cl <symbol>` and the bank
{var}`mu <symbol>` as controls. Differentiating the air-relative energy
e = h + V²/2g along them leaves two terms,

```{math}
:label: eq-alb-energy-rate

\dot e = -\frac{D V}{m g}
- \frac{\mathrm dW}{\mathrm dh}\,\frac{V^2\sin\gamma\cos\gamma\cos\psi}{g} :
```

drag always takes, and the gradient pays whenever sin γ cos ψ < 0, that is
climbing into the wind or descending with it. The wind profile is

```{math}
:label: eq-alb-wind

W(h) = w_\mathrm{ref}\,\frac{\ln(h/h_0)}{\ln(h_\mathrm{ref}/h_0)},
```

with a roughness length h₀ = 0.03 m and h_ref = 10 m, so the gradient is
steepest near the sea, which is why the bird flies low:

```{eval-rst}
.. plot:: _ext/gallery/albatross_wind_profile.py
   :include-source: false
```

The wing reaches the dynamics through a statics chain: the section, the
planform and a Reynolds-number correction give C{sub}`D0`, k and the stall
limit C{sub}`L,max`, and a structural model gives the mass. The default bird
is a wandering albatross: 8.5 kg, 0.65 m² spanning 3.44 m.

```{eval-rst}
.. plot:: _ext/gallery/albatross_polar.py
   :include-source: false
```

Minimizing the wind a closed cycle needs, with the wing free, collocated on
twenty Hermite–Simpson segments, gives the optimizer:

```{formulation} albatross/formulation_codesign.json
```

```{variables} gemseo_tempo.problems.albatross.model:REGISTRY
:columns: name, full_name, unit, tags
```

## Deterministic optimal control

:::{admonition} Takeaway: the deterministic problem
:class: tip

- **Transcription:** collocation, which converges in about fifty iterations
  and seconds. Single shooting is slow and badly conditioned over nine
  seconds of unstable flight, and indirect shooting fails on the active
  path constraints.
- **Discretization:** twenty to forty segments, from several starts for the
  shapes, and co-design started from the sequential wing.
- **Differentiation:** under shooting, forward mode through the dynamics, the
  stage that sets the cost: exact, lean in memory and flat in the number of
  constraints. Finite differences batched into one call cost the same, with
  six digits rather than fifteen. Reverse mode through the checkpointing
  adjoint is faster with many controls but not exact here.
:::

### Variants

One method throughout: collocation on twenty segments, solved by SLSQP from
a cycle built by inverse dynamics, with the real bird's wing. The objective
changes; so does the shape, O or S, through the heading offset of the
periodicity constraint:

| objective | O | S |
| --- | --- | --- |
| weakest wind a cycle closes in | {docvalue}`albatross/shapes.json:loop.minimum_wind:.2f` m/s | {docvalue}`albatross/shapes.json:weave.minimum_wind:.2f` m/s |
| most energy banked, at {docvalue}`albatross/missions.json:w_ref:.0f` m/s | {docvalue}`albatross/missions.json:loop_energy.air_change:+.2f` m | {docvalue}`albatross/missions.json:weave_energy.air_change:+.2f` m |
| furthest downwind in a neutral cycle | {docvalue}`albatross/missions.json:loop_distance.distance:.0f` m | {docvalue}`albatross/missions.json:weave_distance.distance:.0f` m |
| fastest over {docvalue}`albatross/missions.json:target_distance:.0f` m | {docvalue}`albatross/missions.json:loop_time.objective:.2f` s | {docvalue}`albatross/missions.json:weave_time.objective:.2f` s |

The weakest wind is the problem's defining number. For each shape it is the
best of nine starts, the default guess and eight from a Latin hypercube over
period, airspeed, mean height and height swing: the S needs about 3% less
wind than the O, and
{docvalue}`albatross/shapes.json:weave.at_best:d` and
{docvalue}`albatross/shapes.json:loop.at_best:d` unrelated starts land on
the same answers to within 0.1%. Both, re-flown densely:

```{eval-rst}
.. plot:: _ext/gallery/albatross_shapes.py
   :include-source: false
```

The O drifts downwind, the circumnavigation of Barnes's catalogue; the S
snakes across the wind with its heading swinging through
{docvalue}`albatross/shapes.json:weave.heading_swing:.0f*57.29578`°.
Asking either to end where it started, a hover, needs a different cycle:
shortening the passage step by step, the wind needed climbs steeply and the
solver stalls at
{docvalue}`albatross/shapes.json:weave.hover.-1.fraction:.0%` of the drift.

**Verification.** Hong et al. solve the weakest-wind problem for a bird with
a fixed polar {footcite:p}`hong_dynamic_2023`, and their cycle is an S: their
heading closes without a turn. With their bird, polar and limits in place of
the statics chain:

| | Hong et al. (2023), an S | here, an S | here, an O |
| --- | --- | --- | --- |
| minimum wind | {docvalue}`albatross/hong_2023.json:reference.minimum_wind:.2f` m/s | {docvalue}`albatross/hong_2023.json:weave.minimum_wind:.2f` m/s | {docvalue}`albatross/hong_2023.json:loop.minimum_wind:.2f` m/s |
| period | 7.7–9.4 s | {docvalue}`albatross/hong_2023.json:weave.runs.1.period:.2f` s | {docvalue}`albatross/hong_2023.json:loop.runs.1.period:.2f` s |

The S agrees with theirs to
{docvalue}`albatross/hong_2023.json:weave.difference:+.2%`, and the O needs
{docvalue}`albatross/hong_2023.json:loop.difference:+.1%` more, the same
ordering found above. Two further checks close the budget: re-flying the
optimal controls through a tight integrator departs from the collocation
nodes by at most
{docvalue}`albatross/dense_check.json:codesign.departure.x:.1f` m, and the
energy the gradient supplies matches the drag and the change in energy to
integration accuracy:

```{eval-rst}
.. plot:: _ext/gallery/albatross_energy_analysis.py
   :include-source: false
```

### Transcriptions

The weakest-wind O with the real wing, by each transcription, with SLSQP:

```{eval-rst}
.. plot:: _ext/gallery/albatross_transcriptions.py
   :include-source: false
```

| transcription | grid | minimum wind | iterations | wall time |
| --- | --- | --- | --- | --- |
| shooting | 10 | {docvalue}`albatross/transcriptions.json:rows.0.objective:.3f` m/s | {docvalue}`albatross/transcriptions.json:rows.0.calls:d` | {docvalue}`albatross/transcriptions.json:rows.0.seconds:.0f` s |
| shooting | 20 | {docvalue}`albatross/transcriptions.json:rows.1.objective:.3f` m/s | {docvalue}`albatross/transcriptions.json:rows.1.calls:d` | {docvalue}`albatross/transcriptions.json:rows.1.seconds:.0f` s |
| collocation | 20 | {docvalue}`albatross/transcriptions.json:rows.4.objective:.3f` m/s | {docvalue}`albatross/transcriptions.json:rows.4.calls:d` | {docvalue}`albatross/transcriptions.json:rows.4.seconds:.0f` s |
| collocation | 40 | {docvalue}`albatross/transcriptions.json:rows.5.objective:.3f` m/s | {docvalue}`albatross/transcriptions.json:rows.5.calls:d` | {docvalue}`albatross/transcriptions.json:rows.5.seconds:.0f` s |
| indirect | none | not converged | {docvalue}`albatross/transcriptions.json:rows.6.calls:d` | {docvalue}`albatross/transcriptions.json:rows.6.seconds:.0f` s |

Collocation converges in about fifty iterations on every grid, in seconds,
to answers that agree to a fraction of a percent as the grid is refined.
Single shooting integrates nine seconds of unstable flight at every call,
which makes it slow and badly conditioned: on ten intervals it finds a cycle
1% cheaper than collocation's, a different local optimum, but on twenty and
forty it stops at the iteration budget short of both. It also checks the
path constraints only where it saves the trajectory, so they are checked at
a hundred points: at one per interval, the cycle passed through the sea
between them and claimed 1.6 m/s.

Indirect shooting is set up automatically from the collocation optimum:
costates from JAX, the periodicity and the wind as extra conditions, the
path constraints as a penalty, and ten shooting segments to keep the
costates from growing. It does not converge. The lift limit and the height
floor are active over long arcs of the optimal cycle, where the necessary
conditions change form, and the penalty that stands in for them leaves the
residual at {docvalue}`albatross/transcriptions.json:rows.6.residual:.1f`, from
{docvalue}`albatross/transcriptions.json:rows.6.seed_residual:.1f` at the seed. Direct methods treat
those constraints for what they are, which is why collocation is the
default.

### Design strategies

Three ways of arriving at the bird, each flown as an O and as an S: the
real wing with only the cycle optimized; the sequential route, which sizes
the wing for its glide ratio first and only then optimizes the cycle on it;
and co-design, which optimizes both at once, started from the sequential
answer.

| wing | aspect ratio | glide ratio | O | S |
| --- | --- | --- | --- | --- |
| the real bird's | {docvalue}`albatross/design_strategies.json:loop_fixed.aspect_ratio:.1f` | {docvalue}`albatross/design_strategies.json:loop_fixed.glide_ratio:.1f` | {docvalue}`albatross/design_strategies.json:loop_fixed.minimum_wind:.2f` m/s | {docvalue}`albatross/design_strategies.json:weave_fixed.minimum_wind:.2f` m/s |
| sized for glide, then flown | {docvalue}`albatross/design_strategies.json:loop_sequential.aspect_ratio:.1f` | {docvalue}`albatross/design_strategies.json:loop_sequential.glide_ratio:.1f` | {docvalue}`albatross/design_strategies.json:loop_sequential.minimum_wind:.2f` m/s | {docvalue}`albatross/design_strategies.json:weave_sequential.minimum_wind:.2f` m/s |
| co-designed with the cycle | {docvalue}`albatross/design_strategies.json:weave_codesign.aspect_ratio:.1f` | {docvalue}`albatross/design_strategies.json:weave_codesign.glide_ratio:.1f` | {docvalue}`albatross/design_strategies.json:loop_codesign.minimum_wind:.2f` m/s | **{docvalue}`albatross/design_strategies.json:weave_codesign.minimum_wind:.2f` m/s** |

Sizing for glide ratio alone gets most of the way, because the weakest wind
is a cruise-like objective, but it stops short: the best glider is not the
best dynamic soarer, since the cycle also has to turn hard at the bottom.
Co-design finds the wing that balances the two, and the S remains the
cheaper shape whatever the wing. The co-designed O, coloured by the net
rate at which it gains energy:

```{eval-rst}
.. plot:: _ext/gallery/albatross_trajectory.py
   :include-source: false
```

### Differentiation

The energy cycle by single shooting, its Jacobian four ways: by finite
differences; by forward mode; and by reverse mode, once through the Direct
adjoint and once through the checkpointing one, the library's default. The
lift limit, the load limit and the height floor are checked at every save
point, so the outputs grow with the save grid and the inputs with the
control intervals; each is swept with the other held, compiled (solid) and
not (dashed).

Each method is measured at its best. Finite differences are forward
differences at the step, from 10⁻⁴ to 10⁻⁹, that comes closest to the
exact Jacobian, timed both batched into one vectorized call and one
evaluation at a time, the faster counting. Compilation is paid once per
program, and is charged separately, below:

```{eval-rst}
.. plot:: _ext/gallery/albatross_mode_sweep.py
   :include-source: false
```

Against {docvalue}`albatross/differentiation.json:inputs.0.n_outputs:d` outputs, forward mode and
batched finite differences run neck and neck at every size, as on the sail:
{docvalue}`albatross/differentiation.json:inputs.26.seconds:.1f` s and
{docvalue}`albatross/differentiation.json:inputs.25.seconds:.1f` s at
{docvalue}`albatross/differentiation.json:inputs.26.n_inputs:d` inputs. Checkpointed reverse mode
overtakes both from {docvalue}`albatross/differentiation.json:inputs.24.n_inputs:d` inputs, to
{docvalue}`albatross/differentiation.json:inputs.28.seconds:.1f` s at the largest, the crossover the
theory predicts. It is not a crossover to take, though: on these save grids
the checkpointed reverse pass is wrong by
{docvalue}`albatross/differentiation.json:inputs.3.error:.0%` to {docvalue}`albatross/differentiation.json:outputs.10.error:.0%`,
where the exact modes agree to rounding and finite differences to six
digits. Exact reverse mode, through the Direct adjoint, only draws level
with forward mode on the largest grid, and its memory grows to
{docvalue}`albatross/differentiation.json:outputs.20.memory:.0f*1e-6` MB as the outputs do. As the
outputs grow at fixed inputs, forward mode and finite differences stay flat
while both reverse modes grow with every row. The batched methods grow
faster than the inputs, since every evaluation itself slows as the control
grid refines. Compiling:

```{eval-rst}
.. plot:: _ext/gallery/albatross_compile_times.py
   :include-source: false
```

At twenty intervals, each method on the four costs, the best at the rim:

```{eval-rst}
.. plot:: _ext/gallery/albatross_mode_radar.py
   :include-source: false
```

| method | compilation | one Jacobian | error | memory |
| --- | --- | --- | --- | --- |
| finite differences | {docvalue}`albatross/differentiation.json:inputs.14.compile_jacobian:.1f` s | {docvalue}`albatross/differentiation.json:inputs.14.seconds:.2f` s | {docvalue}`albatross/differentiation.json:inputs.14.error:.0e` | {docvalue}`albatross/differentiation.json:inputs.14.memory:.0f*0.001` kB |
| forward | {docvalue}`albatross/differentiation.json:inputs.15.compile_jacobian:.1f` s | {docvalue}`albatross/differentiation.json:inputs.15.seconds:.2f` s | {docvalue}`albatross/differentiation.json:inputs.15.error:.0e` | {docvalue}`albatross/differentiation.json:inputs.15.memory:.0f*0.001` kB |
| reverse | {docvalue}`albatross/differentiation.json:inputs.16.compile_jacobian:.1f` s | {docvalue}`albatross/differentiation.json:inputs.16.seconds:.2f` s | {docvalue}`albatross/differentiation.json:inputs.16.error:.0e` | {docvalue}`albatross/differentiation.json:inputs.16.memory:.0f*0.001` kB |
| reverse, checkpointed | {docvalue}`albatross/differentiation.json:inputs.17.compile_jacobian:.1f` s | {docvalue}`albatross/differentiation.json:inputs.17.seconds:.2f` s | {docvalue}`albatross/differentiation.json:inputs.17.error:.0e` | {docvalue}`albatross/differentiation.json:inputs.17.memory:.0f*0.001` kB |

**A mode per stage.** With the split assembly, the cycle is a chain of
three stages, each with its own shape: the statics turn the wing into
aerodynamic constants, a {docvalue}`albatross/chain_modes.json:chains.0.statics_shape.0:d` by
{docvalue}`albatross/chain_modes.json:chains.0.statics_shape.1:d` Jacobian; the dynamics fly the cycle,
{docvalue}`albatross/chain_modes.json:chains.0.dynamics_shape.0:d` trajectory values from
{docvalue}`albatross/chain_modes.json:chains.0.dynamics_shape.1:d` controls and constants; and the outputs
read {docvalue}`albatross/chain_modes.json:chains.0.outputs_shape.1:d` of those into
{docvalue}`albatross/chain_modes.json:chains.0.outputs_shape.0:d` constraints. Each stage differentiated in
either mode, all eight combinations, the chain Jacobian their product,
labelled statics · dynamics · outputs:

```{eval-rst}
.. plot:: _ext/gallery/albatross_chain_modes.py
   :include-source: false
```

Every combination gives the same Jacobian, to
{docvalue}`albatross/chain_modes.json:chains.2.error:.0e`. The one choice that matters is the dynamics':
forward, {docvalue}`albatross/chain_modes.json:chains.0.dynamics_seconds:.1f` s and
{docvalue}`albatross/chain_modes.json:chains.0.dynamics_memory:.0f*0.001` kB, against
{docvalue}`albatross/chain_modes.json:chains.2.dynamics_seconds:.1f` s and
{docvalue}`albatross/chain_modes.json:chains.2.dynamics_memory:.1f*1e-6` MB in reverse, since it has seven
times more outputs than inputs. The statics and the outputs cost a
millisecond or less either way, so their mode is free, whatever the
reverse-for-the-design intuition suggests. The rule of inputs against
outputs applies stage by stage, not to the chain as a whole. Finite
differences of the whole chain, batched into one call, take
{docvalue}`albatross/chain_modes.json:fd.seconds:.1f` s, as long as the forward chains,
for six digits rather than fifteen.

## Uncertainty propagation

:::{admonition} Takeaway: uncertainty propagation
:class: tip

- **Transcription:** single shooting, so that the frozen cycle responds to
  the wind, started from a forty-segment collocation optimum. From the
  default guess, or from twenty segments, shooting does not converge.
- **Sampling:** vectorized with `vmap`.
- **Estimator:** first-order Taylor, two model calls.
- **Differentiation:** finite differences, batched into one call: with two
  uncertain inputs they serve as well as exact derivatives, and are the
  quicker.
:::

The wind at 10 m is forecast at 22 m/s and known to 10%, and the roughness
of the sea, which sets how fast the wind grows with height, is log-normal
about 3 cm. The cycle that banks the most energy at the forecast, the energy
mission above, is flown here by single shooting on forty intervals, started
from the collocation answer, so that a frozen cycle responds to the wind. It ends
with an energy height of {docvalue}`albatross/propagation.json:objective:.1f*-1` m at the
forecast and {docvalue}`albatross/propagation.json:reference.mean:.1f` ±
{docvalue}`albatross/propagation.json:reference.standard_deviation:.1f` m over the uncertainty.
Each way of estimating that, against 20 000 draws, where AD marks exact
derivatives by reverse-mode automatic differentiation and FD central
differences, every perturbed point in the same vectorized call, at the
step closest to the exact derivative at the mean:

| method | error, mean | error, deviation | model calls | time |
| --- | --- | --- | --- | --- |
| sampling, 256 draws | {docvalue}`albatross/propagation.json:estimates.sampling_vmapped.mean_error:+.0e` σ | {docvalue}`albatross/propagation.json:estimates.sampling_vmapped.deviation_error:+.0e` | {docvalue}`albatross/propagation.json:estimates.sampling_vmapped.model_calls:d` | {docvalue}`albatross/propagation.json:estimates.sampling_vmapped.seconds:.3f` s, looped {docvalue}`albatross/propagation.json:estimates.sampling_looped.seconds:.2f` s |
| Taylor, 1st order, AD | {docvalue}`albatross/propagation.json:estimates.taylor1_ad.mean_error:+.0e` σ | {docvalue}`albatross/propagation.json:estimates.taylor1_ad.deviation_error:+.0e` | {docvalue}`albatross/propagation.json:estimates.taylor1_ad.model_calls:d` | {docvalue}`albatross/propagation.json:estimates.taylor1_ad.seconds:.3f` s |
| Taylor, 1st order, FD | {docvalue}`albatross/propagation.json:estimates.taylor1_fd.mean_error:+.0e` σ | {docvalue}`albatross/propagation.json:estimates.taylor1_fd.deviation_error:+.0e` | {docvalue}`albatross/propagation.json:estimates.taylor1_fd.model_calls:d` | {docvalue}`albatross/propagation.json:estimates.taylor1_fd.seconds:.3f` s |
| Taylor, 2nd order, AD | {docvalue}`albatross/propagation.json:estimates.taylor2_ad.mean_error:+.0e` σ | {docvalue}`albatross/propagation.json:estimates.taylor2_ad.deviation_error:+.0e` | {docvalue}`albatross/propagation.json:estimates.taylor2_ad.model_calls:d` | {docvalue}`albatross/propagation.json:estimates.taylor2_ad.seconds:.3f` s |
| Taylor, 2nd order, FD | {docvalue}`albatross/propagation.json:estimates.taylor2_fd.mean_error:+.0e` σ | {docvalue}`albatross/propagation.json:estimates.taylor2_fd.deviation_error:+.0e` | {docvalue}`albatross/propagation.json:estimates.taylor2_fd.model_calls:d` | {docvalue}`albatross/propagation.json:estimates.taylor2_fd.seconds:.3f` s |
| chaos, 12 points | {docvalue}`albatross/propagation.json:estimates.chaos.mean_error:+.0e` σ | {docvalue}`albatross/propagation.json:estimates.chaos.deviation_error:+.0e` | {docvalue}`albatross/propagation.json:estimates.chaos.model_calls:d` | {docvalue}`albatross/propagation.json:estimates.chaos.seconds:.3f` s |
| chaos + AD gradients | {docvalue}`albatross/propagation.json:estimates.chaos_ad.mean_error:+.0e` σ | {docvalue}`albatross/propagation.json:estimates.chaos_ad.deviation_error:+.0e` | {docvalue}`albatross/propagation.json:estimates.chaos_ad.model_calls:d` | {docvalue}`albatross/propagation.json:estimates.chaos_ad.seconds:.3f` s |
| chaos + FD gradients | {docvalue}`albatross/propagation.json:estimates.chaos_fd.mean_error:+.0e` σ | {docvalue}`albatross/propagation.json:estimates.chaos_fd.deviation_error:+.0e` | {docvalue}`albatross/propagation.json:estimates.chaos_fd.model_calls:d` | {docvalue}`albatross/propagation.json:estimates.chaos_fd.seconds:.3f` s |

The energy banked is nearly linear in the wind and the roughness over their
ranges, so every method lands within a few hundredths of a deviation of the
reference, and first-order Taylor, one evaluation and one gradient, is as
good as any. Vectorizing pays most again: the 256 draws take
{docvalue}`albatross/propagation.json:estimates.sampling_vmapped.seconds:.1f` s in one `vmap`
dispatch against {docvalue}`albatross/propagation.json:estimates.sampling_looped.seconds:.1f` s
in a loop. With two uncertain inputs, finite differences need four
evaluations per gradient and are cheaper than a reverse pass through nine
seconds of flight: exact derivatives with respect to two inputs are not what
this problem is short of. The gradient that matters is the one with respect
to the cycle's controls, which the next section needs.

## Optimization under uncertainty

:::{admonition} Takeaway: optimization under uncertainty
:class: tip

- **Formulation:** the mean less twice the deviation of the energy,
  started from the deterministic cycle.
- **Sampling:** one fixed sample of
  {docvalue}`albatross/robust.json:design_sample:d` draws, checked on
  {docvalue}`albatross/robust.json:check_sample:d` fresh ones.
- **Differentiation:** automatic, with the Jacobian taken group by group:
  the constraints on the forecast flight in forward mode, the statistic
  through the sample in reverse mode, both through the Direct adjoint. One
  gradient costs a seventh of what finite differences do, even batched
  into one call.
:::

The robust cycle maximizes the mean energy less twice its deviation over
{docvalue}`albatross/robust.json:design_sample:d` draws of the wind and the sea, starting
from the deterministic one. Both, flown on
{docvalue}`albatross/robust.json:check_sample:d` fresh draws:

| cycle | at the forecast | mean | deviation | 5th percentile |
| --- | --- | --- | --- | --- |
| deterministic | {docvalue}`albatross/robust.json:designs.deterministic.nominal:.2f` m | {docvalue}`albatross/robust.json:designs.deterministic.mean:.2f` m | {docvalue}`albatross/robust.json:designs.deterministic.std:.2f` m | {docvalue}`albatross/robust.json:designs.deterministic.p05:.2f` m |
| robust, mean − 2σ | {docvalue}`albatross/robust.json:designs.robust.nominal:.2f` m | {docvalue}`albatross/robust.json:designs.robust.mean:.2f` m | {docvalue}`albatross/robust.json:designs.robust.std:.2f` m | {docvalue}`albatross/robust.json:designs.robust.p05:.2f` m |

The robust cycle gives up most of a metre at the forecast and a quarter of
one on average, and in exchange its spread falls by a third and its worst
cycle in twenty banks almost two metres more.

Here the gradient is the bottleneck. Every iteration differentiates the
mean and deviation of {docvalue}`albatross/robust.json:design_sample:d` nine-second flights
with respect to all {docvalue}`albatross/robust.json:gradient_cost.n_design:d` controls:
one gradient takes {docvalue}`albatross/robust.json:gradient_cost.ad_seconds:.1f` s by AD,
one reverse pass through the whole batch, against
{docvalue}`albatross/robust.json:gradient_cost.fd_seconds:.1f` s by finite differences, even
with every perturbed cycle flown in the same vectorized call. With AD the
robust solve ran its {docvalue}`albatross/robust.json:designs.robust.calls:d` iterations, its
budget, in {docvalue}`albatross/robust.json:designs.robust.seconds:.0f*0.016666666666666666` min.
How the Jacobian is split matters as much:
{class}`~gemseo_tempo.uncertainty.robust.GroupedJAXDiscipline`
differentiates the cycle's three hundred constraint rows on the forecast
flight alone, in forward mode, eighty-two tangents rather than three
hundred cotangents, and only the statistic through the sample, in reverse
mode. Taken together, every constraint row would be back-propagated through
all {docvalue}`albatross/robust.json:design_sample:d` flights. One integrator serves both
groups, and the Direct adjoint is the better one for the pair: it lets the
constraints go forward, at a quarter of what reverse mode costs them, and
costs the statistic half as much again as the checkpointing adjoint would.


## References

```{footbibliography}
```
