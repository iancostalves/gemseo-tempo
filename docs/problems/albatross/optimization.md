# Optimization

```{variable-context} gemseo_tempo.problems.albatross.model:REGISTRY
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

## References

```{footbibliography}
```
