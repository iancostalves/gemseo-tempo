# Optimization

```{variable-context} gemseo_tempo.problems.sail.model:REGISTRY
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

The difficulty is the physics rather than the numerics. Beating upwind
means alternating between port and starboard tack at the best upwind angle,
and a tack **crosses the no-go zone**, where the boat is briefly not
propelled at all and carries through on its own momentum
{footcite:p}`ciccarella_optimal_2024`. A single smooth heading over the
whole leg has to make that crossing slowly, so it pays the no-go zone's
price at every tack instead of passing through it, and the count of tacks
becomes part of the answer. That is what makes the upwind leg a different
shape from the downwind one rather than a harder instance of it. The
stochastic formulation in {footcite:t}`ciccarella_optimal_2024` makes it
explicit by charging a fixed cost per tack and optimizing over the
switching times, which is the structure this library would need to add to
solve the leg from any start.

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

## References

```{footbibliography}
```
