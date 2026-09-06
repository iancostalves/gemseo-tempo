# Choosing a transcription

All three transcriptions solve the same optimal-control problem and are
interchangeable through the `transcription` argument of
{class}`~gemseo_tempo.scenario.dynamic_scenario.DynamicScenario`. They differ
in what the optimizer sees, and that changes how they behave.

```{list-table}
:header-rows: 1
:widths: 18 27 27 28

* -
  - Single shooting
  - Hermite-Simpson collocation
  - Indirect shooting
* - Design variables
  - Control parameters (+ `t_f`, free initial states)
  - States and controls at every node (+ `t_f`)
  - Initial costates (+ `t_f`)
* - Problem size
  - Small, independent of the save grid
  - Grows with the number of segments
  - Smallest: one costate per state
* - Constraints
  - Terminal and path only
  - Terminal, path, plus one defect per state per segment
  - None; residuals form a least-squares objective
* - Needs an ODE solve
  - Yes
  - No
  - Yes, on the augmented state-costate system
* - Path constraints
  - Yes
  - Yes, at nodes and midpoints
  - **No** — they change the necessary conditions
```

## Which one to use

The short answer, from the evidence gathered across the three problems
rather than from first principles:

| if | use | because |
| --- | --- | --- |
| the problem has a path constraint | shooting or collocation | the indirect method cannot express one without changing the necessary conditions |
| the trajectory must respond to a parameter you will vary afterwards | **shooting** | under collocation the states are design variables, so a frozen design cannot react to a changed parameter at all |
| the grid is fine and the dynamics are stiff | **collocation** | no ODE solve, and the defects spread the difficulty over the nodes |
| you want the smallest possible problem and the physics are smooth | indirect | one costate per state, and it is the only one that returns a certificate of optimality |
| you are about to sample or optimize under uncertainty | **shooting** | see the first two rows: the whole of {doc}`../guides/uncertainty` depends on it |

The second row is the one that surprises people, and it cost this library a
wasted afternoon before it was written down. Collocation treats the states as
design variables constrained by defects. Freeze such a design and change a
parameter, and the trajectory does not move: only the defects go out of
balance. Every uncertainty study here therefore uses single shooting, not
because it is more accurate but because it is the one whose answer depends on
the parameter being varied.

## Single shooting

The optimizer sees only the control parameters, so the problem stays small
whatever the save grid. The integrator does the work, and the sensitivity of
the final state to an early control parameter passes through the whole
trajectory, which makes the Jacobian dense and can make long horizons
ill-conditioned.

Reach for it first. It is the most forgiving of a poor initial guess: on the
brachistochrone it converges from a crude guess in a few dozen discipline
calls.

## Hermite-Simpson collocation

States and controls at the segment nodes become design variables, and Simpson
defects enforce the dynamics. There is no ODE solve at all: the whole defect
vector is one jitted function.

The problem is much larger, but it is sparse and each iteration is cheap, and
long horizons condition better than shooting because no sensitivity has to
propagate across the whole trajectory.

Two settings matter:

- `defect_ref` scales each defect toward unit magnitude. Defect conditioning
  drives SQP behaviour.
- `sparse_jacobian=True` computes the defect Jacobian from per-segment blocks
  instead of a dense transform. See {doc}`differentiation`.

:::{important}
Collocation needs an initial guess that satisfies the dynamics, because it
carries node states of its own. Feeding it a linear ramp between boundary
states is the single most common way to make it fail — see
{doc}`../guides/scenario`.
:::

## Indirect shooting

The Pontryagin necessary conditions turn the problem into a boundary-value
problem in the costates. The optimizer sees only one costate per state, which
makes it by far the smallest problem, and the solution is exact rather than
discretized, so it is the natural high-accuracy cross-check for the direct
methods.

The costs are real. State path constraints change the necessary conditions
and are out of scope, control bounds are honoured only through smooth
saturation, and initial costates have no physical scale, so a guess is hard
to make and multi-start is usually needed:

```python
gemseo_scenario.execute(
    algo_name="MultiStart", n_start=10,
    opt_algo_name="L-BFGS-B", doe_algo_name="LHS",
)
```

Note that the objective here is the residual norm, not the first functional:
the functionals define the problem that shapes the Hamiltonian, while the
optimizer drives the boundary-condition residuals to zero. The achieved cost
is reported separately as `ocp_cost`.

## What the benchmark shows

The {doc}`benchmark studies <../guides/benchmarking>` compare these on the
brachistochrone over sampled starting points and grid sizes. Two results are
worth stating up front:

- Given the same simulated initial guess, shooting and collocation both reach
  the cycloid time exactly, collocation in 30 to 72 discipline calls at 10 to
  100 segments.
- Gradient-free optimization does not scale here. On collocation, COBYLA
  solved **8%** of starting points within 200 calls against **67%** for
  SLSQP.

The harder problems separate them further, and not always in collocation's
favour:

| problem | shooting | collocation |
| --- | --- | --- |
| brachistochrone | the cycloid time, exactly | the cycloid time, exactly |
| sail, downwind | 158.81 s, 0.26% above the ideal | 161.71 s, 2.09% above |
| sail, upwind | 256.37 s at 25 nodes, 343.67 s at 40 | 300.13 s at 25, 258.05 s at 40 |

On the sail's downwind leg collocation finds a **different and worse local
optimum**: it gybes narrowly, 102 m across the course where shooting goes
189 m, and adding nodes does not help. That gap opened when the polar was
smoothed, and the reason is instructive. The old polar had a corner at dead
downwind with a slope of 0.0027 per degree either side, which pushed a
trajectory off the wind axis; without it the region is flat to first order,
the best gybing angle is less sharply determined, and collocation settles for
a narrower one. The corner had been making the problem easier by being wrong.

Neither transcription solves the upwind leg reliably, and that is reported
rather than tuned away: a tack has to cross head to wind, where the polar
says the boat stops, and one smooth heading over a whole leg makes that
crossing slow.

Compiling is worth it in every case measured. The crossover, the number of
evaluations after which the compiled discipline has repaid the compilation,
is between **3 and 9** on all three problems, so any optimization at all
pays for the compiler.
