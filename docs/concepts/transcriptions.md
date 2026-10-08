# Choosing a transcription

## The optimal control problem

Every problem in this library is an instance of the same thing: choose a
control trajectory that minimizes a cost, subject to dynamics that must
hold at every instant and to boundary and path conditions.

```{math}
:label: control-problem
\begin{split}
\min_{\mathbf{u}(t) \in \mathcal{U}}\ &f_{obj} = \phi(\mathbf{s}(t_f)) +
  \int_{t_0}^{t_f} \ell(\mathbf{s}(t), \mathbf{u}(t), t)\,dt \\
\text{where:}\ & \dot{\mathbf{s}}(t) = \mathbf{f}(t, \mathbf{s}(t),
  \mathbf{u}(t), \theta), \quad \mathbf{s}(t_0) = \mathbf{s}_0 \\
\text{subject to:}\ & \mathbf{g}(\mathbf{s}, \mathbf{u}) = \mathbf{0},
  \quad \mathbf{h}(\mathbf{s}, \mathbf{u}) \le \mathbf{0}, \quad
  \mathbf{u}_{\text{low}} \le \mathbf{u} \le \mathbf{u}_{\text{up}}
\end{split}
```

In this library's own names: `f` is a
{class}`~gemseo_tempo.dynamics.state_space.StateSpace`'s `rhs`, `phi` and
`l` are a {class}`~gemseo_tempo.outputs.functionals.Mayer` and a
{class}`~gemseo_tempo.outputs.functionals.Lagrange`, `h` is a
{class}`~gemseo_tempo.outputs.constraints.PathConstraint`, and `g` is a
{class}`~gemseo_tempo.outputs.constraints.TerminalConstraint` or a
{class}`~gemseo_tempo.outputs.constraints.PeriodicityConstraint`. See
{doc}`../guides/scenario` for how each is declared.

## Two families

Two families of methods solve {eq}`control-problem`, and they differ in the
order in which they optimize and discretize.

**Indirect methods** operate under the philosophy of *optimize then
discretize*. The Calculus of Variations and Pontryagin's Minimum Principle
derive first-order necessary conditions analytically: the dynamics are
adjoined to the cost through a costate `lambda(t)`, the same role a
Lagrange multiplier plays below, and stationarity gives a two-point
boundary value problem — the state is known at the start and integrates
forward, the costate is known at the end (the transversality condition)
and integrates backward. This library's
{class}`~gemseo_tempo.transcriptions.IndirectShooting` solves that
boundary value problem by having an ordinary optimizer drive the
mismatched end to zero.

**Direct methods** instead discretize the trajectory first and hand a
finite-dimensional nonlinear program (NLP) to a general-purpose optimizer:
*discretize then optimize*. Both derive from Pontryagin's own formulation
of the problem {cite}`pontrjagin_mathematical_1964`. They have gained the wider following, for
their flexibility, and they differ among themselves in how much of the
trajectory is promoted to a decision variable:

- **single shooting** parameterizes only the control and recovers the
  state by one forward integration from `s0`, so the optimizer sees the
  control alone — {class}`~gemseo_tempo.transcriptions.SingleShooting`;
- **multiple shooting** splits the horizon into segments integrated
  independently, adding the inter-segment states as variables and
  *defect* constraints that stitch the segments together, improving
  conditioning and parallelization — not implemented here, since this
  library commits to whole-horizon integration (see
  {doc}`architecture`), and the two remedies below cover the cases that
  motivate it;
- **direct collocation** promotes every state and control node to a
  variable and enforces the dynamics through algebraic defect constraints
  at interior points, with no explicit time integration
  {cite}`hargraves_direct_1987` —
  {class}`~gemseo_tempo.transcriptions.HermiteSimpson`.

```{eval-rst}
.. plot::
   :include-source: false

   from tempo_schemes import plot_transcription_scheme
   fig = plot_transcription_scheme()
```

Single shooting and collocation, from the optimizer's point of view. Top:
how the control is held. Middle: how the state is held — one variable at
the start and a computed value at the end for shooting, every node a
variable tied down by defects at the collocation points for collocation.
Bottom: the dataflow between the optimizer, the trajectory or the
defects, and the objective. Moving from shooting to collocation the
problem grows in dimension but becomes sparser and better conditioned.

## The KKT system

At a constrained optimum the objective gradient no longer vanishes; it
balances against the active constraints through the Karush-Kuhn-Tucker
conditions.

```{math}
:label: kkt
\begin{split}
\mathcal{L}(\mathbf{x}, \lambda, \mu) &= f_{obj}(\mathbf{x}) +
  \lambda^\top \mathbf{g}(\mathbf{x}) + \mu^\top \mathbf{h}(\mathbf{x}) \\
\nabla f_{obj} + \nabla\mathbf{g}\,\lambda + \nabla\mathbf{h}\,\mu &=
  \mathbf{0}, \qquad \mu \ge \mathbf{0}, \quad \mu_i h_i = 0
\end{split}
```

Each multiplier measures how much the optimum would improve if its
constraint were relaxed. SLSQP {cite}`kraft1988software`, this library's
default, works by solving a sequence of local quadratic approximations to
{eq}`kkt`; every constraint class in {doc}`../guides/scenario` exists to
be one of `g` or `h` here.

## Why collocation is usually chosen — and what that argument assumes

The received comparison — Betts's reference text on practical optimal
control {cite}`betts_practical_2010`, and the documentation of tools such
as [Dymos](https://openmdao.github.io/dymos/) {cite}`falck_dymos_2021` —
favours collocation over shooting on the grounds sketched above:
sparsity, conditioning, parallelizability. That
comparison is drawn in a setting where gradients come from **finite
differences**, or where the derivative of the time-stepping chain has to
be implemented by hand. There, a shooting gradient costs about one extra
forward integration *per optimization variable*, and the sequential chain
also accumulates each step's truncation error into the sensitivity — the
penalty against shooting is both computational and numerical.

Automatic differentiation changes the terms of that comparison. With AD
through the integrator, the same sequential chain differentiates exactly
at a cost of roughly **one extra pass regardless of the number of
variables** ({doc}`differentiation`), and just-in-time compilation
amortizes the repeated evaluation the optimizer requires — see the AD
benchmarks on {doc}`../problems/brachistochrone/optimization`, which measure exactly
this crossover. What remains of collocation's advantage is narrower but
genuine: the sparsity of the constraint Jacobian, which a sparse optimizer
can exploit, and the conditioning of long or stiff horizons, where the
sensitivity of the terminal state to the initial one grows exponentially
with the horizon length. AD returns that ill-conditioned derivative
exactly; it does not make it well conditioned.

## Which one to use

| if | use | because |
| --- | --- | --- |
| the problem has a path constraint | shooting or collocation | the indirect method cannot express one without changing the necessary conditions |
| the trajectory must respond to a parameter you will vary afterwards | **shooting** | under collocation the states are design variables, so a frozen design cannot react to a changed parameter at all — see {doc}`../guides/uncertainty` |
| the grid is fine and the dynamics are stiff | **collocation** | no ODE solve, and the defects spread the difficulty over the nodes |
| you want the smallest possible problem and the physics are smooth | indirect | one costate per state, and it is the only one that returns a certificate of optimality |

The second row cost this library a wasted afternoon before it was written
down. Collocation treats the states as design variables constrained by
defects; freeze such a design and change a parameter, and the trajectory
does not move — only the defects go out of balance. Every uncertainty
study in this documentation therefore uses single shooting, not because
it is more accurate but because it is the one whose answer depends on the
parameter being varied.

## Indirect shooting in this library

The costs of the indirect method are real, beyond what the table above
says. State path constraints change the necessary conditions and are out
of scope; control bounds are honoured only through smooth saturation; and
initial costates have no physical scale, so a guess is hard to make and
multi-start is usually needed:

```python
gemseo_scenario.execute(
    algo_name="MultiStart", n_start=10,
    opt_algo_name="L-BFGS-B", doe_algo_name="LHS",
)
```

The objective reported is also not the first functional: the functionals
define the problem that shapes the Hamiltonian, while the optimizer drives
the boundary-condition residuals to zero, and the achieved cost is
reported separately as `ocp_cost`.
