# Differentiation

Every model is a JAX function, so derivatives come from automatic
differentiation rather than finite differences. Three choices matter: the
mode, the adjoint used to differentiate through an ODE solve, and whether a
transcription can exploit the structure of its own Jacobian.

## Forward or reverse

`DynamicScenario(differentiation_method=...)` selects the mode of the
assembled disciplines, `"reverse"` by default. Reverse mode costs one pass
per output and suits the usual shape of these problems, few objectives and
constraints against many design variables. Forward mode costs one pass per
input and wins when that ratio is inverted.

The split assembly deliberately mixes the two: the statics discipline always
differentiates in forward mode, being a cheap algebraic chain with more
outputs than inputs, while the ODE-bearing disciplines follow the scenario's
choice.

## The adjoint has to agree with the mode

Differentiating through an ODE solve needs an adjoint, and not every adjoint
supports both modes. `RecursiveCheckpoint` and `Backsolve` define only a
reverse rule, so forward-mode differentiation through them fails deep inside
a JAX trace with `can't apply jvp to a custom_vjp function`.

```{list-table}
:header-rows: 1
:widths: 34 33 33

* - Adjoint
  - Forward
  - Reverse
* - `Direct`
  - yes
  - yes
* - `RecursiveCheckpoint` (default)
  - no
  - yes
* - `Backsolve`
  - no
  - yes
```

{func}`~gemseo_tempo.dynamics.integrators.check_adjoint_compatibility`
enforces this table up front, so an incompatible pairing is rejected when the
scenario is built rather than surfacing as an opaque trace error. To
differentiate in forward mode, ask for the Direct adjoint:

```python
DynamicScenario(
    integrator=DiffraxIntegrator(rtol=1e-10, atol=1e-10, adjoint="Direct"),
    differentiation_method="forward",
    ...,
)
```

Indirect shooting has its own `differentiation_method`, which governs the
derivatives *inside* the necessary conditions — the dynamics Jacobian
`df/ds`, the transversality gradient, and the inner stationarity solve. It is
independent of the mode of the assembled discipline. The costate equation is
written as a Jacobian relation,

$$\dot\lambda = -\left(\frac{\partial f}{\partial s}\right)^{\!T}\lambda - \frac{\partial l}{\partial s},$$

precisely so it holds for any number of states and can be evaluated with
either `jacfwd` or `jacrev`.

## Structure-aware Jacobians

A dense transform differentiates every output row by row. That is wasteful
when the transcription already knows most of those rows touch only a few
inputs. Collocation defects are the motivating case: segment *k* involves
only nodes *k* and *k+1*, so the defect Jacobian is a band of small blocks
and one vectorized pass over the segments produces every nonzero entry.

Enable it per transcription:

```python
HermiteSimpson(num_segments=100, sparse_jacobian=True)
```

Measured against a dense `jacrev` on a two-state problem:

```{list-table}
:header-rows: 1
:widths: 25 25 25 25

* - Segments
  - Dense `jacrev`
  - Segment blocks
  - Speedup
* - 20
  - 0.063 ms
  - 0.010 ms
  - 6x
* - 100
  - 0.170 ms
  - 0.012 ms
  - 15x
* - 400
  - 1.730 ms
  - 0.020 ms
  - **85x**
```

The dense cost grows roughly with the square of the grid because each of the
`N * n` defect rows drives its own reverse pass; the blocks stay flat.

This changes only how the derivative is obtained, never the defect values.
The blocks agree with the dense Jacobian to machine precision, and the
assembled discipline Jacobian is identical either way.

Only the outputs a transcription declares are taken from its structured
Jacobian; everything else, such as a scalar objective, still goes through the
ordinary transform, which is cheap for those rows. If a differentiated input
falls outside the declared structure — a design constant, say — the dense
path handles the whole call rather than guessing.

:::{note}
Single shooting has no exploitable sparsity: the ODE couples every control
parameter to the whole trajectory, so its Jacobian is structurally dense.
There the useful lever is the mode, not the structure.
:::
