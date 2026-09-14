# Optimization under uncertainty

```{variable-context} gemseo_tempo.problems.sail.model:REGISTRY
```

## What a safety margin costs

The argument, on the downwind leg with a 150 m channel the boat has to stay
inside, under an uncertain wind. Every design is measured on 4000 draws it
was not optimized against, and every design is warm-started from the
unconstrained optimum, since started cold from the polar-derived guess the
constrained solves stall in a poorer basin and the comparison would be
measuring the starting point.

| design | time | widest | reliability | 95th pct | worst |
| --- | --- | --- | --- | --- | --- |
| deterministic | 160.24 s | 150.0 m | **53.7%** | +53.0 m | +120.5 m |
| margin 25 m | 162.76 s | 125.0 m | 82.4% | +20.9 m | +78.2 m |
| margin 50 m | 161.72 s | 100.0 m | 97.0% | -8.2 m | +42.7 m |
| margin 75 m | 160.72 s | 75.0 m | 99.9% | -35.2 m | +6.7 m |
| **P(inside) >= 95%** | **160.72 s** | 75.6 m | **99.9%** | -34.6 m | +7.6 m |

The deterministic design satisfies its constraint and sits exactly on it, so
an uncertain wind makes it a **coin flip**: it leaves the channel 46.3% of the
time. That is not a flaw in the solve, it is what a constraint enforced at one
set of parameter values means.

The margin column is the conventional repair, and it shows the blindness a
margin carries: a designer picking 25 m of margin pays 2.5 s and buys 82.4%.
Picking 75 m pays **less** time, 1.0 s cheaper, and buys 99.9%. Nothing
available to them at the time of choosing says which they are getting,
because the margin column is **not monotone** in time, and it has to be,
since a looser channel's feasible set contains a tighter one's — the 25 m
entry is a poor local optimum, and the margin approach is being flattered
nowhere and slandered there.

Putting the probability in the problem instead **dominates the 50 m margin
outright**: at 160.72 s and 99.9%, it is both faster and more reliable. Against
the largest margin tried, 75 m, the two approaches land in essentially the
same place here — 160.72 s either way, at 99.9% reliability by margin and
99.9% by carrying the probability directly. That the two coincide this
closely is itself informative: it says a large enough margin, chosen by
trial, can reach what carrying the probability reaches by construction, on
this particular channel and target. What a margin cannot do, and what this
comparison does not change, is say *how large is enough* before the fact;
carrying the probability answers that by asking for a reliability directly
rather than guessing at a distance. The probabilistic design also
overshoots its own target, reaching 99.9% where 95% was asked.

See {doc}`../../guides/uncertainty` for how `robust_discipline` builds the
constraint this design was found under.

## UQ strategies comparison

`gemseo_tempo.uncertainty.estimators` estimates the mean and standard
deviation of one quantity several ways and scores them against a large
sample. On the downwind leg, with the along-track arrival error as the
quantity and the wind strength and veer as the uncertainties, against
50 000 samples taking 40.4 s:

| estimator | model calls | seconds | mean error | std error |
| --- | --- | --- | --- | --- |
| sampling, vmapped | 512 | 0.46 | +0.9% | -2.1% |
| sampling, looped | 512 | 22.81 | +0.9% | -2.1% |
| Taylor, first order | 2 | 0.28 | +2.4% | +1.1% |
| Taylor, second order | 6 | 0.54 | -3.9% | +1.6% |
| chaos, 32 points | 32 | 0.07 | +0.5% | -0.3% |
| chaos + gradients, 32 | 64 | 0.45 | +0.6% | -0.3% |
| chaos, 12 points | 12 | 0.05 | -2.6% | +0.2% |
| chaos + gradients, 12 | 24 | 0.35 | -1.6% | -0.4% |

Errors are relative to the reference **spread**, the mean included. A converged
design misses its target by little on average, so the reference mean is near
zero, and dividing an error by it turns a small absolute error into an enormous
relative one that says nothing about the estimator.

What the table says:

- The two sampling rows are one estimator on identical points, differing only in
  dispatch. The 50x between them is the whole claim of the batch evaluator,
  measured on a real analysis rather than a synthetic loop.
- **First-order Taylor gets the spread to 1.1% for two model calls**, beating
  512 samples: the cost is about twice a deterministic evaluation when the
  derivatives are exact, and here they are, by construction.
- Second order is *worse* than first on the mean here. The curvature correction
  overshoots. It is reported rather than dressed up.
- Chaos on 12 points costs 0.05 s and gets the spread to 0.2%. Gradient
  enhancement earns its keep where points are scarce, roughly halving the error
  of the mean at 12 points, and adds little at 32.

A warning drawn from getting this wrong first, in the adjoint fault below.
With a subtly incorrect Jacobian, first-order Taylor came out 17% wrong on
the spread, and that looked exactly like the expected failure of a
first-order method on a nonlinear response. It was not. An estimator that
depends on derivatives is only as good as the derivatives, and a
plausible-looking failure is the most expensive kind.
