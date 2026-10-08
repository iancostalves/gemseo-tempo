# Brachistochrone

```{variable-context} gemseo_tempo.problems.brachistochrone.model:REGISTRY
```

The fastest wire between two points for a bead sliding under gravity
{footcite:p}`bernoulli_problema_1696`, in three physical variants. It is
the smallest problem in the library and the one this documentation uses as
a guided tour: every capability shown here reappears, at scale, on the sail
and albatross pages.

```python
from gemseo_tempo.problems.brachistochrone import create_optimization

scenario, gemseo_scenario = create_optimization(variant="ideal", size=20)
gemseo_scenario.execute(algo_name="SLSQP", max_iter=200)
```

## Problem

Frictionless and in vacuum, the answer is the cycloid and is known in
closed form: the descent takes
{docvalue}`brachistochrone/transcriptions.json:reference:.6f` s. That makes
this the one problem here whose optimizer can be marked against a right
answer rather than against itself. The other two variants switch on one
force each, a bead that rubs and a bead in air, so that what each costs
can be read separately.

The final time is free, so the dynamics of {doc}`physics` run on
τ = t / t_f ∈ [0, 1] and {var}`t_f <symbol>` is a design variable beside
the path angle {var}`gamma <symbol>`:

```{math}
:label: eq-brach-ocp

\min_{\gamma(\cdot),\,t_f}\; t_f
\quad\text{s.t.}\quad
(x, y, v)(0) = 0, \qquad (x, y)(t_f) = (2, 2)\ \text{m},
\qquad 0 \le \gamma \le \tfrac{\pi}{2}.
```

Collocated on twenty Hermite–Simpson segments
{footcite:p}`hargraves_direct_1987`, the problem the optimizer sees is:

```{formulation} brachistochrone/formulation.json
```

The pages that follow read in order:

```{toctree}
:maxdepth: 1

physics
optimization
uncertainty
```

- {doc}`physics`: the model, and why each force is written the way it is.
- {doc}`optimization`: the deterministic optimum of each variant, how each
  transcription reaches it, and what each way of differentiating costs.
- {doc}`uncertainty`: what uncertain friction, drag and gravity do to the
  descent, and the wire that accounts for them.

## References

```{footbibliography}
```
