# Sailing to a mark

```{variable-context} gemseo_tempo.problems.sail.model:REGISTRY
```

Minimum-time sailing to a mark 800 m away, inside a race area that stretches
500 m either side of the direct line. This is the first problem in the
library with a genuine **path constraint**, and the first whose interesting
answer is qualitative: the optimizer is never told to gybe or to tack, and
does both. Downwind it arrives in
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

A boat cannot sail straight at a mark that lies in its **no-go zone**, and
gains by not sailing straight at one that lies dead downwind either: it
reaches off at an angle and gybes, trading distance for enough extra speed
to arrive sooner. That is the classical structure of the problem
{footcite:p}`ciccarella_optimal_2024`, and nothing here tells the optimizer
about it. It is handed a polar, a wind field and a mark, and the zigzag is
what comes back.

With the boat a point moving at its polar speed along the heading
{var}`psi <symbol>`, {eq}`sail-kinematics` on {doc}`physics`, the leg is

```{math}
:label: eq-sail-ocp

\min_{\psi(\cdot),\,t_f}\; t_f
\quad\text{s.t.}\quad
(x, y)(0) = 0, \qquad (x, y)(t_f) = (x_m, y_m), \qquad
|y(t)| \le b ,
```

with b = 500 m for the race area. Solved by single shooting on 25
intervals, the problem the optimizer sees is:

```{formulation} sail/formulation.json
```

The pages that follow read in order:

```{toctree}
:maxdepth: 1

physics
optimization
uncertainty
```

- {doc}`physics`: the points of sail, the velocity triangle, where the
  polar comes from, and what its smoothness costs.
- {doc}`optimization`: the downwind and upwind legs, the race area as a
  constraint that binds, each transcription, and each way of
  differentiating.
- {doc}`uncertainty`: what a wind that is not the forecast one does to the
  planned track, and a design that stays inside the race area anyway.

## References

```{footbibliography}
```
