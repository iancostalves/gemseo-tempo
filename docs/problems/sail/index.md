# Sailing to a mark

Minimum-time sailing to a mark through a steady, spatially varying wind.
This is the first problem in the library with a genuine **path
constraint**, and the first whose interesting answer is qualitative: the
optimizer is never told to gybe, and gybes anyway.

```python
from gemseo_tempo.problems.sail import create_optimization

scenario, gemseo_scenario = create_optimization(leg="downwind", size=25)
gemseo_scenario.execute(algo_name="SLSQP", max_iter=300)
print(gemseo_scenario.optimization_result.f_opt)
```

A boat cannot sail straight at a mark that lies in its **no-go zone**, and
gains by not sailing straight at one that lies dead downwind either: it
reaches off at an angle and gybes, trading distance for enough extra speed
to arrive sooner. That is the classical structure of the problem
{cite}`ciccarella2024sailboat`, and nothing here tells the optimizer about
it. It is handed a polar, a wind field and a mark, and the zigzag is what
comes back.

```{toctree}
:maxdepth: 1

physics
optimization
uncertainty
```
