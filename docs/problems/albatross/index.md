# Albatross dynamic soaring

An albatross crosses oceans for days without flapping and without thermals.
It does it by repeatedly climbing into the wind and diving back downwind
through air that moves faster higher up. Each pass through the gradient
takes a little energy out of the wind, and over a closed cycle that can be
exactly enough to pay for the drag.

This is the flagship problem of the library. It is the only one that uses a
**statics chain**, a **path constraint**, **periodicity** and a **co-design
formulation** at once.

```python
from gemseo_tempo.problems.albatross import create_codesign_optimization

scenario, gemseo_scenario = create_codesign_optimization(size=20)
gemseo_scenario.execute(algo_name="SLSQP", max_iter=500)
print(gemseo_scenario.optimization_result.f_opt)  # about 8.6 m/s
```

```{toctree}
:maxdepth: 1

physics
optimization
uncertainty
```
