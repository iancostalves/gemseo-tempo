# Brachistochrone

Minimum-time descent of a bead under gravity, in three physical variants. It
is the smallest problem in the library and the one this documentation uses
as a guided tour: every capability shown here reappears, at scale, on the
sail and albatross pages.

```python
from gemseo_tempo.problems.brachistochrone import create_optimization

scenario, gemseo_scenario = create_optimization(variant="ideal", size=20)
gemseo_scenario.execute(algo_name="SLSQP", max_iter=200)
```

Frictionless and in vacuum, the answer is the cycloid and is known in closed
form, which makes this the one problem here whose optimizer can be marked
against a right answer rather than against itself. The other two variants
switch on one force each — a bead that rubs, and a bead in air — so that
what each costs can be read separately.

```{toctree}
:maxdepth: 1

physics
optimization
uncertainty
```
