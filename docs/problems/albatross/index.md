# Albatross dynamic soaring

```{variable-context} gemseo_tempo.problems.albatross.model:REGISTRY
```

An albatross crosses oceans without flapping by climbing into the wind and
diving back with it, through air that moves faster higher up
{footcite:p}`rayleigh_soaring_1883,sachs_flying_2012`. Each pass takes a
little energy out of the wind gradient, and over a closed cycle it can pay
for the drag exactly. Flying the real bird's wing, that takes a wind of
{docvalue}`albatross/shapes.json:weave.minimum_wind:.2f` m/s at 10 m; with
the wing designed alongside the cycle,
{docvalue}`albatross/design_strategies.json:weave_codesign.minimum_wind:.2f` m/s.

```python
from gemseo_tempo.problems.albatross import create_codesign_optimization

scenario, gemseo_scenario = create_codesign_optimization(size=20, topology="weave")
gemseo_scenario.execute(algo_name="SLSQP", max_iter=500)
```

## Problem

This is the flagship problem of the library, and the only one that uses a
**statics chain**, **path constraints**, **periodicity** and a **co-design
formulation** at once. The bird flies a closed cycle through a wind that
grows with height, {doc}`physics`, with the lift coefficient and the bank
as controls and a wing that a statics chain sizes. Minimizing the wind a
closed cycle needs, with the wing free, collocated on twenty
Hermite–Simpson segments, gives the optimizer:

```{formulation} albatross/formulation_codesign.json
```

```{variables} gemseo_tempo.problems.albatross.model:REGISTRY
:columns: name, full_name, unit, tags
```

The pages that follow read in order:

```{toctree}
:maxdepth: 1

physics
optimization
uncertainty
```

- {doc}`physics`: the frames, the cycle's two families, the equations of
  motion and where the energy comes from, the wing, and the wind.
- {doc}`optimization`: the cycle for each mission and shape, checked against
  Hong et al., each transcription, three ways of arriving at the wing, and
  each way of differentiating.
- {doc}`uncertainty`: what an uncertain wind and sea do to the energy a
  cycle banks, and the cycle that accounts for them.

## References

```{footbibliography}
```
