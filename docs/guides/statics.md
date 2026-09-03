# The statics stage

A statics model is one whose every input is a declared constant or the output
of another statics model. Such models can run once per optimizer iteration,
ahead of the dynamics, instead of at every time step.

Three wrappers around gemseo-jax disciplines, ported from NOADS:

```python
from gemseo_tempo.statics import AutoModel, JAXModel, StaticsChain

class Wing(AutoModel):
    def _jax_func(self, span=30.0, area=120.0):
        aspect_ratio = span**2 / area
        return aspect_ratio
```

{class}`~gemseo_tempo.statics.model.AutoModel` detects inputs and outputs
from the function signature and the returned names, which removes the
grammar boilerplate. {class}`~gemseo_tempo.statics.model.JAXModel` takes an
explicit dictionary function with declared names, for closures and partial
applications whose signature cannot be introspected.

## Ordering resolves itself

{meth}`~gemseo_tempo.statics.chain.StaticsChain.split` partitions models into
the statics side and the rest by a **fixed-point** sweep: a model joins the
statics side once all of its inputs are constants or outputs of models
already there, and the sweep repeats until nothing moves.

That matters because a single pass misclassifies models declared out of
dependency order. The fixed point gives the same answer for every one of the
possible declaration orders.

```python
statics, dynamics_side = StaticsChain.split(models, constant_inputs=["k"])
chain = StaticsChain(statics, ["k"])
outputs = chain({"k": 0.5})
```

The chain delegates to the underlying `JAXChain` function, so it composes
into a larger jitted program and differentiates end to end. Missing inputs
fall back to the discipline defaults, so it can be called with only the
inputs of interest.
