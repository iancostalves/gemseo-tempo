# Variable metadata and units

Variables carry metadata: a grammar name, a readable name, a description, a
unit, bounds, tags and, for later use by the uncertainty layer, a probability
distribution.

```python
from gemseo_tempo.metadata import Variable, VariableRegistry

registry = VariableRegistry([
    Variable(
        name="v",
        full_name="Speed",
        description="Speed of the bead along its path.",
        unit="m/s",
        lower_bound=0.0,
        tags=frozenset({"state"}),
    ),
])
```

A registry validates as it goes: registering a duplicate name raises unless
`overwrite=True`, and an unparsable unit raises at registration rather than
somewhere deep inside a scenario. Pass one to a
{class}`~gemseo_tempo.statics.model.Model` or a
{class}`~gemseo_tempo.scenario.dynamic_scenario.DynamicScenario` and the
names are checked against the discipline grammar.

Registries subset, filter by tag, and round-trip through YAML including the
distribution field:

```python
states = registry.filter("state")
registry.to_yaml("variables.yaml")
VariableRegistry.from_yaml("variables.yaml")
```

Formulation builders read bounds from the registry when no explicit bounds
are given, so declaring a variable once puts it in the design space correctly.

## Units are metadata, not runtime

Inside jitted code every value is in canonical base units. Conversion happens
only at discipline boundaries, and never inside traced code, because a pint
quantity must not meet a JAX tracer.

```python
from gemseo_tempo.metadata import make_unit_shell, to_base

to_base(3.6, "km/h")  # 1.0 m/s
```

The conversion coefficients are computed eagerly, cached per unit string and
kept as plain Python floats, so `to_base` is safe to call on a tracer and
compiles once.

Conversions are **affine**, not a bare factor, so offset units behave:

```python
from gemseo_tempo.metadata import convert

convert(212.0, "degF", "degC")  # 100.0
```

{func}`~gemseo_tempo.metadata.units.make_unit_shell` wraps a
dictionary-to-dictionary function so it accepts and returns values in
declared units while computing in base units. Keys absent from the mappings
pass through untouched.
