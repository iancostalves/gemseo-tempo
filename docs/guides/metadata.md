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

## The documentation reads the same record

The registry is not only for design spaces and unit conversion. The
documentation reads it too, through a small Sphinx extension
(`docs/_ext/tempo_myst.py`), so a page cannot quietly disagree with the model
it describes. A hand-written variable table is a copy, and a copy of a number
is a number that will be wrong later.

Four pieces of grammar. A page first says which model it is about:

````markdown
```{variable-context} gemseo_tempo.problems.albatross.model:REGISTRY
```
````

Then a mention is a role rather than backticks, and renders from the record:

| written | rendered |
| --- | --- |
| `` {var}`V` `` | Airspeed (V, m/s) |
| `` {var}`V <name>` `` | Airspeed |
| `` {var}`V <symbol>` `` | V |
| `` {var}`V <unit>` `` | m/s |
| `` {var}`brachistochrone:mu` `` | Coulomb friction coefficient (mu, –) |

Each one links to the variable's entry in the {doc}`generated glossary
<../reference/variables>`, and carries its description as hover text.

A table is generated rather than typed:

````markdown
```{variables} gemseo_tempo.problems.albatross.model:REGISTRY
:filter: tags=state
:columns: name, full_name, unit, bounds
```
````

`:filter:` keeps the variables carrying any of the named tags, `:names:`
selects them explicitly and in order, and `:columns:` chooses from `name`,
`full_name`, `description`, `unit`, `bounds`, `lower_bound`, `upper_bound`,
`tags` and `distribution`.

**A name that no registry declares fails the build.** That is the point of
the arrangement: the documentation is checked against the code rather than
merely written alongside it. Set `tempo_unknown_variable = "warn"` in
`conf.py` to downgrade it while a model is in flux.

### Why variables are scoped by model

Writing the grammar made a clash visible that prose had been hiding. The same
grammar name means different things in different problems:

| name | brachistochrone | sail | albatross |
| --- | --- | --- | --- |
| `x` | horizontal position | eastward position | downwind position |
| `y` | depth | northward position | crosswind position |
| `mu` | Coulomb friction coefficient | | bank angle |
| `t_f` | descent time | time to the mark | |

A single global glossary would have had to pick one of each and be wrong
about the rest. So an entry belongs to a model, `{variable-context}` says
which model a page is about, and a name that is unique across all of them
still resolves without being told. Every registry of one problem shares a
label and they are merged, because a page about the albatross should resolve
the name of a wing variable as readily as the name of a state; the extension
checks that the merged registries agree wherever they overlap.
