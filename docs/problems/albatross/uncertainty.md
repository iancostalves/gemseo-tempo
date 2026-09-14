# Optimization under uncertainty

```{variable-context} gemseo_tempo.problems.albatross.model:REGISTRY
```

(uq-strategies-comparison)=
## UQ strategies comparison

The two uncertain quantities in this problem are the ones the wind profile
above turns on: the reference wind `w_ref` and the sea-surface roughness
`h0`. Propagated through the fixed-wind cycle's final specific energy
(`energy`, the same quantity the mission `energy` above maximizes), with
single shooting rather than collocation, since the states must be free to
respond to a changed wind — see
{doc}`../../concepts/transcriptions`'s "Which one to use":

Morris screening, at 60 evaluations, ranks the two variables in the same
order Sobol indices confirm at 2048 evaluations, and the vmapped batch
evaluator (8.2x on this study's 256-draw speedup measurement) is what makes
running both affordable in the same study rather than a choice between
them; see {doc}`../../guides/uncertainty` for what that speedup is worth in
general and the corrected, range-normalized elementary effect this ranking
depends on.

```{list-table}
:header-rows: 1
:widths: 25 25 25 25

* - variable
  - Morris mean effect
  - Sobol first order
  - Sobol total
* - `w_ref`
  - 249.0
  - **0.752**
  - **0.840**
* - `roughness`
  - 133.3
  - 0.127
  - 0.489
```

`w_ref` dominates both measures, at a fifth of the evaluations. `roughness`
is worth keeping rather than dropping, though: its total index (0.489) is
well above its first order (0.127), so a large share of its influence is in
interaction with `w_ref` rather than acting alone — a screening that only
looked at first-order Sobol, or that dropped `roughness` for having the
smaller Morris effect, would have missed that its effect is conditional.

Six estimators score against the same 4096-sample reference. Unlike the
sail's arrival error, this response does not reward the analytic
estimators:

| estimator | model calls | seconds | mean error | std error |
| --- | --- | --- | --- | --- |
| sampling, vmapped | 128 | 0.37 | -6.5% | +6.9% |
| sampling, looped | 128 | 2.82 | -6.5% | +6.9% |
| Taylor, first order | 2 | 0.29 | -51.4% | -90.2% |
| Taylor, second order | 6 | 0.65 | -45.0% | -85.8% |
| chaos, 32 points | 32 | 0.12 | +25.1% | +517.4% |
| chaos + gradients, 32 | 64 | 0.90 | -20.0% | +77.2% |

Both Taylor rows miss the standard deviation by nearly an order of
magnitude, and the second-order correction barely helps: the energy
response to the wind is curved enough over this parameter range that a
polynomial expansion around one point is the wrong tool, not merely an
imprecise one. Unweighted chaos at 32 points is worse than either Taylor
row on the spread, by a wide margin, which is the opposite of what
{doc}`../sail/index`'s UQ strategies comparison finds for the same estimator — the
gradient-enhanced version at the same 32 points is the only one of the six
within a factor of two of the reference spread. The contrast with the
sail is the finding: which cheap estimator is trustworthy is a property of
the response being estimated, not of the method in the abstract, and
plain sampling, expensive as it looks in a table, is the one that does not
depend on that judgment call.

(multiobjective-optimization-under-uncertainty)=
## Multiobjective optimization under uncertainty

{ref}`Formulation B <formulation-b-what-did-you-size-it-for>` found that no
sequential weighting of glide against manoeuvrability reproduces the
co-design wing, because the vehicle merit is linear and its optimum sits at
a vertex while the mission's sits in the interior. That comparison was run
at one wind and one roughness. The question this subsection asks is whether
the *ranking itself* — cruise-sized, dynamics-sized, co-designed — is stable
once the forecast the bird was designed against is wrong.

Both B-cruise and B-dynamics are frozen designs: fixed wings, flown by
re-solving only the cycle for each wind draw. The co-design wing, by
contrast, was never handed a wind to be robust to; it was optimized for the
single worst wind it could sustain. Freezing each of the three wings and
propagating the same `w_ref`/`roughness` uncertainty through its flown cycle
is what makes them comparable on the same footing the UQ strategies
comparison above was measured on.

The structural finding above already answers half the question: since no
convex combination of the two sizing objectives reaches the co-design wing,
no re-weighting *in response to* a wind forecast can reach it either. What a
propagated uncertainty adds is whether the cruise wing's near-miss (+2.9% on
minimum wind, at one wind) is a robust near-miss or one that only holds at
the nominal forecast — which is the practical version of the question a
designer choosing between "size for cruise" and "co-design properly" is
actually asking.
