# Copyright 2026 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Evaluating a whole sample at once instead of one draw at a time.

A design of experiments loops. It hands the discipline one point, waits, and
hands it the next, which is the only thing it can do when the discipline is a
black box. The disciplines here are not black boxes: they are jitted JAX
programs, and JAX can map one over an entire sample in a single dispatch.

That is what :func:`batch_evaluator` builds. The saving is not the arithmetic,
which is the same either way, but everything around it: one dispatch instead
of N, one set of Python and GEMSEO overheads instead of N, and inner loops
that vectorize instead of running one trajectory at a time.

The same trick gives the derivatives. :func:`batch_jacobian` returns the
Jacobian of the quantities with respect to the uncertain inputs, for the whole
sample at once. That matters more than it looks: the methods in the CSMA
paper that are described as conditionally affordable, first-order Taylor
propagation and gradient-enhanced polynomial chaos, are conditional on exactly
these derivatives being cheap. Here they are free.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

CONSTANT_PREFIX = "constant."
"""The grammar prefix uncertain constants carry inside a scenario."""


def freeze_design(scenario, gemseo_scenario, nominal=None, prefix=CONSTANT_PREFIX):
    """Fix a scenario at a converged design, ready to be propagated through.

    Propagating uncertainty through a design means through **a** design, and
    which one matters. Freezing the scenario at its starting guess measures
    the sensitivity of a trajectory nobody would fly; freezing it at the
    optimum measures what the uncertainty does to the design actually
    chosen, which is the question a designer has.

    Parameters
    ----------
    scenario
        The dynamic scenario, whose input defaults are updated in place.
    gemseo_scenario
        The executed GEMSEO scenario whose optimum is frozen in.
    nominal
        The values the uncertain constants take at the nominal design; the
        mean of each distribution, usually.
    prefix
        The grammar prefix of those constants.

    Returns
    -------
    DynamicScenario
        The same scenario, for chaining.
    """
    defaults = scenario.problem.input_defaults
    for name, value in (nominal or {}).items():
        defaults[prefix + name] = numpy.atleast_1d(numpy.asarray(value, dtype=float))
    for name, value in gemseo_scenario.optimization_result.x_opt_as_dict.items():
        defaults[name] = numpy.atleast_1d(numpy.asarray(value, dtype=float))
    return scenario


def _quantity(outputs, name: str, reductions=None):
    """Return one output as a scalar.

    Parameters
    ----------
    outputs
        The output dictionary of the transcribed problem.
    name
        The output name.
    reductions
        How to reduce outputs that are not already scalar, by name. A path
        constraint is one value per node, and which scalar it should
        become is the caller's decision rather than this module's: the
        worst violation along the trajectory is the usual answer, but so
        is the violation at a particular instant.

    Returns
    -------
    Array
        The value, reduced to a scalar.

    Raises
    ------
    ValueError
        If the output is not scalar and no reduction is given for it.
    """
    value = outputs[name]
    reduce = (reductions or {}).get(name)
    if reduce is not None:
        return jnp.reshape(reduce(value), ())
    if jnp.size(value) != 1:
        msg = (
            f"The output {name!r} has {jnp.size(value)} components, so it is"
            " not a quantity of interest on its own; pass a reduction for it"
            " saying which scalar it should become."
        )
        raise ValueError(msg)
    return jnp.reshape(value, ())


def batch_evaluator(
    scenario,
    uncertain: Sequence[str],
    quantities: Sequence[str],
    prefix: str = CONSTANT_PREFIX,
    reductions=None,
) -> Callable:
    """Build a function evaluating many draws in one call.

    Parameters
    ----------
    scenario
        The dynamic scenario, whose transcribed problem carries the JAX
        function to map.
    uncertain
        The raw names of the uncertain inputs, in the order the sample
        columns are given in.
    quantities
        The output names to return.
    prefix
        The grammar prefix the uncertain names carry as inputs.
    reductions
        How to reduce any quantity that is not already scalar, by name;
        see :func:`_quantity`.

    Returns
    -------
    Callable
        ``sample -> array`` taking an ``(n, len(uncertain))`` array and
        returning an ``(n, len(quantities))`` one.
    """
    problem = scenario.problem
    defaults = dict(problem.input_defaults)
    names = [prefix + name for name in uncertain]

    def one(row):
        """Evaluate a single draw."""
        inputs = dict(defaults)
        for index, name in enumerate(names):
            inputs[name] = jnp.reshape(row[index], jnp.shape(defaults[name]))
        outputs = problem.jax_function(inputs)
        return jnp.stack([_quantity(outputs, name, reductions) for name in quantities])

    return jax.jit(jax.vmap(one))


def looped_evaluator(
    scenario,
    uncertain: Sequence[str],
    quantities: Sequence[str],
    prefix: str = CONSTANT_PREFIX,
    reductions=None,
) -> Callable:
    """Build the same function, evaluating one draw at a time.

    The honest comparison for :func:`batch_evaluator`: the same arithmetic
    through the same jitted program, but dispatched once per draw the way a
    design of experiments would.

    Parameters
    ----------
    scenario
        The dynamic scenario.
    uncertain
        The raw names of the uncertain inputs.
    quantities
        The output names to return.
    prefix
        The grammar prefix.
    reductions
        How to reduce any quantity that is not already scalar, by name.

    Returns
    -------
    Callable
        ``sample -> array``, looping internally.
    """
    problem = scenario.problem
    defaults = dict(problem.input_defaults)
    names = [prefix + name for name in uncertain]
    single = jax.jit(problem.jax_function)

    def loop(sample):
        """Evaluate every draw in turn."""
        rows = []
        for row in numpy.asarray(sample):
            inputs = dict(defaults)
            for index, name in enumerate(names):
                inputs[name] = jnp.reshape(row[index], jnp.shape(defaults[name]))
            outputs = single(inputs)
            rows.append([float(_quantity(outputs, q, reductions)) for q in quantities])
        return numpy.asarray(rows)

    return loop


def batch_jacobian(
    scenario,
    uncertain: Sequence[str],
    quantities: Sequence[str],
    prefix: str = CONSTANT_PREFIX,
    reductions=None,
) -> Callable:
    """Build a function giving the derivatives of a whole sample at once.

    The quantity the Taylor and gradient-enhanced routes need.

    Reverse mode, for two reasons. The quantities of interest are few, so
    reverse costs one pass per quantity rather than one per uncertain
    variable. And the default adjoint through the integrator is
    reverse-only: it is a ``custom_vjp``, so asking for forward mode raises
    rather than falling back. A scenario built with the Direct adjoint
    could be differentiated either way, but nothing here needs it to be.

    One pass **per quantity**, deliberately, rather than one ``jacrev``
    over a stacked output. The two are the same derivative and should be
    the same number, and on the sail they are not: ``jacrev`` batches the
    reverse pass over the output basis, and a ``vmap`` over the sample
    batches it again. Pushed through diffrax's RecursiveCheckpoint adjoint
    that doubly-batched pass comes back about nine percent wrong at the
    downwind optimum, silently. Taking one cotangent at a time and mapping
    *that* over the sample agrees with finite differences to ten digits
    under both adjoints.

    So this is the safe route rather than merely a tidy one. It costs
    nothing when the quantities are few, which they are: the quantities of
    interest are the ones a designer names, not the whole trajectory.

    Parameters
    ----------
    scenario
        The dynamic scenario.
    uncertain
        The raw names of the uncertain inputs.
    quantities
        The output names to differentiate.
    prefix
        The grammar prefix.
    reductions
        How to reduce any quantity that is not already scalar, by name.

    Returns
    -------
    Callable
        ``sample -> array`` of shape ``(n, len(quantities), len(uncertain))``.
    """
    problem = scenario.problem
    defaults = dict(problem.input_defaults)
    names = [prefix + name for name in uncertain]

    def scalar(row, quantity):
        """Evaluate a single draw, returning one quantity as a scalar."""
        inputs = dict(defaults)
        for index, name in enumerate(names):
            inputs[name] = jnp.reshape(row[index], jnp.shape(defaults[name]))
        return _quantity(problem.jax_function(inputs), quantity, reductions)

    gradients = [
        jax.vmap(jax.grad(partial(scalar, quantity=quantity)))
        for quantity in quantities
    ]

    def jacobian(sample):
        """Stack the gradient of each quantity over the whole sample."""
        return jnp.stack([gradient(sample) for gradient in gradients], axis=1)

    return jax.jit(jacobian)


@dataclass
class Speedup:
    """How much faster the whole sample went through at once."""

    n_samples: int
    """The size of the sample."""

    looped_seconds: float
    """Time to evaluate it one draw at a time."""

    batched_seconds: float
    """Time to evaluate it in a single call."""

    max_difference: float
    """The largest disagreement between the two sets of results."""

    @property
    def speedup(self) -> float:
        """Return how many times faster the batched path was.

        Returns
        -------
        float
            The ratio of the two times.
        """
        return self.looped_seconds / self.batched_seconds


def measure_speedup(
    scenario,
    uncertain: Sequence[str],
    quantities: Sequence[str],
    sample,
    prefix: str = CONSTANT_PREFIX,
) -> Speedup:
    """Time both paths over the same sample and check they agree.

    Compilation is excluded from both times by evaluating each path once
    before timing it. The warm-up runs on the **whole** sample rather than a
    slice of it: a jitted function specializes on the shape of its argument,
    so warming up on two rows and timing five hundred measures a
    recompilation and reports it as the cost of vectorizing.

    Parameters
    ----------
    scenario
        The dynamic scenario.
    uncertain
        The raw names of the uncertain inputs.
    quantities
        The output names.
    sample
        The ``(n, len(uncertain))`` sample.
    prefix
        The grammar prefix.

    Returns
    -------
    Speedup
        The two times, their ratio, and how far the results differ.
    """
    sample = numpy.asarray(sample, dtype=float)
    batched = batch_evaluator(scenario, uncertain, quantities, prefix)
    looped = looped_evaluator(scenario, uncertain, quantities, prefix)

    batched(sample).block_until_ready()
    looped(sample[:2])

    started = time.perf_counter()
    batch_result = numpy.asarray(batched(sample).block_until_ready())
    batched_seconds = time.perf_counter() - started

    started = time.perf_counter()
    loop_result = looped(sample)
    looped_seconds = time.perf_counter() - started

    return Speedup(
        n_samples=int(sample.shape[0]),
        looped_seconds=looped_seconds,
        batched_seconds=batched_seconds,
        max_difference=float(numpy.max(numpy.abs(batch_result - loop_result))),
    )


__all__ = [
    "CONSTANT_PREFIX",
    "Speedup",
    "batch_evaluator",
    "batch_jacobian",
    "freeze_design",
    "looped_evaluator",
    "measure_speedup",
]
