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
"""Propagating uncertainty through a frozen design, and costing it."""

from __future__ import annotations

import jax.numpy as jnp
import numpy

from .common import _solve_checked
from .common import _write_json


def _estimate_record(estimate, comparison) -> dict:
    """Flatten one estimate and its error into a storable record.

    Parameters
    ----------
    estimate
        The estimate.
    comparison
        The comparison it belongs to, for the errors.

    Returns
    -------
    dict
        The record.
    """
    mean_error, deviation_error = comparison.relative_error(estimate)
    return {
        "name": estimate.name,
        "mean": estimate.mean,
        "standard_deviation": estimate.standard_deviation,
        "n_evaluations": estimate.n_evaluations,
        "n_gradients": estimate.n_gradients,
        "cost": estimate.cost,
        "seconds": estimate.seconds,
        "mean_error": mean_error,
        "deviation_error": deviation_error,
    }


def _propagation_study(
    name, scenario, gemseo_scenario, registry, quantity, smoke, budget=400, warm=None
) -> dict:
    """Propagate the uncertainty through an optimum, by every estimator.

    The design is frozen at the optimum and the uncertain constants are
    drawn from the registry's distributions. Each estimator is then scored
    against a large sample, and the ones that need derivatives are run
    twice, with the exact derivatives automatic differentiation provides
    and with central finite differences, which is the comparison that shows
    what the derivatives are worth.

    Parameters
    ----------
    name
        The problem, which names the stored file.
    scenario
        The dynamic scenario, single-shooting so the trajectory responds.
    gemseo_scenario
        Its GEMSEO scenario, solved here at the nominal constants.
    registry
        The variable metadata carrying the distributions.
    quantity
        The scalar output propagated.
    smoke
        Whether to run the tiny version, for the tests.
    budget
        The iteration budget of the nominal solve.
    warm
        A design to start the nominal solve from.

    Returns
    -------
    dict
        What is stored. Each estimate carries its cost in evaluations of
        the model, counting a reverse pass as one and a finite-difference
        gradient as the two evaluations per uncertain variable it takes.
    """
    from gemseo_tempo.uncertainty import freeze_design
    from gemseo_tempo.uncertainty import looped_evaluator
    from gemseo_tempo.uncertainty import make_parameter_space
    from gemseo_tempo.uncertainty import measure_speedup
    from gemseo_tempo.uncertainty import nominal_values
    from gemseo_tempo.uncertainty import uncertain_names
    from gemseo_tempo.uncertainty.estimators import Comparison
    from gemseo_tempo.uncertainty.estimators import finite_differences
    from gemseo_tempo.uncertainty.estimators import from_scenario
    from gemseo_tempo.uncertainty.estimators import monte_carlo
    from gemseo_tempo.uncertainty.estimators import polynomial_chaos
    from gemseo_tempo.uncertainty.estimators import taylor

    names = uncertain_names(registry)
    objective, _, feasible, _ = _solve_checked(
        scenario, gemseo_scenario, 60 if smoke else budget, warm=warm
    )
    freeze_design(scenario, gemseo_scenario, nominal_values(registry, names))
    space = make_parameter_space(registry, names)
    sample = numpy.asarray(space.compute_samples(32 if smoke else 256))
    speedup = measure_speedup(scenario, names, [quantity], sample)

    exact = from_scenario(scenario, registry, names, quantity)
    fd_step = _propagation_step(exact)
    differenced = finite_differences(exact, relative_step=fd_step)
    looped = looped_evaluator(scenario, names, [quantity])
    size = 64 if smoke else 256
    reference = monte_carlo(
        exact, n_samples=256 if smoke else 20000, seed=1, name="reference"
    )
    # Named by slugs, which is how the pages refer to them.
    estimates = [
        monte_carlo(exact, n_samples=size, name="sampling_vmapped"),
        monte_carlo(
            exact,
            n_samples=size,
            name="sampling_looped",
            evaluate=lambda draws: numpy.asarray(looped(draws)).ravel(),
        ),
    ]
    for label, propagation in (("ad", exact), ("fd", differenced)):
        for order, second in ((1, False), (2, True)):
            estimate = taylor(propagation, second_order=second)
            estimate.name = f"taylor{order}_{label}"
            estimates.append(estimate)
    for label, propagation, enhanced in (
        ("chaos", exact, False),
        ("chaos_ad", exact, True),
        ("chaos_fd", differenced, True),
    ):
        estimate = polynomial_chaos(
            propagation, n_training=12, degree=2, gradient_enhanced=enhanced
        )
        estimate.name = label
        estimates.append(estimate)
    comparison = Comparison(reference=reference, estimates=tuple(estimates))
    records = {}
    for estimate in estimates:
        record = _estimate_record(estimate, comparison)
        per_gradient = 2 * len(names) if estimate.name.endswith("_fd") else 1
        record["model_calls"] = (
            estimate.n_evaluations + per_gradient * estimate.n_gradients
        )
        records[estimate.name] = record
    study = {
        "quantity": quantity,
        "names": list(names),
        "objective": objective,
        "feasible": feasible,
        "fd_step": fd_step,
        "speedup": {
            "n_samples": speedup.n_samples,
            "looped_seconds": speedup.looped_seconds,
            "batched_seconds": speedup.batched_seconds,
            "speedup": speedup.speedup,
        },
        "reference": _estimate_record(reference, comparison),
        "estimates": records,
    }
    _write_json(f"{name}/propagation.json", study)
    return study


def _propagation_step(propagation, steps=(1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)):
    """Return the finite-difference step closest to the exact derivative.

    Measured at the mean, against the derivative automatic differentiation
    gives there, so that the differenced estimators are compared at their
    best rather than at an arbitrary step.

    Parameters
    ----------
    propagation
        The propagation with exact derivatives.
    steps
        The steps tried, as fractions of each standard deviation.

    Returns
    -------
    float
        The step whose central differences come closest.
    """
    from gemseo_tempo.uncertainty.estimators import finite_differences

    mean = propagation.mean_point
    exact = numpy.asarray(propagation.jacobian(mean))
    scale = max(float(numpy.abs(exact).max()), numpy.finfo(float).tiny)

    def error(step):
        differenced = finite_differences(propagation, relative_step=step)
        return float(numpy.abs(differenced.jacobian(mean) - exact).max()) / scale

    return min(steps, key=error)


def _statistic_of_design(scenario, names, sample, quantity, statistic, **settings):
    """Return one statistic of a quantity as a function of a flat design.

    Parameters
    ----------
    scenario
        The dynamic scenario, whose design variables make the vector.
    names
        The uncertain names.
    sample
        The fixed sample the statistic is taken over.
    quantity
        The output.
    statistic
        Which statistic, one of the names
        :func:`~gemseo_tempo.uncertainty.robust.statistics_function`
        returns.
    **settings
        Forwarded to that function.

    Returns
    -------
    tuple
        The function of the flat design vector, and the current design.
    """
    from gemseo_tempo.uncertainty.robust import statistics_function

    statistics = statistics_function(scenario, names, sample, quantity, **settings)
    defaults = scenario.problem.input_defaults
    design = list(scenario.problem.design_variables)
    shapes = {key: numpy.shape(defaults[key]) for key in design}
    sizes = [int(numpy.prod(shapes[key])) or 1 for key in design]
    point = jnp.concatenate([jnp.ravel(jnp.asarray(defaults[key])) for key in design])

    def function(flat):
        values, start = {}, 0
        for key, count in zip(design, sizes, strict=True):
            values[key] = jnp.reshape(flat[start : start + count], shapes[key])
            start += count
        return jnp.reshape(statistics(values)[statistic], ())

    return function, point


def _gradient_cost(functions, point, repeats: int = 5) -> dict:
    """Time one gradient by AD and by finite differences, each at its best.

    Measured on the bare JAX function, compiled first, since GEMSEO caches
    a linearization until its inputs change and timing it twice at one
    point would time the cache. Automatic differentiation is timed in
    reverse mode, one pass for one output, through every adjoint given,
    the fastest counting; all of them are exact for a scalar. Forward mode
    is left out: it takes one pass per input, and on the albatross's
    margin, eighty-two inputs through sixty-four draws, it had not
    finished compiling and running once in twenty-five minutes.
    Finite differences take the step that comes closest to that exact
    gradient and are timed batched into one vectorized call and one
    evaluation at a time, the faster counting.

    Parameters
    ----------
    functions
        The scalar function of a flat vector, by the adjoint its
        integration is differentiated through.
    point
        Where to differentiate it.
    repeats
        How many times each is timed; the fastest counts.

    Returns
    -------
    dict
        The design size, the seconds of one evaluation and of one gradient
        by each route, and the configuration each route was fastest in.
    """
    import jax

    from gemseo_tempo.benchmarking.differentiation import _differences
    from gemseo_tempo.benchmarking.differentiation import _fastest
    from gemseo_tempo.benchmarking.differentiation import batched_differences
    from gemseo_tempo.benchmarking.differentiation import best_step

    routes = {
        f"reverse, {adjoint}": jax.jit(jax.grad(function))
        for adjoint, function in functions.items()
    }
    timings = {}
    for name, route in routes.items():
        route(point).block_until_ready()
        timings[name] = _fastest(lambda route=route: route(point), repeats)
    ad = min(timings, key=timings.get)
    exact = numpy.asarray(routes[ad](point))[None, :]

    function = functions.get("Direct", next(iter(functions.values())))
    value = jax.jit(function)
    batched = jax.jit(batched_differences(lambda x: jnp.atleast_1d(function(x))))
    step, _ = best_step(batched, point, exact)
    value(point).block_until_ready()
    batched(point, step).block_until_ready()
    fd = {
        "batched": _fastest(lambda: batched(point, step), repeats),
        "sequential": _fastest(
            lambda: _differences(lambda x: jnp.atleast_1d(value(x)), point, step),
            repeats,
        ),
    }
    variant = min(fd, key=fd.get)
    return {
        "n_design": int(point.size),
        "evaluation_seconds": _fastest(lambda: value(point), repeats),
        "ad_seconds": timings[ad],
        "ad_route": ad,
        "fd_seconds": fd[variant],
        "fd_variant": variant,
        "fd_step": step,
    }


def _spread(scenario, names, sample, quantity, design) -> dict:
    """Return what a frozen design does over a sample of the uncertainty.

    Parameters
    ----------
    scenario
        A scenario of the problem, at the nominal constants.
    names
        The uncertain names.
    sample
        The sample, independent of the one any design was optimized on.
    quantity
        The output.
    design
        The design to freeze.

    Returns
    -------
    dict
        The nominal value, and the mean, deviation and 5th and 95th
        percentiles over the sample.
    """
    from gemseo_tempo.uncertainty import batch_evaluator

    defaults = scenario.problem.input_defaults
    defaults.update({key: numpy.atleast_1d(value) for key, value in design.items()})
    values = numpy.asarray(batch_evaluator(scenario, names, [quantity])(sample)).ravel()
    nominal = scenario.problem.jax_function(dict(defaults))[quantity]
    return {
        "nominal": float(numpy.asarray(nominal).ravel()[0]),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "p05": float(numpy.quantile(values, 0.05)),
        "p95": float(numpy.quantile(values, 0.95)),
    }
