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
"""What a wrong forecast costs, and what to do about it.

Four analyses on the sailing problem, in the order they are worth doing.

``speedup`` times the same sample through the ordinary looped path and
through one vectorized dispatch. They are the same evaluations, so the
ratio is the whole claim of the batch evaluator.

``screen`` ranks the uncertainties by a Morris screening, which is the
cheapest useful thing to know: a variable a screening says does nothing is
a variable no Sobol budget need be spent on.

``estimators`` is the comparison proper. Six ways to the mean and standard
deviation of one quantity, scored against a large sample on cost and on
error. The interesting result is not that sampling works but that first
order Taylor gets the spread to about one percent for two model calls,
which is affordable only because the derivatives are exact.

``margin`` is the argument. A deterministic design, the same design padded
by a fixed safety margin, and a design that carries the probability in its
constraints, all measured on a sample none of them was optimized against.

Examples
--------
The quick pass, a few minutes::

    python examples/uncertainty/run_uncertainty.py --analysis speedup screen

Everything, which takes considerably longer::

    python examples/uncertainty/run_uncertainty.py --analysis all
"""

from __future__ import annotations

import argparse
import logging

import jax.numpy as jnp
import numpy

from gemseo_tempo.problems.sail import create_optimization
from gemseo_tempo.problems.sail import create_scenario
from gemseo_tempo.problems.sail.model import REGISTRY
from gemseo_tempo.problems.sail.problem import sail_integrator
from gemseo_tempo.scenario.formulations import OptimalControl
from gemseo_tempo.transcriptions import SingleShooting
from gemseo_tempo.uncertainty import freeze_design
from gemseo_tempo.uncertainty import looped_evaluator
from gemseo_tempo.uncertainty import make_parameter_space
from gemseo_tempo.uncertainty import measure_speedup
from gemseo_tempo.uncertainty import nominal_values
from gemseo_tempo.uncertainty import uncertain_names
from gemseo_tempo.uncertainty.estimators import Comparison
from gemseo_tempo.uncertainty.estimators import from_scenario
from gemseo_tempo.uncertainty.estimators import monte_carlo
from gemseo_tempo.uncertainty.estimators import polynomial_chaos
from gemseo_tempo.uncertainty.estimators import taylor
from gemseo_tempo.uncertainty.robust import assess
from gemseo_tempo.uncertainty.robust import format_assessments
from gemseo_tempo.uncertainty.robust import statistics_discipline
from gemseo_tempo.uncertainty.sensitivity import format_indices
from gemseo_tempo.uncertainty.sensitivity import screen
from gemseo_tempo.uncertainty.sensitivity import sobol_indices

logging.disable(logging.CRITICAL)

ANALYSES = ("speedup", "screen", "estimators", "margin")
"""The analyses, in the order they are worth doing."""

ARRIVAL = "constraint.final.x"
"""How far short of or past the mark the boat ends up, in m."""

CHANNEL = "constraint.channel"
"""The path constraint, one value per node, positive outside the channel."""

SIZE = 20
"""The discretization."""

HALF_WIDTH = 150.0
"""The channel the boat has to stay inside, in m."""

TARGET = 0.95
"""The reliability the probabilistic design is asked for."""


def integrator():
    """Return the integrator, fixed-step so every evaluation costs the same.

    Returns
    -------
    DiffraxIntegrator
        The integrator, carrying the adjoint this problem needs.
    """
    return sail_integrator(dt0=1.0 / 300, n_save=SIZE + 1)


def solve(scenario, gemseo_scenario, warm=None, max_iter=600):
    """Solve one scenario, optionally from a given starting design.

    Parameters
    ----------
    scenario
        The dynamic scenario.
    gemseo_scenario
        The GEMSEO scenario on it.
    warm
        A design to start from, as ``{name: value}``; the scenario's own
        starting point when omitted.
    max_iter
        The iteration budget.

    Returns
    -------
    tuple
        The objective, the optimal design, and whether it is feasible.
    """
    if warm:
        current = gemseo_scenario.design_space.get_current_value(as_dict=True)
        gemseo_scenario.design_space.set_current_value(
            {
                name: numpy.reshape(value, numpy.shape(current[name]))
                for name, value in warm.items()
                if name in current
            }
        )
    gemseo_scenario.execute(algo_name="SLSQP", max_iter=max_iter)
    result = gemseo_scenario.optimization_result
    design = {
        name: numpy.atleast_1d(value) for name, value in result.x_opt_as_dict.items()
    }
    return float(result.f_opt), design, result.is_feasible


def frozen_scenario(names):
    """Return a scenario solved and frozen at its nominal optimum.

    Parameters
    ----------
    names
        The uncertain variable names.

    Returns
    -------
    tuple
        The frozen scenario and the objective it attained.
    """
    scenario, gemseo_scenario = create_optimization(
        transcription=SingleShooting(), size=SIZE, integrator=integrator()
    )
    objective, _, _ = solve(scenario, gemseo_scenario)
    freeze_design(scenario, gemseo_scenario, nominal_values(REGISTRY, names))
    return scenario, objective


def run_speedup(scenario, names) -> None:
    """Time one sample through both paths.

    Parameters
    ----------
    scenario
        The frozen scenario.
    names
        The uncertain variable names.
    """
    sample = numpy.asarray(make_parameter_space(REGISTRY, names).compute_samples(512))
    result = measure_speedup(scenario, names, [ARRIVAL], sample)
    print(
        f"{result.n_samples} draws: looped {result.looped_seconds:.2f} s,"
        f" batched {result.batched_seconds:.3f} s,"
        f" speedup {result.speedup:.1f}x,"
        f" largest disagreement {result.max_difference:.1e}"
    )


def run_screen(scenario, names) -> None:
    """Rank the uncertainties, then decompose the variance.

    Parameters
    ----------
    scenario
        The frozen scenario.
    names
        The uncertain variable names.
    """
    screening = screen(scenario, REGISTRY, names, ARRIVAL)
    print(f"Morris, {screening.n_evaluations} evaluations: {screening.ranking}")
    result = sobol_indices(scenario, REGISTRY, names, ARRIVAL, n_samples=1024)
    print(f"Sobol, {result.n_evaluations} evaluations in {result.seconds:.2f} s")
    print(format_indices(result))


def run_estimators(scenario, names) -> None:
    """Estimate the same statistic six ways and score them.

    Parameters
    ----------
    scenario
        The frozen scenario.
    names
        The uncertain variable names.
    """
    propagation = from_scenario(scenario, REGISTRY, names, ARRIVAL)
    looped = looped_evaluator(scenario, names, [ARRIVAL])
    comparison = Comparison(
        reference=monte_carlo(propagation, n_samples=50000, name="reference"),
        estimates=(
            monte_carlo(propagation, n_samples=512, name="sampling, vmapped"),
            monte_carlo(
                propagation,
                n_samples=512,
                name="sampling, looped",
                evaluate=lambda sample: numpy.asarray(looped(sample)).ravel(),
            ),
            taylor(propagation),
            taylor(propagation, second_order=True),
            polynomial_chaos(propagation, n_training=32),
            polynomial_chaos(propagation, n_training=32, gradient_enhanced=True),
            polynomial_chaos(propagation, n_training=12, degree=3),
            polynomial_chaos(
                propagation, n_training=12, degree=3, gradient_enhanced=True
            ),
        ),
    )
    print(comparison.table())


def run_margin(names) -> None:
    """Compare a safety margin against carrying the probability.

    Every design is warm-started from the unconstrained optimum. That is
    not a convenience: started cold from the polar-derived guess, which
    sails a narrow gybe, the constrained solves stall in a poorer basin
    and the comparison would be measuring the starting point.

    Parameters
    ----------
    names
        The uncertain variable names.
    """
    space = make_parameter_space(REGISTRY, names)
    design_sample = numpy.asarray(space.compute_samples(256))
    check_sample = numpy.asarray(space.compute_samples(4000))
    nominal = nominal_values(REGISTRY, names)

    free_scenario, free_gemseo = create_optimization(
        transcription=SingleShooting(), size=SIZE, integrator=integrator()
    )
    free_objective, free_design, _ = solve(free_scenario, free_gemseo)
    print(f"unconstrained: {free_objective:.2f} s")

    def measure(scenario, design, objective, label):
        """Freeze a design into a scenario and measure what it achieves."""
        for name, value in design.items():
            scenario.problem.input_defaults[name] = value
        for name, value in nominal.items():
            scenario.problem.input_defaults[f"constant.{name}"] = numpy.atleast_1d(
                value
            )
        return assess(
            scenario,
            names,
            check_sample,
            CHANNEL,
            objective=objective,
            name=label,
            reduction=jnp.max,
        )

    assessments = []
    for margin in (0.0, 25.0, 50.0, 75.0):
        scenario, gemseo_scenario = create_optimization(
            transcription=SingleShooting(),
            size=SIZE,
            integrator=integrator(),
            channel_half_width=HALF_WIDTH - margin,
        )
        objective, design, _ = solve(scenario, gemseo_scenario, warm=free_design)
        # Measured against the real limit, not the tightened one.
        real = create_scenario(
            transcription=SingleShooting(),
            size=SIZE,
            integrator=integrator(),
            channel_half_width=HALF_WIDTH,
        )
        label = "deterministic" if margin == 0.0 else f"margin {margin:.0f} m"
        assessments.append(measure(real, design, objective, label))

    scenario = create_scenario(
        transcription=SingleShooting(),
        size=SIZE,
        integrator=integrator(),
        channel_half_width=HALF_WIDTH,
    )
    formulation = OptimalControl(
        scenario,
        bounds=_heading_bounds(scenario),
        extra_disciplines=[
            statistics_discipline(
                scenario,
                names,
                design_sample,
                CHANNEL,
                reduction=jnp.max,
                smoothing=5.0,
                target=TARGET,
            )
        ],
        extra_constraints=[(f"{CHANNEL}.shortfall", "ineq")],
    )
    gemseo_scenario = formulation.create_gemseo_scenario()
    objective, design, _ = solve(scenario, gemseo_scenario, warm=free_design)
    assessments.append(
        measure(scenario, design, objective, f"P(inside) >= {TARGET:.0%}")
    )
    print(format_assessments(assessments))


def _heading_bounds(scenario) -> dict:
    """Return the heading bounds, one turn centred on the bearing.

    Parameters
    ----------
    scenario
        The dynamic scenario.

    Returns
    -------
    dict
        The bounds by design-variable name.
    """
    bounds = {"t_f": (10.0, 4000.0)}
    for name in scenario.problem.design_variables:
        if name.startswith("control.psi"):
            bounds[name] = (-numpy.pi, numpy.pi)
    return bounds


def parse_arguments(argv=None):
    """Parse the command line.

    Parameters
    ----------
    argv
        The arguments; ``sys.argv`` when omitted.

    Returns
    -------
    Namespace
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        nargs="+",
        default=["speedup", "screen"],
        choices=[*ANALYSES, "all"],
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Run the requested analyses.

    Parameters
    ----------
    argv
        The command-line arguments.

    Returns
    -------
    int
        Zero.
    """
    arguments = parse_arguments(argv)
    wanted = ANALYSES if "all" in arguments.analysis else tuple(arguments.analysis)
    names = uncertain_names(REGISTRY)
    print(f"uncertain: {names}")

    if {"speedup", "screen", "estimators"} & set(wanted):
        scenario, objective = frozen_scenario(names)
        print(f"nominal optimum: {objective:.2f} s")
        if "speedup" in wanted:
            run_speedup(scenario, names)
        if "screen" in wanted:
            run_screen(scenario, names)
        if "estimators" in wanted:
            run_estimators(scenario, names)
    if "margin" in wanted:
        run_margin(names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
