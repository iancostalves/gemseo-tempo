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
"""The brachistochrone's results: variants, transcriptions, the wire."""

from __future__ import annotations

import time

import numpy

from .common import _benchmark_transcriptions
from .common import _compile
from .common import _samples
from .common import _solve
from .common import _solve_checked
from .common import _trajectory_arrays
from .common import _write_arrays
from .common import _write_json
from .propagation import _gradient_cost
from .propagation import _propagation_study
from .propagation import _spread
from .propagation import _statistic_of_design


def generate_brachistochrone(smoke: bool = False) -> None:
    """Compute the guided tour's results.

    Parameters
    ----------
    smoke
        Whether to run the tiny version, for the tests.
    """
    from gemseo_tempo.problems.brachistochrone import VARIANTS
    from gemseo_tempo.problems.brachistochrone import create_optimization
    from gemseo_tempo.problems.brachistochrone import cycloid_time
    from gemseo_tempo.problems.brachistochrone import reference_time
    from gemseo_tempo.scenario.trajectories import optimal_trajectory

    print("brachistochrone")
    size = 8 if smoke else 20
    budget = 120 if smoke else 300

    # The three variants, on the same grid, with the analytic cycloid.
    optima = {}
    for variant in VARIANTS:
        scenario, gemseo_scenario = create_optimization(variant=variant, size=size)
        objective, design, feasible = _solve(gemseo_scenario, budget)
        trajectory = optimal_trajectory(scenario, gemseo_scenario)
        _write_arrays(
            f"brachistochrone/variant_{variant}.npz", **_trajectory_arrays(trajectory)
        )
        optima[variant] = {
            "time": objective,
            "reference": float(reference_time(variant)),
            "feasible": feasible,
        }
    _write_json("brachistochrone/variants.json", optima)

    # The cycloid itself, sampled finely, as the answer to reproduce. theta_f
    # solves the same transcendental aspect-ratio equation cycloid_time()
    # does; it is not pi in general (only the target's aspect ratio would
    # make it so), so it is solved for here rather than assumed, the way an
    # earlier version of this generator did, which quietly stored a curve
    # that reached (pi, 2) instead of the actual (2, 2) target.
    import math

    from scipy.optimize import brentq

    from gemseo_tempo.problems.brachistochrone.model import TARGET

    x_target, y_target = TARGET
    ratio = x_target / y_target
    theta_final = brentq(
        lambda theta: (theta - math.sin(theta)) / (1.0 - math.cos(theta)) - ratio,
        1e-6,
        2.0 * math.pi - 1e-6,
    )
    radius = y_target / (1.0 - math.cos(theta_final))
    theta = numpy.linspace(0.0, theta_final, 200)
    _write_arrays(
        "brachistochrone/cycloid.npz",
        x=radius * (theta - numpy.sin(theta)),
        y=radius * (1.0 - numpy.cos(theta)),
        time=numpy.asarray(cycloid_time()),
    )

    # The three transcriptions on the ideal variant, against the cycloid.
    _benchmark_transcriptions(
        "brachistochrone",
        lambda transcription, count: create_optimization(
            transcription=transcription, variant="ideal", size=count
        ),
        (5, 10) if smoke else (5, 10, 20, 40, 80),
        budget,
        reference=float(cycloid_time()),
        indirect={"control_grid": 32},
    )

    _generate_wire_uncertainty(smoke)


def _generate_wire_uncertainty(smoke: bool) -> None:
    """Propagate the uncertainty through the optimal wire, then design for it.

    Friction, drag and gravity are uncertain; the wire, once built, is not.
    Three wires are compared on a sample none of them was designed on: the
    cycloid, designed for the ideal bead; the wire designed for the mean
    physics; and the wire minimizing the mean plus twice the deviation of
    the descent time over a fixed sample. The last is solved twice from the
    same start, each derivative at its best: exact, in the cheaper mode of
    each output group, through the checkpointed adjoint, which is exact on
    this problem and the fastest measured on it; and by forward differences
    batched into one vectorized call, at the step that comes closest to the
    exact gradient there. The cost of one gradient of that objective is
    measured as the wire is refined.

    Parameters
    ----------
    smoke
        Whether to run the tiny version, for the tests.
    """
    from gemseo_tempo.dynamics.integrators import DiffraxIntegrator
    from gemseo_tempo.problems.brachistochrone import REGISTRY
    from gemseo_tempo.problems.brachistochrone import create_wire_optimization
    from gemseo_tempo.problems.brachistochrone import create_wire_scenario
    from gemseo_tempo.problems.brachistochrone.wire import DESCENT_TIME
    from gemseo_tempo.problems.brachistochrone.wire import wire_bounds
    from gemseo_tempo.scenario.formulations import OptimalControl
    from gemseo_tempo.uncertainty import nominal_values
    from gemseo_tempo.uncertainty import uncertain_names
    from gemseo_tempo.uncertainty.robust import MARGIN
    from gemseo_tempo.uncertainty.robust import robust_discipline

    size = 10 if smoke else 20
    budget = 100 if smoke else 400
    factor = 2.0
    names = uncertain_names(REGISTRY)
    nominal = nominal_values(REGISTRY, names)
    design_sample, check_sample = _samples(
        REGISTRY, names, (16, 256) if smoke else (64, 4000)
    )

    scenario, gemseo_scenario = create_wire_optimization(size=size, constants=nominal)
    _propagation_study(
        "brachistochrone", scenario, gemseo_scenario, REGISTRY, DESCENT_TIME, smoke
    )

    def spread(design):
        return _spread(
            create_wire_scenario(size=size, constants=nominal),
            names,
            check_sample,
            DESCENT_TIME,
            design,
        )

    designs = {}
    for label, constants in (("ideal", None), ("nominal", nominal)):
        _, gemseo_scenario = create_wire_optimization(size=size, constants=constants)
        started = time.perf_counter()
        _, design, met = _solve(gemseo_scenario, budget)
        designs[label] = {
            "met": met,
            "seconds": time.perf_counter() - started,
            "calls": len(gemseo_scenario.formulation.optimization_problem.database),
            **spread(design),
        }
    costs = []
    for count in (10, 20) if smoke else (10, 20, 40, 80):
        functions = {}
        for adjoint in ("Direct", "RecursiveCheckpoint"):
            functions[adjoint], point = _statistic_of_design(
                create_wire_scenario(
                    size=count,
                    constants=nominal,
                    integrator=DiffraxIntegrator(
                        dt0=1.0 / 400, n_save=count + 1, adjoint=adjoint
                    ),
                ),
                names,
                design_sample,
                DESCENT_TIME,
                MARGIN,
                factor=factor,
            )
        costs.append({"size": count, **_gradient_cost(functions, point)})
    # Measured at the start of the robust solve, on its own discretization.
    fd_step = next(cost["fd_step"] for cost in costs if cost["size"] == size)

    for label, modes in (("robust", ("auto", "auto")), ("robust_fd", ("fd", "fd"))):
        scenario = create_wire_scenario(size=size, constants=nominal)
        formulation = OptimalControl(
            scenario,
            bounds=wire_bounds(scenario),
            disciplines=[
                robust_discipline(
                    scenario,
                    names,
                    design_sample,
                    DESCENT_TIME,
                    factor=factor,
                    modes=modes,
                    fd_step=fd_step,
                )
            ],
            objective=f"{DESCENT_TIME}.{MARGIN}",
        )
        gemseo_scenario = formulation.create_gemseo_scenario()
        compile_seconds = _compile(formulation.disciplines[0], True)
        started = time.perf_counter()
        objective, design, met, _ = _solve_checked(scenario, gemseo_scenario, budget)
        designs[label] = {
            "objective": objective,
            "met": met,
            "modes": list(modes),
            "compile_seconds": compile_seconds,
            "seconds": time.perf_counter() - started,
            "calls": len(gemseo_scenario.formulation.optimization_problem.database),
            **spread(design),
        }
    _write_json(
        "brachistochrone/robust.json",
        {
            "factor": factor,
            "design_sample": len(design_sample),
            "check_sample": len(check_sample),
            "fd_step": fd_step,
            "designs": designs,
            "gradient_cost": costs,
        },
    )
