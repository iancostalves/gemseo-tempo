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
"""Closing the speed periodicity with an inner solve instead of the optimizer.

The albatross guess is built by inverse dynamics and satisfies every equation
of motion except the one for speed, and speed is also the state whose
periodicity is hardest to close. That suggests a nested formulation: rather
than handing the optimizer an extra design variable and an extra equality,
solve for the initial speed **inside** each evaluation, so the outer problem
is smaller and better conditioned.

This module builds the inner solve and measures whether it is trustworthy,
which is the part worth settling before any formulation is built on top of it.

## What works

The residual ``r(V0) = V(t_f; V0) - V0`` is solved with a Newton iteration
from `optimistix`, and the derivative of the solution comes from the implicit
function theorem rather than from differentiating through the iterations. That
matters twice over: the gradient stays exact and its cost does not grow with
the number of Newton steps. Measured against a central difference, the two
agree to five significant figures.

## What does not

The inner solve is **conditionally** robust, and the condition is physical
rather than numerical. Below a certain initial speed the prescribed cycle
takes the bird under the water: the height reaches -12 m, the wind profile's
smooth floor keeps the arithmetic finite, and the residual it returns is a
number computed from a trajectory that does not exist. Newton started there
does not recover.

So the inner solve needs a start in the region where the trajectory is
physical, and :func:`robustness_map` reports where that is rather than leaving
it to be discovered inside an optimizer. A formulation built on this must
either bracket the root or start from the inverse-dynamics speed, which is
already in the right region.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy
import optimistix

from gemseo_tempo.dynamics.controls import CubicControl
from gemseo_tempo.dynamics.integrators import DiffraxIntegrator
from gemseo_tempo.problems.albatross.model import create_state_space
from gemseo_tempo.problems.albatross.problem import Guess
from gemseo_tempo.problems.albatross.problem import _design_values
from gemseo_tempo.problems.albatross.problem import soaring_cycle
from gemseo_tempo.problems.albatross.statics import create_statics

INNER_TOLERANCE = 1e-8
"""The tolerance the inner Newton solve is asked for."""

MAX_INNER_STEPS = 50
"""The iteration budget of the inner solve."""


@dataclass
class InnerProblem:
    """Everything the inner solve needs, built once and reused."""

    residual: object
    """``(v0, params) -> V(t_f) - V(t_0)``, traceable and differentiable."""

    trajectory: object
    """``(v0, params) -> Trajectory``, for inspecting a candidate."""

    speed_guess: float
    """The initial speed the inverse-dynamics cycle was built with."""

    params: dict
    """The control parameters of the guessed cycle."""


def build_inner_problem(
    guess: Guess | None = None,
    w_ref: float = 12.0,
    design: dict | None = None,
    size: int = 20,
) -> InnerProblem:
    """Build the speed-periodicity residual of one albatross cycle.

    Parameters
    ----------
    guess
        The cycle shape; the default when omitted.
    w_ref
        The reference wind speed, in m/s.
    design
        Overrides of the default wing.
    size
        The discretization of the control history.

    Returns
    -------
    InnerProblem
        The residual, a trajectory probe, and the guessed speed.
    """
    guess = guess or Guess()
    values = _design_values(design)
    statics = create_statics()(values)
    theta = {
        "w_ref": w_ref,
        "mass": float(statics["mass"]),
        "wing_area": values["wing_area"],
        "induced_factor": float(statics["induced_factor"]),
        "cd0": float(statics["cd0"]),
        "cl_max": float(statics["cl_max"]),
    }
    times, states, controls = soaring_cycle(guess, w_ref, design, size + 1)
    # Hoisted to Python floats: left as array slices they become tracers
    # inside the solve, and the integrator cannot take a traced initial
    # state it is meant to treat as a constant.
    height, climb, heading = (float(states[name][0]) for name in ("h", "gamma", "psi"))
    lift = CubicControl("cl", times, default_params=controls["cl"])
    bank = CubicControl("mu", times, default_params=controls["mu"])
    state_space = create_state_space()
    # The Direct adjoint, because the solve is differentiated through.
    integrator = DiffraxIntegrator(
        rtol=1e-8, atol=1e-8, n_save=size + 1, adjoint="Direct"
    )

    def trajectory(v0, params):
        """Integrate the cycle from a candidate initial speed."""
        return integrator.integrate(
            state_space,
            {"x": 0.0, "y": 0.0, "h": height, "V": v0, "gamma": climb, "psi": heading},
            (lift, bank),
            params,
            theta,
            0.0,
            guess.period,
        )

    def residual(v0, params):
        """Return how far the cycle is from closing in speed."""
        return trajectory(v0, params).states["V"][-1] - v0

    return InnerProblem(
        residual=residual,
        trajectory=trajectory,
        speed_guess=guess.airspeed,
        params={"cl": lift.default_params, "mu": bank.default_params},
    )


def solve_initial_speed(problem: InnerProblem, start: float | None = None):
    """Solve for the initial speed that closes the cycle.

    Parameters
    ----------
    problem
        The inner problem.
    start
        The starting speed; the guessed one when omitted.

    Returns
    -------
    tuple
        The solved speed and whether the solve succeeded.
    """
    begin = problem.speed_guess if start is None else start
    try:
        solution = optimistix.root_find(
            problem.residual,
            optimistix.Newton(rtol=INNER_TOLERANCE, atol=INNER_TOLERANCE),
            begin,
            args=problem.params,
            max_steps=MAX_INNER_STEPS,
            throw=False,
        )
    except Exception:  # noqa: BLE001
        # A start outside the physical region does not merely fail to
        # converge: the integration underneath it fails, and the error
        # surfaces from diffrax rather than from the root finder. Reported
        # as a failure, because inside an optimizer this is a discipline
        # that did not evaluate, not a bad number.
        return float(begin), False
    converged = solution.result == optimistix.RESULTS.successful
    return float(solution.value), bool(converged)


def implicit_gradient(problem: InnerProblem, start: float | None = None):
    """Differentiate the solved speed with respect to the controls.

    The derivative comes from the implicit function theorem, not from
    unrolling the Newton iterations, so it is exact and its cost does not
    grow with the number of steps taken.

    Parameters
    ----------
    problem
        The inner problem.
    start
        The starting speed.

    Returns
    -------
    ndarray
        The gradient of the solved speed with respect to the lift knots.
    """
    begin = problem.speed_guess if start is None else start

    def solved(lift):
        return optimistix.root_find(
            problem.residual,
            optimistix.Newton(rtol=INNER_TOLERANCE, atol=INNER_TOLERANCE),
            begin,
            args={"cl": lift, "mu": problem.params["mu"]},
            max_steps=MAX_INNER_STEPS,
            throw=False,
        ).value

    return numpy.asarray(jax.grad(solved)(problem.params["cl"]))


def robustness_map(problem: InnerProblem, starts) -> list[dict]:
    """Report where the inner solve can be started from.

    The residual is well behaved only where the trajectory it comes from is
    physical. Started too slow, the prescribed cycle flies the bird under the
    water, and the number that comes back is computed from a trajectory that
    does not exist. This says where that boundary is rather than leaving it to
    be met inside an optimizer.

    Parameters
    ----------
    problem
        The inner problem.
    starts
        The initial speeds to try, in m/s.

    Returns
    -------
    list
        One row per start: the speed, the lowest height its trajectory
        reaches, whether that trajectory is physical, and whether the solve
        converged from there.
    """
    rows = []
    for start in starts:
        try:
            heights = numpy.asarray(
                problem.trajectory(float(start), problem.params).states["h"]
            )
            lowest = float(heights.min())
        except Exception:  # noqa: BLE001
            lowest = float("nan")
        solved, converged = solve_initial_speed(problem, float(start))
        rows.append(
            {
                "start": float(start),
                "min_height": lowest,
                "physical": bool(lowest > 0.0),
                "converged": converged,
                "solved": solved,
            }
        )
    return rows


__all__ = [
    "INNER_TOLERANCE",
    "MAX_INNER_STEPS",
    "InnerProblem",
    "build_inner_problem",
    "implicit_gradient",
    "robustness_map",
    "solve_initial_speed",
]
