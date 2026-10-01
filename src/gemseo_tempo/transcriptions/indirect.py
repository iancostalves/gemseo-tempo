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
"""Indirect single shooting from the Pontryagin necessary conditions.

The Hamiltonian of the problem is ``H = l + lam^T f``, with ``l`` the
running cost and ``f`` the dynamics. Every necessary condition follows
from it, or from the boundary cost ``Phi = phi + nu^T psi``, by
differentiation, and automatic differentiation supplies each one, so the
adjoint equations are never derived by hand:

- the costates, ``d(lam)/dt = -(df/ds)^T lam - dl/ds``;
- the control, the minimizer of ``H`` over the control box at each
  instant, found by a grid search and polished by Newton steps, so a
  Hamiltonian that is not convex in the control, a boat polar for one,
  still yields its global minimizer;
- the transversality conditions, ``lam(t1) = dPhi/ds(t1)`` on free final
  states, ``lam(t0) = -dPhi/ds(t0)`` on free initial states, and
  ``H(t1) + dPhi/dt1 = 0`` when the final time is free, with one
  multiplier ``nu`` per terminal or periodicity equality;
- the stationarity of the design constants, ``dPhi/dc + int dH/dc dt = 0``,
  carried through the statics chain when a constant feeds it.

Path inequalities change the necessary conditions into ones with
multiplier functions and junctions, which no generic shooting method can
guess. They are handled here by a smooth exterior penalty added to the
running cost, which turns the problem into an unconstrained one whose
necessary conditions are the ones above; the answer approaches the
constrained one as the penalty weight grows.

The optimizer sees the unknowns of the boundary-value problem: the initial
costates, and the multipliers, free initial states, final time and design
constants when the problem has them. It drives the residuals of the
conditions above to zero as a least-squares objective, so the rest of the
scenario machinery applies unchanged.

**The objective is not the first functional here.** In the direct
transcriptions ``functionals[0]`` is what the optimizer minimizes. In this
one the functionals define the optimal-control problem that shapes ``H``,
while the optimizer minimizes :data:`RESIDUAL_NORM`; the achieved cost of
the underlying problem is reported separately as :data:`OCP_COST`.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy

from gemseo_tempo.dynamics.integrators import FORWARD
from gemseo_tempo.dynamics.integrators import REVERSE
from gemseo_tempo.dynamics.integrators import Trajectory
from gemseo_tempo.dynamics.state_space import StateSpace
from gemseo_tempo.dynamics.tables import smooth_clip
from gemseo_tempo.dynamics.time_scaling import normalize_time
from gemseo_tempo.outputs.constraints import PathConstraint
from gemseo_tempo.outputs.constraints import PeriodicityConstraint
from gemseo_tempo.outputs.constraints import TerminalConstraint
from gemseo_tempo.outputs.functionals import FinalRate
from gemseo_tempo.outputs.functionals import Lagrange
from gemseo_tempo.outputs.functionals import Mayer
from gemseo_tempo.outputs.functionals import TimeIntegral
from gemseo_tempo.transcriptions.base import CONSTANT_PREFIX
from gemseo_tempo.transcriptions.base import CONTROL_PREFIX
from gemseo_tempo.transcriptions.base import COSTATE_PREFIX
from gemseo_tempo.transcriptions.base import INITIAL_PREFIX
from gemseo_tempo.transcriptions.base import MULTIPLIER_PREFIX
from gemseo_tempo.transcriptions.base import T_FINAL
from gemseo_tempo.transcriptions.base import TIMES
from gemseo_tempo.transcriptions.base import TRAJECTORY_PREFIX
from gemseo_tempo.transcriptions.base import AbstractTranscription
from gemseo_tempo.transcriptions.base import TranscribedProblem
from gemseo_tempo.transcriptions.base import VariableSpec
from gemseo_tempo.transcriptions.base import resolve_parameter_sources

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from gemseo_tempo.dynamics.controls import AbstractControl
    from gemseo_tempo.dynamics.integrators import AbstractIntegrator
    from gemseo_tempo.outputs.constraints import AbstractConstraint
    from gemseo_tempo.outputs.functionals import AbstractFunctional
    from gemseo_tempo.statics.chain import StaticsChain
    from gemseo_tempo.transcriptions.base import BoundarySpec

RESIDUAL = "residual"
"""The output name of the necessary-condition residual vector."""

RESIDUAL_NORM = "residual_norm"
"""The output name of the residual sum of squares (the objective)."""

OCP_COST = "ocp_cost"
"""The output name of the achieved cost of the optimal-control problem."""

_COSTATE_STATE = "_lam_"
"""Internal prefix of the costate components of the augmented state."""

_COST_STATE = "_ocp_cost"
"""Internal name of the running-cost quadrature state."""

SEGMENT_PREFIX = "segment."
"""The input prefix of the states at the interior shooting nodes; the
costates there carry ``"costate."`` in front of it."""

_PARAMETER_ADJOINT = "_mu_"
"""Internal prefix of the quadratures of dH/dtheta, one per parameter."""


class IndirectShooting(AbstractTranscription):
    """Indirect single shooting driven by the Pontryagin conditions.

    Parameters
    ----------
    u_star
        The analytic minimizer ``(t, s, lam, theta) -> dict`` of the
        Hamiltonian, when one is available. Otherwise it is found
        numerically; see ``control_grid``.
    newton_steps
        The number of Newton steps polishing the minimizer. The loop is
        unrolled, so it stays jit-safe; one step is exact when ``H`` is
        quadratic in ``u``.
    smooth_bounds_sharpness
        The sharpness of the smooth saturation applied to ``u*`` when
        ``control_bounds`` are given and there is no grid. It maps to the
        smoothing parameter as ``eps = (1 / sharpness) ** 2``.
    control_bounds
        The ``(lower, upper)`` bounds of each control, keyed by name. The
        grid search needs one for every control.
    differentiation_method
        The mode used for every derivative of the necessary conditions,
        ``"reverse"`` (default) or ``"forward"``. It selects
        :func:`jax.jacrev` or :func:`jax.jacfwd`, and is independent of
        the differentiation mode of the assembled discipline.
    costate_guess
        The initial guess of each initial costate, keyed by state name;
        zero when omitted.
    control_grid
        The number of grid points per control of the global search for
        the minimizer of ``H``, zero for none. Without a grid, the Newton
        steps start from zero, which finds the minimizer only when ``H``
        is strictly convex in ``u``. With one, the best grid point starts
        them and each step is kept only if it lowers ``H``, within the
        bounds.
    path_penalty
        The weight of the exterior penalty standing in for each path
        inequality.
    path_smoothing
        The width over which the penalty switches on, in the units of the
        constraint.
    design_parameters
        The constants the necessary conditions also solve for, such as a
        design variable of the vehicle or the wind a cycle can live in.
    multiplier_guess
        The initial guess of the multipliers of each terminal or
        periodicity equality, keyed by constraint name; zero when omitted.
    segments
        The number of shooting segments. Costates of a stable system grow
        forward in time, often exponentially, so over a long horizon one
        segment turns a small error in the initial costates into a huge
        one at the end. Several segments, each started from unknown states
        and costates that continuity conditions join up, keep every
        integration short; this is multiple shooting.
    node_guess
        The initial guess of the states and costates at the interior nodes,
        keyed by state name and by ``"costate." + name``, one value per
        node; the initial ones repeated when omitted.

    Notes
    -----
    Scope limits: path inequalities are penalized rather than enforced
    (see the module docstring); terminal constraints must be equalities;
    control-group constraints and ``FinalRate`` functionals are not part
    of a Bolza problem and are refused. Only the monolithic assembly is
    available.

    The initial costates are notoriously sensitive to their guess. A
    direct solution provides a good one: see :func:`estimate_costates`.
    """

    def __init__(
        self,
        u_star: Callable | None = None,
        newton_steps: int = 5,
        smooth_bounds_sharpness: float = 20.0,
        control_bounds: Mapping[str, tuple[float, float]] | None = None,
        differentiation_method: str = REVERSE,
        costate_guess: Mapping[str, float] | None = None,
        control_grid: int = 0,
        path_penalty: float = 1e3,
        path_smoothing: float = 1e-2,
        design_parameters: Sequence[str] = (),
        multiplier_guess: Mapping[str, Sequence[float]] | None = None,
        segments: int = 1,
        node_guess: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        """Store the settings.

        Parameters
        ----------
        u_star
            The analytic Hamiltonian minimizer, if any.
        newton_steps
            The number of Newton steps.
        smooth_bounds_sharpness
            The saturation sharpness.
        control_bounds
            The bounds of each control.
        differentiation_method
            The differentiation mode of the necessary conditions.
        costate_guess
            The initial guess of each initial costate.
        control_grid
            The grid points per control of the minimizer search.
        path_penalty
            The weight of the path-constraint penalty.
        path_smoothing
            The width of the path-constraint penalty.
        design_parameters
            The constants solved for.
        multiplier_guess
            The initial guess of the multipliers of each terminal or
            periodicity equality, keyed by constraint name.
        segments
            The number of shooting segments.
        node_guess
            The initial guess of the states and costates at the interior
            nodes.

        Raises
        ------
        ValueError
            If the differentiation method is unknown.
        """
        if differentiation_method not in (FORWARD, REVERSE):
            raise ValueError(
                f"Unknown differentiation method {differentiation_method!r};"
                f" use {FORWARD!r} or {REVERSE!r}."
            )
        self.u_star = u_star
        self.newton_steps = int(newton_steps)
        self.smooth_bounds_sharpness = float(smooth_bounds_sharpness)
        self.control_bounds = dict(control_bounds or {})
        self.differentiation_method = differentiation_method
        self.costate_guess = dict(costate_guess or {})
        self.control_grid = int(control_grid)
        self.path_penalty = float(path_penalty)
        self.path_smoothing = float(path_smoothing)
        self.design_parameters = tuple(design_parameters)
        self.multiplier_guess = dict(multiplier_guess or {})
        self.segments = int(segments)
        self.node_guess = dict(node_guess or {})

    @property
    def jacobian(self) -> Callable:
        """The Jacobian transform matching the differentiation mode.

        Returns
        -------
        Callable
            :func:`jax.jacfwd` in forward mode, :func:`jax.jacrev` in
            reverse mode.
        """
        if self.differentiation_method == FORWARD:
            return jax.jacfwd
        return jax.jacrev

    def transcribe(
        self,
        statics: StaticsChain | None,
        state_space,
        integrator: AbstractIntegrator,
        controls: Sequence[AbstractControl],
        functionals: Sequence[AbstractFunctional],
        constraints: Sequence[AbstractConstraint],
        boundary: BoundarySpec,
        constants: Mapping[str, float] | None = None,
        t0: float = 0.0,
        t1: float = 1.0,
    ) -> TranscribedProblem:
        """Build the indirect shooting problem.

        Parameters
        ----------
        statics
            The constant-fed chain running ahead of the dynamics, if any.
        state_space
            The dynamics model.
        integrator
            The integrator applied to the augmented state-costate system.
        controls
            Unused: the controls follow from the Hamiltonian. Passing an
            empty sequence is the natural choice; a non-empty one must
            match the state-space control names.
        functionals
            The functionals defining the optimal-control problem, not the
            optimizer objective (see the module docstring).
        constraints
            Path inequalities, penalized; terminal and periodicity
            equalities, each with its multiplier.
        boundary
            The boundary specification.
        constants
            The values of constant inputs.
        t0
            The initial time.
        t1
            The final time (ignored when the final time is free).

        Returns
        -------
        TranscribedProblem
            The problem (monolithic only; no split stage pieces).

        Raises
        ------
        ValueError
            On any structural mismatch or unsupported feature.
        """
        constants = dict(constants or {})
        state_names = tuple(state_space.states)
        control_names = tuple(state_space.controls)
        self._validate(
            state_space, tuple(controls), functionals, constraints, boundary, t0
        )

        statics_output_names = (
            set(statics.chain.output_grammar.names) if statics else set()
        )
        statics_input_names = list(statics.chain.input_grammar.names) if statics else []
        theta_inputs = resolve_parameter_sources(
            state_space, statics_output_names, constants
        )
        unknown = sorted(
            name
            for name in self.design_parameters
            if name not in constants and name not in statics_input_names
        )
        if unknown:
            raise ValueError(f"Design parameters {unknown} are not constant inputs.")
        missing = sorted(
            name for name in control_names if name not in self.control_bounds
        )
        if self.control_grid and missing:
            raise ValueError(f"The control grid needs bounds on controls {missing}.")

        costs = _Costs(
            state_space,
            functionals,
            constraints,
            self.path_penalty,
            self.path_smoothing,
            t0,
        )
        terminals = costs.terminals
        bolza_integrand = costs.bolza_integrand
        running_cost = costs.running_cost
        terminal_cost = costs.terminal_cost
        boundary_cost = costs.boundary_cost
        hamiltonian = costs.hamiltonian
        free_initial = [n for n in state_names if not boundary.is_initial_fixed(n)]
        jacobian = self.jacobian

        def saturate(u):
            """Apply the smooth saturation of the bounded controls."""
            if not self.control_bounds:
                return u
            eps = (1.0 / self.smooth_bounds_sharpness) ** 2
            return {
                name: (
                    smooth_clip(value, *self.control_bounds[name], eps)
                    if name in self.control_bounds
                    else value
                )
                for name, value in u.items()
            }

        if self.control_grid:
            lower = jnp.asarray([self.control_bounds[n][0] for n in control_names])
            upper = jnp.asarray([self.control_bounds[n][1] for n in control_names])
            axes = [
                jnp.linspace(low, high, self.control_grid)
                for low, high in zip(lower, upper, strict=True)
            ]
            grid = jnp.stack(
                [axis.ravel() for axis in jnp.meshgrid(*axes, indexing="ij")], axis=1
            )

        def optimal_control(t, s, lam, theta):
            """Return the Hamiltonian-minimizing control at one instant."""
            if self.u_star is not None:
                return saturate(dict(self.u_star(t, s, lam, theta)))
            if not control_names:
                return {}

            def h_of_vector(vector):
                u = {n: vector[k] for k, n in enumerate(control_names)}
                return hamiltonian(t, s, lam, u, theta)

            gradient = jacobian(h_of_vector)
            hessian = jacobian(gradient)
            if not self.control_grid:
                vector = jnp.zeros(len(control_names))
                for _ in range(self.newton_steps):
                    vector = vector - jnp.linalg.solve(
                        hessian(vector), gradient(vector)
                    )
                return saturate({n: vector[k] for k, n in enumerate(control_names)})

            # The best grid point starts the polish; only the polish carries
            # derivatives, which is where they belong at a smooth minimum.
            values = jax.vmap(h_of_vector)(grid)
            vector = jax.lax.stop_gradient(grid[jnp.argmin(values)])
            identity = jnp.eye(len(control_names))
            for _ in range(self.newton_steps):
                curvature = hessian(vector)
                # Levenberg damping keeps the step a descent direction when
                # the curvature is not positive.
                damping = jnp.abs(jnp.trace(curvature)) * 1e-6 + 1e-9
                step = jnp.linalg.solve(
                    curvature + damping * identity, gradient(vector)
                )
                candidate = jnp.clip(vector - step, lower, upper)
                better = h_of_vector(candidate) < h_of_vector(vector)
                vector = jnp.where(better, candidate, vector)
            return {n: vector[k] for k, n in enumerate(control_names)}

        def costate_rates(t, s, lam, u, theta):
            """Return -(df/ds)^T lam - dl/ds from the dynamics Jacobian."""
            dynamics_jacobian = jacobian(
                lambda states: state_space.rhs(t, states, u, theta)
            )(s)
            cost_gradient = jacobian(lambda states: running_cost(t, states, u, theta))(
                s
            )
            rates = {}
            for row in state_names:
                total = cost_gradient[row]
                for column in state_names:
                    total = total + lam[column] * dynamics_jacobian[column][row]
                rates[row] = -total
            return rates

        parameter_names = tuple(theta_inputs) if self.design_parameters else ()

        def split_augmented(y):
            """Split the augmented state into states and costates."""
            s = {name: y[name] for name in state_names}
            lam = {name: y[_COSTATE_STATE + name] for name in state_names}
            return s, lam

        def augmented_rhs(t, y, u_unused, theta):
            """Return the rates of the states, costates and quadratures."""
            del u_unused
            s, lam = split_augmented(y)
            u = optimal_control(t, s, lam, theta)
            rates = dict(state_space.rhs(t, s, u, theta))
            for name, rate in costate_rates(t, s, lam, u, theta).items():
                rates[_COSTATE_STATE + name] = rate
            rates[_COST_STATE] = bolza_integrand(t, s, u, theta)
            if parameter_names:
                sensitivity = jacobian(
                    lambda parameters: hamiltonian(t, s, lam, u, parameters)
                )({name: theta[name] for name in parameter_names})
                for name in parameter_names:
                    rates[_PARAMETER_ADJOINT + name] = sensitivity[name]
            return rates

        def augmented_output(t, y, u_unused, theta):
            """Return the model outputs plus the realised controls."""
            del u_unused
            s, lam = split_augmented(y)
            u = optimal_control(t, s, lam, theta)
            outputs = dict(state_space.output(t, s, u, theta))
            outputs.update(u)
            return outputs

        augmented = StateSpace(
            states=(
                *state_names,
                *(_COSTATE_STATE + name for name in state_names),
                _COST_STATE,
                *(_PARAMETER_ADJOINT + name for name in parameter_names),
            ),
            controls=(),
            parameters=state_space.parameters,
            outputs=(*state_space.outputs, *control_names),
            rhs=augmented_rhs,
            output=augmented_output,
        )
        simulated = normalize_time(augmented) if boundary.free_final_time else augmented

        def theta_of(data):
            """Run the statics and read the parameters as scalars.

            Constants are stored as 1D arrays for GEMSEO, while the
            necessary conditions do direct scalar arithmetic on them.
            """
            data = dict(data)
            if statics is not None:
                data.update(
                    statics(
                        {
                            name: data[CONSTANT_PREFIX + name]
                            for name in statics_input_names
                        }
                    )
                )
            return {
                parameter: jnp.squeeze(data[input_name])
                for parameter, input_name in theta_inputs.items()
            }

        def jax_function(inputs):
            """Integrate the augmented system and return every output."""
            data = dict(inputs)
            theta = theta_of(data)
            initial = {
                name: jnp.squeeze(data[INITIAL_PREFIX + name])
                if name in free_initial
                else jnp.asarray(boundary.s0[name], dtype=float)
                for name in state_names
            }
            y0 = dict(initial)
            y0.update(
                {
                    _COSTATE_STATE + name: jnp.squeeze(data[COSTATE_PREFIX + name])
                    for name in state_names
                }
            )
            y0[_COST_STATE] = jnp.zeros(())
            for name in parameter_names:
                y0[_PARAMETER_ADJOINT + name] = jnp.zeros(())

            params, span = theta, (t0, t1)
            if boundary.free_final_time:
                t_final = jnp.squeeze(data[T_FINAL])
                params, span = {**theta, T_FINAL: t_final}, (0.0, 1.0)
            edges = [
                span[0] + (span[1] - span[0]) * k / self.segments
                for k in range(self.segments + 1)
            ]
            pieces, defects, start = [], [], y0
            for k in range(self.segments):
                if k:
                    end = {key: value[-1] for key, value in pieces[-1].states.items()}
                    node = {}
                    for name in state_names:
                        node[name] = data[SEGMENT_PREFIX + name][k - 1]
                        node[_COSTATE_STATE + name] = data[
                            COSTATE_PREFIX + SEGMENT_PREFIX + name
                        ][k - 1]
                    defects.extend(jnp.atleast_1d(end[key] - node[key]) for key in node)
                    start = {**end, **node}
                pieces.append(
                    integrator.integrate(
                        simulated, start, (), {}, params, edges[k], edges[k + 1]
                    )
                )
            trajectory = _stitch(pieces)
            times = trajectory.ts
            if boundary.free_final_time:
                times = times * t_final

            final = {name: trajectory.states[name][-1] for name in state_names}
            initial_costates = {
                name: trajectory.states[_COSTATE_STATE + name][0]
                for name in state_names
            }
            final_costates = {
                name: trajectory.states[_COSTATE_STATE + name][-1]
                for name in state_names
            }
            final_time = times[-1]
            nu = {
                constraint.name: jnp.atleast_1d(
                    data[MULTIPLIER_PREFIX + constraint.name]
                )
                for constraint in terminals
            }

            def phi(initial_states, final_states, time, parameters):
                return boundary_cost(initial_states, final_states, time, parameters, nu)

            entries = [
                _terminal_values(constraint, initial, final, final_time, theta)
                for constraint in terminals
            ]
            final_gradient = jacobian(
                lambda states: phi(initial, states, final_time, theta)
            )(final)
            for name in state_names:
                if boundary.is_final_fixed(name):
                    miss = final[name] - boundary.s1[name]
                else:
                    miss = final_costates[name] - final_gradient[name]
                entries.append(jnp.atleast_1d(miss))
            if free_initial:
                initial_gradient = jacobian(
                    lambda states: phi(states, final, final_time, theta)
                )(initial)
                entries.extend(
                    jnp.atleast_1d(initial_costates[name] + initial_gradient[name])
                    for name in free_initial
                )
            if boundary.free_final_time:
                control = optimal_control(final_time, final, final_costates, theta)
                time_gradient = jacobian(lambda time: phi(initial, final, time, theta))(
                    final_time
                )
                entries.append(
                    jnp.atleast_1d(
                        hamiltonian(final_time, final, final_costates, control, theta)
                        + time_gradient
                    )
                )
            if parameter_names:
                entries.extend(
                    self._parameter_residuals(
                        data,
                        theta_of,
                        theta,
                        phi,
                        jacobian,
                        trajectory,
                        parameter_names,
                        initial,
                        final,
                        final_time,
                    )
                )
            residual = jnp.concatenate(entries + defects)

            result = {TIMES: times}
            for name in state_names:
                result[TRAJECTORY_PREFIX + name] = trajectory.states[name]
                result[TRAJECTORY_PREFIX + COSTATE_PREFIX + name] = trajectory.states[
                    _COSTATE_STATE + name
                ]
            for name in (*state_space.outputs, *control_names):
                result[TRAJECTORY_PREFIX + name] = trajectory.outputs[name]
            result[RESIDUAL] = residual
            result[RESIDUAL_NORM] = jnp.sum(residual**2)
            result[OCP_COST] = trajectory.states[_COST_STATE][-1] + terminal_cost(
                final_time, final, theta
            )
            return {name: jnp.atleast_1d(value) for name, value in result.items()}

        design_variables: dict[str, VariableSpec] = {}
        input_defaults: dict = {}

        def declare(name, value):
            value = jnp.atleast_1d(jnp.asarray(value, dtype=float))
            design_variables[name] = VariableSpec(
                name=name, size=value.shape[0], value=value
            )
            input_defaults[name] = value

        for name in state_names:
            declare(COSTATE_PREFIX + name, self.costate_guess.get(name, 0.0))
        for name in free_initial:
            declare(INITIAL_PREFIX + name, boundary.s0[name])
        for constraint in terminals:
            size = (
                len(constraint.state_names)
                if isinstance(constraint, PeriodicityConstraint)
                else 1
            )
            declare(
                MULTIPLIER_PREFIX + constraint.name,
                self.multiplier_guess.get(constraint.name, jnp.zeros(size)),
            )
        if boundary.free_final_time:
            declare(T_FINAL, boundary.t_f)
        interior = self.segments - 1
        for name in state_names if interior else ():
            declare(
                SEGMENT_PREFIX + name,
                self.node_guess.get(name, jnp.full(interior, boundary.s0[name])),
            )
            declare(
                COSTATE_PREFIX + SEGMENT_PREFIX + name,
                self.node_guess.get(
                    COSTATE_PREFIX + name,
                    jnp.full(interior, self.costate_guess.get(name, 0.0)),
                ),
            )
        for name in statics_input_names:
            input_defaults[CONSTANT_PREFIX + name] = jnp.atleast_1d(
                jnp.asarray(
                    constants.get(name, statics.chain.default_input_data[name]),
                    dtype=float,
                )
            )
        for parameter, input_name in theta_inputs.items():
            if input_name.startswith(CONSTANT_PREFIX):
                input_defaults[input_name] = jnp.atleast_1d(
                    jnp.asarray(constants[parameter], dtype=float)
                )
        for name in self.design_parameters:
            declare(CONSTANT_PREFIX + name, input_defaults[CONSTANT_PREFIX + name])

        output_names = [
            TIMES,
            *(TRAJECTORY_PREFIX + name for name in state_names),
            *(TRAJECTORY_PREFIX + COSTATE_PREFIX + name for name in state_names),
            *(
                TRAJECTORY_PREFIX + name
                for name in (*state_space.outputs, *control_names)
            ),
            RESIDUAL,
            RESIDUAL_NORM,
            OCP_COST,
        ]
        return TranscribedProblem(
            design_variables=design_variables,
            jax_function=jax_function,
            objective_name=RESIDUAL_NORM,
            constraint_names=[],
            input_defaults=input_defaults,
            output_names=output_names,
        )

    def _parameter_residuals(
        self,
        data,
        theta_of,
        theta,
        phi,
        jacobian,
        trajectory,
        parameter_names,
        initial,
        final,
        final_time,
    ):
        """Return ``dPhi/dc + int dH/dc dt`` for each design constant.

        The Hamiltonian and the boundary cost see the parameters the
        dynamics read, which a statics chain may compute from the design
        constants; the chain rule back to the constants is one
        vector-Jacobian product through that chain.

        Returns
        -------
        list
            One residual per design constant.
        """
        boundary_gradient = jacobian(
            lambda parameters: phi(initial, final, final_time, parameters)
        )({name: theta[name] for name in parameter_names})
        cotangent = {
            name: boundary_gradient[name]
            + trajectory.states[_PARAMETER_ADJOINT + name][-1]
            for name in parameter_names
        }
        design = {
            CONSTANT_PREFIX + name: data[CONSTANT_PREFIX + name]
            for name in self.design_parameters
        }
        _, pullback = jax.vjp(lambda values: theta_of({**data, **values}), design)
        (gradient,) = pullback(cotangent)
        return [
            jnp.atleast_1d(jnp.sum(gradient[CONSTANT_PREFIX + name]))
            for name in self.design_parameters
        ]

    @staticmethod
    def _validate(state_space, controls, functionals, constraints, boundary, t0):
        """Check the problem against the scope of the indirect method.

        Raises
        ------
        ValueError
            On any structural mismatch or unsupported feature.
        """
        if controls:
            names = {control.name for control in controls}
            if names != set(state_space.controls):
                raise ValueError(
                    f"Control parameterizations {sorted(names)} do not match"
                    f" the state-space controls {sorted(state_space.controls)}."
                )
        for constraint in constraints:
            supported = isinstance(constraint, PathConstraint) or (
                isinstance(constraint, (TerminalConstraint, PeriodicityConstraint))
                and constraint.kind == "eq"
            )
            if not supported:
                raise ValueError(
                    f"Indirect shooting does not support the constraint"
                    f" {constraint.name!r}: only path inequalities, which it"
                    " penalizes, and terminal or periodicity equalities, which"
                    " it enforces with multipliers."
                )
        for functional in functionals:
            if isinstance(functional, FinalRate):
                raise ValueError(
                    f"FinalRate {functional.name!r} is not part of a Bolza"
                    " cost, so it cannot shape the Hamiltonian."
                )
        if boundary.free_final_time and t0 != 0.0:
            raise ValueError("free_final_time requires t0 = 0.")


def _stitch(pieces) -> Trajectory:
    """Join the segments of a multiple shooting into one trajectory.

    Each segment starts where the last one ended, so the shared save point
    is kept once.
    """
    if len(pieces) == 1:
        return pieces[0]

    def joined(series):
        return jnp.concatenate([series[0]] + [part[1:] for part in series[1:]])

    return Trajectory(
        ts=joined([piece.ts for piece in pieces]),
        states={
            key: joined([piece.states[key] for piece in pieces])
            for key in pieces[0].states
        },
        outputs={
            key: joined([piece.outputs[key] for piece in pieces])
            for key in pieces[0].outputs
        },
    )


def _terminal_values(constraint, initial, final, time, theta):
    """Return the residuals of one terminal or periodicity equality."""
    if isinstance(constraint, PeriodicityConstraint):
        return constraint.evaluate(initial, final)
    return jnp.atleast_1d(constraint.function(time, final, theta))


class _Costs:
    """The cost terms of a Bolza problem, shared by the shooting and its guess.

    Parameters
    ----------
    state_space
        The dynamics model.
    functionals
        The functionals defining the cost.
    constraints
        Path inequalities, penalized, and terminal or periodicity
        equalities, carried by multipliers.
    penalty
        The weight of the path penalty.
    smoothing
        The width of the path penalty.
    t0
        The initial time, the origin of any discounting.
    """

    def __init__(self, state_space, functionals, constraints, penalty, smoothing, t0):
        self.state_space = state_space
        self.paths = [c for c in constraints if isinstance(c, PathConstraint)]
        self.terminals = [
            c
            for c in constraints
            if isinstance(c, (TerminalConstraint, PeriodicityConstraint))
        ]
        self.mayers = [f for f in functionals if isinstance(f, Mayer)]
        self.lagranges = [f for f in functionals if isinstance(f, Lagrange)]
        self.integrals = [f for f in functionals if isinstance(f, TimeIntegral)]
        self.penalty = penalty
        self.smoothing = smoothing
        self.t0 = t0

    def bolza_integrand(self, t, s, u, theta):
        """Return the integrand l of the Bolza cost, discounts included."""
        total = jnp.zeros(())
        for functional in self.lagranges:
            total = total + functional.integrand(t, s, u, theta) * jnp.exp(
                -functional.rho * (t - self.t0)
            )
        if self.integrals:
            outputs = self.state_space.output(t, s, u, theta)
            for functional in self.integrals:
                total = total + outputs[functional.output_name] * jnp.exp(
                    -functional.rho * (t - self.t0)
                )
        return total

    def running_cost(self, t, s, u, theta):
        """Return the integrand plus the penalty of the path inequalities."""
        total = self.bolza_integrand(t, s, u, theta)
        for constraint in self.paths:
            excess = self.smoothing * jax.nn.softplus(
                constraint.function(t, s, u, theta) / self.smoothing
            )
            total = total + self.penalty * excess**2
        return total

    def terminal_cost(self, t, s, theta):
        """Return the Mayer terminal cost phi."""
        total = jnp.zeros(())
        for functional in self.mayers:
            total = total + functional.function(t, s, theta)
        return total

    def boundary_cost(self, initial, final, time, theta, nu):
        """Return Phi = phi + nu^T psi over the terminal equalities."""
        total = self.terminal_cost(time, final, theta)
        for constraint in self.terminals:
            total = total + jnp.dot(
                nu[constraint.name],
                _terminal_values(constraint, initial, final, time, theta),
            )
        return total

    def hamiltonian(self, t, s, lam, u, theta):
        """Return H = l + lam^T f."""
        rates = self.state_space.rhs(t, s, u, theta)
        total = self.running_cost(t, s, u, theta)
        for name in self.state_space.states:
            total = total + lam[name] * rates[name]
        return total


def indirect_from_direct(
    scenario, gemseo_scenario, control_grid: int = 16, integrator=None, **settings
):
    """Warm-start indirect shooting from a solved direct problem.

    A direct solution fixes everything the shooting needs except the
    costates and multipliers: the initial states, the final time and the
    design constants are read off it. Along its trajectory the costate
    equation is linear in the costates, ``lam(t) = M(t) lam(t0) + c(t)``,
    and so is every condition that pins them: stationarity ``dH/du = 0`` at
    each node, transversality on the free final and initial states, and
    ``H(t1) + dPhi/dt1 = 0`` when the final time is free. One
    least-squares solve returns the initial costates and the multipliers
    together, which is the guess the shooting is notoriously sensitive to.

    Parameters
    ----------
    scenario
        The solved direct :class:`.DynamicScenario`.
    gemseo_scenario
        The executed GEMSEO scenario.
    control_grid
        The grid points per control of the minimizer search.
    integrator
        The integrator of the state-costate system; the direct problem's
        when omitted. Costates can grow fast forward in time, which an
        adaptive integrator answers with ever smaller steps, so a fixed
        step is often the better choice here.
    **settings
        Forwarded to :class:`IndirectShooting`. The control bounds default
        to the registry's; the design parameters to the constants the
        direct problem optimized.

    Returns
    -------
    DynamicScenario
        The same problem under indirect shooting, seeded.
    """
    from gemseo_tempo.scenario.trajectories import optimal_trajectory

    definition = scenario.definition
    state_space = definition["state_space"]
    boundary = definition["boundary"]
    registry = definition["registry"]
    state_names = tuple(state_space.states)
    control_names = tuple(state_space.controls)
    optimum = gemseo_scenario.optimization_result.x_opt_as_dict
    design = {
        name[len(CONSTANT_PREFIX) :]: float(numpy.ravel(value)[0])
        for name, value in optimum.items()
        if name.startswith(CONSTANT_PREFIX)
    }
    constants = {**definition["constants"], **design}
    theta = _parameters(definition["statics"], state_space, constants)
    trajectory = optimal_trajectory(scenario, gemseo_scenario)
    settings.setdefault(
        "control_bounds",
        _control_bounds(gemseo_scenario.design_space, registry, control_names),
    )
    settings.setdefault("design_parameters", tuple(design))
    # A direct transcription minimizes its first functional and reports the
    # others; here every functional shapes the Hamiltonian, so only the
    # objective is carried over.
    objective = definition["functionals"][:1]
    costs = _Costs(
        state_space,
        objective,
        definition["constraints"],
        settings.get("path_penalty", 1e3),
        settings.get("path_smoothing", 1e-2),
        definition["t0"],
    )
    sensitivity = (
        jax.jacfwd(
            lambda values: _parameters(
                definition["statics"], state_space, {**constants, **values}
            )
        )({name: jnp.asarray(value) for name, value in design.items()})
        if design
        else {}
    )
    costates, multipliers, history = _estimate_costates(
        costs,
        boundary,
        trajectory,
        theta,
        state_names,
        control_names,
        {name: {key: sensitivity[key][name] for key in sensitivity} for name in design},
    )
    times = numpy.asarray(trajectory.ts)
    segments = settings.get("segments", 1)
    if segments > 1:
        # The interior nodes, seeded with the direct path and its costates.
        edges = times[0] + (times[-1] - times[0]) * numpy.arange(1, segments) / segments
        node_guess = {}
        for i, name in enumerate(state_names):
            node_guess[name] = numpy.interp(
                edges, times, numpy.asarray(trajectory.states[name])
            ).tolist()
            node_guess[COSTATE_PREFIX + name] = numpy.interp(
                edges, times, history[:, i]
            ).tolist()
        settings.setdefault("node_guess", node_guess)
    free_initial = [n for n in state_names if not boundary.is_initial_fixed(n)]
    seeded = dataclasses.replace(
        boundary,
        s0={
            **boundary.s0,
            **{n: float(trajectory.states[n][0]) for n in free_initial},
        },
        t_f=float(times[-1] - times[0]) if boundary.free_final_time else boundary.t_f,
    )
    transcription = IndirectShooting(
        control_grid=control_grid,
        costate_guess=costates,
        multiplier_guess=multipliers,
        **settings,
    )
    changes = {
        "controls": (),
        "functionals": objective,
        "boundary": seeded,
        "constants": constants,
    }
    if integrator is not None:
        changes["integrator"] = integrator
    return scenario.retranscribe(transcription, **changes)


def _control_bounds(design_space, registry, control_names) -> dict:
    """Return the box each control was optimized in.

    The direct problem's own bounds come first, since they are what its
    optimum respects; the registry's fill in for a control the design space
    does not carry, and a control bounded by neither is left out.
    """
    bounds = {}
    for name in control_names:
        variable = CONTROL_PREFIX + name
        if variable in design_space:
            bounds[name] = (
                float(numpy.min(design_space.get_lower_bound(variable))),
                float(numpy.max(design_space.get_upper_bound(variable))),
            )
        elif registry is not None and name in registry:
            entry = registry[name]
            if entry.lower_bound is not None and entry.upper_bound is not None:
                bounds[name] = (entry.lower_bound, entry.upper_bound)
    return bounds


def _parameters(statics, state_space, constants) -> dict:
    """Return the parameters the dynamics read, for given constants."""
    names = set(statics.chain.output_grammar.names) if statics else set()
    sources = resolve_parameter_sources(state_space, names, constants)
    outputs = {}
    if statics is not None:
        defaults = statics.chain.default_input_data
        outputs = statics(
            {
                name: jnp.atleast_1d(constants.get(name, defaults[name]))
                for name in statics.chain.input_grammar.names
            }
        )
    return {
        parameter: jnp.asarray(constants[parameter], dtype=float)
        if source.startswith(CONSTANT_PREFIX)
        else jnp.squeeze(outputs[source])
        for parameter, source in sources.items()
    }


def _estimate_costates(
    costs,
    boundary,
    trajectory,
    theta,
    state_names,
    control_names,
    design_jacobian=None,
    tolerance: float = 1e-3,
):
    """Solve for the costates and multipliers a known optimal path implies.

    The unknowns are the initial costates, the multipliers of the terminal
    equalities, and one non-negative multiplier per node and path
    inequality active there: an active constraint pushes on the costates,
    and leaving it out would seed the shooting with the costates of a
    different problem. Along the path, ``lam(t) = L(t) z + c(t)`` in the
    vector ``z`` of all of them, stepped by Heun's method between nodes.
    A design constant adds its stationarity, ``dPhi/dc + int dH/dc dt = 0``,
    which is what sets the scale of the costates when the constant is the
    objective; ``design_jacobian`` maps each constant to ``dtheta/dc``.

    Returns
    -------
    tuple
        The initial costates and the terminal multipliers, keyed by name,
        and the costates at every node.
    """
    from scipy.optimize import lsq_linear

    state_space = costs.state_space
    times = jnp.asarray(trajectory.ts)
    states = jnp.stack([jnp.asarray(trajectory.states[n]) for n in state_names], 1)
    controls = jnp.stack(
        [jnp.asarray(trajectory.controls[n]) for n in control_names], 1
    )

    def named(names, vector):
        return {name: vector[k] for k, name in enumerate(names)}

    def rates(t, s, u):
        values = state_space.rhs(
            t, named(state_names, s), named(control_names, u), theta
        )
        return jnp.stack([values[n] for n in state_names])

    def cost(t, s, u):
        return costs.bolza_integrand(
            t, named(state_names, s), named(control_names, u), theta
        )

    def limits(t, s, u):
        return jnp.stack(
            [
                c.function(t, named(state_names, s), named(control_names, u), theta)
                for c in costs.paths
            ]
            or [jnp.zeros(())]
        )

    def rates_of(t, s, u, parameters):
        values = state_space.rhs(
            t, named(state_names, s), named(control_names, u), parameters
        )
        return jnp.stack([values[n] for n in state_names])

    def cost_of(t, s, u, parameters):
        return costs.bolza_integrand(
            t, named(state_names, s), named(control_names, u), parameters
        )

    def limits_of(t, s, u, parameters):
        return jnp.stack(
            [
                c.function(
                    t, named(state_names, s), named(control_names, u), parameters
                )
                for c in costs.paths
            ]
            or [jnp.zeros(())]
        )

    at_nodes = jax.vmap(
        lambda t, s, u: (
            jax.jacfwd(rates, 1)(t, s, u),
            jax.jacfwd(rates, 2)(t, s, u),
            jax.grad(cost, 1)(t, s, u),
            jax.grad(cost, 2)(t, s, u),
            rates(t, s, u),
            cost(t, s, u),
            limits(t, s, u),
            jax.jacfwd(limits, 1)(t, s, u),
            jax.jacfwd(limits, 2)(t, s, u),
        )
    )
    a, b, q, r, f, running, g, g_s, g_u = (
        numpy.asarray(v) for v in at_nodes(times, states, controls)
    )
    if design_jacobian:
        by_parameter = jax.vmap(
            lambda t, s, u: (
                jax.jacfwd(rates_of, 3)(t, s, u, theta),
                jax.grad(cost_of, 3)(t, s, u, theta),
                jax.jacfwd(limits_of, 3)(t, s, u, theta),
            )
        )(times, states, controls)
    size, count = len(state_names), len(times)
    terminals = costs.terminals
    widths = [
        len(c.state_names) if isinstance(c, PeriodicityConstraint) else 1
        for c in terminals
    ]
    active = [
        (k, j)
        for k in range(count)
        for j in range(len(costs.paths))
        if g[k, j] > -tolerance
    ]
    first_path = size + sum(widths)
    unknowns = first_path + len(active)
    forcing = numpy.zeros((count, size, unknowns))
    for column, (k, j) in enumerate(active):
        forcing[k, :, first_path + column] = g_s[k, j]

    maps = [numpy.eye(size, unknowns)]
    offsets = [numpy.zeros(size)]
    for k in range(count - 1):
        step = float(times[k + 1] - times[k])
        slope_m = -a[k].T @ maps[k] - forcing[k]
        slope_c = -a[k].T @ offsets[k] - q[k]
        guess_m = maps[k] + step * slope_m
        guess_c = offsets[k] + step * slope_c
        maps.append(
            maps[k] + 0.5 * step * (slope_m - a[k + 1].T @ guess_m - forcing[k + 1])
        )
        offsets.append(
            offsets[k] + 0.5 * step * (slope_c - a[k + 1].T @ guess_c - q[k + 1])
        )

    initial = named(state_names, states[0])
    final = named(state_names, states[-1])
    final_time = times[-1]

    def psi(start, end, time):
        return jnp.concatenate(
            [_terminal_values(c, start, end, time, theta) for c in terminals]
            or [jnp.zeros(0)]
        )

    d_end = jax.jacfwd(psi, 1)(initial, final, final_time)
    d_start = jax.jacfwd(psi, 0)(initial, final, final_time)
    d_time = jax.jacfwd(psi, 2)(initial, final, final_time)
    phi_end = jax.grad(costs.terminal_cost, 1)(final_time, final, theta)
    phi_time = jax.grad(costs.terminal_cost, 0)(final_time, final, theta)

    rows, values = [], []
    for k in range(count):
        for i in range(len(control_names)):
            row = b[k][:, i] @ maps[k]
            for column, (node, j) in enumerate(active):
                if node == k:
                    row[first_path + column] += g_u[k, j, i]
            rows.append(row)
            values.append(-(r[k][i] + b[k][:, i] @ offsets[k]))
    for i, name in enumerate(state_names):
        if not boundary.is_final_fixed(name):
            row = maps[-1][i].copy()
            row[size:first_path] -= numpy.asarray(d_end[name])
            rows.append(row)
            values.append(float(phi_end[name]) - offsets[-1][i])
        if not boundary.is_initial_fixed(name):
            row = numpy.zeros(unknowns)
            row[i] = 1.0
            row[size:first_path] = numpy.asarray(d_start[name])
            rows.append(row)
            values.append(0.0)
    if boundary.free_final_time:
        row = f[-1] @ maps[-1]
        row[size:first_path] += numpy.asarray(d_time)
        rows.append(row)
        values.append(-(running[-1] + f[-1] @ offsets[-1] + float(phi_time)))
    if design_jacobian:
        steps = numpy.diff(numpy.asarray(times))
        weights = (
            numpy.concatenate([steps, [0.0]]) + numpy.concatenate([[0.0], steps])
        ) / 2
        phi_theta = jax.grad(costs.terminal_cost, 2)(final_time, final, theta)
        psi_theta = jax.jacfwd(
            lambda parameters: jnp.concatenate(
                [
                    _terminal_values(c, initial, final, final_time, parameters)
                    for c in terminals
                ]
                or [jnp.zeros(0)]
            )
        )(theta)
        f_theta, l_theta, g_theta = by_parameter
        for sensitivity in design_jacobian.values():

            def along(series, sensitivity=sensitivity):
                return sum(
                    numpy.asarray(series[name]) * float(value)
                    for name, value in sensitivity.items()
                )

            f_c, l_c, g_c = along(f_theta), along(l_theta), along(g_theta)
            row = numpy.zeros(unknowns)
            constant = float(along(phi_theta))
            for k in range(count):
                row += weights[k] * (f_c[k] @ maps[k])
                constant += weights[k] * (f_c[k] @ offsets[k] + l_c[k])
            for column, (k, j) in enumerate(active):
                row[first_path + column] += weights[k] * g_c[k, j]
            row[size:first_path] += along(psi_theta)
            rows.append(row)
            values.append(-constant)
    lower = numpy.full(unknowns, -numpy.inf)
    lower[first_path:] = 0.0
    solution = lsq_linear(
        numpy.array(rows), numpy.array(values), bounds=(lower, numpy.inf)
    ).x
    costates = {name: float(solution[k]) for k, name in enumerate(state_names)}
    multipliers, start = {}, size
    for constraint, width in zip(terminals, widths, strict=True):
        multipliers[constraint.name] = solution[start : start + width].tolist()
        start += width
    history = numpy.array([maps[k] @ solution + offsets[k] for k in range(count)])
    return costates, multipliers, history
