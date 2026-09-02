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
running cost and ``f`` the dynamics. The costate equation follows from it
as a Jacobian relation::

    d(lam)/dt = -(df/ds)^T lam - dl/ds

``df/ds`` is the full n-by-n Jacobian of the dynamics, so this holds for
any number of states, and automatic differentiation supplies it directly:
the adjoint equations never have to be derived by hand. Writing the
relation with a Jacobian rather than a gradient also leaves the
differentiation mode free, so every derivative here can be evaluated in
forward or in reverse mode (see ``differentiation_method``).

The optimizer sees only the initial costates ``lam(t0)`` (plus ``t_f``
when the final time is free). Their values are driven by the terminal and
transversality residuals, which are posed as a least-squares objective so
the rest of the scenario machinery applies unchanged.

**The objective is not the first functional here.** In the direct
transcriptions ``functionals[0]`` is what the optimizer minimizes. In this
one the functionals define the optimal-control problem that shapes ``H``,
while the optimizer minimizes :data:`RESIDUAL_NORM`; the achieved cost of
the underlying problem is reported separately as :data:`OCP_COST`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from gemseo_tempo.dynamics.integrators import FORWARD
from gemseo_tempo.dynamics.integrators import REVERSE
from gemseo_tempo.dynamics.state_space import StateSpace
from gemseo_tempo.dynamics.tables import smooth_clip
from gemseo_tempo.dynamics.time_scaling import normalize_time
from gemseo_tempo.outputs.functionals import FinalRate
from gemseo_tempo.outputs.functionals import Lagrange
from gemseo_tempo.outputs.functionals import Mayer
from gemseo_tempo.outputs.functionals import TimeIntegral
from gemseo_tempo.transcriptions.base import CONSTANT_PREFIX
from gemseo_tempo.transcriptions.base import COSTATE_PREFIX
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


class IndirectShooting(AbstractTranscription):
    """Indirect single shooting driven by the Pontryagin conditions.

    Parameters
    ----------
    u_star
        The analytic minimizer ``(t, s, lam, theta) -> dict`` of the
        Hamiltonian, when one is available. Otherwise the stationarity
        condition ``dH/du = 0`` is solved numerically.
    newton_steps
        The number of Newton steps of the inner stationarity solve. The
        loop is unrolled, so it stays jit-safe; one step is exact when
        ``H`` is quadratic in ``u``. The solve needs ``H`` to be strictly
        convex in ``u``: for bang-bang structures supply ``u_star``.
    smooth_bounds_sharpness
        The sharpness of the smooth saturation applied to ``u*`` when
        ``control_bounds`` are given. It maps to the smoothing parameter
        as ``eps = (1 / sharpness) ** 2``; the saturation is exact away
        from the bounds and rounds over a band of about ``1 / sharpness``.
    control_bounds
        The ``(lower, upper)`` bounds of each control, keyed by name.
    differentiation_method
        The mode used for every derivative of the necessary conditions,
        ``"reverse"`` (default) or ``"forward"``. It selects
        :func:`jax.jacrev` or :func:`jax.jacfwd`, and is independent of
        the differentiation mode of the assembled discipline.
    costate_guess
        The initial guess of each initial costate, keyed by state name;
        zero when omitted.

    Notes
    -----
    Scope limits: state path constraints change the necessary conditions
    and are out of scope, so no explicit constraint is accepted; terminal
    conditions are declared through ``BoundarySpec``. Control bounds are
    honoured only through the smooth saturation of ``u*``. Only the
    monolithic assembly is available.

    Multi-start over the initial costates, which are notoriously sensitive
    to their guess, needs no extra machinery: run the assembled scenario
    with ``execute(algo_name="MultiStart", n_start=..., opt_algo_name=
    "L-BFGS-B", doe_algo_name="LHS")``. The sampling requires finite
    bounds on the ``costate.`` design variables.
    """

    def __init__(
        self,
        u_star: Callable | None = None,
        newton_steps: int = 5,
        smooth_bounds_sharpness: float = 20.0,
        control_bounds: Mapping[str, tuple[float, float]] | None = None,
        differentiation_method: str = REVERSE,
        costate_guess: Mapping[str, float] | None = None,
    ) -> None:
        """Store the settings.

        Parameters
        ----------
        u_star
            The analytic Hamiltonian minimizer, if any.
        newton_steps
            The number of inner Newton steps.
        smooth_bounds_sharpness
            The saturation sharpness.
        control_bounds
            The bounds of each control.
        differentiation_method
            The differentiation mode of the necessary conditions.
        costate_guess
            The initial guess of each initial costate.

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
            Must be empty; see the scope limits.
        boundary
            The boundary specification; every initial state must be fixed.
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
        controls = tuple(controls)
        state_names = tuple(state_space.states)
        control_names = tuple(state_space.controls)
        self._validate(state_space, controls, functionals, constraints, boundary, t0)

        statics_output_names = (
            set(statics.chain.output_grammar.names) if statics else set()
        )
        statics_input_names = list(statics.chain.input_grammar.names) if statics else []
        theta_inputs = resolve_parameter_sources(
            state_space, statics_output_names, constants
        )

        mayers = [f for f in functionals if isinstance(f, Mayer)]
        lagranges = [f for f in functionals if isinstance(f, Lagrange)]
        integrals = [f for f in functionals if isinstance(f, TimeIntegral)]
        jacobian = self.jacobian

        def running_cost(t, s, u, theta):
            """Return the integrand l of the Bolza cost, discounts included."""
            total = jnp.zeros(())
            for functional in lagranges:
                total = total + functional.integrand(t, s, u, theta) * jnp.exp(
                    -functional.rho * (t - t0)
                )
            if integrals:
                outputs = state_space.output(t, s, u, theta)
                for functional in integrals:
                    total = total + outputs[functional.output_name] * jnp.exp(
                        -functional.rho * (t - t0)
                    )
            return total

        def terminal_cost(t, s, theta):
            """Return the Mayer terminal cost phi."""
            total = jnp.zeros(())
            for functional in mayers:
                total = total + functional.function(t, s, theta)
            return total

        def hamiltonian(t, s, lam, u, theta):
            """Return H = l + lam^T f."""
            rates = state_space.rhs(t, s, u, theta)
            total = running_cost(t, s, u, theta)
            for name in state_names:
                total = total + lam[name] * rates[name]
            return total

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
            vector = jnp.zeros(len(control_names))
            for _ in range(self.newton_steps):
                vector = vector - jnp.linalg.solve(hessian(vector), gradient(vector))
            return saturate({n: vector[k] for k, n in enumerate(control_names)})

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

        def split_augmented(y):
            """Split the augmented state into states and costates."""
            s = {name: y[name] for name in state_names}
            lam = {name: y[_COSTATE_STATE + name] for name in state_names}
            return s, lam

        def augmented_rhs(t, y, u_unused, theta):
            """Return the rates of the states, costates and cost quadrature."""
            del u_unused
            s, lam = split_augmented(y)
            u = optimal_control(t, s, lam, theta)
            rates = dict(state_space.rhs(t, s, u, theta))
            for name, rate in costate_rates(t, s, lam, u, theta).items():
                rates[_COSTATE_STATE + name] = rate
            rates[_COST_STATE] = running_cost(t, s, u, theta)
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
            ),
            controls=(),
            parameters=state_space.parameters,
            outputs=(*state_space.outputs, *control_names),
            rhs=augmented_rhs,
            output=augmented_output,
        )
        simulated = normalize_time(augmented) if boundary.free_final_time else augmented

        def read_theta(data):
            """Read the parameters as scalars.

            Constants are stored as 1D arrays for GEMSEO, while the
            necessary conditions do direct scalar arithmetic on them.
            """
            return {
                parameter: jnp.squeeze(data[input_name])
                for parameter, input_name in theta_inputs.items()
            }

        def jax_function(inputs):
            """Integrate the augmented system and return every output."""
            data = dict(inputs)
            if statics is not None:
                data.update(
                    statics(
                        {
                            name: data[CONSTANT_PREFIX + name]
                            for name in statics_input_names
                        }
                    )
                )
            theta = read_theta(data)
            y0 = {name: jnp.asarray(boundary.s0[name]) for name in state_names}
            y0.update(
                {
                    _COSTATE_STATE + name: jnp.squeeze(data[COSTATE_PREFIX + name])
                    for name in state_names
                }
            )
            y0[_COST_STATE] = jnp.zeros(())

            if boundary.free_final_time:
                t_final = jnp.squeeze(data[T_FINAL])
                trajectory = integrator.integrate(
                    simulated, y0, (), {}, {**theta, T_FINAL: t_final}, 0.0, 1.0
                )
                times = trajectory.ts * t_final
            else:
                trajectory = integrator.integrate(simulated, y0, (), {}, theta, t0, t1)
                times = trajectory.ts

            final_states = {name: trajectory.states[name][-1] for name in state_names}
            final_costates = {
                name: trajectory.states[_COSTATE_STATE + name][-1]
                for name in state_names
            }
            final_time = times[-1]

            residual = self._residuals(
                state_names,
                boundary,
                terminal_cost,
                hamiltonian,
                optimal_control,
                jacobian,
                final_time,
                final_states,
                final_costates,
                theta,
            )

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
                final_time, final_states, theta
            )
            return {name: jnp.atleast_1d(value) for name, value in result.items()}

        design_variables: dict[str, VariableSpec] = {}
        input_defaults: dict = {}
        for name in state_names:
            value = jnp.asarray([self.costate_guess.get(name, 0.0)], dtype=float)
            design_variables[COSTATE_PREFIX + name] = VariableSpec(
                name=COSTATE_PREFIX + name, size=1, value=value
            )
            input_defaults[COSTATE_PREFIX + name] = value
        if boundary.free_final_time:
            value = jnp.asarray([boundary.t_f], dtype=float)
            design_variables[T_FINAL] = VariableSpec(name=T_FINAL, size=1, value=value)
            input_defaults[T_FINAL] = value
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

    @staticmethod
    def _residuals(
        state_names,
        boundary,
        terminal_cost,
        hamiltonian,
        optimal_control,
        jacobian,
        final_time,
        final_states,
        final_costates,
        theta,
    ):
        """Assemble the terminal, transversality and free-time residuals.

        Returns
        -------
        Array
            One residual per state, plus the ``H(t1)`` condition when the
            final time is free.
        """
        cost_gradient = jacobian(
            lambda states: terminal_cost(final_time, states, theta)
        )(final_states)
        entries = []
        for name in state_names:
            if boundary.is_final_fixed(name):
                entries.append(final_states[name] - boundary.s1[name])
            else:
                entries.append(final_costates[name] - cost_gradient[name])
        if boundary.free_final_time:
            control = optimal_control(final_time, final_states, final_costates, theta)
            time_gradient = jacobian(
                lambda time: terminal_cost(time, final_states, theta)
            )(final_time)
            entries.append(
                hamiltonian(final_time, final_states, final_costates, control, theta)
                + time_gradient
            )
        return jnp.stack(entries)

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
        if constraints:
            raise ValueError(
                "Indirect shooting does not support explicit constraints:"
                " path constraints change the necessary conditions, and"
                " terminal conditions are declared through"
                " BoundarySpec.fix_final / s1 instead."
            )
        for functional in functionals:
            if isinstance(functional, FinalRate):
                raise ValueError(
                    f"FinalRate {functional.name!r} is not part of a Bolza"
                    " cost, so it cannot shape the Hamiltonian."
                )
        free_initial = [
            name for name in state_space.states if not boundary.is_initial_fixed(name)
        ]
        if free_initial:
            raise ValueError(
                f"States {sorted(free_initial)} have fix_initial=False."
                " Indirect single shooting propagates from a fully known"
                " initial state; free initial states need extra necessary"
                " conditions that are out of scope."
            )
        if boundary.free_final_time and t0 != 0.0:
            raise ValueError("free_final_time requires t0 = 0.")
