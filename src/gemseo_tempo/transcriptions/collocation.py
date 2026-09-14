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
"""Hermite-Simpson direct collocation, in separated form.

States and controls at the segment nodes are design variables; the
midpoint states come from the Hermite interpolant, the midpoint controls
from linear interpolation, and the Simpson defects enforce the dynamics.
There is no ODE solve: the whole defect vector is one jitted function.
Integral functionals are evaluated by Simpson quadrature on the same grid
(no augmented state). When ``compressed`` is set, fixed boundary states
are removed from the design space (they become parameters) instead of
kept as variables tied down by an equality. Monolithic assembly only; the
split assembly of the defect vector is a backlog item.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy

from gemseo_tempo.outputs.constraints import ControlGroupConstraint
from gemseo_tempo.outputs.constraints import PathConstraint
from gemseo_tempo.outputs.constraints import TerminalConstraint
from gemseo_tempo.outputs.functionals import FinalRate
from gemseo_tempo.outputs.functionals import Lagrange
from gemseo_tempo.outputs.functionals import Mayer
from gemseo_tempo.outputs.functionals import TimeIntegral
from gemseo_tempo.transcriptions.base import CONSTANT_PREFIX
from gemseo_tempo.transcriptions.base import CONSTRAINT_PREFIX
from gemseo_tempo.transcriptions.base import CONTROL_PREFIX
from gemseo_tempo.transcriptions.base import STATE_PREFIX
from gemseo_tempo.transcriptions.base import T_FINAL
from gemseo_tempo.transcriptions.base import TIMES
from gemseo_tempo.transcriptions.base import TRAJECTORY_PREFIX
from gemseo_tempo.transcriptions.base import AbstractTranscription
from gemseo_tempo.transcriptions.base import StructuredJacobian
from gemseo_tempo.transcriptions.base import TranscribedProblem
from gemseo_tempo.transcriptions.base import VariableSpec
from gemseo_tempo.transcriptions.base import resolve_parameter_sources

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from gemseo_tempo.dynamics.controls import AbstractControl
    from gemseo_tempo.dynamics.integrators import AbstractIntegrator
    from gemseo_tempo.outputs.constraints import AbstractConstraint
    from gemseo_tempo.outputs.functionals import AbstractFunctional
    from gemseo_tempo.statics.chain import StaticsChain
    from gemseo_tempo.transcriptions.base import BoundarySpec

DEFECT_PREFIX = f"{CONSTRAINT_PREFIX}defect."
"""The output-name prefix of the Simpson defect constraints."""


def _node_rates(state_space, control_nodes: Mapping, step) -> dict:
    """Return the time-derivative of each control the model asks for.

    There is no control *function* here to differentiate: under collocation
    the node values are design variables and the control is linear between
    them, so its rate is the slope of the segment, constant across that
    segment. Each node is given the slope of the segment starting there and
    the last node inherits the one before it, which is the same one-sided
    reading the shooting path gets when it differentiates the piecewise
    linear interpolant at a node.

    Parameters
    ----------
    state_space
        The model, read for which controls need rates.
    control_nodes
        The node values of each control.
    step
        The time between nodes.

    Returns
    -------
    dict
        The rates, keyed as the model expects to receive them.
    """
    rates = {}
    for name in state_space.control_rates:
        values = control_nodes[name]
        slopes = (values[1:] - values[:-1]) / step
        rates[state_space.rate_name(name)] = jnp.concatenate(
            [
                slopes,
                slopes[-1:],
            ]
        )
    return rates


class HermiteSimpson(AbstractTranscription):
    """Hermite-Simpson direct collocation.

    Parameters
    ----------
    num_segments
        The number of segments; the grid has ``num_segments + 1`` nodes.
    defect_ref
        The defect scaling, a single factor or one per state name. Defect
        conditioning drives SQP behavior, so each defect is divided by its
        reference to bring it near unit scale.
    compressed
        Whether fixed boundary states are removed from the design space
        (they become parameters) rather than kept with an equality.
    sparse_jacobian
        Whether to compute the defect Jacobian from per-segment blocks.
        Segment ``k`` involves only nodes ``k`` and ``k+1``, so one
        vectorised pass over the segments replaces a dense transform
        whose cost grows with the square of the grid size. The defect
        values are unchanged; only their derivative is obtained
        differently. Off by default.
    """

    def __init__(
        self,
        num_segments: int,
        defect_ref: dict[str, float] | float = 1.0,
        compressed: bool = True,
        sparse_jacobian: bool = False,
    ) -> None:
        """Store the discretization settings.

        Parameters
        ----------
        num_segments
            The number of segments.
        defect_ref
            The defect scaling.
        compressed
            Whether fixed boundary states leave the design space.
        sparse_jacobian
            Whether to derive the defect Jacobian from per-segment
            blocks instead of a dense automatic transform.
        """
        self.num_segments = int(num_segments)
        self.defect_ref = defect_ref
        self.compressed = compressed
        self.sparse_jacobian = sparse_jacobian

    def _defect_ref_of(self, name: str) -> float:
        """Return the defect scaling of a state.

        Parameters
        ----------
        name
            The state name.

        Returns
        -------
        float
            The scaling factor.
        """
        if isinstance(self.defect_ref, dict):
            return float(self.defect_ref.get(name, 1.0))
        return float(self.defect_ref)

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
        """Build the collocation problem.

        Parameters
        ----------
        statics
            The constant-fed chain running ahead of the dynamics, if any.
        state_space
            The dynamics model.
        integrator
            Unused: collocation solves no ODE. Accepted to keep the
            transcription interface uniform.
        controls
            The control parameterizations; only their names and default
            values (to seed the node guess) are used.
        functionals
            The scalar functionals; the first one is the objective.
        constraints
            The trajectory constraints.
        boundary
            The boundary specification.
        constants
            The values of constant inputs, overriding the statics-chain
            defaults; required for dynamics parameters not produced by the
            statics chain.
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
            If the controls do not match the state space, a parameter has
            no source, no functional is given, or the final time is free
            with a nonzero initial time.
        """
        del integrator  # collocation solves no ODE
        constants = dict(constants or {})
        controls = tuple(controls)
        self._validate(state_space, controls, functionals, boundary, t0)

        n_seg = self.num_segments
        n_nodes = n_seg + 1
        state_names = state_space.states
        control_names = tuple(control.name for control in controls)

        statics_output_names = (
            set(statics.chain.output_grammar.names) if statics else set()
        )
        statics_input_names = list(statics.chain.input_grammar.names) if statics else []
        theta_inputs = resolve_parameter_sources(
            state_space, statics_output_names, constants
        )

        # Per-state boundary compression: which end nodes are fixed.
        fixed_lo = {name: boundary.is_initial_fixed(name) for name in state_names}
        fixed_hi = {name: boundary.is_final_fixed(name) for name in state_names}
        free_indices = {
            name: tuple(
                index
                for index in range(n_nodes)
                if not (index == 0 and fixed_lo[name])
                and not (index == n_nodes - 1 and fixed_hi[name])
            )
            for name in state_names
        }

        # Default node grid (used to seed control/state guesses).
        duration = boundary.t_f if boundary.free_final_time else (t1 - t0)
        origin = 0.0 if boundary.free_final_time else t0
        node_times_guess = jnp.linspace(origin, origin + duration, n_nodes)

        def full_state(name, free_values):
            """Rebuild the N+1 node vector from its free components."""
            full = jnp.zeros(n_nodes)
            if fixed_lo[name]:
                full = full.at[0].set(boundary.s0[name])
            if fixed_hi[name]:
                full = full.at[n_nodes - 1].set(boundary.s1[name])
            indices = jnp.asarray(free_indices[name], dtype=int)
            return full.at[indices].set(free_values)

        def read_theta(data):
            # Parameters are conceptually scalars; constant inputs are stored
            # as 1D arrays for GEMSEO, so squeeze them back to 0-D to keep the
            # direct node-array arithmetic from broadcasting to (N, 1).
            return {
                parameter: jnp.squeeze(data[input_name])
                for parameter, input_name in theta_inputs.items()
            }

        def jax_function(inputs):
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
            state_nodes = {
                name: full_state(name, data[STATE_PREFIX + name])
                for name in state_names
            }
            control_nodes = {
                name: data[CONTROL_PREFIX + name] for name in control_names
            }
            if boundary.free_final_time:
                total = data[T_FINAL]
                times = jnp.linspace(0.0, 1.0, n_nodes) * total
            else:
                times = jnp.linspace(t0, t1, n_nodes)
            step = times[1] - times[0]
            # Rates are handed to the model but never published as a
            # trajectory of their own, so they are kept out of the dict the
            # results are read from.
            node_rates = _node_rates(state_space, control_nodes, step)
            control_nodes_full = {**control_nodes, **node_rates}

            def rhs_point(t, s, u):
                return state_space.rhs(t, s, u, theta)

            def output_point(t, s, u):
                return state_space.output(t, s, u, theta)

            rates = jax.vmap(rhs_point)(times, state_nodes, control_nodes_full)
            # Hermite midpoint states and linear midpoint controls.
            times_mid = 0.5 * (times[:-1] + times[1:])
            state_mid = {
                name: 0.5 * (values[:-1] + values[1:])
                + (step / 8.0) * (rates[name][:-1] - rates[name][1:])
                for name, values in state_nodes.items()
            }
            control_mid = {
                name: 0.5 * (values[:-1] + values[1:])
                for name, values in control_nodes.items()
            }
            # A midpoint's rate is its own segment's slope, not the average
            # of the two node rates: the control is linear across the
            # segment, so the slope is constant over it.
            control_mid_full = {
                **control_mid,
                **{name: values[:-1] for name, values in node_rates.items()},
            }
            rates_mid = jax.vmap(rhs_point)(times_mid, state_mid, control_mid_full)
            output_nodes = jax.vmap(output_point)(
                times, state_nodes, control_nodes_full
            )
            output_mid = jax.vmap(output_point)(times_mid, state_mid, control_mid_full)

            result = {TIMES: times}
            for name, values in state_nodes.items():
                result[TRAJECTORY_PREFIX + name] = values
            for name, values in control_nodes.items():
                result[TRAJECTORY_PREFIX + name] = values
            for name in state_space.outputs:
                result[TRAJECTORY_PREFIX + name] = output_nodes[name]

            # Simpson defects, per state per segment, scaled by defect_ref.
            for name in state_names:
                left = state_nodes[name][:-1]
                right = state_nodes[name][1:]
                defect = (
                    right
                    - left
                    - (step / 6.0)
                    * (rates[name][:-1] + 4.0 * rates_mid[name] + rates[name][1:])
                )
                result[DEFECT_PREFIX + name] = defect / self._defect_ref_of(name)

            initial_states = {n: v[0] for n, v in state_nodes.items()}
            final_states = {n: v[-1] for n, v in state_nodes.items()}
            self._add_functionals(
                result,
                functionals,
                times,
                times_mid,
                step,
                theta,
                state_nodes,
                control_nodes,
                state_mid,
                control_mid,
                output_nodes,
                output_mid,
                final_states,
                t0,
            )
            self._add_constraints(
                result,
                constraints,
                times,
                times_mid,
                theta,
                state_nodes,
                control_nodes,
                state_mid,
                control_mid,
                initial_states,
                final_states,
            )
            return {name: jnp.atleast_1d(value) for name, value in result.items()}

        design_variables, input_defaults = self._design_space(
            state_names,
            control_names,
            controls,
            boundary,
            free_indices,
            node_times_guess,
            statics,
            statics_input_names,
            theta_inputs,
            constants,
        )

        constraint_names = [
            (CONSTRAINT_PREFIX + constraint.name, constraint.kind)
            for constraint in constraints
        ]
        constraint_names.extend((DEFECT_PREFIX + name, "eq") for name in state_names)

        output_names = [
            TIMES,
            *(TRAJECTORY_PREFIX + name for name in state_names),
            *(TRAJECTORY_PREFIX + name for name in control_names),
            *(TRAJECTORY_PREFIX + name for name in state_space.outputs),
            *(DEFECT_PREFIX + name for name in state_names),
            *(functional.name for functional in functionals),
            *(
                name
                for name, _ in constraint_names
                if not name.startswith(DEFECT_PREFIX)
            ),
        ]

        structured = None
        if self.sparse_jacobian:
            structured = self._build_structured_jacobian(
                state_space,
                state_names,
                control_names,
                free_indices,
                boundary,
                read_theta,
                full_state,
                statics,
                statics_input_names,
                t0,
                t1,
            )

        problem = TranscribedProblem(
            design_variables=design_variables,
            jax_function=jax_function,
            objective_name=functionals[0].name,
            constraint_names=constraint_names,
            input_defaults=input_defaults,
            output_names=output_names,
            jacobian=structured,
        )
        problem.guess_setter = self._make_guess_setter(
            problem,
            state_names,
            control_names,
            free_indices,
            node_times_guess,
        )
        return problem

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _validate(state_space, controls, functionals, boundary, t0):
        """Check the problem structure (mirrors single shooting).

        Parameters
        ----------
        state_space
            The dynamics model.
        controls
            The control parameterizations.
        functionals
            The scalar functionals.
        boundary
            The boundary specification.
        t0
            The initial time.

        Raises
        ------
        ValueError
            On any structural mismatch.
        """
        control_names = {control.name for control in controls}
        if control_names != set(state_space.controls):
            raise ValueError(
                f"Control parameterizations {sorted(control_names)} do not"
                f" match the state-space controls"
                f" {sorted(state_space.controls)}."
            )
        if not functionals:
            raise ValueError(
                "At least one functional is required (the first one is the objective)."
            )
        if boundary.free_final_time and t0 != 0.0:
            raise ValueError("free_final_time requires t0 = 0.")
        known = set(state_space.states) | set(state_space.outputs)
        for functional in functionals:
            if isinstance(functional, FinalRate) and (
                functional.output_name not in known
            ):
                raise ValueError(
                    f"FinalRate {functional.name!r} references"
                    f" {functional.output_name!r}, which is neither a state"
                    " nor an output."
                )
            if isinstance(functional, TimeIntegral) and (
                functional.output_name not in state_space.outputs
            ):
                raise ValueError(
                    f"TimeIntegral {functional.name!r} references"
                    f" {functional.output_name!r}, which is not an output."
                )

    def _add_functionals(
        self,
        result,
        functionals,
        times,
        times_mid,
        step,
        theta,
        state_nodes,
        control_nodes,
        state_mid,
        control_mid,
        output_nodes,
        output_mid,
        final_states,
        t0,
    ):
        """Evaluate the functionals and store them in ``result``."""
        for functional in functionals:
            if isinstance(functional, Mayer):
                result[functional.name] = functional.function(
                    times[-1], final_states, theta
                )
            elif isinstance(functional, (Lagrange, TimeIntegral)):
                if isinstance(functional, TimeIntegral):
                    l_nodes = output_nodes[functional.output_name]
                    l_mid = output_mid[functional.output_name]
                else:
                    l_nodes = jax.vmap(
                        lambda t, s, u, f=functional: f.integrand(t, s, u, theta)
                    )(times, state_nodes, control_nodes)
                    l_mid = jax.vmap(
                        lambda t, s, u, f=functional: f.integrand(t, s, u, theta)
                    )(times_mid, state_mid, control_mid)
                l_nodes = l_nodes * jnp.exp(-functional.rho * (times - t0))
                l_mid = l_mid * jnp.exp(-functional.rho * (times_mid - t0))
                result[functional.name] = jnp.sum(
                    (step / 6.0) * (l_nodes[:-1] + 4.0 * l_mid + l_nodes[1:])
                )
            else:  # FinalRate
                series = {**state_nodes, **output_nodes}[functional.output_name]
                result[functional.name] = (series[-1] - series[-2]) / (
                    times[-1] - times[-2]
                )

    @staticmethod
    def _add_constraints(
        result,
        constraints,
        times,
        times_mid,
        theta,
        state_nodes,
        control_nodes,
        state_mid,
        control_mid,
        initial_states,
        final_states,
    ):
        """Evaluate the constraints and store them in ``result``."""
        for constraint in constraints:
            name = CONSTRAINT_PREFIX + constraint.name
            if isinstance(constraint, PathConstraint):
                at_nodes = jax.vmap(
                    lambda t, s, u, c=constraint: c.function(t, s, u, theta)
                )(times, state_nodes, control_nodes)
                at_mid = jax.vmap(
                    lambda t, s, u, c=constraint: c.function(t, s, u, theta)
                )(times_mid, state_mid, control_mid)
                values = jnp.concatenate([at_nodes, at_mid])
                result[name] = constraint.evaluate_on_nodes(values)
            elif isinstance(constraint, ControlGroupConstraint):
                result[name] = constraint.evaluate(control_nodes)
            elif isinstance(constraint, TerminalConstraint):
                result[name] = constraint.function(times[-1], final_states, theta)
            else:  # PeriodicityConstraint
                result[name] = constraint.evaluate(initial_states, final_states)

    def _design_space(
        self,
        state_names,
        control_names,
        controls,
        boundary,
        free_indices,
        node_times_guess,
        statics,
        statics_input_names,
        theta_inputs,
        constants,
    ):
        """Build the design-variable specs and the input defaults."""
        design_variables: dict[str, VariableSpec] = {}
        input_defaults: dict = {}

        for name in state_names:
            start = boundary.s0.get(name, 0.0)
            end = boundary.s1.get(name, start)
            ramp = jnp.linspace(start, end, self.num_segments + 1)
            indices = jnp.asarray(free_indices[name], dtype=int)
            value = ramp[indices]
            design_variables[STATE_PREFIX + name] = VariableSpec(
                name=STATE_PREFIX + name, size=value.shape[0], value=value
            )
            input_defaults[STATE_PREFIX + name] = value

        for control in controls:
            value = jax.vmap(lambda t, c=control: c(t, c.default_params))(
                node_times_guess
            )
            design_variables[CONTROL_PREFIX + control.name] = VariableSpec(
                name=CONTROL_PREFIX + control.name,
                size=value.shape[0],
                value=value,
            )
            input_defaults[CONTROL_PREFIX + control.name] = value

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
        return design_variables, input_defaults

    @staticmethod
    def _make_guess_setter(
        problem, state_names, control_names, free_indices, node_times_guess
    ):
        """Build the warm-start callback for :meth:`set_initial_guess`."""

        def guess_setter(trajectory):
            for name in state_names:
                if name not in trajectory.states:
                    continue
                interpolated = jnp.interp(
                    node_times_guess, trajectory.ts, trajectory.states[name]
                )
                indices = jnp.asarray(free_indices[name], dtype=int)
                value = interpolated[indices]
                problem.input_defaults[STATE_PREFIX + name] = value
                problem.design_variables[STATE_PREFIX + name].value = value
            for name in control_names:
                if name not in trajectory.controls:
                    continue
                value = jnp.interp(
                    node_times_guess, trajectory.ts, trajectory.controls[name]
                )
                problem.input_defaults[CONTROL_PREFIX + name] = value
                problem.design_variables[CONTROL_PREFIX + name].value = value

        return guess_setter

    def _build_structured_jacobian(
        self,
        state_space,
        state_names,
        control_names,
        free_indices,
        boundary,
        read_theta,
        full_state,
        statics,
        statics_input_names,
        t0,
        t1,
    ):
        """Build the defect Jacobian from per-segment blocks.

        Segment ``k`` of a Hermite-Simpson defect involves only the nodes
        ``k`` and ``k+1``, so differentiating one segment and mapping that
        over the grid produces every nonzero entry in a single vectorised
        pass. The blocks are then scattered into the columns GEMSEO
        expects. The cost grows with the number of segments instead of
        with its square.

        Returns
        -------
        StructuredJacobian
            The defect Jacobian with respect to the node design variables
            and, when the final time is free, ``t_f``.
        """
        n_seg = self.num_segments
        n_nodes = n_seg + 1
        segments = numpy.arange(n_seg)

        def segment_defects(s_left, s_right, u_left, u_right, t_left, t_right, theta):
            """Return the Simpson defects of a single segment."""
            step = t_right - t_left
            # The control is linear across the segment, so its rate is the
            # segment slope and is the same at both ends and the midpoint.
            rates = {
                state_space.rate_name(name): (u_right[name] - u_left[name]) / step
                for name in state_space.control_rates
            }
            u_left = {**u_left, **rates}
            u_right = {**u_right, **rates}
            f_left = state_space.rhs(t_left, s_left, u_left, theta)
            f_right = state_space.rhs(t_right, s_right, u_right, theta)
            s_mid = {
                name: 0.5 * (s_left[name] + s_right[name])
                + (step / 8.0) * (f_left[name] - f_right[name])
                for name in state_names
            }
            u_mid = {
                name: 0.5 * (u_left[name] + u_right[name]) for name in control_names
            }
            u_mid.update(rates)
            f_mid = state_space.rhs(0.5 * (t_left + t_right), s_mid, u_mid, theta)
            return {
                name: (
                    s_right[name]
                    - s_left[name]
                    - (step / 6.0) * (f_left[name] + 4.0 * f_mid[name] + f_right[name])
                )
                / self._defect_ref_of(name)
                for name in state_names
            }

        block_jacobian = jax.vmap(
            jax.jacobian(segment_defects, argnums=(0, 1, 2, 3, 4, 5)),
            in_axes=(0, 0, 0, 0, 0, 0, None),
        )

        def scatter(left, right, node_to_column, n_columns):
            """Place the per-segment blocks into their dense columns."""
            matrix = jnp.zeros((n_seg, n_columns))
            for offset, values in ((0, left), (1, right)):
                columns = node_to_column[segments + offset]
                keep = numpy.nonzero(columns >= 0)[0]
                if keep.size:
                    matrix = matrix.at[segments[keep], columns[keep]].add(values[keep])
            return matrix

        state_columns = {}
        for name in state_names:
            lookup = numpy.full(n_nodes, -1, dtype=int)
            for column, node in enumerate(free_indices[name]):
                lookup[node] = column
            state_columns[name] = lookup
        control_columns = numpy.arange(n_nodes)

        def function(data):
            """Return the defect Jacobian, scattered into dense columns."""
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
            theta = read_theta(data)
            state_nodes = {
                name: full_state(name, data[STATE_PREFIX + name])
                for name in state_names
            }
            control_nodes = {
                name: data[CONTROL_PREFIX + name] for name in control_names
            }
            if boundary.free_final_time:
                times = jnp.linspace(0.0, 1.0, n_nodes) * jnp.squeeze(data[T_FINAL])
            else:
                times = jnp.linspace(t0, t1, n_nodes)

            blocks = block_jacobian(
                {name: values[:-1] for name, values in state_nodes.items()},
                {name: values[1:] for name, values in state_nodes.items()},
                {name: values[:-1] for name, values in control_nodes.items()},
                {name: values[1:] for name, values in control_nodes.items()},
                times[:-1],
                times[1:],
                theta,
            )
            jacobian = {}
            for name in state_names:
                # jax.jacobian puts the output structure outermost, so this
                # is the tuple of derivatives of one defect with respect to
                # the six differentiated arguments of the segment function.
                by_argument = blocks[name]
                row = {}
                for other in state_names:
                    row[STATE_PREFIX + other] = scatter(
                        by_argument[0][other],
                        by_argument[1][other],
                        state_columns[other],
                        len(free_indices[other]),
                    )
                for other in control_names:
                    row[CONTROL_PREFIX + other] = scatter(
                        by_argument[2][other],
                        by_argument[3][other],
                        control_columns,
                        n_nodes,
                    )
                if boundary.free_final_time:
                    # times = linspace(0, 1, n_nodes) * t_f, so node k moves
                    # at rate k / n_seg when the final time changes.
                    row[T_FINAL] = (
                        by_argument[4] * (segments / n_seg)
                        + by_argument[5] * ((segments + 1) / n_seg)
                    ).reshape(-1, 1)
                jacobian[DEFECT_PREFIX + name] = row
            return jacobian

        input_names = [STATE_PREFIX + name for name in state_names]
        input_names.extend(CONTROL_PREFIX + name for name in control_names)
        if boundary.free_final_time:
            input_names.append(T_FINAL)
        return StructuredJacobian(
            function=function,
            output_names=tuple(DEFECT_PREFIX + name for name in state_names),
            input_names=tuple(input_names),
        )
