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
"""Direct single shooting.

The design variables are the control parameters (plus the free final
time and the free initial-state components); the trajectory is produced
by the integrator; there are no defect constraints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from gemseo_tempo.dynamics.integrators import QuasiStaticIntegrator
from gemseo_tempo.dynamics.state_space import StateSpace
from gemseo_tempo.dynamics.time_scaling import normalize_time
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
from gemseo_tempo.transcriptions.base import INITIAL_PREFIX
from gemseo_tempo.transcriptions.base import T_FINAL
from gemseo_tempo.transcriptions.base import TIMES
from gemseo_tempo.transcriptions.base import TRAJECTORY_PREFIX
from gemseo_tempo.transcriptions.base import AbstractTranscription
from gemseo_tempo.transcriptions.base import StagePiece
from gemseo_tempo.transcriptions.base import TranscribedProblem
from gemseo_tempo.transcriptions.base import VariableSpec

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from gemseo_tempo.dynamics.controls import AbstractControl
    from gemseo_tempo.dynamics.integrators import AbstractIntegrator
    from gemseo_tempo.outputs.constraints import AbstractConstraint
    from gemseo_tempo.outputs.functionals import AbstractFunctional
    from gemseo_tempo.statics.chain import StaticsChain
    from gemseo_tempo.transcriptions.base import BoundarySpec

_QUAD_PREFIX = "_quad_"


class SingleShooting(AbstractTranscription):
    """Direct single shooting.

    Integral functionals are integrated as augmented quadrature states
    (exact under adaptive stepping); under the quasi-static execution
    model, which has no integrated state, they fall back to trapezoid
    quadrature on the save grid.
    """

    def transcribe(
        self,
        statics: StaticsChain | None,
        state_space: StateSpace,
        integrator: AbstractIntegrator,
        controls: Sequence[AbstractControl],
        functionals: Sequence[AbstractFunctional],
        constraints: Sequence[AbstractConstraint],
        boundary: BoundarySpec,
        constants: Mapping[str, float] | None = None,
        t0: float = 0.0,
        t1: float = 1.0,
    ) -> TranscribedProblem:
        """Build the shooting problem.

        Parameters
        ----------
        statics
            The constant-fed chain running ahead of the dynamics, if any.
        state_space
            The dynamics model.
        integrator
            The execution model of the dynamics.
        controls
            The control parameterizations, one per state-space control.
        functionals
            The scalar functionals; the first one is the objective.
        constraints
            The trajectory constraints.
        boundary
            The boundary specification.
        constants
            The values of constant inputs, overriding the statics-chain
            defaults; required for dynamics parameters not produced by
            the statics chain.
        t0
            The initial time.
        t1
            The final time (ignored when the final time is free).

        Returns
        -------
        TranscribedProblem
            The problem.

        Raises
        ------
        ValueError
            If the controls do not match the state space, a parameter has
            no source, a functional references an unknown output, no
            functional is given, or the final time is free with a nonzero
            initial time.
        """
        constants = dict(constants or {})
        controls = tuple(controls)
        self._validate(state_space, controls, functionals, boundary, t0)

        statics_output_names = (
            set(statics.chain.output_grammar.names) if statics else set()
        )
        statics_input_names = list(statics.chain.input_grammar.names) if statics else []
        param_sources = {}
        for parameter in state_space.parameters:
            if parameter in statics_output_names:
                param_sources[parameter] = "statics"
            elif parameter in constants:
                param_sources[parameter] = "constant"
            else:
                raise ValueError(
                    f"Parameter {parameter!r} is neither an output of the"
                    " statics chain nor given in constants."
                )

        use_augmentation = not isinstance(integrator, QuasiStaticIntegrator)
        quadratures = [
            f for f in functionals if isinstance(f, (Lagrange, TimeIntegral))
        ]
        quad_names = (
            tuple(_QUAD_PREFIX + f.name for f in quadratures)
            if use_augmentation
            else ()
        )
        sim_space = self._build_simulation_space(
            state_space, quadratures, boundary, t0, use_augmentation
        )

        theta_inputs = {
            parameter: (
                parameter if source == "statics" else CONSTANT_PREFIX + parameter
            )
            for parameter, source in param_sources.items()
        }
        free_initials = [
            name for name in state_space.states if not boundary.is_initial_fixed(name)
        ]

        def read_theta(data):
            return {
                parameter: data[input_name]
                for parameter, input_name in theta_inputs.items()
            }

        def dynamics_function(data):
            theta = read_theta(data)
            control_params = {
                control.name: data[CONTROL_PREFIX + control.name]
                for control in controls
            }
            s0 = {
                # A free initial value arrives as a length-one design
                # variable and a fixed one as a bare scalar. Left as they
                # are, a problem that frees some initial states and fixes
                # others cannot have its state vector stacked at all, so
                # both are reshaped to a scalar here.
                name: (
                    jnp.reshape(data[INITIAL_PREFIX + name], ())
                    if not boundary.is_initial_fixed(name)
                    else jnp.asarray(boundary.s0[name], dtype=float)
                )
                for name in state_space.states
            }
            s0.update(dict.fromkeys(quad_names, jnp.asarray(0.0)))
            if boundary.free_final_time:
                t_final = data[T_FINAL]
                trajectory = integrator.integrate(
                    sim_space,
                    s0,
                    controls,
                    control_params,
                    {**theta, T_FINAL: t_final},
                    0.0,
                    1.0,
                )
                times = trajectory.ts * t_final
            else:
                trajectory = integrator.integrate(
                    sim_space, s0, controls, control_params, theta, t0, t1
                )
                times = trajectory.ts
            result = {TIMES: times}
            for name, series in trajectory.states.items():
                result[TRAJECTORY_PREFIX + name] = series
            for control in controls:
                result[TRAJECTORY_PREFIX + control.name] = trajectory.controls[
                    control.name
                ]
            for name in state_space.outputs:
                result[TRAJECTORY_PREFIX + name] = trajectory.outputs[name]
            return result

        outputs_function = self._build_outputs_function(
            state_space,
            controls,
            functionals,
            constraints,
            boundary,
            theta_inputs,
            use_augmentation,
            t0,
        )

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
            dynamics_result = dynamics_function(data)
            data.update(dynamics_result)
            return {**dynamics_result, **outputs_function(data)}

        # Assemble the specs and defaults.
        design_variables: dict[str, VariableSpec] = {}
        for control in controls:
            design_variables[CONTROL_PREFIX + control.name] = VariableSpec(
                name=CONTROL_PREFIX + control.name,
                size=control.n_params,
                value=control.default_params,
            )
        for name in free_initials:
            design_variables[INITIAL_PREFIX + name] = VariableSpec(
                name=INITIAL_PREFIX + name,
                size=1,
                value=jnp.asarray([boundary.s0[name]], dtype=float),
            )
        if boundary.free_final_time:
            design_variables[T_FINAL] = VariableSpec(
                name=T_FINAL,
                size=1,
                value=jnp.asarray([boundary.t_f], dtype=float),
            )

        input_defaults = {spec.name: spec.value for spec in design_variables.values()}
        # GEMSEO concatenates inputs for finite differences: 1D at least.
        for name in statics_input_names:
            input_defaults[CONSTANT_PREFIX + name] = jnp.atleast_1d(
                jnp.asarray(
                    constants.get(name, statics.chain.default_input_data[name]),
                    dtype=float,
                )
            )
        for parameter, source in param_sources.items():
            if source == "constant":
                input_defaults[CONSTANT_PREFIX + parameter] = jnp.atleast_1d(
                    jnp.asarray(constants[parameter], dtype=float)
                )

        constraint_names = [
            (CONSTRAINT_PREFIX + constraint.name, constraint.kind)
            for constraint in constraints
        ]
        constraint_names.extend(
            (f"{CONSTRAINT_PREFIX}final.{name}", "eq")
            for name in state_space.states
            if boundary.is_final_fixed(name)
        )

        dynamics_piece, outputs_piece = self._build_stage_pieces(
            statics,
            statics_input_names,
            dynamics_function,
            outputs_function,
            design_variables,
            input_defaults,
            theta_inputs,
            constants,
        )

        trajectory_names = [
            TRAJECTORY_PREFIX + name
            for name in (
                *state_space.states,
                *quad_names,
                *(control.name for control in controls),
                *state_space.outputs,
            )
        ]
        output_names = [
            TIMES,
            *trajectory_names,
            *(functional.name for functional in functionals),
            *(name for name, _ in constraint_names),
        ]
        return TranscribedProblem(
            design_variables=design_variables,
            jax_function=jax_function,
            objective_name=functionals[0].name,
            constraint_names=constraint_names,
            input_defaults=input_defaults,
            output_names=output_names,
            dynamics=dynamics_piece,
            outputs=outputs_piece,
        )

    @staticmethod
    def _validate(state_space, controls, functionals, boundary, t0):
        """Check the problem structure before building anything.

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
        known_series = set(state_space.states) | set(state_space.outputs)
        for functional in functionals:
            if isinstance(functional, FinalRate) and (
                functional.output_name not in known_series
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

    @staticmethod
    def _build_simulation_space(
        state_space, quadratures, boundary, t0, use_augmentation
    ):
        """Build the (augmented, possibly time-normalized) state space.

        Parameters
        ----------
        state_space
            The dynamics model.
        quadratures
            The integral functionals.
        boundary
            The boundary specification.
        t0
            The initial time.
        use_augmentation
            Whether integral functionals become quadrature states.

        Returns
        -------
        StateSpace
            The simulation model.
        """
        if not (use_augmentation and quadratures):
            augmented = state_space
        else:

            def augmented_rhs(t, s, u, theta):
                base_states = {name: s[name] for name in state_space.states}
                rates = dict(state_space.rhs(t, base_states, u, theta))
                outputs = state_space.output(t, base_states, u, theta)
                for functional in quadratures:
                    if isinstance(functional, TimeIntegral):
                        integrand = outputs[functional.output_name]
                    else:
                        integrand = functional.integrand(t, base_states, u, theta)
                    rates[_QUAD_PREFIX + functional.name] = integrand * jnp.exp(
                        -functional.rho * (t - t0)
                    )
                return rates

            def augmented_output(t, s, u, theta):
                base_states = {name: s[name] for name in state_space.states}
                return state_space.output(t, base_states, u, theta)

            augmented = StateSpace(
                states=(
                    *state_space.states,
                    *(_QUAD_PREFIX + f.name for f in quadratures),
                ),
                controls=state_space.controls,
                parameters=state_space.parameters,
                outputs=state_space.outputs,
                rhs=augmented_rhs,
                output=augmented_output,
                control_rates=state_space.control_rates,
            )
        if boundary.free_final_time:
            return normalize_time(augmented)
        return augmented

    @staticmethod
    def _build_outputs_function(
        state_space,
        controls,
        functionals,
        constraints,
        boundary,
        theta_inputs,
        use_augmentation,
        t0,
    ):
        """Build the function evaluating functionals and constraints.

        Parameters
        ----------
        state_space
            The dynamics model.
        controls
            The control parameterizations.
        functionals
            The scalar functionals.
        constraints
            The trajectory constraints.
        boundary
            The boundary specification.
        theta_inputs
            The input name of each dynamics parameter.
        use_augmentation
            Whether integral functionals are quadrature states.
        t0
            The initial time.

        Returns
        -------
        Callable
            The dict-to-dict function.
        """

        def outputs_function(data):
            theta = {
                parameter: data[input_name]
                for parameter, input_name in theta_inputs.items()
            }
            times = data[TIMES]
            state_series = {
                name: data[TRAJECTORY_PREFIX + name] for name in state_space.states
            }
            control_series = {
                control.name: data[TRAJECTORY_PREFIX + control.name]
                for control in controls
            }
            output_series = {
                name: data[TRAJECTORY_PREFIX + name] for name in state_space.outputs
            }
            initial_states = {name: series[0] for name, series in state_series.items()}
            final_states = {name: series[-1] for name, series in state_series.items()}
            result = {}
            for functional in functionals:
                if isinstance(functional, Mayer):
                    result[functional.name] = functional.function(
                        times[-1], final_states, theta
                    )
                elif isinstance(functional, (Lagrange, TimeIntegral)):
                    if use_augmentation:
                        result[functional.name] = data[
                            TRAJECTORY_PREFIX + _QUAD_PREFIX + functional.name
                        ][-1]
                    else:
                        if isinstance(functional, TimeIntegral):
                            integrand = output_series[functional.output_name]
                        else:
                            integrand = jax.vmap(
                                lambda t, s, u, f=functional: f.integrand(
                                    t, s, u, theta
                                )
                            )(times, state_series, control_series)
                        discount = jnp.exp(-functional.rho * (times - t0))
                        result[functional.name] = jnp.trapezoid(
                            integrand * discount, x=times
                        )
                else:  # FinalRate
                    series = {**state_series, **output_series}[functional.output_name]
                    result[functional.name] = (series[-1] - series[-2]) / (
                        times[-1] - times[-2]
                    )
            for constraint in constraints:
                name = CONSTRAINT_PREFIX + constraint.name
                if isinstance(constraint, PathConstraint):
                    node_values = jax.vmap(
                        lambda t, s, u, c=constraint: c.function(t, s, u, theta)
                    )(times, state_series, control_series)
                    result[name] = constraint.evaluate_on_nodes(node_values)
                elif isinstance(constraint, ControlGroupConstraint):
                    result[name] = constraint.evaluate(control_series)
                elif isinstance(constraint, TerminalConstraint):
                    result[name] = constraint.function(times[-1], final_states, theta)
                else:  # PeriodicityConstraint
                    result[name] = constraint.evaluate(initial_states, final_states)
            for name in state_space.states:
                if boundary.is_final_fixed(name):
                    result[f"{CONSTRAINT_PREFIX}final.{name}"] = (
                        final_states[name] - boundary.s1[name]
                    )
            # GEMSEO concatenates outputs for finite differences: 1D at
            # least, so scalar functionals become size-1 arrays.
            return {name: jnp.atleast_1d(value) for name, value in result.items()}

        return outputs_function

    @staticmethod
    def _build_stage_pieces(
        statics,
        statics_input_names,
        dynamics_function,
        outputs_function,
        design_variables,
        input_defaults,
        theta_inputs,
        constants,
    ):
        """Build the split-assembly stage pieces.

        The dynamics stage consumes the design variables and the dynamics
        parameters (raw statics outputs and prefixed constants); the
        outputs stage consumes the timeseries and the same parameters.
        Their default inputs are obtained by one eager evaluation of the
        upstream stages on the defaults.

        Parameters
        ----------
        statics
            The constant-fed chain, if any.
        statics_input_names
            The raw statics input names.
        dynamics_function
            The dynamics dict-to-dict function.
        outputs_function
            The outputs dict-to-dict function.
        design_variables
            The design-variable specs.
        input_defaults
            The default value of every monolithic input.
        theta_inputs
            The input name of each dynamics parameter.
        constants
            The values of the constant inputs.

        Returns
        -------
        tuple
            The dynamics and outputs stage pieces.
        """
        data = dict(input_defaults)
        if statics is not None:
            data.update(
                statics(
                    {name: data[CONSTANT_PREFIX + name] for name in statics_input_names}
                )
            )
        # 1D defaults for the stage grammars (see the note above).
        data = {
            name: jnp.atleast_1d(jnp.asarray(value, dtype=float))
            for name, value in data.items()
        }
        dynamics_inputs = sorted(set(design_variables) | set(theta_inputs.values()))
        dynamics_defaults = {name: data[name] for name in dynamics_inputs}
        dynamics_result = dynamics_function(data)
        data.update(dynamics_result)
        outputs_result = outputs_function(data)
        outputs_inputs = sorted(set(dynamics_result) | set(theta_inputs.values()))
        dynamics_piece = StagePiece(
            function=dynamics_function,
            input_names=dynamics_inputs,
            output_names=sorted(dynamics_result),
            default_inputs=dynamics_defaults,
        )
        outputs_piece = StagePiece(
            function=outputs_function,
            input_names=outputs_inputs,
            output_names=sorted(outputs_result),
            default_inputs={name: data[name] for name in outputs_inputs},
        )
        return dynamics_piece, outputs_piece
