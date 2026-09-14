# Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
# Copyright 2025 IRT Saint Exupéry, https://www.irt-saintexupery.com
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
"""The two execution models of the dynamics stage.

:class:`.DiffraxIntegrator` integrates the state-space ODE with diffrax;
:class:`.QuasiStaticIntegrator` evaluates a per-timestep algebraic model
vmapped over the time grid (the AeroMAPS-style execution model), with
first-order delay chains solved by diffrax. The adjoint enum and its
compatibility checks are ported from NOADS ``TemporalScenario``
(LGPL-3.0).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from typing import TYPE_CHECKING

import diffrax
import jax
import jax.numpy as jnp

from gemseo_tempo.dynamics.controls import DelayedControl

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from jax import Array

    from gemseo_tempo.dynamics.controls import AbstractControl
    from gemseo_tempo.dynamics.state_space import StateSpace


class AdjointMethod(StrEnum):
    """The diffrax adjoint used to differentiate through ODE solves."""

    RECURSIVE = "RecursiveCheckpoint"
    DIRECT = "Direct"
    BACKSOLVE = "Backsolve"


_ADJOINTS = {
    AdjointMethod.RECURSIVE: diffrax.RecursiveCheckpointAdjoint,
    AdjointMethod.DIRECT: diffrax.DirectAdjoint,
    AdjointMethod.BACKSOLVE: diffrax.BacksolveAdjoint,
}

FORWARD = "forward"
"""The forward-mode (jvp) differentiation method."""

REVERSE = "reverse"
"""The reverse-mode (vjp) differentiation method."""

# Which differentiation modes each adjoint supports. The Direct adjoint
# unrolls the solve, so both jvp and vjp apply; the RecursiveCheckpoint and
# Backsolve adjoints define only a custom reverse rule, so forward-mode
# differentiation through them raises ``can't apply jvp to a custom_vjp``.
_SUPPORTED_MODES = {
    AdjointMethod.DIRECT: (FORWARD, REVERSE),
    AdjointMethod.RECURSIVE: (REVERSE,),
    AdjointMethod.BACKSOLVE: (REVERSE,),
}


def check_adjoint_compatibility(
    adjoint_method: AdjointMethod,
    differentiation_method: str,
) -> None:
    """Check that an adjoint and a differentiation mode work together.

    The compatibility matrix is: the Direct adjoint supports both forward
    and reverse differentiation, while the RecursiveCheckpoint and
    Backsolve adjoints support reverse only (they define no forward rule).
    Ported and generalized from the NOADS ``TemporalScenario`` validation.

    Parameters
    ----------
    adjoint_method
        The diffrax adjoint.
    differentiation_method
        The JAX differentiation mode, ``"forward"`` or ``"reverse"``.

    Raises
    ------
    ValueError
        If the mode is unknown, or the adjoint does not support it.
    """
    adjoint_method = AdjointMethod(adjoint_method)
    if differentiation_method not in (FORWARD, REVERSE):
        raise ValueError(
            f"Unknown differentiation method {differentiation_method!r};"
            f" use {FORWARD!r} or {REVERSE!r}."
        )
    supported = _SUPPORTED_MODES[adjoint_method]
    if differentiation_method not in supported:
        raise ValueError(
            f"The {adjoint_method} adjoint does not support"
            f" {differentiation_method!r} differentiation (only"
            f" {', '.join(supported)}); use the Direct adjoint for forward"
            " mode."
        )


@dataclass
class Trajectory:
    """A simulated trajectory on the save grid.

    Any declared state, control, or output is available on the save grid
    regardless of the transcription, giving a uniform timeseries view.
    """

    ts: Array
    """The save times."""

    states: dict[str, Array] = field(default_factory=dict)
    """The state timeseries, keyed by name."""

    controls: dict[str, Array] = field(default_factory=dict)
    """The control timeseries, keyed by name (delayed values for delayed
    controls)."""

    outputs: dict[str, Array] = field(default_factory=dict)
    """The instantaneous-output timeseries, keyed by name."""


class AbstractIntegrator:
    """Base class of the dynamics execution models."""

    def integrate(
        self,
        state_space: StateSpace,
        s0: Mapping[str, Array],
        controls: Sequence[AbstractControl],
        control_params: Mapping[str, Array],
        theta: Mapping[str, Array],
        t0: float,
        t1: float,
    ) -> Trajectory:
        """Simulate the state space over a time span.

        Parameters
        ----------
        state_space
            The model.
        s0
            The initial states, keyed by name.
        controls
            The control parameterizations, one per state-space control.
        control_params
            The parameter vector of each control, keyed by control name.
        theta
            The constant parameters, keyed by name.
        t0
            The initial time.
        t1
            The final time.

        Returns
        -------
        Trajectory
            The trajectory on the save grid.

        Raises
        ------
        NotImplementedError
            Always; subclasses implement the simulation.
        """
        raise NotImplementedError


class DiffraxIntegrator(AbstractIntegrator):
    """ODE integration of the state-space RHS with diffrax.

    Delay stages of :class:`.DelayedControl` parameterizations are
    appended to the flattened state automatically. The compiled program is
    cached per (state space, controls) pair, so repeated calls with
    identical structure trace and compile exactly once.
    """

    def __init__(
        self,
        solver: diffrax.AbstractSolver | None = None,
        adjoint: AdjointMethod | str = AdjointMethod.RECURSIVE,
        dt0: float | None = None,
        rtol: float | None = None,
        atol: float | None = None,
        n_save: int = 101,
        max_steps: int = 16384,
    ) -> None:
        """Configure the solver, stepping, and adjoint.

        Parameters
        ----------
        solver
            The diffrax solver; Tsit5 when omitted.
        adjoint
            The adjoint used to differentiate through the solve.
        dt0
            The step size (fixed stepping), or the initial step size when
            tolerances are given (adaptive stepping).
        rtol
            The relative tolerance; giving ``rtol`` or ``atol`` enables
            adaptive stepping with a PID controller.
        atol
            The absolute tolerance.
        n_save
            The number of evenly spaced save points.
        max_steps
            The maximum number of solver steps.

        Raises
        ------
        ValueError
            If neither a step size nor tolerances are given.
        """
        self.solver = solver if solver is not None else diffrax.Tsit5()
        self.adjoint_method = AdjointMethod(adjoint)
        self.adjoint = _ADJOINTS[self.adjoint_method]()
        self.n_save = int(n_save)
        self.max_steps = int(max_steps)
        if rtol is not None or atol is not None:
            self.stepsize_controller = diffrax.PIDController(
                rtol=rtol if rtol is not None else 1e-8,
                atol=atol if atol is not None else 1e-8,
            )
            self.dt0 = dt0
        elif dt0 is not None:
            self.stepsize_controller = diffrax.ConstantStepSize()
            self.dt0 = float(dt0)
        else:
            raise ValueError(
                "Give either dt0 (fixed stepping) or rtol/atol (adaptive stepping)."
            )
        self._compiled: dict = {}

    def check_differentiation_method(self, differentiation_method: str) -> None:
        """Check this integrator's adjoint against a differentiation mode.

        Parameters
        ----------
        differentiation_method
            The JAX differentiation mode, ``"forward"`` or ``"reverse"``.
        """
        check_adjoint_compatibility(self.adjoint_method, differentiation_method)

    def integrate(
        self,
        state_space: StateSpace,
        s0: Mapping[str, Array],
        controls: Sequence[AbstractControl],
        control_params: Mapping[str, Array],
        theta: Mapping[str, Array],
        t0: float,
        t1: float,
    ) -> Trajectory:
        """Integrate the ODE and sample the trajectory on the save grid.

        Parameters
        ----------
        state_space
            The model.
        s0
            The initial states, keyed by name.
        controls
            The control parameterizations, one per state-space control.
        control_params
            The parameter vector of each control, keyed by control name.
        theta
            The constant parameters, keyed by name.
        t0
            The initial time.
        t1
            The final time.

        Returns
        -------
        Trajectory
            The trajectory on the save grid.
        """
        key = (id(state_space), tuple(id(control) for control in controls))
        if key not in self._compiled:
            self._compiled[key] = jax.jit(
                self._build_program(state_space, tuple(controls))
            )
        ts, states, control_values, outputs = self._compiled[key](
            dict(s0), dict(control_params), dict(theta), t0, t1
        )
        return Trajectory(
            ts=ts, states=states, controls=control_values, outputs=outputs
        )

    def _build_program(self, state_space: StateSpace, controls: tuple):
        """Build the traceable simulation program for a model structure.

        Parameters
        ----------
        state_space
            The model.
        controls
            The control parameterizations.

        Returns
        -------
        Callable
            The program ``(s0, control_params, theta, t0, t1) ->
            (ts, states, controls, outputs)``.
        """
        n_states = len(state_space.states)
        delayed = [c for c in controls if isinstance(c, DelayedControl)]
        by_name = {control.name: control for control in controls}
        rate_sources = {}
        for name in state_space.control_rates:
            control = by_name[name]
            if isinstance(control, DelayedControl):
                msg = (
                    f"The state space asks for the time-derivative of {name!r},"
                    " which is a delayed control carried as an extra state;"
                    " its rate is a stage equation rather than a derivative"
                    " of a function of time."
                )
                raise TypeError(msg)
            rate_sources[name] = control

        def control_values_from(t, y, control_params):
            """Read control values at time t from the augmented state."""
            values = {}
            index = n_states
            for control in controls:
                if isinstance(control, DelayedControl):
                    values[control.name] = y[index + control.order - 1]
                    index += control.order
                else:
                    values[control.name] = control(t, control_params[control.name])
            for name in state_space.control_rates:
                # The control is a differentiable function of time here, so
                # its rate is the derivative of the same call the value came
                # from, rather than a finite difference of it.
                control = rate_sources[name]
                values[state_space.rate_name(name)] = jax.grad(
                    lambda time, params, control=control: jnp.reshape(
                        control(time, params), ()
                    )
                )(t, control_params[name])
            return values

        def rhs(t, y, args):
            control_params, theta = args
            s = state_space.unflatten(y)
            u = control_values_from(t, y, control_params)
            rates = state_space.rhs(t, s, u, theta)
            parts = [jnp.atleast_1d(jnp.asarray(rates[k])) for k in state_space.states]
            index = n_states
            for control in delayed:
                stages = y[index : index + control.order]
                index += control.order
                parts.append(control.stage_rhs(t, stages, control_params[control.name]))
            return jnp.concatenate(parts)

        def program(s0, control_params, theta, t0, t1):
            parts = [state_space.flatten(s0)]
            parts.extend(
                control.initial_states(t0, control_params[control.name])
                for control in delayed
            )
            y0 = jnp.concatenate(parts)
            ts = jnp.linspace(t0, t1, self.n_save)
            solution = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(rhs),
                solver=self.solver,
                t0=t0,
                t1=t1,
                dt0=self.dt0,
                y0=y0,
                args=(control_params, theta),
                saveat=diffrax.SaveAt(ts=ts),
                stepsize_controller=self.stepsize_controller,
                adjoint=self.adjoint,
                max_steps=self.max_steps,
            )
            ys = solution.ys
            states = {
                name: ys[:, index] for index, name in enumerate(state_space.states)
            }
            control_series = jax.vmap(
                lambda t, y: control_values_from(t, y, control_params)
            )(ts, ys)
            outputs = jax.vmap(
                lambda t, y, u: state_space.output(
                    t, state_space.unflatten(y), u, theta
                )
            )(ts, ys, control_series)
            return ts, states, control_series, outputs

        return program


class QuasiStaticIntegrator(AbstractIntegrator):
    """Per-timestep algebraic evaluation vmapped over the time grid.

    No integrated state: the state space must be purely algebraic (no
    states), constants are broadcast, controls are sampled on the grid,
    and first-order delay chains are solved by diffrax. This is the
    AeroMAPS-style execution model, ported from the NOADS
    ``TemporalScenario`` quasi-static path.
    """

    def __init__(
        self,
        time_step: float = 1.0,
        adjoint: AdjointMethod | str = AdjointMethod.RECURSIVE,
        solver: diffrax.AbstractSolver | None = None,
    ) -> None:
        """Configure the grid and the delay-chain solver.

        Parameters
        ----------
        time_step
            The grid spacing.
        adjoint
            The adjoint used to differentiate through the delay solves.
        solver
            The diffrax solver for the delay chains; Dopri5 when omitted.
        """
        self.time_step = float(time_step)
        self.adjoint_method = AdjointMethod(adjoint)
        self.adjoint = _ADJOINTS[self.adjoint_method]()
        self.solver = solver if solver is not None else diffrax.Dopri5()
        self._compiled: dict = {}

    def check_differentiation_method(self, differentiation_method: str) -> None:
        """Check this integrator's adjoint against a differentiation mode.

        Parameters
        ----------
        differentiation_method
            The JAX differentiation mode, ``"forward"`` or ``"reverse"``.
        """
        check_adjoint_compatibility(self.adjoint_method, differentiation_method)

    def integrate(
        self,
        state_space: StateSpace,
        s0: Mapping[str, Array],
        controls: Sequence[AbstractControl],
        control_params: Mapping[str, Array],
        theta: Mapping[str, Array],
        t0: float,
        t1: float,
    ) -> Trajectory:
        """Evaluate the algebraic model over the time grid.

        Parameters
        ----------
        state_space
            The model; must declare no states.
        s0
            Ignored; the model has no states.
        controls
            The control parameterizations, one per state-space control.
        control_params
            The parameter vector of each control, keyed by control name.
        theta
            The constant parameters, keyed by name.
        t0
            The initial time.
        t1
            The final time; ``t0``, ``t1`` and the step define the static
            grid, so they must be Python floats.

        Returns
        -------
        Trajectory
            The trajectory on the grid.

        Raises
        ------
        ValueError
            If the state space declares states.
        """
        if state_space.states:
            raise ValueError(
                "QuasiStaticIntegrator only supports algebraic state spaces"
                f" (no states); got states {state_space.states}. Model"
                " memory belongs in DelayedControl parameterizations or in"
                " the DiffraxIntegrator."
            )
        n_steps = int(round((float(t1) - float(t0)) / self.time_step))
        key = (
            id(state_space),
            tuple(id(control) for control in controls),
            float(t0),
            float(t1),
        )
        if key not in self._compiled:
            ts = jnp.linspace(float(t0), float(t1), n_steps + 1)
            self._compiled[key] = jax.jit(
                self._build_program(state_space, tuple(controls), ts)
            )
        ts, control_series, outputs = self._compiled[key](
            dict(control_params), dict(theta)
        )
        return Trajectory(ts=ts, states={}, controls=control_series, outputs=outputs)

    def _build_program(self, state_space: StateSpace, controls: tuple, ts: Array):
        """Build the traceable evaluation program for a model structure.

        Parameters
        ----------
        state_space
            The model.
        controls
            The control parameterizations.
        ts
            The static time grid.

        Returns
        -------
        Callable
            The program ``(control_params, theta) ->
            (ts, controls, outputs)``.
        """

        def delayed_series(control, params):
            """Solve the delay chain of a control over the grid."""

            def chain_rhs(t, stages, args):
                return control.stage_rhs(t, stages, args)

            solution = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(chain_rhs),
                solver=self.solver,
                t0=ts[0],
                t1=ts[-1],
                dt0=ts[1] - ts[0],
                y0=control.initial_states(ts[0], params),
                args=params,
                saveat=diffrax.SaveAt(ts=ts),
                adjoint=self.adjoint,
            )
            return solution.ys[:, -1]

        def program(control_params, theta):
            control_series = {}
            for control in controls:
                params = control_params[control.name]
                if isinstance(control, DelayedControl):
                    control_series[control.name] = delayed_series(control, params)
                else:
                    control_series[control.name] = jax.vmap(
                        lambda t, c=control, p=params: c(t, p)
                    )(ts)
            theta_broadcast = {
                name: value * jnp.ones(ts.shape) for name, value in theta.items()
            }
            outputs = jax.vmap(lambda t, u, th: state_space.output(t, {}, u, th))(
                ts, control_series, theta_broadcast
            )
            return ts, control_series, outputs

        return program
