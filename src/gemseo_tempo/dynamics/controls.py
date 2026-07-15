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
"""Parameterized control trajectories, with optional delay chains."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from gemseo_tempo.dynamics.interpolation import CubicSpline

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array


class AbstractControl:
    """A control trajectory parameterized by a finite parameter vector.

    Calling the control with a time and its parameter vector returns the
    control value; the call is JAX-traceable so it can be evaluated inside
    an ODE right-hand side (shooting, indirect) or vmapped over a time
    grid (collocation, quasi-static).
    """

    name: str
    """The control name, matching a :class:`.StateSpace` control."""

    n_params: int
    """The number of parameters."""

    default_params: Array
    """The default parameter vector, of size :attr:`n_params`."""

    def __init__(self, name: str, default_params) -> None:
        """Store the name and defaults.

        Parameters
        ----------
        name
            The control name.
        default_params
            The default parameter vector.
        """
        self.name = name
        self.default_params = jnp.asarray(default_params, dtype=float)
        self.n_params = self.default_params.shape[0]

    def __call__(self, t, params: Array):
        """Evaluate the control at a time.

        Parameters
        ----------
        t
            The time.
        params
            The parameter vector.

        Returns
        -------
        Array
            The control value.

        Raises
        ------
        NotImplementedError
            Always; subclasses implement the evaluation.
        """
        raise NotImplementedError


class PiecewiseLinearControl(AbstractControl):
    """A control interpolated linearly between values at fixed times."""

    def __init__(self, name: str, times, default_params=None) -> None:
        """Store the interpolation times.

        Parameters
        ----------
        name
            The control name.
        times
            The strictly increasing node times.
        default_params
            The default node values; zeros when omitted.
        """
        self.times = jnp.asarray(times, dtype=float)
        if default_params is None:
            default_params = jnp.zeros(self.times.shape)
        super().__init__(name, default_params)

    def __call__(self, t, params: Array):
        """Interpolate the node values at a time.

        Parameters
        ----------
        t
            The time.
        params
            The node values.

        Returns
        -------
        Array
            The interpolated value; constant beyond the first and last
            nodes.
        """
        return jnp.interp(t, self.times, params)


class CubicControl(AbstractControl):
    """A control interpolated by a C2 cubic spline through node values."""

    def __init__(self, name: str, times, default_params=None) -> None:
        """Store the interpolation times.

        Parameters
        ----------
        name
            The control name.
        times
            The strictly increasing node times; at least 4.
        default_params
            The default node values; zeros when omitted.
        """
        self.times = jnp.asarray(times, dtype=float)
        if default_params is None:
            default_params = jnp.zeros(self.times.shape)
        super().__init__(name, default_params)

    def __call__(self, t, params: Array):
        """Evaluate the interpolating spline at a time.

        Parameters
        ----------
        t
            The time.
        params
            The node values.

        Returns
        -------
        Array
            The spline value; extrapolated with the edge cubic pieces
            beyond the first and last nodes.
        """
        return CubicSpline(self.times, params)(t)


class ParametricControl(AbstractControl):
    """A control given by a user function of time and parameters.

    Typical uses are shaped signals such as sigmoid penetration curves,
    where the parameters are the shape coefficients rather than node
    values.
    """

    def __init__(self, name: str, function: Callable, default_params) -> None:
        """Store the function and defaults.

        Parameters
        ----------
        name
            The control name.
        function
            The JAX-traceable function ``(t, params) -> value``.
        default_params
            The default parameter vector.
        """
        self._function = function
        super().__init__(name, default_params)

    def __call__(self, t, params: Array):
        """Evaluate the user function.

        Parameters
        ----------
        t
            The time.
        params
            The parameter vector.

        Returns
        -------
        Array
            The control value.
        """
        return self._function(t, params)


class DelayedControl(AbstractControl):
    """A control filtered by an order-N linear delay chain.

    The wrapped control provides the input signal; the delayed value is
    the last stage of a chain of ``order`` first-order lags of time
    constant ``time_constant / order`` each (an Erlang filter, the ISDC
    paper approach). The stage values are dynamic states: the integrators
    append them to the global flattened state automatically, and the
    state-space RHS receives the delayed value under this control's name.
    """

    inner: AbstractControl
    """The wrapped control providing the input signal."""

    time_constant: float
    """The total delay time constant."""

    order: int
    """The number of delay stages appended to the global state."""

    def __init__(
        self,
        control: AbstractControl,
        time_constant: float,
        order: int = 1,
        initial_value: float | None = None,
    ) -> None:
        """Wrap a control with the delay chain.

        Parameters
        ----------
        control
            The control providing the input signal.
        time_constant
            The total delay time constant.
        order
            The number of first-order stages.
        initial_value
            The initial value of every stage; the input signal at the
            initial time when omitted (steady-state start).
        """
        self.inner = control
        self.time_constant = float(time_constant)
        self.order = int(order)
        self.initial_value = initial_value
        super().__init__(control.name, control.default_params)

    def __call__(self, t, params: Array):
        """Evaluate the undelayed input signal.

        The delayed value is a dynamic state maintained by the
        integrators, so calling a delayed control directly returns the
        input of the chain, not its output.

        Parameters
        ----------
        t
            The time.
        params
            The parameter vector of the wrapped control.

        Returns
        -------
        Array
            The undelayed input value.
        """
        return self.inner(t, params)

    def initial_states(self, t0, params: Array) -> Array:
        """Return the initial value of the delay stages.

        Parameters
        ----------
        t0
            The initial time.
        params
            The parameter vector of the wrapped control.

        Returns
        -------
        Array
            The stage values at ``t0``, of size :attr:`order`.
        """
        if self.initial_value is None:
            value = self.inner(t0, params)
        else:
            value = jnp.asarray(self.initial_value, dtype=float)
        return jnp.full(self.order, value)

    def stage_rhs(self, t, stages: Array, params: Array) -> Array:
        """Compute the rates of the delay stages.

        Parameters
        ----------
        t
            The time.
        stages
            The stage values, of size :attr:`order`.
        params
            The parameter vector of the wrapped control.

        Returns
        -------
        Array
            The stage rates: each stage relaxes toward its predecessor
            (the input signal for the first stage) with time constant
            ``time_constant / order``.
        """
        rate = self.order / self.time_constant
        drivers = jnp.concatenate(
            [
                jnp.atleast_1d(self.inner(t, params)),
                stages[:-1],
            ]
        )
        return (drivers - stages) * rate
