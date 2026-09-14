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
"""The continuous-time state-space model of the dynamics stage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from jax import Array


class StateSpace:
    """A named continuous-time state-space model.

    The model is defined by a right-hand side ``ds/dt = rhs(t, s, u, theta)``
    and instantaneous outputs ``output(t, s, u, theta)``, all operating on
    dictionaries keyed by variable name. Both functions can be passed to the
    constructor or overridden in a subclass. Scalar state components are
    stacked into a flat array (in declaration order) for the integrators via
    :meth:`flatten` / :meth:`unflatten`.
    """

    states: tuple[str, ...]
    """The state names, defining the flattening order."""

    controls: tuple[str, ...]
    """The control names."""

    parameters: tuple[str, ...]
    """The constant-parameter names."""

    outputs: tuple[str, ...]
    """The instantaneous-output names."""

    control_rates: tuple[str, ...]
    """The controls whose time-derivative the model needs.

    Empty for almost every model: a right-hand side normally reads the
    control's *value* and nothing else. A few physical laws need its rate
    as well — the normal force on a bead constrained to a wire carries a
    centripetal term in the turning rate of the path angle, which is the
    control — and for those the rate arrives in ``u`` under
    ``"<name>_rate"``, so that :meth:`rhs` keeps its signature and no model
    that does not ask for a rate is affected.
    """

    RATE_SUFFIX = "_rate"
    """How a control's time-derivative is named in the control dictionary."""

    def __init__(
        self,
        states: Sequence[str],
        controls: Sequence[str] = (),
        parameters: Sequence[str] = (),
        outputs: Sequence[str] = (),
        rhs: Callable | None = None,
        output: Callable | None = None,
        control_rates: Sequence[str] = (),
    ) -> None:
        """Declare the variables and optionally the model functions.

        Parameters
        ----------
        states
            The state names.
        controls
            The control names.
        parameters
            The constant-parameter names.
        outputs
            The instantaneous-output names.
        rhs
            The right-hand side ``(t, s, u, theta) -> ds`` returning the
            rate of each state, keyed by state name; alternatively override
            :meth:`rhs` in a subclass.
        output
            The instantaneous outputs ``(t, s, u, theta) -> dict``;
            alternatively override :meth:`output` in a subclass.
        control_rates
            The controls whose time-derivative ``rhs`` reads, delivered in
            ``u`` as ``"<name>_rate"``; see :attr:`control_rates`.

        Raises
        ------
        ValueError
            If a control rate is asked for a name that is not a control.
        """
        self.states = tuple(states)
        self.controls = tuple(controls)
        self.parameters = tuple(parameters)
        self.outputs = tuple(outputs)
        self.control_rates = tuple(control_rates)
        self._rhs = rhs
        self._output = output
        unknown = sorted(set(self.control_rates) - set(self.controls))
        if unknown:
            msg = (
                f"Cannot take the time-derivative of {unknown}, which"
                f" {'is' if len(unknown) == 1 else 'are'} not among the"
                f" controls {list(self.controls)}."
            )
            raise ValueError(msg)

    def rate_name(self, name: str) -> str:
        """Return the key a control's time-derivative arrives under.

        Parameters
        ----------
        name
            The control name.

        Returns
        -------
        str
            The key in the control dictionary passed to :meth:`rhs`.
        """
        return f"{name}{self.RATE_SUFFIX}"

    def rhs(self, t, s: dict, u: dict, theta: dict) -> dict:
        """Compute the state rates.

        Parameters
        ----------
        t
            The time.
        s
            The states, keyed by name.
        u
            The controls, keyed by name.
        theta
            The constant parameters, keyed by name.

        Returns
        -------
        dict
            The rate of each state, keyed by state name.

        Raises
        ------
        NotImplementedError
            If no ``rhs`` function was given and the method is not
            overridden.
        """
        if self._rhs is None:
            raise NotImplementedError(
                "Pass rhs= to the constructor or override StateSpace.rhs()."
            )
        return self._rhs(t, s, u, theta)

    def output(self, t, s: dict, u: dict, theta: dict) -> dict:
        """Compute the instantaneous outputs.

        Parameters
        ----------
        t
            The time.
        s
            The states, keyed by name.
        u
            The controls, keyed by name.
        theta
            The constant parameters, keyed by name.

        Returns
        -------
        dict
            The outputs, keyed by name; empty when no output function was
            given.
        """
        if self._output is None:
            return {}
        return self._output(t, s, u, theta)

    def flatten(self, s: dict) -> Array:
        """Stack named scalar states into a flat array.

        Parameters
        ----------
        s
            The states, keyed by name.

        Returns
        -------
        Array
            The stacked states, in declaration order.
        """
        if not self.states:
            return jnp.zeros(0)
        return jnp.stack([jnp.asarray(s[name]) for name in self.states])

    def unflatten(self, y: Array) -> dict:
        """Split a flat state array back into named scalar states.

        Parameters
        ----------
        y
            The stacked states, in declaration order.

        Returns
        -------
        dict
            The states, keyed by name.
        """
        return {name: y[index] for index, name in enumerate(self.states)}
