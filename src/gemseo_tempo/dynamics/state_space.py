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

    def __init__(
        self,
        states: Sequence[str],
        controls: Sequence[str] = (),
        parameters: Sequence[str] = (),
        outputs: Sequence[str] = (),
        rhs: Callable | None = None,
        output: Callable | None = None,
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
        """
        self.states = tuple(states)
        self.controls = tuple(controls)
        self.parameters = tuple(parameters)
        self.outputs = tuple(outputs)
        self._rhs = rhs
        self._output = output

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
