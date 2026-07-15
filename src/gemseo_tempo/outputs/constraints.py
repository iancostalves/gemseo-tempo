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
"""Constraints over trajectories.

All constraints follow the GEMSEO sign convention: equality constraints
are ``g = 0`` and inequality constraints are ``g <= 0``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax.scipy.special import logsumexp

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence


class AbstractConstraint:
    """A named constraint with a sense."""

    name: str
    """The output name of the constraint value."""

    kind: str
    """The constraint sense, ``"eq"`` or ``"ineq"``."""

    def __init__(self, name: str, kind: str) -> None:
        """Store the name and sense.

        Parameters
        ----------
        name
            The output name.
        kind
            The sense, ``"eq"`` or ``"ineq"``.

        Raises
        ------
        ValueError
            If the sense is unknown.
        """
        if kind not in ("eq", "ineq"):
            raise ValueError(f"Unknown constraint kind {kind!r}.")
        self.name = name
        self.kind = kind


class PathConstraint(AbstractConstraint):
    """An inequality ``g(t, s, u, theta) <= 0`` sampled at the save nodes.

    Vector-valued by default (one entry per node); with
    ``aggregate="ks"`` the node values are aggregated into one scalar by
    the Kreisselmeier-Steinhauser envelope, a smooth upper bound of the
    maximum that sharpens as ``ks_rho`` grows.
    """

    def __init__(
        self,
        name: str,
        function: Callable,
        aggregate: str | None = None,
        ks_rho: float = 50.0,
    ) -> None:
        """Store the constraint function and aggregation.

        Parameters
        ----------
        name
            The output name.
        function
            The JAX-traceable constraint ``(t, s, u, theta) -> scalar``,
            negative when satisfied.
        aggregate
            The aggregation over nodes: ``None`` (vector value) or
            ``"ks"``.
        ks_rho
            The KS sharpness parameter.

        Raises
        ------
        ValueError
            If the aggregation is unknown.
        """
        super().__init__(name, "ineq")
        if aggregate not in (None, "ks"):
            raise ValueError(f"Unknown aggregation {aggregate!r}.")
        self.function = function
        self.aggregate = aggregate
        self.ks_rho = float(ks_rho)

    def evaluate_on_nodes(self, node_values):
        """Aggregate the node values if requested.

        Parameters
        ----------
        node_values
            The constraint values at the save nodes.

        Returns
        -------
        Array
            The vector of node values, or the KS scalar.
        """
        if self.aggregate == "ks":
            return logsumexp(self.ks_rho * node_values) / self.ks_rho
        return node_values


class ControlGroupConstraint(AbstractConstraint):
    """The NOADS control-group constraint: ``sum(controls) - 1 <= 0``.

    The sum of the named control values must stay below one at every save
    node (e.g. pathway shares of one energy carrier, aircraft shares of
    one market).
    """

    def __init__(self, name: str, control_names: Sequence[str]) -> None:
        """Store the group members.

        Parameters
        ----------
        name
            The output name.
        control_names
            The names of the controls in the group.
        """
        super().__init__(name, "ineq")
        self.control_names = tuple(control_names)

    def evaluate(self, control_series: dict) -> jnp.ndarray:
        """Compute the constraint from the control timeseries.

        Parameters
        ----------
        control_series
            The control values on the save grid, keyed by name.

        Returns
        -------
        Array
            ``sum of the group values - 1`` at every node.
        """
        total = sum(control_series[name] for name in self.control_names)
        return total - 1.0


class TerminalConstraint(AbstractConstraint):
    """A constraint ``psi(t1, s(t1), theta)`` on the final state."""

    def __init__(self, name: str, function: Callable, kind: str = "eq") -> None:
        """Store the terminal function and sense.

        Parameters
        ----------
        name
            The output name.
        function
            The JAX-traceable function ``(t1, s1, theta) -> scalar`` with
            ``s1`` the final states keyed by name.
        kind
            The sense, ``"eq"`` or ``"ineq"``.
        """
        super().__init__(name, kind)
        self.function = function


class PeriodicityConstraint(AbstractConstraint):
    """The equality ``s(t1) - s(t0) = 0`` on selected states.

    Periodic problems promote the initial states to design variables
    (``fix_initial=False``) and close the cycle with this constraint.
    """

    def __init__(self, name: str, state_names: Sequence[str]) -> None:
        """Store the periodic states.

        Parameters
        ----------
        name
            The output name.
        state_names
            The names of the states to close periodically.
        """
        super().__init__(name, "eq")
        self.state_names = tuple(state_names)

    def evaluate(self, initial_states: dict, final_states: dict) -> jnp.ndarray:
        """Compute the periodicity residuals.

        Parameters
        ----------
        initial_states
            The initial states, keyed by name.
        final_states
            The final states, keyed by name.

        Returns
        -------
        Array
            The stacked residuals ``s(t1) - s(t0)``.
        """
        return jnp.stack(
            [final_states[name] - initial_states[name] for name in self.state_names]
        )
