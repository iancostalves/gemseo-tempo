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
"""Free final time by time normalization."""

from __future__ import annotations

from gemseo_tempo.dynamics.state_space import StateSpace

T_FINAL = "t_f"
"""The name of the final-time parameter appended by :func:`normalize_time`."""


def normalize_time(state_space: StateSpace) -> StateSpace:
    """Reformulate a state space on the normalized time τ in [0, 1].

    The physical time is ``t = τ * t_f`` and the RHS is scaled by ``t_f``,
    which becomes an ordinary constant parameter: free-final-time problems
    expose ``t_f`` as a design variable and integrate over the fixed span
    [0, 1]. Controls used with the normalized model are parameterized on
    τ, not on the physical time.

    Parameters
    ----------
    state_space
        The model on physical time, starting at ``t = 0``.

    Returns
    -------
    StateSpace
        The model on normalized time, with ``t_f`` appended to the
        parameters.

    Raises
    ------
    ValueError
        If the model already has a ``t_f`` parameter.
    """
    if T_FINAL in state_space.parameters:
        raise ValueError(f"The state space already has a {T_FINAL!r} parameter.")

    def to_physical_rates(u, t_final):
        """Rescale control rates from normalized to physical time.

        The controls are parameterized on τ, so a rate computed here is
        ``du/dτ``; the inner model is written on physical time and wants
        ``du/dt = (du/dτ) / t_f``. Forgetting this leaves every
        rate-reading model wrong by the final time it was solved at, which
        is a factor that changes as the optimizer moves.
        """
        if not state_space.control_rates:
            return u
        scaled = dict(u)
        for name in state_space.control_rates:
            key = state_space.rate_name(name)
            scaled[key] = u[key] / t_final
        return scaled

    def rhs(tau, s, u, theta):
        t_final = theta[T_FINAL]
        inner_theta = {k: v for k, v in theta.items() if k != T_FINAL}
        rates = state_space.rhs(
            tau * t_final, s, to_physical_rates(u, t_final), inner_theta
        )
        return {name: rate * t_final for name, rate in rates.items()}

    def output(tau, s, u, theta):
        t_final = theta[T_FINAL]
        inner_theta = {k: v for k, v in theta.items() if k != T_FINAL}
        return state_space.output(
            tau * t_final, s, to_physical_rates(u, t_final), inner_theta
        )

    return StateSpace(
        states=state_space.states,
        controls=state_space.controls,
        parameters=(*state_space.parameters, T_FINAL),
        outputs=state_space.outputs,
        rhs=rhs,
        output=output,
        control_rates=state_space.control_rates,
    )
