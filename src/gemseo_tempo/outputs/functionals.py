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
"""Functionals turning a trajectory into scalars.

Integral terms (:class:`.Lagrange`, :class:`.TimeIntegral`) are computed
by state augmentation: the transcription appends one quadrature state per
integral functional, so the value is exact under adaptive stepping (no
resampling error), and the optional discount rate multiplies the
integrand by ``exp(-rho (t - t0))``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class AbstractFunctional:
    """A named scalar computed from a simulated trajectory."""

    name: str
    """The output name of the functional value."""

    def __init__(self, name: str) -> None:
        """Store the name.

        Parameters
        ----------
        name
            The output name of the functional value.
        """
        self.name = name


class Mayer(AbstractFunctional):
    """A terminal-cost functional ``phi(t1, s(t1), theta)``."""

    def __init__(self, name: str, function: Callable) -> None:
        """Store the terminal function.

        Parameters
        ----------
        name
            The output name.
        function
            The JAX-traceable terminal cost ``(t1, s1, theta) -> scalar``
            with ``s1`` the final states keyed by name.
        """
        super().__init__(name)
        self.function = function


class Lagrange(AbstractFunctional):
    """A running-cost functional ``integral of l(t, s, u, theta) dt``.

    Integrated as an augmented quadrature state, so the value is exact
    under adaptive stepping. With a nonzero discount rate ``rho`` the
    integrand is multiplied by ``exp(-rho (t - t0))``.
    """

    def __init__(self, name: str, integrand: Callable, rho: float = 0.0) -> None:
        """Store the integrand.

        Parameters
        ----------
        name
            The output name.
        integrand
            The JAX-traceable running cost ``(t, s, u, theta) -> scalar``.
        rho
            The discount rate.
        """
        super().__init__(name)
        self.integrand = integrand
        self.rho = float(rho)


class TimeIntegral(AbstractFunctional):
    """The time integral of a declared state-space output.

    A convenience :class:`.Lagrange` whose integrand is a named output of
    the state space (the NOADS ``cumulative.`` concept), sharing the
    quadrature-state machinery and the optional discount rate.
    """

    def __init__(self, name: str, output_name: str, rho: float = 0.0) -> None:
        """Store the integrated output name.

        Parameters
        ----------
        name
            The output name of the integral value.
        output_name
            The name of the state-space output to integrate.
        rho
            The discount rate.
        """
        super().__init__(name)
        self.output_name = output_name
        self.rho = float(rho)


class FinalRate(AbstractFunctional):
    """The rate of change of a trajectory quantity at final time.

    Computed as the backward difference over the last save interval (the
    NOADS ``final_rate.`` concept, used e.g. to ensure fleet turnover is
    complete by the end year).
    """

    def __init__(self, name: str, output_name: str) -> None:
        """Store the differentiated output name.

        Parameters
        ----------
        name
            The output name of the rate value.
        output_name
            The name of the state or state-space output to differentiate.
        """
        super().__init__(name)
        self.output_name = output_name
