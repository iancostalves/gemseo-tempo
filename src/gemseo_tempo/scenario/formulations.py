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
"""Formulation builders emitting GEMSEO design spaces and scenarios."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy
from gemseo import create_scenario
from gemseo.algos.design_space import DesignSpace

from gemseo_tempo.transcriptions.base import CONSTANT_PREFIX
from gemseo_tempo.transcriptions.base import T_FINAL

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.scenarios.base_scenario import BaseScenario

    from gemseo_tempo.scenario.dynamic_scenario import DynamicScenario


class BaseFormulationBuilder:
    """Base class of the formulation builders.

    A builder selects the design variables of a
    :class:`.DynamicScenario`, produces the GEMSEO
    :class:`~gemseo.algos.design_space.DesignSpace` (with bounds from the
    variable registry when present), and wires the objective and
    constraints into a GEMSEO scenario.
    """

    def __init__(
        self,
        scenario: DynamicScenario,
        maximize: bool = False,
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Store the scenario.

        Parameters
        ----------
        scenario
            The dynamic scenario.
        maximize
            Whether to maximize the objective instead of minimizing it.
        bounds
            Explicit ``(lower, upper)`` bounds per design-variable name,
            overriding the registry.
        """
        self.scenario = scenario
        self.maximize = maximize
        self.bounds = dict(bounds or {})

    def _design_variable_names(self) -> list[str]:
        """Return the names of the design variables of this formulation.

        Returns
        -------
        list
            The grammar names.

        Raises
        ------
        NotImplementedError
            Always; subclasses select the design variables.
        """
        raise NotImplementedError

    def _bounds_of(self, name: str) -> tuple[float, float]:
        """Return the bounds of a design variable.

        Explicit bounds win; otherwise the registry entry of the raw
        (unprefixed) name is used; otherwise the variable is unbounded.

        Parameters
        ----------
        name
            The grammar name.

        Returns
        -------
        tuple
            The lower and upper bounds.
        """
        if name in self.bounds:
            return self.bounds[name]
        registry = self.scenario.registry
        raw = name.rsplit(".", 1)[-1]
        if registry is not None and raw in registry:
            variable = registry[raw]
            return (
                variable.lower_bound
                if variable.lower_bound is not None
                else -numpy.inf,
                variable.upper_bound if variable.upper_bound is not None else numpy.inf,
            )
        return -numpy.inf, numpy.inf

    def create_design_space(self) -> DesignSpace:
        """Build the design space of this formulation.

        Returns
        -------
        DesignSpace
            The design space, with defaults as initial values and bounds
            from the explicit ``bounds``, then the registry, then
            unbounded.
        """
        design_space = DesignSpace()
        defaults = self.scenario.problem.input_defaults
        specs = self.scenario.problem.design_variables
        for name in self._design_variable_names():
            lower, upper = self._bounds_of(name)
            value = numpy.atleast_1d(numpy.asarray(defaults[name]))
            size = specs[name].size if name in specs else value.shape[0]
            design_space.add_variable(
                name,
                size=size,
                lower_bound=lower,
                upper_bound=upper,
                value=value,
            )
        return design_space

    def create_gemseo_scenario(
        self, compile_jit: bool = True, **formulation_settings
    ) -> BaseScenario:
        """Build the GEMSEO scenario with wired objective and constraints.

        Parameters
        ----------
        compile_jit
            Whether to JIT-compile the disciplines. A JAX discipline is not
            compiled on construction, so leaving this off runs every
            evaluation in eager mode, which costs orders of magnitude on
            these problems.
        **formulation_settings
            The GEMSEO formulation settings; the formulation defaults to
            ``DisciplinaryOpt``, which chains the disciplines in order.

        Returns
        -------
        BaseScenario
            The scenario, ready for ``execute(algo_name=..., ...)``.
        """
        formulation_settings.setdefault("formulation_name", "DisciplinaryOpt")
        problem = self.scenario.problem
        gemseo_scenario = create_scenario(
            self.scenario.disciplines,
            objective_name=problem.objective_name,
            design_space=self.create_design_space(),
            maximize_objective=self.maximize,
            **formulation_settings,
        )
        for name, kind in problem.constraint_names:
            gemseo_scenario.add_constraint(name, constraint_type=kind)
        if compile_jit:
            # Order matters: building the scenario and adding the
            # constraints is what declares the differentiated inputs and
            # outputs, and declaring them rebuilds the jacobian function.
            # Compiling any earlier would leave that rebuilt jacobian
            # uncompiled.
            for discipline in self.scenario.disciplines:
                discipline.compile_jit(pre_run=False)
        return gemseo_scenario


class OptimalControl(BaseFormulationBuilder):
    """Optimize the controls of a fixed design.

    The design variables are the transcription's: control parameters,
    the free final time, and the free initial-state components.
    """

    def _design_variable_names(self) -> list[str]:
        """Return the transcription design-variable names.

        Returns
        -------
        list
            The grammar names.
        """
        return list(self.scenario.problem.design_variables)


class SystemDesign(BaseFormulationBuilder):
    """Optimize design constants under a fixed control shape."""

    def __init__(
        self,
        scenario: DynamicScenario,
        design_names: Sequence[str],
        maximize: bool = False,
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Store the design constants.

        Parameters
        ----------
        scenario
            The dynamic scenario.
        design_names
            The raw (unprefixed) names of the constant inputs to
            optimize.
        maximize
            Whether to maximize the objective instead of minimizing it.
        bounds
            Explicit ``(lower, upper)`` bounds per design-variable name,
            overriding the registry.

        Raises
        ------
        ValueError
            If a design constant is not an input of the problem.
        """
        super().__init__(scenario, maximize=maximize, bounds=bounds)
        self.design_names = [CONSTANT_PREFIX + name for name in design_names]
        unknown = sorted(set(self.design_names) - set(scenario.problem.input_defaults))
        if unknown:
            raise ValueError(
                f"Design constants {unknown} are not inputs of the problem."
            )

    def _design_variable_names(self) -> list[str]:
        """Return the prefixed design-constant names.

        Returns
        -------
        list
            The grammar names.
        """
        return list(self.design_names)


class ControlAndDesign(SystemDesign):
    """Co-optimize the controls and the design constants.

    The design variables are the union of :class:`.OptimalControl` and
    :class:`.SystemDesign`, including the free final time when the
    boundary declares it (``t_f`` conditions well with the co-design
    variables).
    """

    def _design_variable_names(self) -> list[str]:
        """Return the union of control and design names.

        Returns
        -------
        list
            The grammar names, controls first.
        """
        names = list(self.scenario.problem.design_variables)
        names.extend(name for name in self.design_names if name not in names)
        return names


__all__ = [
    "BaseFormulationBuilder",
    "ControlAndDesign",
    "OptimalControl",
    "SystemDesign",
    "T_FINAL",
]
