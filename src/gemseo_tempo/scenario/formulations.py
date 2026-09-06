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
        disciplines: Sequence | None = None,
        extra_constraints: Sequence[tuple[str, str]] = (),
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
        disciplines
            The disciplines to optimize, replacing the scenario's own.
            They must between them produce the objective and every
            constraint. This exists so a discipline can compute something
            the transcription does not, a statistic over a sample of
            uncertain parameters being the case it was added for.

            It replaces rather than extends, and that is the whole point.
            Chaining a second discipline alongside the scenario's own
            leaves the **first** one undifferentiated: a
            :class:`~gemseo_jax.jax_discipline.JAXDiscipline` filters its
            Jacobian down to the inputs and outputs GEMSEO declares to it,
            those declarations do not reach a discipline in a chain, and
            an empty filter returns zeros rather than an error. The
            objective and every transcription constraint come back with
            gradient exactly zero, the only surviving row is the added
            one, and SLSQP stops on the first iteration with a singular
            LSQ matrix. Nothing raises. So the way to add a quantity here
            is to compute it in the *same* discipline, which
            :func:`~gemseo_tempo.uncertainty.robust.robust_discipline`
            does, and these problems are one jitted program each anyway.
        extra_constraints
            ``(output name, constraint type)`` pairs to wire in alongside
            the transcription's own.
        """
        self.scenario = scenario
        self.maximize = maximize
        self.bounds = dict(bounds or {})
        self.disciplines = list(disciplines) if disciplines is not None else None
        self.extra_constraints = list(extra_constraints)

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
            Note that chaining several JAX disciplines silently
            un-differentiates all but the last; see ``disciplines``.

        Returns
        -------
        BaseScenario
            The scenario, ready for ``execute(algo_name=..., ...)``.
        """
        formulation_settings.setdefault("formulation_name", "DisciplinaryOpt")
        problem = self.scenario.problem
        disciplines = (
            self.scenario.disciplines if self.disciplines is None else self.disciplines
        )
        gemseo_scenario = create_scenario(
            disciplines,
            objective_name=problem.objective_name,
            design_space=self.create_design_space(),
            maximize_objective=self.maximize,
            **formulation_settings,
        )
        for name, kind in (*problem.constraint_names, *self.extra_constraints):
            gemseo_scenario.add_constraint(name, constraint_type=kind)
        if compile_jit:
            # Order matters: building the scenario and adding the
            # constraints is what declares the differentiated inputs and
            # outputs, and declaring them rebuilds the jacobian function.
            # Compiling any earlier would leave that rebuilt jacobian
            # uncompiled.
            for discipline in disciplines:
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
        **settings,
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
        **settings
            Forwarded to :class:`.BaseFormulationBuilder`, which is where
            the extra disciplines and constraints are declared.

        Raises
        ------
        ValueError
            If a design constant is not an input of the problem.
        """
        super().__init__(scenario, maximize=maximize, bounds=bounds, **settings)
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
