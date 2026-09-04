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
"""What the optimizer was actually handed.

A transcription turns a trajectory problem into a finite-dimensional one, and
by the time it has done so the count of design variables and constraints is
rarely obvious from the code that built it: a single ``state.h`` is one name
covering a node per segment, and a path constraint is one name covering a
value per save point.

Writing those numbers into documentation by hand means maintaining a second
source of truth that drifts the first time a default changes.
:func:`describe_problem` reads them off the transcribed problem instead, and
measures the constraint sizes by evaluating the problem once rather than
inferring them, so a problem statement in the documentation is the problem
that will actually be solved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    from gemseo_tempo.scenario.dynamic_scenario import DynamicScenario
    from gemseo_tempo.scenario.formulations import BaseFormulationBuilder


@dataclass(frozen=True)
class DesignVariableDescription:
    """One design variable of the finite-dimensional problem."""

    name: str
    """The grammar name."""

    size: int
    """The number of components."""

    lower_bound: float
    """The lower bound, or negative infinity."""

    upper_bound: float
    """The upper bound, or positive infinity."""

    @property
    def bounds_text(self) -> str:
        """Return the bounds as readable text.

        Returns
        -------
        str
            The interval, with unbounded sides shown as ``-inf`` / ``inf``.
        """

        def side(value: float) -> str:
            if numpy.isinf(value):
                return "-inf" if value < 0.0 else "inf"
            return f"{value:g}"

        return f"[{side(self.lower_bound)}, {side(self.upper_bound)}]"


@dataclass(frozen=True)
class ConstraintDescription:
    """One constraint of the finite-dimensional problem."""

    name: str
    """The output name."""

    kind: str
    """The sense, ``"eq"`` or ``"ineq"``."""

    size: int
    """The number of components, measured by evaluating the problem."""


@dataclass(frozen=True)
class ProblemDescription:
    """The problem statement of a transcribed scenario."""

    design_variables: tuple[DesignVariableDescription, ...]
    """The design variables, in the order the formulation declares them."""

    objective: str
    """The objective output name."""

    maximize: bool
    """Whether the objective is maximized rather than minimized."""

    constraints: tuple[ConstraintDescription, ...]
    """The constraints, equalities and inequalities together."""

    @property
    def design_size(self) -> int:
        """Return the total number of scalar design variables.

        Returns
        -------
        int
            The sum of the design-variable sizes.
        """
        return sum(variable.size for variable in self.design_variables)

    @property
    def constraint_size(self) -> int:
        """Return the total number of scalar constraints.

        Returns
        -------
        int
            The sum of the constraint sizes.
        """
        return sum(constraint.size for constraint in self.constraints)

    def to_markdown(self) -> str:
        """Render the problem statement as markdown tables.

        Returns
        -------
        str
            Two tables, design variables and constraints, under a line
            naming the objective and its sense.
        """
        sense = "maximize" if self.maximize else "minimize"
        lines = [
            f"**{sense}** `{self.objective}`",
            "",
            f"over {self.design_size} design variables,"
            f" subject to {self.constraint_size} constraints.",
            "",
            "| design variable | size | bounds |",
            "| --- | --- | --- |",
        ]
        lines.extend(
            f"| `{variable.name}` | {variable.size} | `{variable.bounds_text}` |"
            for variable in self.design_variables
        )
        lines.extend(["", "| constraint | sense | size |", "| --- | --- | --- |"])
        lines.extend(
            f"| `{constraint.name}` | {constraint.kind} | {constraint.size} |"
            for constraint in self.constraints
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        """Return a plain-text summary.

        Returns
        -------
        str
            One line per design variable and constraint.
        """
        sense = "maximize" if self.maximize else "minimize"
        lines = [
            f"{sense} {self.objective}"
            f"  ({self.design_size} variables, {self.constraint_size} constraints)",
            "  design variables:",
        ]
        lines.extend(
            f"    {variable.name:<24} size {variable.size:<5} {variable.bounds_text}"
            for variable in self.design_variables
        )
        lines.append("  constraints:")
        lines.extend(
            f"    {constraint.name:<24} {constraint.kind:<5} size {constraint.size}"
            for constraint in self.constraints
        )
        return "\n".join(lines)


def describe_problem(
    scenario: DynamicScenario,
    formulation: BaseFormulationBuilder | None = None,
) -> ProblemDescription:
    """Report the problem statement a formulation hands the optimizer.

    Parameters
    ----------
    scenario
        The dynamic scenario.
    formulation
        The formulation builder, which decides which variables are design
        variables and what their bounds are. Without it, every transcription
        design variable is reported with the bounds the transcription gave
        it.

    Returns
    -------
    ProblemDescription
        The design variables, objective and constraints.
    """
    problem = scenario.problem
    if formulation is None:
        names = list(problem.design_variables)
        maximize = False

        def bounds_of(name):
            spec = problem.design_variables[name]
            return (
                -numpy.inf if spec.lower_bound is None else spec.lower_bound,
                numpy.inf if spec.upper_bound is None else spec.upper_bound,
            )
    else:
        names = formulation._design_variable_names()
        maximize = formulation.maximize
        bounds_of = formulation._bounds_of

    defaults = problem.input_defaults
    variables = tuple(
        DesignVariableDescription(
            name=name,
            size=(
                problem.design_variables[name].size
                if name in problem.design_variables
                else int(numpy.atleast_1d(numpy.asarray(defaults[name])).size)
            ),
            lower_bound=bounds_of(name)[0],
            upper_bound=bounds_of(name)[1],
        )
        for name in names
    )

    # Constraint sizes are measured rather than inferred: a path constraint
    # carries one value per save point and a periodicity one per closed
    # state, and neither count is recorded anywhere but the output itself.
    outputs = problem.jax_function(defaults)
    constraints = tuple(
        ConstraintDescription(
            name=name,
            kind=kind,
            size=int(numpy.atleast_1d(numpy.asarray(outputs[name])).size),
        )
        for name, kind in problem.constraint_names
    )
    return ProblemDescription(
        design_variables=variables,
        objective=problem.objective_name,
        maximize=maximize,
        constraints=constraints,
    )


__all__ = [
    "ConstraintDescription",
    "DesignVariableDescription",
    "ProblemDescription",
    "describe_problem",
]
