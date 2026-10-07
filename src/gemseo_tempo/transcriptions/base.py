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
"""The transcription base classes and grammar conventions.

A transcription decides the finite-dimensional problem the optimizer
sees: design variables, one dict-to-dict JAX function producing every
output, an objective, and named constraints. Grammar prefixes follow
NOADS: ``control.``, ``constant.``, ``initial.``, ``trajectory.``, and
``constraint.``.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from jax import Array

    from gemseo_tempo.dynamics.controls import AbstractControl
    from gemseo_tempo.dynamics.integrators import AbstractIntegrator
    from gemseo_tempo.dynamics.state_space import StateSpace
    from gemseo_tempo.outputs.constraints import AbstractConstraint
    from gemseo_tempo.outputs.functionals import AbstractFunctional
    from gemseo_tempo.statics.chain import StaticsChain

CONTROL_PREFIX = "control."
"""The grammar prefix of control parameter vectors."""

CONSTANT_PREFIX = "constant."
"""The grammar prefix of time-independent inputs."""

INITIAL_PREFIX = "initial."
"""The grammar prefix of free initial-state components."""

STATE_PREFIX = "state."
"""The grammar prefix of state-node design variables (collocation)."""

COSTATE_PREFIX = "costate."
"""The grammar prefix of initial-costate design variables (indirect)."""

MULTIPLIER_PREFIX = "multiplier."
"""The grammar prefix of the multipliers of terminal equalities (indirect)."""

TRAJECTORY_PREFIX = "trajectory."
"""The grammar prefix of timeseries on the save grid."""

CONSTRAINT_PREFIX = "constraint."
"""The grammar prefix of constraint values."""

TIMES = "times"
"""The output name of the physical save times."""

T_FINAL = "t_f"
"""The input name of the free final time."""


@dataclass
class BoundarySpec:
    """Per-state boundary flags.

    A state with ``fix_initial=False`` has its initial value promoted to
    a design variable (defaulting to ``s0``); a state with
    ``fix_final=True`` gets a terminal equality on the target value in
    ``s1``. Missing entries default to fixed initial and free final.
    """

    s0: dict[str, float]
    """The initial state values (defaults for free components)."""

    fix_initial: dict[str, bool] = field(default_factory=dict)
    """Whether each initial value is fixed; missing names are fixed."""

    fix_final: dict[str, bool] = field(default_factory=dict)
    """Whether each final value is constrained; missing names are free."""

    s1: dict[str, float] = field(default_factory=dict)
    """The final target values of the ``fix_final`` states."""

    free_final_time: bool = False
    """Whether the final time is a design variable."""

    t_f: float | None = None
    """The initial guess of the free final time."""

    def __post_init__(self) -> None:
        """Validate the flags.

        Raises
        ------
        ValueError
            If a ``fix_final`` state has no target in ``s1``, or the
            final time is free without an initial guess.
        """
        for name, fixed in self.fix_final.items():
            if fixed and name not in self.s1:
                raise ValueError(
                    f"State {name!r} has fix_final=True but no target value in s1."
                )
        if self.free_final_time and self.t_f is None:
            raise ValueError("free_final_time=True requires an initial guess t_f.")

    def is_initial_fixed(self, name: str) -> bool:
        """Return whether the initial value of a state is fixed.

        Parameters
        ----------
        name
            The state name.

        Returns
        -------
        bool
            Whether the initial value is fixed (the default).
        """
        return self.fix_initial.get(name, True)

    def is_final_fixed(self, name: str) -> bool:
        """Return whether the final value of a state is constrained.

        Parameters
        ----------
        name
            The state name.

        Returns
        -------
        bool
            Whether the final value is constrained (not the default).
        """
        return self.fix_final.get(name, False)


@dataclass
class VariableSpec:
    """The specification of one design variable."""

    name: str
    """The grammar name."""

    size: int
    """The number of components."""

    value: Array
    """The default value."""

    lower_bound: float | None = None
    """The lower bound, if any."""

    upper_bound: float | None = None
    """The upper bound, if any."""


@dataclass
class StagePiece:
    """One stage of the split assembly, as a plain JAX function."""

    function: Callable
    """The dict-to-dict JAX function."""

    input_names: list[str]
    """The input names."""

    output_names: list[str]
    """The output names."""

    default_inputs: dict[str, Array]
    """The default value of each input."""


@dataclass
class StructuredJacobian:
    """A transcription-supplied Jacobian exploiting known sparsity.

    A transcription that knows the structure of some of its outputs can
    compute their derivatives far more cheaply than a dense automatic
    transform. It declares here which outputs and which inputs it covers;
    the discipline falls back to the dense transform for anything else,
    so a transcription only has to describe the part it knows is sparse.
    """

    function: Callable
    """The map from the input dict to ``{output: {input: array}}``."""

    output_names: tuple[str, ...]
    """The outputs whose derivatives this function provides."""

    input_names: tuple[str, ...]
    """The inputs with respect to which it differentiates them."""


@dataclass
class TranscribedProblem:
    """The finite-dimensional problem seen by the optimizer."""

    design_variables: dict[str, VariableSpec]
    """The design variables, keyed by grammar name."""

    jax_function: Callable
    """The monolithic dict-to-dict function producing all outputs."""

    objective_name: str
    """The output name of the objective."""

    constraint_names: list[tuple[str, str]]
    """The constraint output names with their sense, ``"eq"`` or
    ``"ineq"``."""

    input_defaults: dict[str, Array]
    """The default value of every input (design variables and
    constants)."""

    output_names: list[str]
    """All output names of the monolithic function."""

    dynamics: StagePiece | None = None
    """The dynamics stage of the split assembly, when supported."""

    outputs: StagePiece | None = None
    """The outputs stage of the split assembly, when supported."""

    jacobian: StructuredJacobian | None = None
    """A structure-aware Jacobian for part of the outputs, when the
    transcription can beat the dense automatic transform."""

    guess_setter: Callable | None = None
    """A callback set by transcriptions that support warm starts; it
    updates the design-variable defaults from a :class:`.Trajectory`."""

    def set_initial_guess(self, trajectory) -> None:
        """Seed the design-variable defaults from a trajectory.

        Interpolates the states and controls of a :class:`.Trajectory`
        (from a cheap shooting run or a coarse solve) onto the transcription
        grid and writes them into :attr:`input_defaults` and the
        design-variable specs, so the next scenario build starts warm.

        Parameters
        ----------
        trajectory
            The trajectory to interpolate onto the grid.

        Raises
        ------
        NotImplementedError
            If the transcription does not support warm starts (e.g. single
            shooting, whose design variables are control parameters, not
            node values).
        """
        if self.guess_setter is None:
            raise NotImplementedError(
                "This transcription does not support set_initial_guess;"
                " it has no node grid to interpolate onto."
            )
        self.guess_setter(trajectory)


class AbstractTranscription:
    """Base class of the transcriptions."""

    def transcribe(
        self,
        statics: StaticsChain | None,
        state_space: StateSpace,
        integrator: AbstractIntegrator,
        controls: Sequence[AbstractControl],
        functionals: Sequence[AbstractFunctional],
        constraints: Sequence[AbstractConstraint],
        boundary: BoundarySpec,
        constants: Mapping[str, float] | None = None,
        t0: float = 0.0,
        t1: float = 1.0,
    ) -> TranscribedProblem:
        """Build the finite-dimensional problem.

        Parameters
        ----------
        statics
            The constant-fed chain running ahead of the dynamics, if any.
        state_space
            The dynamics model.
        integrator
            The execution model of the dynamics.
        controls
            The control parameterizations, one per state-space control.
        functionals
            The scalar functionals; the first one is the objective.
        constraints
            The trajectory constraints.
        boundary
            The boundary specification.
        constants
            The values of constant inputs, overriding the statics-chain
            defaults; required for dynamics parameters not produced by
            the statics chain.
        t0
            The initial time.
        t1
            The final time (ignored when the final time is free).

        Returns
        -------
        TranscribedProblem
            The problem.

        Raises
        ------
        NotImplementedError
            Always; subclasses implement the transcription.
        """
        raise NotImplementedError


def resolve_parameter_sources(
    state_space: StateSpace,
    statics_output_names: set[str],
    constants: Mapping[str, float],
) -> dict[str, str]:
    """Map each dynamics parameter to the input name that supplies it.

    A parameter is produced either by the statics chain (used under its
    own name) or declared as a constant (used under the ``constant.``
    prefix).

    Parameters
    ----------
    state_space
        The dynamics model.
    statics_output_names
        The names produced by the statics chain.
    constants
        The declared constant values.

    Returns
    -------
    dict
        The input name of each parameter, keyed by parameter name.

    Raises
    ------
    ValueError
        If a parameter has neither source.
    """
    theta_inputs = {}
    for parameter in state_space.parameters:
        if parameter in statics_output_names:
            theta_inputs[parameter] = parameter
        elif parameter in constants:
            theta_inputs[parameter] = CONSTANT_PREFIX + parameter
        else:
            raise ValueError(
                f"Parameter {parameter!r} is neither an output of the"
                " statics chain nor given in constants."
            )
    return theta_inputs
