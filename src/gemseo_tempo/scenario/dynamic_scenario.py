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
"""Assembly of the three stages into GEMSEO disciplines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from gemseo_jax.jax_discipline import JAXDiscipline

from gemseo_tempo.dynamics.integrators import FORWARD
from gemseo_tempo.dynamics.integrators import REVERSE
from gemseo_tempo.scenario.structured_discipline import StructuredJAXDiscipline
from gemseo_tempo.transcriptions.base import CONSTANT_PREFIX
from gemseo_tempo.transcriptions.single_shooting import SingleShooting

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from gemseo_tempo.dynamics.controls import AbstractControl
    from gemseo_tempo.dynamics.integrators import AbstractIntegrator
    from gemseo_tempo.dynamics.state_space import StateSpace
    from gemseo_tempo.metadata.registry import VariableRegistry
    from gemseo_tempo.outputs.constraints import AbstractConstraint
    from gemseo_tempo.outputs.functionals import AbstractFunctional
    from gemseo_tempo.statics.chain import StaticsChain
    from gemseo_tempo.transcriptions.base import AbstractTranscription
    from gemseo_tempo.transcriptions.base import BoundarySpec


class DynamicScenario:
    """The assembled dynamic optimization problem.

    Two assemblies are supported:

    - ``"monolithic"``: the whole workflow (statics, integration,
      functionals) is one dict-to-dict JAX function wrapped in a single
      reverse-differentiated :class:`~gemseo_jax.jax_discipline.JAXDiscipline`.
    - ``"split"``: statics (forward AD), dynamics, and outputs are three
      JAXDisciplines chained by GEMSEO, with the trajectory passed as
      flat arrays, so GEMSEO handles coupling, caching, and mixed
      differentiation.
    """

    MONOLITHIC = "monolithic"
    SPLIT = "split"

    disciplines: list[JAXDiscipline]
    """The GEMSEO disciplines: 1 (monolithic) or up to 3 (split)."""

    state_space: StateSpace
    """The dynamics model, kept so a solved trajectory can be read back."""

    definition: dict
    """Everything the problem was built from, bar the transcription, so it
    can be built again under another one; see :meth:`retranscribe`."""

    def __init__(
        self,
        state_space: StateSpace,
        integrator: AbstractIntegrator,
        controls: Sequence[AbstractControl],
        functionals: Sequence[AbstractFunctional],
        constraints: Sequence[AbstractConstraint] = (),
        boundary: BoundarySpec | None = None,
        statics: StaticsChain | None = None,
        transcription: AbstractTranscription | None = None,
        assembly: str = MONOLITHIC,
        registry: VariableRegistry | None = None,
        constants: Mapping[str, float] | None = None,
        differentiation_method: str = REVERSE,
        t0: float = 0.0,
        t1: float = 1.0,
    ) -> None:
        """Transcribe the problem and build the disciplines.

        Parameters
        ----------
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
        statics
            The constant-fed chain running ahead of the dynamics, if any.
        transcription
            The transcription; direct single shooting when omitted.
        assembly
            The assembly strategy, ``"monolithic"`` or ``"split"``.
        registry
            The variable metadata, used by the formulation builders for
            design-space bounds.
        constants
            The values of constant inputs, overriding the statics-chain
            defaults; required for dynamics parameters not produced by
            the statics chain.
        differentiation_method
            The JAX differentiation mode of the ODE-bearing disciplines,
            ``"reverse"`` (default) or ``"forward"``. Must be compatible
            with the integrator adjoint: forward mode requires the Direct
            adjoint (see :func:`.check_adjoint_compatibility`). The
            statics discipline of the split assembly always differentiates
            in forward mode, being a cheap algebraic chain with more
            outputs than inputs.
        t0
            The initial time.
        t1
            The final time (ignored when the final time is free).

        Raises
        ------
        ValueError
            If the assembly is unknown, the transcription does not support
            the split assembly, or the differentiation method is
            incompatible with the integrator adjoint.
        """
        if assembly not in (self.MONOLITHIC, self.SPLIT):
            raise ValueError(f"Unknown assembly {assembly!r}.")
        self.definition = {
            "state_space": state_space,
            "integrator": integrator,
            "controls": tuple(controls),
            "functionals": tuple(functionals),
            "constraints": tuple(constraints),
            "boundary": boundary,
            "statics": statics,
            "assembly": assembly,
            "registry": registry,
            "constants": dict(constants or {}),
            "differentiation_method": differentiation_method,
            "t0": t0,
            "t1": t1,
        }
        self.assembly = assembly
        self.registry = registry
        self.state_space = state_space
        self.differentiation_method = differentiation_method
        self.transcription = (
            transcription if transcription is not None else SingleShooting()
        )
        self.statics = statics
        integrator.check_differentiation_method(differentiation_method)
        self.problem = self.transcription.transcribe(
            statics,
            state_space,
            integrator,
            controls,
            functionals,
            constraints,
            boundary,
            constants=constants,
            t0=t0,
            t1=t1,
        )
        if assembly == self.MONOLITHIC:
            self.disciplines = [self._build_monolithic()]
        else:
            self.disciplines = self._build_split()

    def retranscribe(self, transcription, **changes) -> DynamicScenario:
        """Return the same problem under another transcription.

        Parameters
        ----------
        transcription
            The transcription to build the problem with.
        **changes
            Any part of :attr:`definition` to replace, such as the
            boundary or the constants a warm start comes with.

        Returns
        -------
        DynamicScenario
            The new scenario.
        """
        return DynamicScenario(
            **{**self.definition, "transcription": transcription, **changes}
        )

    def _build_monolithic(self) -> JAXDiscipline:
        """Build the single differentiated discipline.

        Returns
        -------
        JAXDiscipline
            The discipline, differentiated in the scenario's mode.
        """
        problem = self.problem
        arguments = {
            "function": problem.jax_function,
            "input_names": list(problem.input_defaults),
            "output_names": problem.output_names,
            "default_inputs": problem.input_defaults,
            "differentiation_method": JAXDiscipline.DifferentiationMethod(
                self.differentiation_method
            ),
            "name": "dynamic_scenario",
        }
        if problem.jacobian is not None:
            return StructuredJAXDiscipline(
                structured_jacobian=problem.jacobian, **arguments
            )
        return JAXDiscipline(**arguments)

    def _build_split(self) -> list[JAXDiscipline]:
        """Build the statics, dynamics, and outputs disciplines.

        Returns
        -------
        list
            The disciplines, in execution order.

        Raises
        ------
        ValueError
            If the transcription does not provide the stage pieces.
        """
        problem = self.problem
        if problem.dynamics is None or problem.outputs is None:
            raise ValueError("The transcription does not support the split assembly.")
        disciplines = []
        if self.statics is not None:
            chain = self.statics.chain
            input_names = list(chain.input_grammar.names)

            def statics_function(data, _chain=chain, _names=input_names):
                outputs = _chain.jax_out_func(
                    {name: data[CONSTANT_PREFIX + name] for name in _names}
                )
                # 1D outputs so GEMSEO can concatenate them for FD checks.
                return {name: jnp.atleast_1d(value) for name, value in outputs.items()}

            disciplines.append(
                JAXDiscipline(
                    function=statics_function,
                    input_names=[CONSTANT_PREFIX + name for name in input_names],
                    output_names=list(chain.output_grammar.names),
                    default_inputs={
                        CONSTANT_PREFIX + name: problem.input_defaults[
                            CONSTANT_PREFIX + name
                        ]
                        for name in input_names
                    },
                    differentiation_method=(
                        JAXDiscipline.DifferentiationMethod(FORWARD)
                    ),
                    name="statics",
                )
            )
        for piece, name, method in (
            (problem.dynamics, "dynamics", self.differentiation_method),
            (problem.outputs, "outputs", self.differentiation_method),
        ):
            disciplines.append(
                JAXDiscipline(
                    function=piece.function,
                    input_names=piece.input_names,
                    output_names=piece.output_names,
                    default_inputs=piece.default_inputs,
                    differentiation_method=(
                        JAXDiscipline.DifferentiationMethod(method)
                    ),
                    name=name,
                )
            )
        return disciplines
