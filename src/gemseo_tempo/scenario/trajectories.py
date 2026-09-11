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
"""Reading back the trajectory a solved scenario settled on."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from gemseo_tempo.dynamics.integrators import Trajectory
from gemseo_tempo.transcriptions.base import TIMES
from gemseo_tempo.transcriptions.base import TRAJECTORY_PREFIX

if TYPE_CHECKING:
    from gemseo_tempo.scenario.dynamic_scenario import DynamicScenario


def optimal_trajectory(scenario: DynamicScenario, gemseo_scenario) -> Trajectory:
    """Recover the trajectory a solved scenario settled on.

    The transcription publishes every state, control and output on the save
    grid, so the answer is read back from the discipline rather than
    re-simulated, which would reintroduce the integration error the
    transcription just worked to remove.

    Parameters
    ----------
    scenario
        The dynamic scenario that was optimized.
    gemseo_scenario
        The executed GEMSEO scenario.

    Returns
    -------
    Trajectory
        The states, controls and outputs on the save grid.
    """
    state_space = scenario.state_space
    discipline = scenario.disciplines[0]
    optimum = gemseo_scenario.optimization_result.x_opt_as_dict
    discipline.execute(
        {
            name: value
            for name, value in optimum.items()
            if name in discipline.io.input_grammar
        }
    )
    data = discipline.io.data

    states: dict = {}
    controls: dict = {}
    outputs: dict = {}
    for name in discipline.io.output_grammar:
        if not name.startswith(TRAJECTORY_PREFIX):
            continue
        short = name[len(TRAJECTORY_PREFIX) :]
        if short in state_space.states:
            target = states
        elif short in state_space.controls:
            target = controls
        else:
            target = outputs
        target[short] = jnp.asarray(data[name])

    return Trajectory(
        ts=jnp.asarray(data[TIMES]),
        states=states,
        controls=controls,
        outputs=outputs,
    )


__all__ = ["optimal_trajectory"]
