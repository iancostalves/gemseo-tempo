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
"""Shared toy problems for the scenario and transcription tests."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from gemseo_tempo.dynamics import DiffraxIntegrator
from gemseo_tempo.dynamics import PiecewiseLinearControl
from gemseo_tempo.dynamics import StateSpace
from gemseo_tempo.outputs import FinalRate
from gemseo_tempo.outputs import Lagrange
from gemseo_tempo.outputs import Mayer
from gemseo_tempo.outputs import PathConstraint
from gemseo_tempo.outputs import TerminalConstraint
from gemseo_tempo.statics import JAXModel
from gemseo_tempo.statics import StaticsChain
from gemseo_tempo.transcriptions import BoundarySpec

KNOTS = jnp.linspace(0.0, 1.0, 5)


def make_lqr_space():
    """x' = -c x + u with running cost output y = x**2 + u**2."""
    return StateSpace(
        states=["x"],
        controls=["u"],
        parameters=["c"],
        outputs=["y"],
        rhs=lambda t, s, u, theta: {"x": -theta["c"] * s["x"] + u["u"]},
        output=lambda t, s, u, theta: {"y": s["x"] ** 2 + u["u"] ** 2},
    )


def make_gain_statics():
    """Statics chain computing c = 2 * k from the constant k."""
    gain = JAXModel(
        function=lambda inputs: {"c": 2.0 * inputs["k"]},
        input_names=["k"],
        output_names=["c"],
        default_inputs={"k": 0.5},
        name="gain",
    )
    return StaticsChain([gain], ["k"])


@pytest.fixture
def lqr_pieces():
    """The LQR-like toy problem shared by the assembly tests."""
    return {
        "state_space": make_lqr_space(),
        "statics": make_gain_statics(),
        "integrator": DiffraxIntegrator(rtol=1e-12, atol=1e-12, n_save=21),
        "controls": (
            PiecewiseLinearControl("u", KNOTS, default_params=0.5 * jnp.ones(5)),
        ),
        "functionals": (
            Lagrange(
                "cost",
                lambda t, s, u, theta: s["x"] ** 2 + u["u"] ** 2,
            ),
            Mayer("terminal", lambda t1, s1, theta: s1["x"] ** 2),
            FinalRate("final_rate.y", "y"),
        ),
        "constraints": (
            PathConstraint("u_max", lambda t, s, u, theta: u["u"] - 1.0),
            TerminalConstraint(
                "arrival",
                lambda t1, s1, theta: s1["x"] - 0.1,
                kind="ineq",
            ),
        ),
        "boundary": BoundarySpec(s0={"x": 1.0}),
        "t0": 0.0,
        "t1": 1.0,
    }
