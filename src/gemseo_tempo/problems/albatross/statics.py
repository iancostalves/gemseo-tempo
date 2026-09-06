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
"""The wing sizing that feeds the albatross dynamics.

The dynamics need three numbers that are not states: how much induced drag
the wing makes, how heavily it is loaded, and how slowly it can fly. None of
them are independent quantities. They all follow from the mass and the wing
plan, which is exactly what a statics chain is for: the design variables go in
one end, and the coefficients the equations of motion actually use come out
the other, differentiated along with everything else.

Five models make the chain.

:class:`AirfoilSection` sets the trade between gliding well and turning hard,
which is the one that decides what kind of bird this is.

:class:`WingGeometry` turns the plan into an aspect ratio and the induced-drag
factor ``k`` of the drag polar ``C_D = C_D0 + k C_L^2``.

:class:`FiniteWing` decides how much of the section's properties the wing
actually keeps, which is where aspect ratio reaches the maximum lift
coefficient rather than only the drag.

:class:`StructuralMass` charges the bird for the wing it is carrying, which is
what stops the co-design formulation from answering with an aspect ratio no
bird has.

:class:`FlightEnvelope` turns the mass and the plan into the wing loading and
the speed below which the wing cannot hold the bird up.
"""

from __future__ import annotations

import jax.numpy as jnp

from gemseo_tempo.metadata.registry import VariableRegistry
from gemseo_tempo.metadata.variable import Variable
from gemseo_tempo.statics.chain import StaticsChain
from gemseo_tempo.statics.model import AutoModel

GRAVITY = 9.81
"""The gravitational acceleration, in m/s^2."""

AIR_DENSITY = 1.225
"""The air density at sea level, in kg/m^3."""

DESIGN_DEFAULTS = {
    "body_mass": 6.4,
    "wing_area": 0.65,
    "wing_span": 3.44,
    "oswald": 0.9,
    "section": 0.5,
}
"""The default wandering albatross, roughly to scale.

A bird of about 8.5 kg all told, 6.4 kg of it body, on 0.65 m^2 of wing
spanning 3.44 m, which is an aspect ratio near 18 and a best glide ratio
near 20. The wing section sits midway along its trade.
"""

SECTION_DRAG = (0.025, 0.016)
"""The zero-lift drag at ``section = 0``, and its rise across the range."""

SECTION_LIFT = (1.2, 0.6)
"""The maximum lift coefficient at ``section = 0``, and its rise."""

AIR_VISCOSITY = 1.81e-5
"""The dynamic viscosity of air, in Pa s."""

REFERENCE_SPEED = 20.0
"""The airspeed the Reynolds number is quoted at, in m/s."""

LIFT_SLOPE_2D = 2.0 * float(jnp.pi)
"""The lift-curve slope of the section, per radian."""

REYNOLDS_DRAG_EXPONENT = 0.4
"""How fast profile drag falls with Reynolds number.

Between the laminar 0.5 and the turbulent 0.2, which is where a wing this
size actually sits.
"""

REYNOLDS_HALF = 1.5e5
"""The Reynolds number at which maximum lift is half of its ceiling."""

REFERENCE_ASPECT_RATIO = (
    DESIGN_DEFAULTS["wing_span"] ** 2 / DESIGN_DEFAULTS["wing_area"]
)
"""The aspect ratio of the nominal bird."""

REFERENCE_LIFT_SLOPE = LIFT_SLOPE_2D / (
    1.0
    + LIFT_SLOPE_2D
    / (float(jnp.pi) * DESIGN_DEFAULTS["oswald"] * REFERENCE_ASPECT_RATIO)
)
"""The lift-curve slope of the nominal wing, per radian."""

REFERENCE_REYNOLDS = (
    AIR_DENSITY
    * REFERENCE_SPEED
    * (DESIGN_DEFAULTS["wing_area"] / DESIGN_DEFAULTS["wing_span"])
    / AIR_VISCOSITY
)
"""The chord Reynolds number of the nominal bird.

Both corrections are normalized here, so the nominal wing keeps its section
properties exactly and every other wing reads as a departure from it.
"""

WING_MASS_COEFFICIENT = 0.03354
"""The constant of the wing mass law.

Chosen so the default wing comes out at about 2.1 kg, a quarter of the
bird. It is the number that decides where the aspect ratio settles, so it
is worth stating plainly rather than burying.
"""

REGISTRY = VariableRegistry(
    [
        Variable(
            name="body_mass",
            full_name="Body mass",
            description="Everything but the wing.",
            unit="kg",
            lower_bound=3.0,
            upper_bound=12.0,
            tags=frozenset({"design"}),
        ),
        Variable(
            name="wing_mass",
            full_name="Wing mass",
            description=(
                "What the wing itself weighs, growing with span and area."
                " Without it nothing charges a design for reaching for"
                " aspect ratio."
            ),
            unit="kg",
            tags=frozenset({"statics"}),
        ),
        Variable(
            name="mass",
            full_name="Total mass",
            description=(
                "Body plus wing. The wing part grows with span and area,"
                " so a wing that is better in cruise is heavier and needs"
                " more lift to carry."
            ),
            unit="kg",
            tags=frozenset({"statics"}),
        ),
        Variable(
            name="wing_area",
            full_name="Wing area",
            description=(
                "The reference area of the wing, one of the two variables"
                " that shape it."
            ),
            unit="m**2",
            lower_bound=0.3,
            upper_bound=1.2,
            tags=frozenset({"design"}),
        ),
        Variable(
            name="wing_span",
            full_name="Wing span",
            description=(
                "Tip to tip. With the area it fixes the aspect ratio, and"
                " through that both the induced drag and the chord the"
                " Reynolds number is measured on."
            ),
            unit="m",
            lower_bound=2.0,
            upper_bound=5.0,
            tags=frozenset({"design"}),
        ),
        Variable(
            name="oswald",
            full_name="Oswald efficiency",
            description="Span efficiency of the lift distribution.",
            lower_bound=0.5,
            upper_bound=1.0,
            tags=frozenset({"design"}),
        ),
        Variable(
            name="section",
            full_name="Wing section",
            description=(
                "Where the aerofoil sits between fast and high-lift: 0 is"
                " thin and clean, 1 is thick and cambered."
            ),
            lower_bound=0.0,
            upper_bound=1.0,
            tags=frozenset({"design"}),
        ),
        Variable(
            name="cd0_section",
            full_name="Section zero-lift drag",
            description="Two-dimensional profile drag, before the wing gets it.",
            tags=frozenset({"statics"}),
        ),
        Variable(
            name="cl_max_section",
            full_name="Section maximum lift",
            description="Two-dimensional stall lift, before the wing gets it.",
            tags=frozenset({"statics"}),
        ),
        Variable(
            name="chord",
            full_name="Mean chord",
            description="Wing area over span; what the aspect ratio costs.",
            unit="m",
            tags=frozenset({"statics"}),
        ),
        Variable(
            name="reynolds",
            full_name="Reynolds number",
            description="Chord Reynolds number at the reference airspeed.",
            tags=frozenset({"statics"}),
        ),
        Variable(
            name="cd0",
            full_name="Zero-lift drag coefficient",
            description=(
                "Drag at zero lift, from the section and corrected for the"
                " Reynolds number the wing actually flies at. A small bird"
                " flies at a low Reynolds number, where this is worse."
            ),
            tags=frozenset({"statics"}),
        ),
        Variable(
            name="cl_max",
            full_name="Maximum lift coefficient",
            description=(
                "The stall limit of the finite wing, which bounds the lift"
                " coefficient in flight. It is what stops a design from"
                " buying glide ratio without paying for manoeuvre: the"
                " tight turns of a soaring cycle need it."
            ),
            tags=frozenset({"statics"}),
        ),
        Variable(
            name="aspect_ratio",
            full_name="Aspect ratio",
            description="Span squared over area.",
            tags=frozenset({"statics"}),
        ),
        Variable(
            name="induced_factor",
            full_name="Induced drag factor",
            description="The k of the drag polar C_D = C_D0 + k C_L^2.",
            tags=frozenset({"statics"}),
        ),
        Variable(
            name="wing_loading",
            full_name="Wing loading",
            description=(
                "Weight per unit wing area, which sets how fast the bird"
                " has to fly to hold itself up."
            ),
            unit="Pa",
            tags=frozenset({"statics"}),
        ),
        Variable(
            name="stall_speed",
            full_name="Stall speed",
            description="Level-flight airspeed at the maximum lift coefficient.",
            unit="m/s",
            tags=frozenset({"statics"}),
        ),
    ]
)
"""The metadata of the design and statics variables."""


class AirfoilSection(AutoModel):
    """The trade a wing section makes between cruising and turning.

    This is the trade that actually shapes a soaring bird, and leaving it
    out makes the problem answer the wrong question. A thin, clean section
    has little profile drag and glides beautifully, but stalls early. A
    thick, cambered one holds far more lift, which is what a hard turn
    needs, and pays for it in drag every second of the flight.

    Without the coupling the optimizer sees only the glide ratio, since
    nothing charges it for the lift it uses; the maximum lift coefficient
    then has no way to influence the design at all. With it, cruising and
    manoeuvring pull in opposite directions, which is the whole difficulty
    of designing for dynamic soaring.
    """

    def _jax_func(self, section=DESIGN_DEFAULTS["section"]):
        """Return the profile drag and the maximum lift of the section.

        These are two-dimensional section properties. What the wing gets to
        keep of them is decided downstream, by :class:`FiniteWing`.
        """
        cd0_section = SECTION_DRAG[0] + SECTION_DRAG[1] * section
        cl_max_section = SECTION_LIFT[0] + SECTION_LIFT[1] * section
        return cd0_section, cl_max_section


class WingGeometry(AutoModel):
    """Aspect ratio and induced drag from the wing plan."""

    def _jax_func(
        self,
        wing_area=DESIGN_DEFAULTS["wing_area"],
        wing_span=DESIGN_DEFAULTS["wing_span"],
        oswald=DESIGN_DEFAULTS["oswald"],
    ):
        """Return the aspect ratio and the induced drag factor.

        A longer, narrower wing of the same area has a higher aspect ratio
        and makes less drag for the same lift, which is the whole reason an
        albatross is shaped the way it is.
        """
        aspect_ratio = wing_span**2 / wing_area
        induced_factor = 1.0 / (jnp.pi * oswald * aspect_ratio)
        return aspect_ratio, induced_factor


class FiniteWing(AutoModel):
    """What a real wing keeps of its section's properties.

    Aspect ratio does not only set induced drag. It also decides how much of
    the section's maximum lift the wing actually gets, by two routes that
    pull opposite ways.

    **The lift-curve slope.** A finite wing flies in its own downwash, so it
    reaches a given lift coefficient at a larger angle than the section
    would: ``a = a0 / (1 + a0 / (pi e AR))``. A longer wing suffers less of
    that, reaches stall at a smaller angle, and holds a little more lift.
    Across the aspect ratios this problem uses it is worth about 4%, which
    is real and small.

    **The Reynolds number, which is the one that bites.** The mean chord is
    ``S / b``, so at a fixed area a longer wing is a narrower one, and a
    narrower wing flies at a lower Reynolds number. At bird scale that sits
    between 1.5e5 and 5e5, exactly where a wing starts losing maximum lift
    and gaining profile drag as the chord shrinks. This is the reason small
    flyers do not have sailplane aspect ratios, and without it the optimizer
    will happily ask for one.

    Both factors are one at the nominal wing, so the section values are what
    the default bird actually flies with and the corrections read as
    departures from it.
    """

    def _jax_func(
        self,
        wing_area=DESIGN_DEFAULTS["wing_area"],
        wing_span=DESIGN_DEFAULTS["wing_span"],
        oswald=DESIGN_DEFAULTS["oswald"],
        aspect_ratio=REFERENCE_ASPECT_RATIO,
        cd0_section=0.033,
        cl_max_section=1.5,
    ):
        """Return the chord, the Reynolds number, and the wing's own polar."""
        chord = wing_area / wing_span
        reynolds = AIR_DENSITY * REFERENCE_SPEED * chord / AIR_VISCOSITY

        slope = LIFT_SLOPE_2D / (1.0 + LIFT_SLOPE_2D / (jnp.pi * oswald * aspect_ratio))
        finite_wing = slope / REFERENCE_LIFT_SLOPE

        low_reynolds_drag = (REFERENCE_REYNOLDS / reynolds) ** REYNOLDS_DRAG_EXPONENT
        low_reynolds_lift = (reynolds / (reynolds + REYNOLDS_HALF)) / (
            REFERENCE_REYNOLDS / (REFERENCE_REYNOLDS + REYNOLDS_HALF)
        )

        cd0 = cd0_section * low_reynolds_drag
        cl_max = cl_max_section * finite_wing * low_reynolds_lift
        return chord, reynolds, cd0, cl_max


class StructuralMass(AutoModel):
    """Total mass from the body and the wing it has to carry.

    Without this the problem has no design trade at all. Induced drag falls
    with aspect ratio and nothing else in the model cares, so the optimizer
    simply drives the span to whatever bound it is given and returns an
    aspect ratio no bird has. A wing that costs mass to lengthen restores
    the trade, and the answer stops being an artefact of the bounds.

    The law is the usual bending-driven scaling,
    ``m_wing ~ AR^1.5 sqrt(S)``. The exponent matters more than it looks.
    A weaker law, mass growing as the square root of aspect ratio, leaves
    the optimizer still sitting on its span bound, because shrinking the
    area pays back almost exactly what lengthening the span costs. The
    bending law is steep enough for the trade to bite.
    """

    def _jax_func(
        self,
        body_mass=DESIGN_DEFAULTS["body_mass"],
        wing_area=DESIGN_DEFAULTS["wing_area"],
        aspect_ratio=18.2,
    ):
        """Return the wing mass and the total mass."""
        wing_mass = WING_MASS_COEFFICIENT * aspect_ratio**1.5 * jnp.sqrt(wing_area)
        mass = body_mass + wing_mass
        return wing_mass, mass


class FlightEnvelope(AutoModel):
    """Wing loading and stall speed from the mass and the wing plan."""

    def _jax_func(
        self,
        mass=8.5,
        wing_area=DESIGN_DEFAULTS["wing_area"],
        cl_max=1.5,
    ):
        """Return the wing loading and the stall speed.

        The stall speed is the level-flight airspeed at which the wing is
        already at its maximum lift coefficient, so flying any slower means
        falling.
        """
        wing_loading = mass * GRAVITY / wing_area
        stall_speed = jnp.sqrt(2.0 * wing_loading / (AIR_DENSITY * cl_max))
        return wing_loading, stall_speed


def create_statics() -> StaticsChain:
    """Build the chain from the wing design to the flight coefficients.

    Returns
    -------
    StaticsChain
        The chain, taking the design constants and producing the aspect
        ratio, induced drag factor, wing loading and stall speed.
    """
    # A model's registry may only describe that model's own variables, so
    # each gets the slice of the shared registry it actually uses.
    section = REGISTRY.subset(["section", "cd0_section", "cl_max_section"])
    finite = REGISTRY.subset(
        [
            "wing_area",
            "wing_span",
            "oswald",
            "aspect_ratio",
            "cd0_section",
            "cl_max_section",
            "chord",
            "reynolds",
            "cd0",
            "cl_max",
        ]
    )
    geometry = REGISTRY.subset(
        ["wing_area", "wing_span", "oswald", "aspect_ratio", "induced_factor"]
    )
    structure = REGISTRY.subset(
        ["body_mass", "wing_area", "aspect_ratio", "wing_mass", "mass"]
    )
    envelope = REGISTRY.subset(
        ["mass", "wing_area", "cl_max", "wing_loading", "stall_speed"]
    )
    return StaticsChain(
        [
            AirfoilSection("section", section),
            WingGeometry("geometry", geometry),
            FiniteWing("finite", finite),
            StructuralMass("structure", structure),
            FlightEnvelope("envelope", envelope),
        ],
        list(DESIGN_DEFAULTS),
    )
