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
"""Tests of gemseo_tempo.metadata.variable and .registry."""

from __future__ import annotations

import pytest

from gemseo_tempo.metadata import Variable
from gemseo_tempo.metadata import VariableRegistry

C_L = Variable(
    name="c_l",
    full_name="Lift coefficient",
    description="Sectional lift coefficient.",
    tags=frozenset({"aero", "state"}),
    distribution={"name": "OTNormalDistribution", "mu": 0.03, "sigma": 0.005},
)
ALTITUDE = Variable(
    name="h",
    full_name="Altitude",
    unit="m",
    lower_bound=0.0,
    upper_bound=12000.0,
    tags=frozenset({"state"}),
)
SPEED = Variable(name="v", full_name="Speed", unit="m/s", tags=frozenset({"control"}))


@pytest.fixture
def registry():
    return VariableRegistry([C_L, ALTITUDE, SPEED])


def test_mapping_interface(registry):
    assert len(registry) == 3
    assert set(registry) == {"c_l", "h", "v"}
    assert registry["h"] is ALTITUDE
    assert "c_l" in registry
    assert "unknown" not in registry
    assert "c_l" in repr(registry)


def test_duplicate_registration_raises(registry):
    with pytest.raises(ValueError, match="already registered"):
        registry.register(Variable(name="c_l", full_name="Other"))


def test_overwrite(registry):
    replacement = Variable(name="c_l", full_name="Replacement")
    registry.register(replacement, overwrite=True)
    assert registry["c_l"] is replacement
    assert len(registry) == 3


def test_unknown_unit_raises(registry):
    with pytest.raises(ValueError, match="unknown unit"):
        registry.register(Variable(name="x", full_name="X", unit="notaunit"))


def test_yaml_round_trip(registry, tmp_path):
    path = tmp_path / "variables.yaml"
    registry.to_yaml(path)
    loaded = VariableRegistry.from_yaml(path)
    assert dict(loaded) == dict(registry)
    # The distribution field survives the round trip.
    assert loaded["c_l"].distribution == {
        "name": "OTNormalDistribution",
        "mu": 0.03,
        "sigma": 0.005,
    }
    assert loaded["h"].lower_bound == 0.0
    assert loaded["h"].tags == frozenset({"state"})


def test_subset(registry):
    sub = registry.subset(["h", "v"])
    assert set(sub) == {"h", "v"}
    with pytest.raises(KeyError):
        registry.subset(["h", "unknown"])


def test_filter(registry):
    assert set(registry.filter("state")) == {"c_l", "h"}
    assert set(registry.filter("control")) == {"v"}
    assert len(registry.filter("nonexistent")) == 0


def test_distribution_retrieval(registry):
    assert registry["c_l"].distribution["name"] == "OTNormalDistribution"
    assert registry["h"].distribution is None


def test_records_round_trip():
    record = C_L.to_record()
    assert record["tags"] == ["aero", "state"]
    assert Variable.from_record(record) == C_L
