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
"""Smoke tests: the package imports and enables x64 mode."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import gemseo_tempo


def test_version():
    assert gemseo_tempo.__version__


def test_x64_enabled():
    assert jax.config.jax_enable_x64
    assert jnp.asarray(1.0).dtype == jnp.float64
