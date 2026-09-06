"""Fixtures for the Sphinx extension tests."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest


@pytest.fixture
def rootdir() -> Path:
    """Return the directory holding the miniature documentation projects."""
    return Path(__file__).parent / "roots"


@pytest.fixture(autouse=True)
def _restore_logging():
    """Let Sphinx's warnings through for the duration of these tests.

    Most test modules in this suite disable logging at import time, because
    GEMSEO and JAX are noisy. That is a global switch, so by the time these
    tests run in a full suite it has already silenced the warnings they
    assert on, and the tests pass alone and fail together. It is restored
    afterwards so the rest of the suite stays quiet.
    """
    previous = logging.root.manager.disable
    logging.disable(logging.NOTSET)
    yield
    logging.disable(previous)
