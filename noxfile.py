"""Nox sessions for gemseo-tempo."""

from __future__ import annotations

import nox

nox.options.sessions = ["lint", "tests"]
nox.options.default_venv_backend = "uv|virtualenv"

PYTHON_VERSIONS = ["3.11", "3.12", "3.13"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the test suite with coverage."""
    session.install("-e", ".[test]")
    # pytest-cov breaks the jaxlib MLIR import; run coverage around pytest.
    session.run("coverage", "run", "-m", "pytest", *session.posargs)
    session.run("coverage", "report", "--show-missing")


@nox.session
def lint(session: nox.Session) -> None:
    """Run pre-commit on all files."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", "--show-diff-on-failure")


@nox.session
def docs(session: nox.Session) -> None:
    """Build the documentation, treating warnings as errors."""
    session.install("-e", ".[doc]")
    session.run(
        "sphinx-build", "-b", "html", "-W", "--keep-going", "docs", "docs/_build/html"
    )
