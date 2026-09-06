"""Top-level pytest configuration.

``pytest_plugins`` has to be declared in the root conftest, so the Sphinx
test fixtures that ``tests/docs`` uses are registered here rather than
beside the tests that want them.

Sphinx lives in the ``doc`` extra rather than the ``test`` one, so an
installation with only the test dependencies skips those tests instead of
failing to collect them. The documentation workflow installs both and runs
them.
"""

try:
    import sphinx.testing.fixtures  # noqa: F401
except ImportError:  # pragma: no cover - depends on the extras installed
    pytest_plugins = ()
else:
    pytest_plugins = ("sphinx.testing.fixtures",)
