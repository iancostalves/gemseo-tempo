"""A miniature project exercising the variable grammar."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[4] / "docs" / "_ext"))

extensions = ["myst_parser", "tempo_myst"]
master_doc = "index"
exclude_patterns = ["_build"]
tempo_variable_packages = ["gemseo_tempo.problems"]
tempo_unknown_variable = "warn"
