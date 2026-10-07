"""A miniature project quoting a value no stored result holds."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[4] / "docs" / "_ext"))

extensions = ["myst_parser", "tempo_docdata"]
master_doc = "index"
exclude_patterns = ["_build"]
tempo_data_dir = str(Path(__file__).parents[1] / "test-paper" / "_data")
