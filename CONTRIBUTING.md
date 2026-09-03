# Contributing to gemseo-tempo

## License and file headers

gemseo-tempo is licensed under the GNU Lesser General Public License v3
(see `LICENSE.txt`). Every source file starts with the header below:

```python
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
```

Code ported from [NOADS](https://github.com/iancostalves/noads) (LGPL-3.0)
keeps its original copyright lines, with the gemseo-tempo header appended
below them. No code or re-implemented equations from AeroMAPS (GPL-3.0) may
enter this repository, and no code may be copied from any other
incompatibly licensed project.

## Development setup

```bash
pip install -e ".[test,doc]"
pre-commit install
```

## Workflow

- Lint and format: `ruff check .` and `ruff format .` (or `nox -s lint`).
- Tests: `pytest` (or `nox -s tests`), targeting Python 3.11 to 3.13.
- Docs: `nox -s docs` builds with `-W` (warnings are errors).
- Docstrings are numpy-style.
- **Documentation ships with the change.** A commit that adds or alters
  public behaviour updates `docs/` in the same commit: a new module goes in
  `docs/reference/index.md`, and new behaviour is explained on the guide page
  it belongs to. `tests/test_docs.py` enforces the mechanical half of this,
  failing when a module is undocumented, when the reference points at
  something that no longer exists, when a page is in no toctree, or when a
  number quoted in the docs has drifted from what the code computes.
- Preview the result with `nox -s docs-serve`, which rebuilds on save.
