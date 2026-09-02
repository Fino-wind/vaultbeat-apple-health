"""The MCP Registry proves PyPI ownership by finding the server name in the
package README, which PyPI takes from README.md via ``readme = "README.md"``.

The marker used to live only in the public repo's README; the 2026-09-02
export overwrote that file with this one, 0.6.5 shipped without it, and the
Registry refused the version — a PyPI description is immutable, so the fix
cost a version number. This pins the marker to the file that gets exported.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_readme_carries_the_registry_ownership_marker() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert pyproject["project"]["readme"] == "README.md"
    name = pyproject["project"]["name"]
    readme = (ROOT / "README.md").read_text()
    marker = f"<!-- mcp-name: io.github.Fino-wind/{name} -->"
    assert marker in readme, f"README.md must end with {marker!r}"
    # The marker is case-sensitive on the Registry side (GitHub login is
    # `Fino-wind`, not `fino-wind`); guard the exact spelling.
    assert re.search(r"mcp-name: io\.github\.Fino-wind/", readme)
