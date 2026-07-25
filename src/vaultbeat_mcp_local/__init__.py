from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _installed_version

__all__ = ["__version__"]

# Read from installed package metadata rather than a literal. The literal was
# "0.1.0" and had never been updated, so `vaultbeat-mcp --version` reported
# 0.1.0 for every release through 0.2.1 — the one command a user runs to answer
# "which version do I actually have" was the one that lied. That matters more
# than usual here: app and MCP versions drift independently by design, so
# diagnosing a missing tool starts with trusting this number.
try:
    __version__ = _installed_version("vaultbeat-mcp")
except PackageNotFoundError:  # running from a source checkout, not installed
    __version__ = "0.0.0+dev"
