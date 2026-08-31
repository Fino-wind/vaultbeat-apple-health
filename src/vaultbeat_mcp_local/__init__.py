from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _installed_version

__all__ = ["__version__"]

# Read from installed package metadata rather than a literal. The literal was
# "0.1.0" and had never been updated, so `vaultbeat-mcp --version` reported
# 0.1.0 for every release through 0.2.1 — the one command a user runs to answer
# "which version do I actually have" was the one that lied. That matters more
# than usual here: app and MCP versions drift independently by design, so
# diagnosing a missing tool starts with trusting this number.
# Two distribution names on purpose: the package was renamed to
# vaultbeat-apple-health in 0.6.2, and vaultbeat-mcp installs stay in the wild.
# Asking only for the new name would report 0.0.0+dev to every pre-rename user;
# asking only for the old one is exactly the bug that shipped in 0.6.1 under the
# new name. Try both before giving up (2026-08-31).
__version__ = "0.0.0+dev"  # running from a source checkout, not installed
for _distribution in ("vaultbeat-apple-health", "vaultbeat-mcp"):
    try:
        __version__ = _installed_version(_distribution)
        break
    except PackageNotFoundError:
        continue
