from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _installed_version

__all__ = ["__distribution__", "__version__"]

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
#
# `__distribution__` records WHICH of the two names won, which the loop already
# knew and used to throw away. It is the only way this process can tell that it
# was installed under the retired name: the version number alone cannot, since
# both packages share a version series. That distinction is what the User-Agent
# carries to the server, and it is the question behind "should we publish a
# deprecation build of the old package" — a pre-rename install's own `doctor`
# reports it up to date, because the copy of the version check frozen inside it
# queries the old package's PyPI page, where its version IS the newest.
# None = neither distribution is installed, i.e. a source checkout.
__version__ = "0.0.0+dev"  # running from a source checkout, not installed
__distribution__: str | None = None
for _distribution in ("vaultbeat-apple-health", "vaultbeat-mcp"):
    try:
        __version__ = _installed_version(_distribution)
        __distribution__ = _distribution
        break
    except PackageNotFoundError:
        continue
