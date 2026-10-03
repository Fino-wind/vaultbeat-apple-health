"""Where things are in the Vaultbeat iOS app — the one place this package says so.

Every hint, error and CLI line that sends a person (or their agent) to a row in
the app reads its path from here. Rows that lived in Settings in app 1.2.8 moved to
a new MCP tab in 1.2.9 (GitHub #22), and 39 sentences across the app, this package
and the website kept pointing at the old place (GitHub #41); a path written out
in fourteen places drifts fourteen ways.

Each path names BOTH layouts because this package does not know which app
version the person has: 1.2.8 and 1.2.9 are installed side by side for as long
as people take to update. Drop the parenthesised legacy half once 1.2.8 is gone
from the field — `scripts/ci/check_stale_ui_paths.py` only accepts the old
section name on a line that also says "1.2.8".
"""

from __future__ import annotations

_LEGACY = "app 1.2.8 and earlier: Settings → Data & AI"

CONNECT_SERVER = f"the MCP tab → Connect an AI server ({_LEGACY})"
AUTHORIZED_SERVERS = f"the MCP tab → Authorized AI Servers ({_LEGACY})"
RESYNC = f"the MCP tab → 'Re-sync all health data to AI' ({_LEGACY})"
# The one row that did not move to the MCP tab: it repairs HealthKit
# permissions, which every user needs with or without an AI.
HEALTH_ACCESS = f"Settings → 'Health data access' ({_LEGACY} → 'Apple Health access')"
