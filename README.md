# Tether MCP Server

**Let your own AI agent read your health data — without the cloud ever seeing it.**

This is the official local [MCP](https://modelcontextprotocol.io) server for [Tether — AI Health Sync](https://apps.apple.com/us/app/tether-ai-health-sync/id6759241985), the iOS app that syncs Apple Health data (sleep, heart rate, menstrual cycle, weight, water, symptoms) between partners and to your own AI — end-to-end encrypted.

Tether embeds no AI and runs no model on your phone. Intelligence lives where you control it: Claude Code, Claude Desktop, or any MCP-capable agent running on your own machine. This server is the bridge — it holds a private key that never leaves your computer, pulls ciphertext from the cloud, and decrypts **only locally**.

```
iPhone (Apple Health) ──E2EE──▶ cloud (ciphertext only) ──E2EE──▶ this server (your machine) ──▶ your AI agent
```

## Requirements

- [Tether — AI Health Sync](https://apps.apple.com/us/app/tether-ai-health-sync/id6759241985) on iOS, with a **Pro** subscription (the AI-agent interface is the Pro tier)
- Python 3.11+ on the machine where your agent runs (macOS / Linux / Windows)

## Quick start

### 1. Install

With [uv](https://docs.astral.sh/uv/) (recommended — no clone needed):

```bash
uvx --from 'git+https://github.com/Fino-wind/tether-mcp' tether-mcp status
```

Or with pip:

```bash
pip install 'tether-mcp[qr] @ git+https://github.com/Fino-wind/tether-mcp'
```

### 2. Bind your phone

```bash
tether-mcp bind
```

This generates a keypair on your machine and prints a QR code. In the Tether iOS app, open **Settings → Data & AI → MCP Server** and scan it (or import a QR screenshot from Photos). The app authorizes this machine and starts sealing your health envelopes to its public key. The private key stays in `~/.tether/mcp-local/` (owner-only `0600` permissions, OS keychain where available) — it is never uploaded anywhere.

### 3. Connect your agent

**Claude Code** (one line):

```bash
claude mcp add tether-health -- uvx --from 'git+https://github.com/Fino-wind/tether-mcp' tether-mcp serve --transport stdio
```

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "tether-health": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/Fino-wind/tether-mcp", "tether-mcp", "serve", "--transport", "stdio"]
    }
  }
}
```

Any other MCP client: run `tether-mcp serve --transport stdio`, or `serve --transport http` for a loopback streamable-HTTP endpoint with bearer-token auth.

> **Claude Desktop note**: it does not inherit your shell `PATH`. If `uvx` isn't found, use the absolute path (`which uvx`) as `command`.

Debugging: `npx @modelcontextprotocol/inspector uvx --from 'git+https://github.com/Fino-wind/tether-mcp' tether-mcp serve --transport stdio`

Then just ask your agent: *“How did we sleep last night?”*

## MCP tools (16)

| Tool | Returns |
|---|---|
| `tether_status` | Local binding state (never exposes keys or tokens) |
| `tether_start_binding` | A fresh QR binding payload for the iOS app to scan |
| `tether_poll_binding` | One poll for the iOS authorization to complete binding |
| `tether_sync_sleep` | Recent sleep sessions incl. heart-rate samples, per-day primary-session selection matching the iOS app |
| `get_sleep_detail` | Per-night heart-rate + respiratory-rate + sleep-stage timeline |
| `get_water_intake` | Daily water intake + computed daily average |
| `get_weight_trend` | Daily weights + latest/avg/min/max + weekly trend rate |
| `get_menstrual_cycle` | Cycle samples + next-period prediction *(sensitive — explicit iOS opt-in required)* |
| `get_symptoms` | HealthKit symptom days grouped by data owner *(sensitive)* |
| `get_notes` | Free-text day annotations with their writer *(sensitive)* |
| `get_activity` | Daily activity rings: steps / energy / exercise minutes / stand hours / distance |
| `get_resting_hr` | Resting heart-rate records + window mean |
| `get_workouts` | Workout records: type / duration / calories / distance |
| `get_mindfulness` | Mindful sessions and minutes per day |
| `get_hrv` | Heart-rate variability (SDNN) records + window mean |
| `get_wrist_temp` | Sleeping wrist-temperature baseline deviation |

Every data tool accepts `owner` (a user-ID prefix) to filter to one person — the server may hold both your and your partner's shared records, and omitting `owner` mixes them into one pool, so per-person questions should always pass it. (Earlier releases named some tools `get_partner_*`; they were renamed in the 16-tool release since they return whichever owners' records this server holds, not specifically the partner's.)

Reads are cache-first: decrypted records are cached locally (owner-only files, 600 s TTL, `TETHER_MCP_CACHE_TTL` to override) so repeat queries answer in ~0.2 s with zero network; pass `fresh=true` to force a cloud round trip. The same service layer backs a full CLI (`tether-mcp sleep / water / weight / …` — every data subcommand takes `--owner` too) if you prefer scripts over MCP.

## Privacy & security model

- **End-to-end encryption**: Curve25519 ECDH + HKDF-SHA256 + AES-GCM. Every health record is sealed on-device to each authorized recipient's public key (your partner, and this server once bound).
- **The cloud only ever holds ciphertext.** Tether's backend cannot read your health data — architecturally, not just by policy.
- **Decryption happens here**, on hardware you own. The private key and server token are never exposed through any tool result.
- **Sensitive kinds** (menstrual cycle, symptoms, notes) reach this server only if explicitly opted in inside the iOS app, and are never re-exported by the server.
- HTTP transport binds to loopback by default and requires a bearer token; binding a non-loopback address fails closed unless explicitly allowed — front it with TLS if you must expose it.

You can audit all of the above in this repository — that is why it is open source.

## Development

```bash
pip install -e '.[dev,qr]'
pytest
```

## License

[MIT](LICENSE). The Tether iOS app and cloud service are separate proprietary components; this repository covers the local MCP server only.

---

*Website: [tetherme.app](https://tetherme.app) · App Store: [Tether — AI Health Sync](https://apps.apple.com/us/app/tether-ai-health-sync/id6759241985) · Bugs & feedback: [tether-community](https://github.com/Fino-wind/tether-community/issues)*
