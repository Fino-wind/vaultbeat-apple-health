# Vaultbeat Apple Health MCP Server

**Your Apple Health data — sleep stages, cycle, HRV, resting heart rate, workouts, weight, VO₂ max, meals, lifts, notes, symptoms — readable and writable by your own AI agent (Claude Code, Claude Desktop, Codex, Hermes, OpenClaw, anything that speaks MCP), end-to-end encrypted so that only your machine ever sees plaintext.**

The [Vaultbeat](https://vaultbeat.app) iPhone app reads HealthKit and encrypts every record on the phone. This package is the local server your agent talks to: it downloads the ciphertext, decrypts it on your machine, and serves it over MCP.

```
iPhone (HealthKit → Vaultbeat app, encrypts) ──► cloud (ciphertext only) ──► this server (decrypts locally) ──MCP──► your agent
```

The cloud stores ciphertext it cannot read. The key that opens it is generated on your machine when you pair, and never leaves it.

## Requirements

- **The Vaultbeat iOS app**, 1.2.3 or newer, signed in — [App Store](https://apps.apple.com/app/id6759241985)
- **[uv](https://docs.astral.sh/uv/)** (for `uvx`). The server needs Python 3.11+, and `uvx` downloads a suitable interpreter by itself if your system Python is older.
- **Vaultbeat Pro for agent access.** Pairing a machine starts a 3-day trial of the AI interface; after that, reads and writes need Pro, bought in the iOS app (Settings → Membership). When access lapses nothing is deleted, and buying Pro resumes this machine without re-pairing.

How much history this server can read is decided by what the app uploads. With Pro, that is your whole history. A trial uploads your whole history on app 1.2.9 and later, and your last 7 days on 1.2.8 and earlier. Without either, app 1.2.8 and earlier upload your last 7 days and app 1.2.9 and later upload nothing. So a short history is usually a plan boundary, not a sync delay — `vaultbeat_doctor` tells the causes apart.

## Quick start

**1. Pair this machine with your iPhone.**

```bash
uvx vaultbeat-apple-health@latest bind
```

It prints a QR code and waits. In the Vaultbeat app open **MCP tab → Connect an AI server** (app 1.2.8 and earlier: Settings → Data & AI) and scan it. The command waits 5 minutes (`--timeout` to change); once scanned there are 10 minutes to finish.

**2. Add the server to your MCP client.**

Claude Code:

```bash
claude mcp add vaultbeat-health -- uvx vaultbeat-apple-health@latest serve --transport stdio
```

Claude Desktop — add to `claude_desktop_config.json` (macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`) and restart it:

```json
{
  "mcpServers": {
    "vaultbeat-health": {
      "command": "uvx",
      "args": ["vaultbeat-apple-health@latest", "serve", "--transport", "stdio"]
    }
  }
}
```

Claude Desktop does not inherit your shell's `PATH`. If it reports that `uvx` cannot be found, put the absolute path from `which uvx` in `"command"`.

Any other MCP client: run `uvx vaultbeat-apple-health@latest serve --transport stdio` as a stdio server. The `@latest` matters — without it `uvx` keeps running whichever version it cached first.

**3. Check it works.** Ask your agent to call `vaultbeat_doctor`, or run:

```bash
uvx vaultbeat-apple-health@latest doctor
```

It checks the config, the private key, cloud reachability, the pairing and a real fetch-and-decrypt round trip, and lists which data types have nothing to read yet. Right after pairing the phone is still sealing your history for the new machine, so a few kinds may be empty for a while; **MCP tab → Re-sync all health data to AI** in the app (app 1.2.8 and earlier: Settings → Data & AI) speeds it up.

### Try it without an iPhone

```bash
uvx vaultbeat-apple-health@latest --demo doctor
uvx vaultbeat-apple-health@latest --demo serve --transport stdio   # wire this into a client
```

`--demo` is a global flag — it goes **before** the subcommand — and serves a fixed synthetic dataset: no pairing, no cloud, no key. Every result says `demo_mode: true` and carries a `[SYNTHETIC DEMO DATA]` banner, and the `log_*` write tools refuse.

## MCP tools

26 tools. **Reads return your own data** — the account that paired this machine. Tools marked *partner* also take `partner=true` to read what your partner chose to share with you in their app — sleep, water and weight, and cycle, symptoms and notes only if they turned that on. The two people are never mixed in one result. Every read tool takes `fresh=true` to skip the local cache.

**Diagnostics**

| Tool | What it does |
| --- | --- |
| `vaultbeat_doctor` | Diagnose this install end to end, report which data types have no data and the likely reasons, and show the pairing state. The tool to call before telling anyone their data is missing. |

**Health records**

| Tool | What it returns |
| --- | --- |
| `get_sleep_nights` *partner* | Every night as one compact row: bed and wake time, asleep, deep / REM / core / awake, awakenings, longest unbroken sleep, heart and breathing rate while asleep, naps. A year fits in one call; `since="YYYY-MM-DD"` for a calendar window. |
| `get_sleep_detail` *partner* | One or two nights in depth: stage intervals with per-stage heart and breathing rate. |
| `get_menstrual_cycle` *partner* | Cycle samples and a next-period prediction. Sensitive. |
| `get_symptoms` *partner* | Symptoms from Apple Health, and beside them the episodes a person reported (type, severity, onset and end, place, suspected triggers). Sensitive. |
| `get_notes` *partner* | Free-text notes on days, with who wrote them. Sensitive. |
| `get_strength_log` | Strength sessions: exercises, sets × reps, volume per session. |
| `get_food_log` | Meals and items per day, with optional kcal / protein / fat / carbs. The newest 14 days by default, or a `since` / `until` window. |
| `get_workouts` | Workouts: type, duration, calories, distance. |
| `get_user_profile` | Sex, age, date of birth and height. Needs Vaultbeat for iOS 1.2.9 or later. |

**Daily numbers and analysis**

| Tool | What it returns |
| --- | --- |
| `get_metric` *partner* | Per-day values for one series, several, or all of them: sleep duration, timing and stages, resting HR, HRV, wrist temperature, VO₂ max, weight and body composition, water, steps, active / basal / total energy, exercise and stand minutes, distance, mindfulness. `aggregation` (`avg` / `sum` / `min` / `max` / `latest`) and `granularity` (`day` / `week` / `month` / `weekday`) are computed server-side; `since` / `until` for a calendar window. Today on an accruing series is marked partial and kept out of aggregates. |
| `get_intraday` | Samples within a day, for HRV: one row per hour that has samples (`granularity="hourly"`) or one per sample (`"raw"`). About 8–14 rows a day; `limit` counts rows. |
| `list_metric_series` *partner* | Every series name the tools above and below accept, with its unit and how much data backs it. |
| `get_metric_trend` *partner* | Least-squares slope, endpoints, mean / median / min / max. |
| `compare_metric_periods` *partner* | The newest N days against the N before them, or two named windows. |
| `correlate_metric_series` *partner* | Pearson r between two series over the days that have both; `lag_days` pairs a day with a later one. |

The analysis tools return numbers only — no scores, grades or verdicts — and refuse rather than fit a line to two points.

**Writes** — all into the account that paired this machine, and nowhere else.

| Tool | What it does |
| --- | --- |
| `log_food_append` / `log_strength_append` / `log_note_append` | Add to a day. Cannot delete anything — the safe default when a day may already have entries. |
| `log_food_entry` / `log_strength_entry` / `log_note` | **Replace that whole day** with what is passed, and report exactly what was removed. |
| `log_weight_entry` | Record a weigh-in, keeping that day's body composition. |
| `log_symptom` | Record one symptom the person reports, as structured fields a later read can match against sleep, food and heart rate. |
| `update_symptom` / `delete_symptom` | Change a reported symptom (usually its end time), or erase one logged by mistake. |

`log_note` and `log_note_append` take `partner=true` to record a note *about* your partner. It stays in your own account, is never sent to them, and is kept apart from your own notes.

### Reading the results

- **Every read carries a `coverage` block.** `days_covered` is how many distinct days the answer rests on — quote it beside any average or trend. `more_available: true` means older records exist and your `limit` stopped short of them (`oldest_available` says how far back they go); it is never a sign of missing data.
- **Results are bounded.** A result larger than about 60 KB is not sent; the tool returns `result_too_large` with a suggestion for narrowing the request (a smaller `limit`, a `since` date, one series instead of all). `get_metric` and `get_intraday` return their rows as a compact table (`columns` + `rows`).
- **Unknown arguments are rejected.** A misspelled or retired parameter is an error, never silently ignored.

## MCP prompts

The server also serves eight prompts over `prompts/list` — `daily_brief`, `sleep_review`, `energy_balance`, `training_block_review`, `cycle_aware_read`, `partner_check_in`, `log_from_conversation` and `why_is_this_empty`. Each names the tools to call and carries the same two rules: say what the data covers before concluding, and treat an empty result as something to explain (it has several possible causes), never as a zero. Every argument is optional.

## Privacy and security

- **Decryption happens only on this machine.** The cloud holds ciphertext and per-recipient wrapped keys; the private key that opens them is generated here when you pair.
- **Health data leaves this package only through MCP.** The command-line subcommands pair a machine, report on that pairing and start the server; none of them prints health data.
- **Sensitive categories.** Your cycle and the symptoms Apple Health records reach this server only after you turn them on, per category, in the iOS app — they are off by default. Your notes and the symptoms you report yourself (in the app, or through `log_symptom`) are always readable by your own agent. None of these reaches your partner's agent unless you turn on sharing it with them.
- **Unpair at any time** from the app's MCP tab (app 1.2.8 and earlier: Settings → Data & AI); the machine can then decrypt nothing new.
- Tool results never contain the private key or the server token.

### Where the private key lives

Not in `config.json`. It is looked for in this order:

1. the `VAULTBEAT_PRIVATE_KEY` environment variable, if set (read, never written back) — for operators injecting it from a secret store;
2. the system keyring — the normal case on a desktop;
3. `~/.tether/mcp-local/identity.key`, mode `0600` — the automatic fallback on a machine with no keyring.

`~/.tether/mcp-local/config.json` (mode `0600`) holds the cloud-issued server token and your **public** key. `.tether` is the app's former name and the path is kept on purpose: existing installs find their key through it.

> **Never delete `config.json` to "start clean".** The private key is not in it, so deleting does not clear a bad key — it mints a new identity, and every record already encrypted for the old one becomes permanently unreadable. To repair a pairing, run `bind` again: it re-pairs this machine in place and keeps what is already encrypted for it.

**Headless servers.** If the keyring is unreachable, let the `identity.key` fallback handle it — it engages by itself. Do not set `PYTHON_KEYRING_BACKEND` to the null backend: it accepts writes and stores nothing, so it only hides a keyring you might be able to reach. If a D-Bus session exists but this process cannot see it (common when the server is started by an agent framework, systemd or cron), pass `DBUS_SESSION_BUS_ADDRESS` through explicitly — resolve it first (`echo "unix:path=/run/user/$(id -u)/bus"`) and put the literal result in the client's `env` block, since that block is JSON, not a shell.

## Telemetry

The server sends usage events to PostHog (EU): which tool was called, whether it succeeded,
a duration range, and the AI client and package versions — tied to the account this machine is
paired to. Tool arguments, results and health data are never sent. Set `VAULTBEAT_TELEMETRY=0`
(or `DO_NOT_TRACK=1`) to turn it off. Details: [vaultbeat.app/privacy](https://vaultbeat.app/privacy).

## Troubleshooting

**Start with `doctor`.** `uvx vaultbeat-apple-health@latest doctor` prints an `[OK]` / `[FAIL]` list with a fix for the first thing that is broken; `doctor --json` is the same report for an agent. Exit code 0 means healthy.

**A read comes back short or empty.** Read `coverage.more_available` first: `true` means your `limit` was the boundary — ask for more. `false` with a short history is the plan boundary described under Requirements, or a pairing that is still filling in. `vaultbeat_doctor` lists the kinds with no data and the possible reasons for each.

**The QR code looks wrong** (mojibake or uneven rows — usually Windows with a non-UTF-8 code page). The pairing is still waiting, so do not re-run `bind`: that replaces the code on screen. Instead render the JSON payload printed just above the code as an image (`qrencode -o pair.png '<payload>'`), send it to your phone and use **import from Photos** in the app's scanner; or run `chcp 65001` in a fresh terminal; or use `bind --no-qr` for the payload as text.

**Reads fail with an access message.** The trial or Pro membership has lapsed. Nothing has been deleted; buying Pro in the iOS app (Settings → Membership) resumes this machine without re-pairing.

## HTTP transport

For clients that connect over a network instead of launching a subprocess:

```bash
vaultbeat-apple-health serve --generate-token            # mint and store a bearer token, print client config
vaultbeat-apple-health serve --transport http            # 127.0.0.1:8000/mcp, bearer token required
```

The token is read from `VAULTBEAT_MCP_HTTP_TOKEN` (preferred, keeps it out of shell history) or the stored config, and clients send it as `Authorization: Bearer <token>`. `--show-token` prints the stored one. `--no-token` serves loopback without auth.

Binding beyond loopback (`--host 0.0.0.0` for a LAN or VPS) fails closed: it needs both a token and `--allow-remote`. The token crosses the wire in clear text, so put TLS in front of it (Caddy, nginx, Cloudflare). Other flags: `--sse-response` for SSE-style responses, `--stateful-http` for clients that need sessions.

Example `mcp.json` (VS Code / Cursor style):

```json
{
  "servers": {
    "vaultbeat-health": {
      "type": "http",
      "url": "http://127.0.0.1:8000/mcp",
      "headers": { "Authorization": "Bearer <token>" }
    }
  }
}
```

A client that only speaks stdio can reach the HTTP server through [`mcp-remote`](https://github.com/geelen/mcp-remote): `npx -y mcp-remote http://127.0.0.1:8000/mcp --header "Authorization: Bearer <token>"`.

## Reporting issues

**This repo takes issues for both the MCP server and the Vaultbeat iPhone app** —
install failures, tool errors, binding that never completes, `doctor` reporting
something wrong, and app problems too (UI, subscriptions, HealthKit permission
prompts, sync not showing up on the phone).
[Open an issue](https://github.com/Fino-wind/vaultbeat-apple-health/issues/new/choose)
and pick the template that fits. Questions and ideas can go in
[Discussions](https://github.com/Fino-wind/vaultbeat-apple-health/discussions).

This is a public repo: never paste health data, pairing codes or account details.

## Development

The command line has six subcommands, and none of them reads health data:

```bash
vaultbeat-apple-health bind      # pair this machine with the iOS app (QR)
vaultbeat-apple-health status    # local pairing state
vaultbeat-apple-health doctor    # self-diagnosis
vaultbeat-apple-health init      # generate a keypair and config without pairing
vaultbeat-apple-health poll      # poll once for a pending pairing
vaultbeat-apple-health serve     # run the MCP server (stdio or http)
```

Decrypted records are cached per data type under `~/.tether/mcp-local/cache/` (files `0600`, directory `0700`) for 600 seconds; `VAULTBEAT_MCP_CACHE_TTL` overrides it (`0` disables). After the first read, a refresh downloads only the records that changed. Pairing to a different server identity clears the cache.

Run the checks from a clone:

```bash
uv sync --frozen --all-extras
uv run --frozen pytest -q tests
uv run --frozen ruff check src tests
uv run --frozen mypy src
```

`vaultbeat-mcp` and `vaultbeat-mcp-local` remain as console-script aliases for installs from before the package was renamed `vaultbeat-apple-health` (0.6.2).

<!-- mcp-name: io.github.Fino-wind/vaultbeat-apple-health -->
