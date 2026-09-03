# Vaultbeat Local MCP Server

**Your Apple Health data — sleep stages, cycle, HRV, resting heart rate, workouts, weight, VO₂ max, meals, lifts, notes — readable and writable by your own AI agent (Claude Code, Hermes, OpenClaw, anything MCP), end-to-end encrypted so that only your machine ever sees plaintext.** The [Vaultbeat](https://vaultbeat.app) iPhone app captures from HealthKit; this package is the local server that decrypts for the agent.

Technically: the local service program for Vaultbeat's encrypted health-data recipient flow.
Published externally as [`Fino-wind/vaultbeat-apple-health`](https://github.com/Fino-wind/vaultbeat-apple-health)
(public package name `vaultbeat-apple-health` since 0.6.2; `vaultbeat-mcp` and
`vaultbeat-mcp-local` remain back-compat console scripts). This directory is the source of truth — after any user-visible change here,
re-export the public repo and update its README tool table + the website `/mcp` page
(see CLAUDE.md "Sync duty").

It runs on the user's computer, generates the Curve25519 keypair used by the iOS app,
shows a QR binding payload, receives a one-time server token from the cloud API, and
then exposes decrypted health data — sleep, water, weight, cycle, activity, vitals —
through either a CLI or a stdio MCP server.

## Commands

```bash
python -m pip install -e './mcp-local-server[qr]'
# try every read tool against synthetic data — no pairing, no cloud, no Apple Health
vaultbeat-apple-health --demo sleep --limit 5
vaultbeat-apple-health --demo doctor

vaultbeat-apple-health bind
vaultbeat-apple-health status

# read decrypted health data — every data subcommand accepts
# --owner <user-id prefix> to filter to one person (omitting it mixes
# both partners' records into one pool; aggregates become meaningless)
vaultbeat-apple-health sleep --limit 5 --owner a1a1        # sleep sessions + provenance
vaultbeat-apple-health sleep-detail --limit 1 --owner a1a1 # HR+RR+stage timeline
vaultbeat-apple-health water --limit 30 --owner a1a1       # water intake + daily average
vaultbeat-apple-health weight --limit 90 --owner b2b2      # weight trend (latest/avg/weekly rate)
vaultbeat-apple-health menstrual --limit 60 --owner b2b2   # menstrual cycle (sensitive)
vaultbeat-apple-health activity --limit 30 --owner a1a1    # daily activity rings
vaultbeat-apple-health resting-hr --limit 30 --owner a1a1  # resting heart rate
vaultbeat-apple-health workouts --limit 20 --owner a1a1    # workout records
vaultbeat-apple-health mindfulness --limit 30 --owner a1a1 # mindful minutes
vaultbeat-apple-health hrv --limit 30 --owner a1a1         # HRV / SDNN (hourly buckets by default; --granularity raw for per-sample)
vaultbeat-apple-health wrist-temp --limit 30 --owner a1a1  # sleeping wrist temperature
vaultbeat-apple-health symptoms --limit 30                 # symptom days (grouped by owner)
vaultbeat-apple-health notes --kind sleep --limit 30       # free-text day annotations

# run as an MCP server
vaultbeat-apple-health serve --transport stdio
vaultbeat-apple-health serve --transport http --host 127.0.0.1 --port 8000 --path /mcp
vaultbeat-apple-health --demo serve --transport stdio   # same synthetic dataset, wired into a client
```

`--demo` is a **global flag**, not a subcommand: it goes before the subcommand
(`vaultbeat-apple-health --demo sleep`), and `VAULTBEAT_DEMO=1` does the same thing. It serves a
deterministic synthetic dataset — the same records on every machine, every run — so demo
output can be pasted into a bug report as a shared baseline. Nothing is fetched and nothing
is decrypted; there is no private key involved at all. Every payload carries `demo_mode:
true` plus a `[SYNTHETIC DEMO DATA]` banner, the tool descriptions say so, and the server
lists itself as `Vaultbeat Health [DEMO — SYNTHETIC DATA]`, so demo output cannot pass for a
real export. **Read tools only** — the `log_*` write tools refuse, because writing needs a
real key and a real account, and a write that pretends to succeed is worse than one that says
it needs pairing. It applies to that one invocation and is never written to the config file.

`http` is a CLI alias for MCP's `streamable-http` transport.
The default transport remains `stdio` for local desktop MCP clients.

The config file defaults to `~/.tether/mcp-local/config.json` and is written with
`0600` permissions. It contains the cloud-issued server token and your **public**
key; do not commit or share it.

### Where the private key lives

Not in `config.json`. It is looked for in three places, in order:

1. **`VAULTBEAT_PRIVATE_KEY`** — read if set, never written back, for operators
   who inject it from systemd-creds / a vault / a KMS.
2. **The system keyring** — the normal case on a desktop.
3. **`~/.tether/mcp-local/identity.key`**, mode `0600` — the automatic fallback
   on a machine with no keyring backend at all.

Keeping it out of `config.json` is a boundary, not tidiness: the server token
alone can download your ciphertext but not read it, and the private key alone
has nothing to decrypt. `config.json` is the file people `cat` into bug reports.

> 🔴 **Never delete `config.json` to "start clean".** The private key is not in
> it, so deleting does not clear a bad key — it mints a brand-new identity, and
> every record already encrypted for the old one becomes permanently unreadable.
> If a command reports missing key material, the error names all three locations
> and what was found in each; read that before removing anything.

**Headless servers**: if the keyring is unreachable, do *not* set
`PYTHON_KEYRING_BACKEND` to the null backend. That backend accepts writes and
stores nothing; since 0.4.3 every keyring write is verified by reading it back,
so a null backend just lands the key in layer 3's `identity.key` file — the same
outcome as having no keyring, with a keyring you might have reached hidden behind
it. Either let layer 3 handle it (automatic when no backend exists) or, if a
D-Bus session exists but this process cannot see it, pass
`DBUS_SESSION_BUS_ADDRESS` through explicitly — `XDG_RUNTIME_DIR` on its own is
not enough.

> **`.tether`, not `.vaultbeat` — that is deliberate, do not "fix" it.** The app
> was renamed but this path is frozen at the pre-rename location, because the
> Keychain username embeds the resolved config path (`_keychain_username` in
> `store.py`). Moving the directory orphans the bound config *and* its
> private-key Keychain entry for every existing install. Until 2026-07-28 this
> README wrote `~/.vaultbeat/...`, which does not exist — so anyone who came here
> to destroy their credentials `rm -rf`'d an empty path, got no error, and left
> the real key in place.

When using HTTP transport, the server binds to `127.0.0.1:8000` and serves MCP at
`/mcp` by default, and requires a bearer token (see "Authenticating HTTP transport"
below). Binding a non-loopback address fails closed unless you pass both a token and
`--allow-remote`; always front a network-exposed server with TLS (a reverse proxy).

## Binding Flow

1. `vaultbeat-apple-health bind` generates a fresh `pollID` and prints a QR payload:
   `{"pollID":"...","publicKeyBase64":"...","serverName":"..."}`
2. The iOS app scans that payload and calls the `mcp-bind-local` Edge Function.
3. The local service polls the `mcp-poll-binding` Edge Function.
4. Once bound, the local config stores `serverID` and `serverToken`.
5. All read commands call the `mcp-sync` Edge Function, decrypt the returned envelopes
   locally, and return plaintext JSON. (All privileged routes are Supabase Edge
   Functions at `/functions/v1/<name>`.)

### Troubleshooting: `vaultbeat-apple-health doctor`

If binding or reads fail, run the self-diagnosis:

```bash
vaultbeat-apple-health doctor          # human-readable [OK]/[FAIL] checklist with hints
vaultbeat-apple-health doctor --json   # machine-readable, for agents
```

It checks, in order: config file → identity key (Keychain) → cloud reachability →
binding state → a real fetch-and-decrypt round trip, and prints a targeted hint for
the first thing that's broken (e.g. "codes expire after 10 minutes — re-run bind for
a fresh QR", or "the stored key can no longer decrypt your data — delete this server
in the iOS app and bind again"). Exit code 0 = all healthy, 1 = something needs the
hint above.

## MCP Tools

`vaultbeat-apple-health serve` can start either a stdio MCP server or a streamable HTTP MCP
server. Every data tool accepts `owner` (user-ID prefix) to filter to one person and
`fresh` to bypass the local cache — omit `owner` and both partners' records mix into
one pool, so per-person analysis must always pass it. The tool names dropped the old
misleading `get_partner_*` prefix in the 16-tool release (2026-07-16): the tools
return whichever owners' envelopes this server holds, not specifically "the partner".

Binding / status:

- `vaultbeat_status` — local binding state (no keys/tokens in the result)
- `vaultbeat_start_binding` — generate a fresh QR binding payload
- `vaultbeat_poll_binding` — poll once for the iOS authorization
- `vaultbeat_doctor` — diagnose this install end to end, and report which data types
  are unavailable and why. Call it before telling a user their data is missing.

Health data:

- `vaultbeat_sync_sleep` — recent sleep records (incl. heart-rate samples) with per-day
  primary-session selection matching the iOS app
- `get_sleep_detail` — per-night HR+RR+stage timeline with stage intervals
- `get_water_intake` — recent daily intake + computed `average_daily_intake_liters`
- `get_weight_trend` — daily weights + latest/avg/min/max + OLS weekly rate
- `get_menstrual_cycle` — recent cycle samples + a next-period prediction (sensitive)
- `get_symptoms` — recent HealthKit symptom days grouped by data owner (sensitive)
- `get_notes` — free-text sleep/menstrual day annotations with their writer (sensitive)
- `get_strength_log` — strength-training sessions with exercise-level sets × reps and
  per-session `total_volume_kg` (owner's own sessions only; logged manually in the app)
- `get_food_log` — per-day meals and items, with optional per-item kcal/protein/fat/carbs
- `get_activity` — daily activity rings (steps/energy/exercise/stand/distance)
- `get_resting_hr` — resting heart rate records + mean
- `get_workouts` — workout records (type/duration/calories/distance)
- `get_mindfulness` — mindful sessions per day
- `get_hrv` — HRV/SDNN records + mean. `granularity="hourly"` (default, `hrv_hourly` kind, one bucket/UTC hour w/ `sample_count`, 30-day window) or `"raw"` (`hrv` kind, per-sample, 3-day window)
- `get_wrist_temp` — sleeping wrist-temperature baseline deviation
- `get_vo2max` — cardiorespiratory fitness. Sparse by design: Apple only computes it
  during outdoor walk/run/hike, so a handful of samples across a year is normal
- `get_basal_energy` — basal metabolism (BMR) kcal, hourly buckets
- `get_total_energy_burned` — basal + active = TDEE, with a 7-day average

**Every read tool also returns a `coverage` block**, and an agent that wants to say
"this is based on N days" has nothing else to read. Fields: `days_covered`,
`days_in_payload` (how many of those days have a row printed — smaller only when a
display cap cut the list), `first_day`, `last_day`, `span_days`,
`days_missing_in_span`, `rows_counted`, `requested`, `requested_unit`,
`window_satisfied`. Two things it exists to stop: counting the returned
array instead (`limit` has already cut it, so the length answers a different question),
and reading a long `span_days` as coverage — `days_covered: 12` with `span_days: 200` is
twelve scattered days, not seven months. Quote `days_covered` beside any average, trend
or comparison drawn from a result: an average over 3 days and one over 30 are the same
shape and the same number of digits, and this is the only field that tells them apart.

Writes (all scoped to the account that paired this machine — none takes an `owner`
argument, so an agent can write to its own account and nowhere else):

- `log_weight_entry` — record a weigh-in. Carries that day's existing body composition
  forward instead of erasing it. No `_append` twin: a day has one weight.
- `log_strength_entry` / `log_food_entry` / `log_note` — **replace that whole day**. Each
  returns a `replaced_*` field naming exactly what it removed, so an agent can notice and
  re-send. `merge=True` appends instead, kept for existing callers.
- `log_strength_append` / `log_food_append` / `log_note_append` — add to a day and
  **cannot delete anything**.

> ⚠️ **`log_*` and `log_*_append` are different operations, and the names are the whole
> point.** An agent picks a tool by name, and `log_food_entry` reads as "record something I
> ate" while it actually means "overwrite this day with what I pass" — a default that lives
> in a schema nobody re-reads. **When a day may already have entries, `log_*_append` is
> almost always the one you want**, and it is the safe default for an agent that cannot see
> what is already there. The split also lets the two be annotated differently
> (`destructiveHint` true vs false), which a boolean argument structurally cannot be.

(The health-memory fact tools — `health_recall_*` / `health_remember` — were deleted with
the fact system in `491c850`, 2026-06-29: long-lived health knowledge lives in local
markdown managed by the user's agent, not in an E2EE cloud round trip.)

Every health kind shares one decryption path (Curve25519 ECDH + HKDF-SHA256 + AES-GCM);
the server routes on `encrypted_sleep_blobs.metric_type` (the live kind list is whatever `check_metric_type_contract.py` prints — see
`KNOWN_METRIC_TYPES` in `service.py`) and only the per-kind JSON decode/aggregate
differs. The same service-layer functions back both the MCP tools and the matching CLI
subcommands — no duplicated logic.

**Local record cache (2026-07-09):** all reads are cache-first. Decrypted records are
kept per metric type under `~/.tether/mcp-local/cache/` (owner-only 0600 files, 0700
dir, stamped with server_id + fetch time + the fetch's decrypt-error list). Default TTL
600 s — override with `VAULTBEAT_MCP_CACHE_TTL` (0 disables). Within the TTL a repeat query
is answered locally with zero network (~0.2 s vs 5-35 s); pass `--fresh` (CLI) or
`fresh=true` (MCP tools) to force a cloud round trip. (Re)binding clears the cache.
`mcp-sync` also accepts `?metric_type=` so single-metric fetches stop paying for every
other kind's ciphertext; the client keeps its own post-decrypt filter, so older edge
deployments stay correct.

Menstrual data is **sensitive**: it only reaches this server when the user explicitly
opted in on iOS (absent otherwise), is decrypted locally, and is never re-exported.

The MCP server never exposes the private key or server token through tool results.

## MCP Prompts

`prompts/list` is the only channel through which an agent can ask what this server is
*for*, rather than what it can call. Without it every client invents its own analysis
routine, and the two mistakes that do real damage with health data — reporting an
association as a cause, and reading a gap as a zero — are left to whichever agent
happened to connect. Each entry names the tools it should call, and every one of them
ends with the same two shared constants: a style rule (say what the data covers before
concluding; describe, do not prescribe; no invented scores, grades or verdicts) and,
wherever a read can legitimately come back empty, the absence rule (an empty result has
four different causes that need opposite fixes — call `vaultbeat_doctor`, which is the
tool that tells them apart).

Every argument is optional. Omitted ones are filled with a default written into the
prompt, so a client that sends nothing still gets a whole sentence rather than a hole.

| Prompt | Argument | What it asks for |
| --- | --- | --- |
| `daily_brief` | — | The most recent day, set against the fortnight behind it, with the newest date named up front |
| `sleep_review` | `nights` | Duration and stages across recent nights — keeping the nights that were never measured out of the average instead of folding them in as zeros |
| `energy_balance` | `days` | Calories in against calories out, with the hours-incomplete days excluded rather than quietly dragging the average down |
| `training_block_review` | `days` | Lifting and cardio volume beside the recovery signals from the same weeks, as context rather than a verdict |
| `cycle_aware_read` | `metric` | A metric read against the same phase of earlier cycles, instead of against last week — which mixes phases and manufactures a trend |
| `partner_check_in` | `days` | Both people's shared data side by side, read once per owner rather than averaged across two bodies, and never turned into a judgement of either |
| `log_from_conversation` | `entry` | Turn something said in passing into an entry, asking for the parts that were left out rather than filling them in |
| `why_is_this_empty` | `context` | Work out which of the four causes is behind an empty or stale result, and give the one next action for that one |

## Transport Options

Stdio transport, for local MCP clients that launch the server as a subprocess:

```bash
vaultbeat-apple-health serve --transport stdio
```

HTTP transport, for MCP clients that connect over a network or reverse proxy:

```bash
vaultbeat-apple-health serve --transport http --host 127.0.0.1 --port 8000 --path /mcp
```

Optional HTTP flags:

- `--sse-response` to use SSE-style HTTP responses instead of JSON responses.
- `--stateful-http` to disable stateless HTTP mode for clients that require sessions.
- `--generate-token` to mint and persist a bearer token, print client config, then exit.
- `--show-token` to print the stored bearer token and exit.
- `--allow-remote` to permit a non-loopback bind (requires a token; confirms intent).
- `--no-token` to serve loopback HTTP without bearer auth.

## Authenticating HTTP transport

The HTTP tool surface exposes **decrypted** health data, so it is gated by a static
bearer token and refuses to bind a network-reachable address without explicit opt-in.

Generate (and persist) a token, then print ready-to-paste client config:

```bash
vaultbeat-apple-health serve --generate-token
```

Serve over HTTP on loopback. Auth is on by default; the token is read from
`VAULTBEAT_MCP_HTTP_TOKEN` (preferred, keeps it out of shell history) or the stored config:

```bash
vaultbeat-apple-health serve --transport http              # 127.0.0.1, bearer required
vaultbeat-apple-health serve --transport http --no-token   # loopback only, no auth
```

Clients send the token as a request header:

```
Authorization: Bearer <token>
```

Example `mcp.json` (VS Code / Cursor style):

```json
{
  "servers": {
    "vaultbeat-local": {
      "type": "http",
      "url": "http://127.0.0.1:8000/mcp",
      "headers": { "Authorization": "Bearer <token>" }
    }
  }
}
```

**Binding beyond loopback** (e.g. `--host 0.0.0.0` for LAN/VPS) fails closed: it
requires both a token *and* the explicit `--allow-remote` flag. The token crosses the
wire in clear text, so you must terminate TLS in front of it (e.g. Caddy / Cloudflare /
nginx):

```bash
VAULTBEAT_MCP_HTTP_TOKEN=<token> vaultbeat-apple-health serve \
  --transport http --host 0.0.0.0 --allow-remote
```

Claude Desktop's config only speaks stdio, so bridge it to the HTTP server with
[`mcp-remote`](https://github.com/geelen/mcp-remote):

```json
{
  "mcpServers": {
    "vaultbeat-local": {
      "command": "npx",
      "args": [
        "-y", "mcp-remote", "http://127.0.0.1:8000/mcp",
        "--header", "Authorization: Bearer <token>"
      ]
    }
  }
}
```

Reveal the stored token any time with `vaultbeat-apple-health serve --show-token`.

## Verification

```bash
python -m pytest -q mcp-local-server/tests
python -m ruff check mcp-local-server/src mcp-local-server/tests
python -m mypy mcp-local-server/src
```

<!-- mcp-name: io.github.Fino-wind/vaultbeat-apple-health -->
