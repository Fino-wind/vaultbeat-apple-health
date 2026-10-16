# Vaultbeat — Apple Health for your agent (Claude Code plugin)

This plugin lets your own AI agent read the Apple Health data you keep in the [Vaultbeat](https://vaultbeat.app/mcp) iPhone app. The iPhone app encrypts every record on the phone. The cloud only ever holds ciphertext, and the private key that opens it is generated on your computer when you pair and never leaves it, so decryption happens only on your machine.

It adds one local MCP server, `vaultbeat-health` (29 tools: sleep stages, HRV, resting heart rate, cycle, workouts, weight, meals, moods; plus writes for symptoms, food, strength, notes, weight and moods), and a `setup` skill that walks Claude through pairing your computer with the iPhone app.

## Requirements

- The Vaultbeat iOS app, 1.2.3 or newer, signed in ([App Store](https://apps.apple.com/app/id6759241985)). Recording on the phone is free; agent access needs Vaultbeat Pro, and pairing a computer starts a 3-day trial.
- [uv](https://docs.astral.sh/uv/) installed on your computer.

## Install

```
/plugin marketplace add Fino-wind/vaultbeat-apple-health
/plugin install vaultbeat-apple-health@vaultbeat
```

Then run `/vaultbeat-apple-health:setup`, or ask Claude to set Vaultbeat up. To try it without an iPhone, start Claude Code with `VAULTBEAT_DEMO=1` set: the server then serves a fixed synthetic dataset, every result is marked `demo_mode: true`, and the write tools refuse.

## What this plugin runs and connects to

- **Runs:** `uvx vaultbeat-apple-health==<version> serve --transport stdio` — the open-source [vaultbeat-apple-health](https://pypi.org/project/vaultbeat-apple-health/) package on PyPI, pinned to an exact version in `.mcp.json`. Its source is in this repository's `src/` folder.
- **Connects to:** Vaultbeat's backend at `api.vaultbeat.app` to fetch your encrypted records (ciphertext only), and to PostHog (EU) for usage events: which tool was called, whether it succeeded, a duration range and the client and package versions. Tool arguments, results and health data are never sent. Set `VAULTBEAT_TELEMETRY=0` or `DO_NOT_TRACK=1` to turn telemetry off.
- **Reads:** your Apple Health data, only on your machine and only through MCP tools. Your cycle and Apple Health symptoms reach the server only after you turn them on, per category, in the iOS app.
- **Does not:** read credentials from your environment, change Claude's settings, or run anything else.

Privacy policy: [vaultbeat.app/privacy](https://vaultbeat.app/privacy). Support: [vaultbeat.app/support](https://vaultbeat.app/support). Full documentation and the tool list: [the repository README](https://github.com/Fino-wind/vaultbeat-apple-health#readme). License: MIT.
