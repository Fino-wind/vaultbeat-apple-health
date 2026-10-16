---
name: setup
description: Walks the user through pairing this computer with their Vaultbeat iPhone app so the vaultbeat-health MCP server can read their Apple Health data — or through the no-iPhone demo. Use when the user has just installed the plugin, when vaultbeat_doctor reports it is not paired, or when they ask how to connect Vaultbeat.
---

# Set up Vaultbeat on this computer

The `vaultbeat-health` MCP server is already configured by this plugin. What is missing is the **pairing**: the iPhone app encrypts the user's Apple Health data, and this computer generates the key that opens it. The private key never leaves this machine.

Do these steps in order, one at a time, and wait for the user between them. Never ask the user to paste a key, a token or any health data into the chat.

1. **Check what already works.** Call the `vaultbeat_doctor` tool. If it reports a healthy pairing and data types with records, stop: tell the user it is set up and offer to look at last night's sleep.
2. **If it is not paired, check the basics.**
   - The Vaultbeat iPhone app is installed and signed in (App Store, app 1.2.3 or newer). Agent access needs Vaultbeat Pro; pairing a computer starts a 3-day trial.
   - `uv` is installed: run `uv --version`. If it is missing, tell the user to install it from https://docs.astral.sh/uv/ and come back.
3. **Start the pairing.** Run `uvx vaultbeat-apple-health@latest bind`. It prints a QR code and waits for 5 minutes. Tell the user to open the Vaultbeat app → **MCP** tab → **Connect an AI server** and scan it. (App 1.3.0 and later can also show a one-line command with a setup code: if the user pastes one that looks like `uvx vaultbeat-apple-health@latest bind --setup-code VB-…`, run exactly that.)
4. **Verify.** After the app says the machine is connected, call `vaultbeat_doctor` again. Right after pairing the phone is still sealing the history for this computer, so some data types may be empty for a few minutes: that is normal, not a failure. If something stays empty, read what `vaultbeat_doctor` says about it before guessing.
5. **Offer a first question**, for example "How did I sleep last week? Show the trend and flag any night that broke the pattern."

## No iPhone, or just looking around

The server has a demo mode that serves a fixed synthetic dataset with no pairing, no cloud and no key. Every result is marked `demo_mode: true`, and the `log_*` write tools refuse. To use it, run `uvx vaultbeat-apple-health@latest --demo serve --transport stdio` as an MCP server (the `--demo` flag goes before `serve`).

## Rules for answering from the data

- Say what period and which data types an answer covers before drawing a conclusion.
- A missing night or an empty data type is **not** a zero. Report it as missing.
- Describe what the numbers show; do not diagnose.
