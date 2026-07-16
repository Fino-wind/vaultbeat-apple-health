from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from tether_mcp_local import __version__
from tether_mcp_local.service import KNOWN_METRIC_TYPES, TetherLocalService
from tether_mcp_local.store import DEFAULT_API_BASE_URL, ConfigStore, write_secret_file


def _store(args: argparse.Namespace) -> ConfigStore:
    return ConfigStore(Path(args.config).expanduser() if args.config else None)


def _service(args: argparse.Namespace) -> TetherLocalService:
    return TetherLocalService(_store(args))


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _print_qr(payload: str) -> None:
    try:
        import qrcode  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        print("Install with `pip install 'tether-mcp-local[qr]'` to render an ASCII QR code.")
        return

    qr = qrcode.QRCode(border=2)
    qr.add_data(payload)
    qr.make(fit=True)
    qr.print_ascii(invert=True)


def handle_init(args: argparse.Namespace) -> int:
    config = _store(args).ensure_initialized(
        server_name=args.server_name,
        api_base_url=args.api_base_url,
    )
    _print_json(config.redacted())
    return 0


def handle_status(args: argparse.Namespace) -> int:
    _print_json(_service(args).status())
    return 0


def handle_bind(args: argparse.Namespace) -> int:
    service = _service(args)
    session = service.start_binding(
        server_name=args.server_name,
        api_base_url=args.api_base_url,
    )
    print("Scan this payload with Tether on iOS:")
    print(session.qr_payload_json)
    if not args.no_qr:
        _print_qr(session.qr_payload_json)

    result = asyncio.run(
        service.poll_until_bound(
            timeout_sec=args.timeout,
            interval_sec=args.interval,
        )
    )
    if result.status != "bound":
        print("Binding is still pending. Re-run `tether-mcp-local poll` to continue.")
        return 2

    print(f"Bound local MCP server: {result.server_id}")
    return 0


def handle_poll(args: argparse.Namespace) -> int:
    result = asyncio.run(_service(args).poll_once())
    _print_json(
        {
            "status": result.status,
            "server_id": result.server_id,
            "request_id": result.request_id,
        }
    )
    return 0 if result.status == "bound" else 2


def handle_sync(args: argparse.Namespace) -> int:
    svc = _service(args)
    metric_type = getattr(args, "metric_type", None)
    if metric_type:
        records, errors = asyncio.run(svc._records_for_metric(metric_type, limit=args.limit, fresh=args.fresh))
    else:
        records, errors = asyncio.run(svc.sync_decrypted_records(limit=args.limit, fresh=args.fresh))
    payload = {
        "records": [record.to_dict() for record in records],
        "errors": errors,
    }
    if args.output:
        output_path = Path(args.output).expanduser()
        write_secret_file(output_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(
            f"Wrote {len(records)} DECRYPTED records to {output_path} (file mode 0600). "
            "This file holds unencrypted health data — treat it as sensitive."
        )
    else:
        _print_json(payload)
    return 0 if not errors else 3


def _emit_decrypted(payload: dict[str, Any], args: argparse.Namespace, *, label: str) -> None:
    """Print or persist a decrypted-health summary, mirroring `handle_sync` output rules."""

    if args.output:
        output_path = Path(args.output).expanduser()
        write_secret_file(output_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(
            f"Wrote DECRYPTED {label} summary to {output_path} (file mode 0600). "
            "This file holds unencrypted health data — treat it as sensitive."
        )
    else:
        _print_json(payload)


def handle_sleep(args: argparse.Namespace) -> int:
    result = asyncio.run(_service(args).sleep_records(
        limit=args.limit, fresh=args.fresh,
        owner=getattr(args, "owner", None),
    ))
    _emit_decrypted(result, args, label="sleep sessions")
    return 0 if not result.get("errors") else 3


def handle_sleep_detail(args: argparse.Namespace) -> int:
    result = asyncio.run(_service(args).sleep_detail_records(
        limit=args.limit, fresh=args.fresh,
        owner=getattr(args, "owner", None),
    ))
    _emit_decrypted(result, args, label="sleep detail (HR+RR+stage timeline)")
    return 0 if not result.get("errors") else 3


def handle_water(args: argparse.Namespace) -> int:
    summary = asyncio.run(_service(args).water_intake_summary(limit=args.limit, fresh=args.fresh, owner=getattr(args, "owner", None)))
    _emit_decrypted(summary, args, label="water intake")
    return 0 if not summary.get("errors") else 3


def handle_weight(args: argparse.Namespace) -> int:
    summary = asyncio.run(
        _service(args).weight_trend_summary(limit=args.limit, goal_kg=args.goal_kg, fresh=args.fresh, owner=getattr(args, "owner", None))
    )
    _emit_decrypted(summary, args, label="weight trend")
    return 0 if not summary.get("errors") else 3


def handle_menstrual(args: argparse.Namespace) -> int:
    print(
        "Note: menstrual data is sensitive; it is decrypted locally and never re-exported. "
        "It only appears if the user explicitly opted in on iOS."
    )
    summary = asyncio.run(_service(args).menstrual_cycle_summary(limit=args.limit, fresh=args.fresh, owner=getattr(args, "owner", None)))
    _emit_decrypted(summary, args, label="menstrual cycle")
    return 0 if not summary.get("errors") else 3


def handle_activity(args: argparse.Namespace) -> int:
    summary = asyncio.run(_service(args).activity_summary(limit=args.limit, fresh=args.fresh, owner=getattr(args, "owner", None)))
    _emit_decrypted(summary, args, label="activity rings")
    return 0 if not summary.get("errors") else 3


def handle_resting_hr(args: argparse.Namespace) -> int:
    result = asyncio.run(_service(args).resting_hr_records(limit=args.limit, fresh=args.fresh, owner=getattr(args, "owner", None)))
    _emit_decrypted(result, args, label="resting heart rate")
    return 0 if not result.get("errors") else 3


def handle_workouts(args: argparse.Namespace) -> int:
    result = asyncio.run(_service(args).workout_records(limit=args.limit, fresh=args.fresh, owner=getattr(args, "owner", None)))
    _emit_decrypted(result, args, label="workouts")
    return 0 if not result.get("errors") else 3


def handle_mindfulness(args: argparse.Namespace) -> int:
    result = asyncio.run(_service(args).mindfulness_summary(limit=args.limit, fresh=args.fresh, owner=getattr(args, "owner", None)))
    _emit_decrypted(result, args, label="mindfulness")
    return 0 if not result.get("errors") else 3


def handle_hrv(args: argparse.Namespace) -> int:
    result = asyncio.run(_service(args).hrv_records(limit=args.limit, fresh=args.fresh, owner=getattr(args, "owner", None)))
    _emit_decrypted(result, args, label="HRV (SDNN)")
    return 0 if not result.get("errors") else 3


def handle_wrist_temp(args: argparse.Namespace) -> int:
    result = asyncio.run(_service(args).wrist_temp_records(limit=args.limit, fresh=args.fresh, owner=getattr(args, "owner", None)))
    _emit_decrypted(result, args, label="wrist temperature")
    return 0 if not result.get("errors") else 3


def handle_notes(args: argparse.Namespace) -> int:
    print(
        "Note: notes are sensitive free text; they are decrypted locally and never re-exported."
    )
    summary = asyncio.run(
        _service(args).notes_summary(limit=args.limit, target_kind=args.kind, fresh=args.fresh)
    )
    _emit_decrypted(summary, args, label="notes")
    return 0 if not summary.get("errors") else 3


def handle_symptoms(args: argparse.Namespace) -> int:
    print(
        "Note: symptom data is sensitive; it is decrypted locally and never re-exported. "
        "It only appears if a user explicitly opted in on iOS."
    )
    summary = asyncio.run(_service(args).symptom_summary(limit=args.limit, fresh=args.fresh))
    _emit_decrypted(summary, args, label="symptoms")
    return 0 if not summary.get("errors") else 3


def _resolve_http_token(store: ConfigStore) -> str | None:
    env_token = os.getenv("TETHER_MCP_HTTP_TOKEN", "").strip()
    if env_token:
        return env_token
    config = store.load()
    return config.http_token if config else None


def _print_http_token(token: str, args: argparse.Namespace) -> None:
    url = f"http://{args.host}:{args.port}{args.path}"
    snippet = {
        "servers": {
            "tether-local": {
                "type": "http",
                "url": url,
                "headers": {"Authorization": f"Bearer {token}"},
            }
        }
    }
    print("HTTP bearer token (store it securely; shown only once):")
    print(f"  {token}")
    print()
    print("Send it from your MCP client as:")
    print(f"  Authorization: Bearer {token}")
    print()
    print("Example mcp.json (VS Code / Cursor style):")
    print(json.dumps(snippet, indent=2))


def handle_serve(args: argparse.Namespace) -> int:
    # Imported lazily: the MCP SDK import chain is ~1.4s, which every OTHER
    # subcommand (the CLI data path) must not pay.
    from tether_mcp_local.mcp_server import run_mcp_server

    store = _store(args)
    is_http = args.transport in ("http", "streamable-http")

    if args.generate_token:
        _print_http_token(store.ensure_http_token(), args)
        return 0

    if args.show_token:
        config = store.load()
        stored = config.http_token if config else None
        if not stored:
            print("No HTTP token set. Run `tether-mcp-local serve --generate-token` first.")
            return 1
        print(stored)
        return 0

    token: str | None = None
    if is_http and not args.no_token:
        token = _resolve_http_token(store)
        if token:
            print(
                f"HTTP bearer auth enabled (token {token[:6]}..., len {len(token)}). "
                "Run `serve --show-token` to reveal it."
            )

    run_mcp_server(
        store,
        transport=args.transport,
        host=args.host,
        port=args.port,
        path=args.path,
        json_response=not args.sse_response,
        stateless_http=not args.stateful_http,
        token=token,
        allow_remote=args.allow_remote,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tether-mcp-local")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", help="Path to config JSON. Defaults to ~/.tether/mcp-local/config.json.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Generate a local keypair and config file.")
    init_parser.add_argument("--server-name", default="Local AI Server")
    init_parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    init_parser.set_defaults(func=handle_init)

    bind_parser = subparsers.add_parser("bind", help="Show a QR payload and wait for iOS authorization.")
    bind_parser.add_argument("--server-name", default="Local AI Server")
    bind_parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    bind_parser.add_argument("--timeout", type=int, default=300)
    # 7.0s keeps polling under the cloud's 10/min IP rate limit (60/7 ≈ 8.6/min);
    # the old 3.0s default (20/min) self-tripped a 429 on first bind.
    bind_parser.add_argument("--interval", type=float, default=7.0)
    bind_parser.add_argument("--no-qr", action="store_true")
    bind_parser.set_defaults(func=handle_bind)

    poll_parser = subparsers.add_parser("poll", help="Poll once for a pending iOS authorization.")
    poll_parser.set_defaults(func=handle_poll)

    sync_parser = subparsers.add_parser("sync", help="Fetch and decrypt all encrypted health payloads.")
    sync_parser.add_argument("--limit", type=int)
    sync_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the local record cache and force a cloud fetch.",
    )
    sync_parser.add_argument(
        "--metric-type",
        dest="metric_type",
        choices=sorted(KNOWN_METRIC_TYPES),
        help="Filter to a single metric type.",
    )
    sync_parser.add_argument("--output")
    sync_parser.set_defaults(func=handle_sync)

    sleep_parser = subparsers.add_parser(
        "sleep", help="Decrypt recent sleep sessions with stage breakdown."
    )
    sleep_parser.add_argument("--limit", type=int)
    sleep_parser.add_argument(
        "--owner",
        help="Only include records from this owner (user-ID prefix, e.g. dce9 or f835).",
    )
    sleep_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the local record cache and force a cloud fetch.",
    )
    sleep_parser.add_argument("--output")
    sleep_parser.set_defaults(func=handle_sleep)

    sleep_detail_parser = subparsers.add_parser(
        "sleep-detail",
        help="Per-night timeline: each HR/RR sample tagged with the concurrent sleep stage.",
    )
    sleep_detail_parser.add_argument("--limit", type=int, help="Keep at most N most recent nights.")
    sleep_detail_parser.add_argument(
        "--owner",
        help="Only include records from this owner (user-ID prefix, e.g. dce9 or f835).",
    )
    sleep_detail_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the local record cache and force a cloud fetch.",
    )
    sleep_detail_parser.add_argument("--output")
    sleep_detail_parser.set_defaults(func=handle_sleep_detail)

    water_parser = subparsers.add_parser(
        "water", help="Decrypt recent water intake and compute the daily average."
    )
    water_parser.add_argument("--limit", type=int)
    water_parser.add_argument(
        "--owner",
        help="Only include records from this owner (user-ID prefix, e.g. dce9 or f835).",
    )
    water_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the local record cache and force a cloud fetch.",
    )
    water_parser.add_argument("--output")
    water_parser.set_defaults(func=handle_water)

    weight_parser = subparsers.add_parser(
        "weight",
        help="Decrypt recent body weight (kg) and compute the trend (latest/avg/min/max/weekly rate).",
    )
    weight_parser.add_argument("--limit", type=int, help="Keep at most N most recent body days.")
    weight_parser.add_argument(
        "--owner",
        help="Only include records from this owner (user-ID prefix, e.g. dce9 or f835).",
    )
    weight_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the local record cache and force a cloud fetch.",
    )
    weight_parser.add_argument(
        "--goal-kg",
        dest="goal_kg",
        type=float,
        help="Goal weight in kg to compute delta_to_goal_kg (negative = below goal); omit to skip.",
    )
    weight_parser.add_argument("--output")
    weight_parser.set_defaults(func=handle_weight)

    menstrual_parser = subparsers.add_parser(
        "menstrual",
        help="Decrypt recent menstrual cycle (sensitive) and predict the next period.",
    )
    menstrual_parser.add_argument("--limit", type=int)
    menstrual_parser.add_argument(
        "--owner",
        help="Only include records from this owner (user-ID prefix, e.g. dce9 or f835).",
    )
    menstrual_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the local record cache and force a cloud fetch.",
    )
    menstrual_parser.add_argument("--output")
    menstrual_parser.set_defaults(func=handle_menstrual)

    activity_parser = subparsers.add_parser(
        "activity", help="Decrypt recent daily activity rings (steps, energy, exercise, stand, distance)."
    )
    activity_parser.add_argument("--limit", type=int)
    activity_parser.add_argument(
        "--owner",
        help="Only include records from this owner (user-ID prefix, e.g. dce9 or f835).",
    )
    activity_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the local record cache and force a cloud fetch.",
    )
    activity_parser.add_argument("--output")
    activity_parser.set_defaults(func=handle_activity)

    resting_hr_parser = subparsers.add_parser(
        "resting-hr", help="Decrypt recent resting heart rate samples."
    )
    resting_hr_parser.add_argument("--limit", type=int)
    resting_hr_parser.add_argument(
        "--owner",
        help="Only include records from this owner (user-ID prefix, e.g. dce9 or f835).",
    )
    resting_hr_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the local record cache and force a cloud fetch.",
    )
    resting_hr_parser.add_argument("--output")
    resting_hr_parser.set_defaults(func=handle_resting_hr)

    workouts_parser = subparsers.add_parser(
        "workouts", help="Decrypt recent workout sessions."
    )
    workouts_parser.add_argument("--limit", type=int)
    workouts_parser.add_argument(
        "--owner",
        help="Only include records from this owner (user-ID prefix, e.g. dce9 or f835).",
    )
    workouts_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the local record cache and force a cloud fetch.",
    )
    workouts_parser.add_argument("--output")
    workouts_parser.set_defaults(func=handle_workouts)

    mindfulness_parser = subparsers.add_parser(
        "mindfulness", help="Decrypt recent daily mindfulness summaries."
    )
    mindfulness_parser.add_argument("--limit", type=int)
    mindfulness_parser.add_argument(
        "--owner",
        help="Only include records from this owner (user-ID prefix, e.g. dce9 or f835).",
    )
    mindfulness_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the local record cache and force a cloud fetch.",
    )
    mindfulness_parser.add_argument("--output")
    mindfulness_parser.set_defaults(func=handle_mindfulness)

    hrv_parser = subparsers.add_parser(
        "hrv", help="Decrypt recent HRV (SDNN) samples."
    )
    hrv_parser.add_argument("--limit", type=int)
    hrv_parser.add_argument(
        "--owner",
        help="Only include records from this owner (user-ID prefix, e.g. dce9 or f835).",
    )
    hrv_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the local record cache and force a cloud fetch.",
    )
    hrv_parser.add_argument("--output")
    hrv_parser.set_defaults(func=handle_hrv)

    wrist_temp_parser = subparsers.add_parser(
        "wrist-temp", help="Decrypt recent sleeping wrist temperature samples."
    )
    wrist_temp_parser.add_argument("--limit", type=int)
    wrist_temp_parser.add_argument(
        "--owner",
        help="Only include records from this owner (user-ID prefix, e.g. dce9 or f835).",
    )
    wrist_temp_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the local record cache and force a cloud fetch.",
    )
    wrist_temp_parser.add_argument("--output")
    wrist_temp_parser.set_defaults(func=handle_wrist_temp)

    symptoms_parser = subparsers.add_parser(
        "symptoms",
        help="Decrypt recent HealthKit symptom days (sensitive), grouped by data owner.",
    )
    symptoms_parser.add_argument("--limit", type=int)
    symptoms_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the local record cache and force a cloud fetch.",
    )
    symptoms_parser.add_argument("--output")
    symptoms_parser.set_defaults(func=handle_symptoms)

    notes_parser = subparsers.add_parser(
        "notes",
        help="Decrypt recent free-text notes (sleep/menstrual annotations, sensitive).",
    )
    notes_parser.add_argument("--limit", type=int)
    notes_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass the local record cache and force a cloud fetch.",
    )
    notes_parser.add_argument(
        "--kind", choices=["sleep", "menstrual"], help="Keep only one target kind."
    )
    notes_parser.add_argument("--output")
    notes_parser.set_defaults(func=handle_notes)

    status_parser = subparsers.add_parser("status", help="Show local binding state.")
    status_parser.set_defaults(func=handle_status)

    serve_parser = subparsers.add_parser("serve", help="Run the MCP server over stdio or streamable HTTP.")
    serve_parser.add_argument(
        "--transport",
        choices=["stdio", "http", "streamable-http"],
        default="stdio",
        help="MCP transport. `http` is an alias for `streamable-http`.",
    )
    serve_parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host when using HTTP transport.")
    serve_parser.add_argument("--port", type=int, default=8000, help="HTTP bind port when using HTTP transport.")
    serve_parser.add_argument("--path", default="/mcp", help="HTTP endpoint path when using HTTP transport.")
    serve_parser.add_argument(
        "--sse-response",
        action="store_true",
        help="Use SSE-style HTTP responses instead of JSON responses.",
    )
    serve_parser.add_argument(
        "--stateful-http",
        action="store_true",
        help="Disable stateless HTTP mode for clients that require sessions.",
    )
    serve_parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Permit binding a non-loopback host (requires a token); confirms intentional network exposure.",
    )
    serve_parser.add_argument(
        "--generate-token",
        action="store_true",
        help="Generate and persist an HTTP bearer token, print it once with client config, then exit.",
    )
    serve_parser.add_argument(
        "--show-token",
        action="store_true",
        help="Print the stored HTTP bearer token and exit.",
    )
    serve_parser.add_argument(
        "--no-token",
        action="store_true",
        help="Loopback only: serve HTTP without bearer auth.",
    )
    serve_parser.set_defaults(func=handle_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
