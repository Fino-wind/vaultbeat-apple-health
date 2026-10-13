"""`bind --setup-code`: the computer tells the phone it has started (GitHub #106).

The phone's onboarding hands the computer a command carrying a code it minted;
`bind` reports that code to `mcp-bind-started` before drawing the QR code, and the
phone moves on to "your computer is ready, scan now". The report is observation,
never a dependency: a missing, malformed, refused or slow report must leave the
bind exactly as it is without one.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

import vaultbeat_mcp_local.cli as cli
from vaultbeat_mcp_local.service import VaultbeatLocalService
from vaultbeat_mcp_local.store import ConfigStore


class RecordingCloud:
    def __init__(self, *, error: Exception | None = None, delay: float = 0.0) -> None:
        self.reports: list[dict[str, str]] = []
        self.error = error
        self.delay = delay

    async def report_bind_started(
        self, setup_code: str, *, os_name: str | None, client_version: str | None
    ) -> None:
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        self.reports.append({"code": setup_code, "os": os_name, "version": client_version})


def _service(tmp_path: Path, cloud: Any) -> VaultbeatLocalService:
    service = VaultbeatLocalService(ConfigStore(tmp_path / "config.json"), cloud)
    service.start_binding(server_name="Test Mac", api_base_url="https://api.test")
    return service


def test_a_minted_code_is_reported_normalised(tmp_path: Path) -> None:
    cloud = RecordingCloud()
    service = _service(tmp_path, cloud)

    assert asyncio.run(service.report_bind_started("  vb-7k3qx9mz\n")) is True

    assert len(cloud.reports) == 1
    report = cloud.reports[0]
    assert report["code"] == "VB-7K3QX9MZ"
    assert report["os"], "the OS family helps read where a stalled user's computer got"
    assert report["version"]


@pytest.mark.parametrize("code", ["", "VB-", "VB-7K3QX9M", "VB-7K3QX9M0", "7K3QX9MZ", "VB-7K3QX9MZZ"])
def test_anything_but_the_minted_shape_is_not_sent(tmp_path: Path, code: str) -> None:
    cloud = RecordingCloud()
    service = _service(tmp_path, cloud)

    assert asyncio.run(service.report_bind_started(code)) is False
    assert cloud.reports == []


@pytest.mark.parametrize("env", [("VAULTBEAT_TELEMETRY", "0"), ("DO_NOT_TRACK", "1")])
def test_an_opt_out_sends_the_code_but_nothing_about_the_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: tuple[str, str]
) -> None:
    monkeypatch.setenv(*env)
    cloud = RecordingCloud()
    service = _service(tmp_path, cloud)

    assert asyncio.run(service.report_bind_started("VB-7K3QX9MZ")) is True
    assert cloud.reports == [{"code": "VB-7K3QX9MZ", "os": None, "version": None}]


def test_a_config_that_will_not_load_does_not_stop_the_bind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path, RecordingCloud())

    def locked() -> None:
        raise RuntimeError("keyring locked")

    monkeypatch.setattr(service.store, "load", locked)
    assert asyncio.run(service.report_bind_started("VB-7K3QX9MZ")) is False


def test_a_refused_report_is_swallowed(tmp_path: Path) -> None:
    service = _service(tmp_path, RecordingCloud(error=RuntimeError("HTTP 500")))

    assert asyncio.run(service.report_bind_started("VB-7K3QX9MZ")) is False


def test_a_slow_report_gives_up_on_time(tmp_path: Path) -> None:
    service = _service(tmp_path, RecordingCloud(delay=5.0))

    started = time.monotonic()
    assert asyncio.run(service.report_bind_started("VB-7K3QX9MZ", timeout_sec=0.05)) is False
    assert time.monotonic() - started < 2.0, "a slow network must not hold the QR code back"


def test_an_old_style_client_without_the_method_is_swallowed(tmp_path: Path) -> None:
    # Fakes and older clients that predate the endpoint: AttributeError, not a crash.
    service = _service(tmp_path, object())

    assert asyncio.run(service.report_bind_started("VB-7K3QX9MZ")) is False


class _BindProbe:
    """Stands in for the service `handle_bind` builds, recording the order of calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def start_binding(self, **_: Any) -> Any:
        self.calls.append("start")
        return type("Session", (), {"qr_payload_json": '{"pollID":"p"}'})()

    async def report_bind_started(self, setup_code: str) -> bool:
        self.calls.append(f"report:{setup_code}")
        return True

    async def poll_until_bound(self, **_: Any) -> Any:
        self.calls.append("poll")
        return type("Result", (), {"status": "bound", "trial_ends_at": None, "server_id": "s"})()


def _run_bind(monkeypatch: Any, tmp_path: Path, *extra: str) -> _BindProbe:
    probe = _BindProbe()
    monkeypatch.setattr(cli, "_service", lambda args: probe)
    monkeypatch.setattr(cli, "_print_qr", lambda payload: probe.calls.append("qr"))
    try:
        cli.main(["--config", str(tmp_path / "config.json"), "bind", *extra])
    except Exception:  # noqa: BLE001 - only the order of the calls above is under test
        pass
    return probe


def test_bind_reports_the_code_before_drawing_the_qr(monkeypatch: Any, tmp_path: Path) -> None:
    probe = _run_bind(monkeypatch, tmp_path, "--setup-code", "VB-7K3QX9MZ")

    assert probe.calls[:3] == ["start", "report:VB-7K3QX9MZ", "qr"]
    assert "poll" in probe.calls, "the bind goes on to wait for the scan"


def test_bind_without_a_code_reports_nothing(monkeypatch: Any, tmp_path: Path) -> None:
    probe = _run_bind(monkeypatch, tmp_path)

    assert not any(call.startswith("report:") for call in probe.calls)
    assert probe.calls[:2] == ["start", "qr"]
