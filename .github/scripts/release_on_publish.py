#!/usr/bin/env python3
"""Tag and publish a GitHub Release for whatever version PyPI actually serves.

Runs INSIDE the public repo (Fino-wind/vaultbeat-apple-health) from
.github/workflows/release-on-publish.yml. Its source of truth is the monorepo's
`mcp-local-server/.github/`, and it reaches the public repo through the normal
export — so no sync can overwrite it (a guard living downstream of a sync
protects nothing against the sync).

WHY THIS EXISTS

  The public repo's "Latest release" sat at v0.3.11 from 2026-08-16 to
  2026-09-29 while PyPI went to 0.8.1 — 21 versions. Anyone arriving from the
  MCP Registry, PyPI or Glama saw a project that looked abandoned, and the
  stale release still told them to install the old package name. Publishing a
  Release was a written step; nothing ran it, and nothing noticed it was
  skipped. So it is no longer a step — this script does it.

WHAT IT DOES

  1. Reads the version V from pyproject.toml on main.
  2. Tag vV and its Release both exist -> nothing to do. A tag with no Release
                               (someone tagged by hand) gets its Release here,
                               after the same byte check below.
  3. PyPI has no V yet      -> nothing to do; on the daily run, alerts if the
                               public repo has claimed V for over 48 hours.
  4. Otherwise downloads the sdist PyPI serves for V and finds the commit on
     main whose src/ is BYTE-IDENTICAL to it. It tags THAT commit, not HEAD —
     docs-only commits land after a publish and are not in the package.
  5. No commit matches      -> Bark alert + red run. Never tags a guess.
  6. Also alerts (without blocking the release) when server.json's version
     fields disagree with pyproject — that drift once went two releases unseen.

DRY_RUN=1 prints what it would do and pushes, creates and sends nothing.
Stdlib only: it runs on a bare ubuntu-latest python3.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path.cwd()
DRY_RUN = os.environ.get("DRY_RUN") == "1"
BARK_URL = "https://push.globaltop.top/push"
MCP_SERVER_KEY = "vaultbeat-health"  # the name users give it in `claude mcp add`
STALE_CLAIM_S = 48 * 3600
MAX_CANDIDATES = 200


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout


def pyproject_field(text: str, key: str) -> str:
    m = re.search(rf'^{key}\s*=\s*"([^"]+)"', text, re.M)
    if not m:
        raise SystemExit(f"FAIL: no `{key}` in pyproject.toml")
    return m.group(1)


def pyproject_url(text: str, key: str) -> str | None:
    block = re.search(r"^\[project\.urls\]\n(.*?)(?:^\[|\Z)", text, re.M | re.S)
    if not block:
        return None
    m = re.search(rf'^{key}\s*=\s*"([^"]+)"', block.group(1), re.M)
    return m.group(1) if m else None


def fetch(url: str) -> bytes | None:
    """GET, or None on 404. Any other failure raises: an outage is not a verdict."""
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data: bytes = resp.read()
            return data
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def notify(title: str, body: str) -> None:
    print(f"ALERT: {title}\n{body}")
    key = os.environ.get("BARK_KEY")
    if DRY_RUN or not key:
        print("(Bark not sent: dry run or no BARK_KEY)")
        return
    payload = json.dumps(
        {"device_key": key, "title": title, "body": body, "group": "MCP 发版守卫"},
        ensure_ascii=False,
    ).encode()
    req = urllib.request.Request(
        BARK_URL, data=payload, headers={"Content-Type": "application/json; charset=utf-8"}
    )
    try:
        urllib.request.urlopen(req, timeout=20).read()
    except Exception as e:  # the alert failing must not hide the original problem
        print(f"(Bark send failed: {type(e).__name__})")


def sdist_src_hashes(name: str, version: str) -> dict[str, str]:
    meta = json.loads(fetch(f"https://pypi.org/pypi/{name}/{version}/json") or b"{}")
    urls = [u["url"] for u in meta.get("urls", []) if u.get("packagetype") == "sdist"]
    if not urls:
        raise SystemExit(f"FAIL: PyPI {name} {version} has no sdist to compare against")
    blob = fetch(urls[0])
    assert blob is not None
    out: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for m in tar.getmembers():
            parts = m.name.split("/", 1)
            if len(parts) < 2 or not m.isfile():
                continue
            rel = parts[1]
            if not rel.startswith("src/") or ".egg-info/" in rel:
                continue
            f = tar.extractfile(m)
            assert f is not None
            out[rel] = hashlib.sha256(f.read()).hexdigest()
    return out


def commit_src_hashes(sha: str) -> dict[str, str]:
    blob = subprocess.run(
        ["git", "archive", sha, "src"], cwd=REPO, check=True, capture_output=True
    ).stdout
    out: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:") as tar:
        for m in tar.getmembers():
            if m.isfile():
                f = tar.extractfile(m)
                assert f is not None
                out[m.name] = hashlib.sha256(f.read()).hexdigest()
    return out


def commits_claiming(version: str) -> list[str]:
    """Commits on main whose pyproject says `version`, oldest first."""
    found: list[str] = []
    for sha in git("rev-list", f"--max-count={MAX_CANDIDATES}", "HEAD").split():
        try:
            text = git("show", f"{sha}:pyproject.toml")
        except subprocess.CalledProcessError:
            break
        m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
        if not m or m.group(1) != version:
            break
        found.append(sha)
    return list(reversed(found))


def release_notes(name: str, version: str, sha: str, pyproject: str, registry: str) -> tuple[str, str]:
    subject = git("show", "-s", "--format=%s", sha).strip()
    body_lines = [
        line for line in git("show", "-s", "--format=%b", sha).splitlines()
        if not line.lower().startswith("co-authored-by:")
    ]
    body = "\n".join(body_lines).strip()
    headline = re.sub(rf"^(Release\s+)?{re.escape(version)}\s*[:—-]\s*", "", subject).strip()
    title = f"v{version} — {headline}" if headline else f"v{version}"

    prev = ""
    try:
        prev = git("describe", "--tags", "--abbrev=0", "--match", "v*", f"{sha}^").strip()
    except subprocess.CalledProcessError:
        pass
    history = ""
    if prev:
        subjects = [
            s for s in git("log", "--format=%s", f"{prev}..{sha}").splitlines()
            if re.match(r"^(Release\s+)?\d+\.\d+\.\d+", s)
        ]
        if subjects:
            history = f"\n### Every release since {prev}\n\n" + "\n".join(f"- {s}" for s in subjects) + "\n"

    desc = pyproject_field(pyproject, "description")
    docs = pyproject_url(pyproject, "Documentation")
    links = [f"- PyPI: [`{name}`](https://pypi.org/project/{name}/{version}/)"]
    if registry:
        links.append(f"- MCP Registry: `{registry}`")
    if docs:
        links.append(f"- Docs: {docs}")

    notes = (
        f"{desc}\n\n"
        "```\n"
        f"uvx {name}@latest bind     # draws a QR code — scan it from the Vaultbeat iOS app\n"
        f"claude mcp add {MCP_SERVER_KEY} -- uvx {name}@latest serve --transport stdio\n"
        "```\n\n"
        f"### What changed in {version}\n\n"
        f"{body or subject}\n"
        f"{history}\n"
        "### Links\n\n" + "\n".join(links) + "\n\n"
        f"<sub>Generated from commit `{sha[:7]}`, whose `src/` was verified byte-identical "
        f"to the sdist PyPI serves for {version}.</sub>\n"
    )
    return title, notes


def main() -> int:
    pyproject = (REPO / "pyproject.toml").read_text()
    name = pyproject_field(pyproject, "name")
    version = pyproject_field(pyproject, "version")
    tag = f"v{version}"
    event = os.environ.get("GITHUB_EVENT_NAME", "local")
    problems = 0

    server = json.loads((REPO / "server.json").read_text()) if (REPO / "server.json").exists() else {}
    registry = str(server.get("name", ""))
    server_versions = {str(server.get("version"))} | {
        str(p.get("version")) for p in server.get("packages", []) if isinstance(p, dict)
    }
    if server and server_versions != {version}:
        notify(
            f"{name}: server.json 没跟上",
            f"pyproject 是 {version}，server.json 写的是 {sorted(server_versions)}。"
            "改两个 version 字段 → mcp-publisher publish → 再 commit。",
        )
        problems += 1

    git("fetch", "--tags", "--quiet", "origin")
    tag_exists = bool(git("tag", "--list", tag).strip())
    if tag_exists:
        has_release = subprocess.run(
            ["gh", "release", "view", tag, "--json", "tagName"],
            cwd=REPO, capture_output=True, text=True,
        ).returncode == 0
        if has_release:
            print(f"OK: {tag} and its Release already exist — nothing to do")
            return 1 if problems else 0
        print(f"{tag} exists without a Release — creating one on the tagged commit")

    if fetch(f"https://pypi.org/pypi/{name}/{version}/json") is None:
        claimed = commits_claiming(version)
        age = time.time() - int(git("show", "-s", "--format=%ct", claimed[0]).strip()) if claimed else 0
        if event == "schedule" and age > STALE_CLAIM_S:
            notify(
                f"{name}: 公开 repo 说 {version}，PyPI 上没有",
                f"main 从 {int(age // 3600)} 小时前起就写着 {version}，但 PyPI 查不到这个版本。"
                "要么发布失败了，要么忘了发。",
            )
            return 1
        print(f"OK: {version} is not on PyPI yet — nothing to release")
        return 1 if problems else 0

    target = sdist_src_hashes(name, version)
    candidates = (
        [git("rev-list", "-n", "1", tag).strip()] if tag_exists else commits_claiming(version)
    )
    match = next((sha for sha in candidates if commit_src_hashes(sha) == target), None)
    if match is None:
        notify(
            f"{name} {version}: 找不到跟 PyPI 一致的 commit",
            f"PyPI 上 {version} 的 src/ 跟 main 上任何一个写着 {version} 的 commit 都对不上。"
            "说明发布用的代码没提交，或者提交后又改了。没有打 tag，需要人看。",
        )
        return 1

    title, notes = release_notes(name, version, match, pyproject, registry)
    print(f"RELEASE: {tag} at {match[:7]}\nTITLE: {title}\n\n{notes}")
    if DRY_RUN:
        print("(dry run: no tag pushed, no release created)")
        return 1 if problems else 0

    if not tag_exists:
        git("tag", "-a", tag, match, "-m", f"{title}\n\nsrc/ verified byte-identical to PyPI {version}.")
        git("push", "origin", tag)
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
        fh.write(notes)
    subprocess.run(
        ["gh", "release", "create", tag, "--verify-tag", "--latest",
         "--title", title, "--notes-file", fh.name],
        cwd=REPO, check=True,
    )
    print(f"DONE: released {tag}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
