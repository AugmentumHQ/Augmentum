"""Live probe harness for Augmentum Powers.

Usage examples:

  python scripts/powers_live_probe.py --list-scenarios
  python scripts/powers_live_probe.py --username shadow --password secret --scenario contract-keeper
  python scripts/powers_live_probe.py --username shadow --password secret --scenario all --output-dir power-reports
  python scripts/powers_live_probe.py --username shadow --password secret --prompt "Create a small FastAPI route and matching frontend fetch example."
  python scripts/powers_live_probe.py --username shadow --password secret --git-url https://github.com/pallets/click.git --prompt "Review the repo and improve one small but real issue."

This script talks to a running Augmentum instance over HTTP(S). It is designed
for interactive evaluation rather than CI. It:

- logs in if auth is enabled
- lists and optionally enables/disables Powers
- optionally activates a manual Power for the workspace
- runs a coder-mode turn in a disposable workspace
- captures streamed `power_activated` events, tool calls, statuses, and text
- writes a JSON report for later review
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_URL = "http://localhost:6100"
DEFAULT_TIMEOUT = 120


@dataclass(slots=True)
class ProbeScenario:
    slug: str
    description: str
    prompt: str
    expect_activation: str = ""
    manual_power: str = ""
    notes: str = ""


SCENARIOS: dict[str, ProbeScenario] = {
    "browser-verification": ProbeScenario(
        slug="browser-verification",
        description="Frontend-oriented task expected to favor browser verification near validation/finish.",
        prompt=(
            "Create a tiny static frontend with an HTML file, CSS, and a small JavaScript form flow. "
            "After implementing it, validate that the user-facing behavior makes sense and finish with "
            "a concise note about what was verified."
        ),
        expect_activation="browser-verification",
        notes="Best-effort controller activation after frontend writes land.",
    ),
    "contract-keeper": ProbeScenario(
        slug="contract-keeper",
        description="Cross-boundary frontend/backend payload work expected to activate Contract Keeper pre-plan.",
        prompt=(
            "Create a tiny backend route that returns JSON and a matching frontend fetch example that consumes "
            "that exact response shape. Keep the request/response contract explicit and internally consistent."
        ),
        expect_activation="contract-keeper",
    ),
    "failure-triage": ProbeScenario(
        slug="failure-triage",
        description="Bug/triage task intended to surface a verification failure and trigger Failure Triage.",
        prompt=(
            "Create a tiny Python module and a pytest file, then deliberately expose one failing behavior, "
            "run the relevant verification, diagnose the failure clearly, and fix it before finishing."
        ),
        expect_activation="failure-triage",
        notes="This is the least deterministic controller scenario because the model may avoid or quickly repair failures.",
    ),
    "mcp-builder": ProbeScenario(
        slug="mcp-builder",
        description="Manual integration Power for MCP design/scaffolding.",
        prompt=(
            "Design and scaffold a minimal MCP server for a simple weather-style API. Include the core file layout, "
            "one tool-shaped capability, and a short explanation of how the integration should be wired."
        ),
        manual_power="mcp-builder",
        expect_activation="mcp-builder",
    ),
    "migration-safety": ProbeScenario(
        slug="migration-safety",
        description="Schema/data-change task expected to trigger Migration Safety before and during implementation.",
        prompt=(
            "Create a tiny persistence-oriented example that adds a new column with a migration/backfill path. "
            "Include the model change, migration, and notes about rollback or safety concerns."
        ),
        expect_activation="migration-safety",
    ),
    "power-audit": ProbeScenario(
        slug="power-audit",
        description="Manual audit Power for evaluating existing skill/power content safely.",
        prompt=(
            "Audit the local compat pack augmentum-dev for what should be kept, what is risky to import directly, "
            "and how its useful parts should be rewritten into Augmentum-native Power guidance."
        ),
        manual_power="power-audit",
        expect_activation="power-audit",
    ),
    "power-forge": ProbeScenario(
        slug="power-forge",
        description="Manual workflow Power for authoring a new POWER.md package.",
        prompt=(
            "Create a new POWER.md package for a documentation-maintenance workflow. Keep it concise, "
            "model-agnostic, and organized like the existing native Powers."
        ),
        manual_power="power-forge",
        expect_activation="power-forge",
    ),
    "release-review": ProbeScenario(
        slug="release-review",
        description="Implementation task expected to trigger Release Review before termination.",
        prompt=(
            "Create a small code change, validate it, and finish with a release-minded quality gate rather than a casual summary."
        ),
        expect_activation="release-review",
    ),
    "test-author": ProbeScenario(
        slug="test-author",
        description="Focused code-and-test task intended to trigger Test Author after source edits land.",
        prompt=(
            "Create a tiny Python helper module first, then add focused pytest regression tests for the behavior you introduced. "
            "Prefer the smallest meaningful test surface over padding coverage."
        ),
        expect_activation="test-author",
    ),
    "natural-contract": ProbeScenario(
        slug="natural-contract",
        description="Natural contract-consistency request with happy-path verification.",
        prompt=(
            "Build a tiny backend endpoint and a matching HTML page that consumes its JSON response. "
            "Keep the request/response contract explicit and internally consistent, verify the happy path, "
            "and finish with a concise explanation of what you checked."
        ),
        expect_activation="contract-keeper",
    ),
    "natural-migration-from-scratch": ProbeScenario(
        slug="natural-migration-from-scratch",
        description="Natural migration task from an empty workspace.",
        prompt=(
            "Create a tiny user persistence example from scratch, then add a slug field with a safe migration "
            "and backfill path for existing rows. Keep rollback concerns explicit and stop once the minimal "
            "production-minded shape is in place."
        ),
        expect_activation="migration-safety",
    ),
    "natural-browser": ProbeScenario(
        slug="natural-browser",
        description="Natural browser-facing form flow with sanity-check verification.",
        prompt=(
            "Create a simple newsletter signup page with HTML, CSS, and JavaScript. Include client-side "
            "validation and a clear success message, then sanity-check the browser-facing flow before finishing."
        ),
        expect_activation="browser-verification",
    ),
    "natural-tests": ProbeScenario(
        slug="natural-tests",
        description="Natural focused-regression request for a tiny Python helper.",
        prompt=(
            "Create a small Python utility that turns article titles into URL slugs. Add focused regression tests "
            "for casing and whitespace behavior, run the relevant verification, and keep the implementation as "
            "small as possible while still matching the tests."
        ),
        expect_activation="test-author",
    ),
    "natural-triage-from-scratch": ProbeScenario(
        slug="natural-triage-from-scratch",
        description="Natural reproducer/debug request from an empty workspace.",
        prompt=(
            "Create a minimal reproducer module from scratch for a bug where ISO timestamps ending in Z are parsed "
            "incorrectly. Add a focused failing test, run the verification, explain the root cause, and fix it "
            "with the smallest targeted change."
        ),
        expect_activation="failure-triage",
    ),
}

BENCHMARKS: dict[str, list[str]] = {
    "natural-core": [
        "natural-contract",
        "natural-migration-from-scratch",
        "natural-browser",
        "natural-tests",
        "natural-triage-from-scratch",
    ],
}


@dataclass(slots=True)
class StreamSummary:
    assistant_text: str = ""
    power_events: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    statuses: list[dict[str, str]] = field(default_factory=list)
    raw_chunks: int = 0
    done: bool = False
    done_reason: str = ""
    stream_error: str = ""


class ProbeError(RuntimeError):
    """Probe-specific error with user-facing text."""


def _coerce_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return text


def _json_or_error(resp: requests.Response) -> Any:
    if resp.headers.get("content-type", "").startswith("application/json"):
        try:
            return resp.json()
        except Exception:
            pass
    return {"status": resp.status_code, "text": resp.text[:1000]}


def _parse_stream(response: requests.Response) -> StreamSummary:
    summary = StreamSummary()
    assistant_parts: list[str] = []
    try:
        for raw in response.iter_lines(decode_unicode=True):
            if not raw:
                continue
            summary.raw_chunks += 1
            data = _coerce_json(raw)
            if not isinstance(data, dict):
                continue

            message = data.get("message") or {}
            delta = message.get("content") or ""
            if isinstance(delta, str) and delta:
                assistant_parts.append(delta)

            if data.get("done") is True:
                summary.done = True
                summary.done_reason = str(data.get("done_reason") or "")

            aug = data.get("augmentum") or {}
            phase = str(aug.get("phase") or "")
            status = str(aug.get("status") or "")
            if phase and status:
                summary.statuses.append({"phase": phase, "status": status})
            if status == "power_activated" and isinstance(aug.get("power_activation"), dict):
                summary.power_events.append(dict(aug["power_activation"]))
            if status == "tool_call" and isinstance(aug.get("tool_call"), dict):
                tc = aug["tool_call"]
                summary.tool_calls.append(
                    {
                        "id": tc.get("id", ""),
                        "tool": tc.get("tool") or tc.get("name") or "",
                        "input": tc.get("input") or {},
                    },
                )
    except Exception as exc:
        summary.stream_error = str(exc)

    summary.assistant_text = "".join(assistant_parts).strip()
    return summary


class PowerProbeClient:
    def __init__(self, *, base_url: str, verify: bool, timeout: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.verify = verify

    def request(self, method: str, path: str, *, retries: int = 0, **kwargs) -> requests.Response:
        url = f"{self.base_url}{path}"
        kwargs.setdefault("timeout", self.timeout)
        kwargs.setdefault("verify", self.verify)
        attempt = 0
        while True:
            attempt += 1
            try:
                return self.session.request(method, url, **kwargs)
            except requests.RequestException:
                if attempt > retries:
                    raise
                time.sleep(0.5 * attempt)

    def auth_status(self) -> dict[str, Any]:
        resp = self.request("GET", "/api/auth/status", retries=3)
        if resp.status_code >= 400:
            raise ProbeError(f"Auth status failed: {resp.status_code} {resp.text[:200]}")
        data = _json_or_error(resp)
        if not isinstance(data, dict):
            raise ProbeError("Auth status returned unexpected payload")
        return data

    def login(self, username: str, password: str) -> dict[str, Any]:
        resp = self.request(
            "POST",
            "/api/auth/login",
            json={"username": username, "password": password},
        )
        if resp.status_code != 200:
            payload = _json_or_error(resp)
            raise ProbeError(f"Login failed: {payload}")
        data = _json_or_error(resp)
        if not isinstance(data, dict):
            raise ProbeError("Login returned unexpected payload")
        return data

    def list_powers(self, *, workspace_id: str = "") -> dict[str, Any]:
        suffix = f"?workspace_id={workspace_id}" if workspace_id else ""
        resp = self.request("GET", f"/api/powers{suffix}", retries=2)
        if resp.status_code != 200:
            payload = _json_or_error(resp)
            raise ProbeError(f"List powers failed: {payload}")
        data = _json_or_error(resp)
        if not isinstance(data, dict):
            raise ProbeError("Powers endpoint returned unexpected payload")
        return data

    def get_active_power(self, *, workspace_id: str = "") -> dict[str, Any]:
        suffix = f"?workspace_id={workspace_id}" if workspace_id else ""
        resp = self.request("GET", f"/api/powers/active{suffix}", retries=2)
        if resp.status_code != 200:
            payload = _json_or_error(resp)
            raise ProbeError(f"Get active power failed: {payload}")
        data = _json_or_error(resp)
        if not isinstance(data, dict):
            raise ProbeError("Active power endpoint returned unexpected payload")
        return data

    def set_enabled(self, power_id: str, enabled: bool, *, workspace_id: str = "") -> dict[str, Any]:
        suffix = "enable" if enabled else "disable"
        resp = self.request(
            "POST",
            f"/api/powers/{power_id}/{suffix}",
            json={"workspace_id": workspace_id},
        )
        if resp.status_code != 200:
            payload = _json_or_error(resp)
            raise ProbeError(f"Set enabled failed for {power_id}: {payload}")
        data = _json_or_error(resp)
        if not isinstance(data, dict):
            raise ProbeError("Enable/disable endpoint returned unexpected payload")
        return data

    def clear_active_power(self, *, workspace_id: str) -> dict[str, Any]:
        resp = self.request(
            "POST",
            "/api/powers/clear-activation",
            json={"workspace_id": workspace_id},
        )
        if resp.status_code != 200:
            payload = _json_or_error(resp)
            raise ProbeError(f"Clear active power failed: {payload}")
        data = _json_or_error(resp)
        if not isinstance(data, dict):
            raise ProbeError("Clear activation endpoint returned unexpected payload")
        return data

    def activate_power(self, power_id: str, *, workspace_id: str, reason: str = "probe manual activation") -> dict[str, Any]:
        resp = self.request(
            "POST",
            f"/api/powers/{power_id}/activate",
            json={
                "workspace_id": workspace_id,
                "source": "manual",
                "scope": "workspace",
                "reason": reason,
            },
        )
        if resp.status_code != 200:
            payload = _json_or_error(resp)
            raise ProbeError(f"Activate power failed for {power_id}: {payload}")
        data = _json_or_error(resp)
        if not isinstance(data, dict):
            raise ProbeError("Activate endpoint returned unexpected payload")
        return data

    def create_workspace(
        self,
        *,
        name: str,
        publish_ports: bool = False,
        git_url: str = "",
        git_branch: str = "",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": name,
            "publish_ports": publish_ports,
        }
        if git_url:
            payload["git_url"] = git_url
        if git_branch:
            payload["git_branch"] = git_branch
        resp = self.request(
            "POST",
            "/api/coder/workspaces",
            json=payload,
        )
        if resp.status_code not in {200, 201}:
            payload = _json_or_error(resp)
            raise ProbeError(f"Create workspace failed: {payload}")
        data = _json_or_error(resp)
        if not isinstance(data, dict) or not str(data.get("id", "")).strip():
            raise ProbeError(f"Create workspace returned unexpected payload: {data}")
        return data

    def wait_workspace_ready(self, workspace_id: str, *, timeout_s: int = 120, poll_s: float = 2.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        last_payload: dict[str, Any] = {"ready": False}
        while time.monotonic() < deadline:
            resp = self.request("GET", f"/api/coder/workspaces/{workspace_id}/ready", retries=2)
            if resp.status_code == 404:
                payload = _json_or_error(resp)
                raise ProbeError(f"Workspace disappeared before ready: {payload}")
            if resp.status_code >= 400:
                payload = _json_or_error(resp)
                raise ProbeError(f"Workspace readiness check failed: {payload}")
            data = _json_or_error(resp)
            if isinstance(data, dict):
                last_payload = data
                if data.get("ready") is True:
                    return data
            time.sleep(poll_s)
        raise ProbeError(f"Workspace did not become ready within {timeout_s}s: {last_payload}")

    def delete_workspace(self, workspace_id: str) -> dict[str, Any]:
        resp = self.request("DELETE", f"/api/coder/workspaces/{workspace_id}", retries=2)
        if resp.status_code not in {200, 204}:
            payload = _json_or_error(resp)
            raise ProbeError(f"Delete workspace failed: {payload}")
        data = _json_or_error(resp)
        return data if isinstance(data, dict) else {"deleted": True}

    def stream_coder_turn(self, *, workspace_id: str, model: str, prompt: str) -> StreamSummary:
        headers = {
            "X-Augmentum-Mode": "coder",
            "X-Augmentum-Workspace": workspace_id,
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        with self.session.post(
            f"{self.base_url}/api/chat",
            json=payload,
            headers=headers,
            timeout=self.timeout,
            verify=self.verify,
            stream=True,
        ) as resp:
            if resp.status_code != 200:
                payload = _json_or_error(resp)
                raise ProbeError(f"Coder turn failed: {payload}")
            return _parse_stream(resp)


def _workspace_id(prefix: str, slug: str) -> str:
    return f"{prefix}-{slug}-{uuid.uuid4().hex[:8]}"


def _scenario_report(
    *,
    scenario: ProbeScenario,
    workspace_id: str,
    workspace_info: dict[str, Any],
    readiness: dict[str, Any],
    model: str,
    powers_before: dict[str, Any],
    active_before: dict[str, Any],
    active_after: dict[str, Any],
    stream: StreamSummary,
    duration_s: float,
) -> dict[str, Any]:
    return {
        "scenario": asdict(scenario),
        "workspace_id": workspace_id,
        "workspace": workspace_info,
        "readiness": readiness,
        "model": model,
        "duration_s": round(duration_s, 3),
        "powers_before_count": len((powers_before.get("powers") or [])),
        "active_before": active_before,
        "active_after": active_after,
        "stream": asdict(stream),
        "did_expect_power_fire": any(
            (evt.get("id") or "") == scenario.expect_activation
            for evt in stream.power_events
        ) if scenario.expect_activation else None,
    }


def _load_credentials(args: argparse.Namespace) -> tuple[str, str]:
    username = args.username or os.environ.get("AUGMENTUM_USERNAME", "")
    password = args.password or os.environ.get("AUGMENTUM_PASSWORD", "")
    return username, password


def _resolve_scenarios(args: argparse.Namespace) -> list[ProbeScenario]:
    if args.prompt:
        slug = args.slug or "ad-hoc"
        return [
            ProbeScenario(
                slug=slug,
                description="Ad-hoc prompt from CLI",
                prompt=args.prompt,
                manual_power=args.manual_power or "",
                expect_activation=args.expect_activation or "",
            ),
        ]
    if args.benchmark:
        if args.benchmark not in BENCHMARKS:
            available = ", ".join(sorted(BENCHMARKS))
            raise ProbeError(f"Unknown benchmark '{args.benchmark}'. Available: {available}")
        return [SCENARIOS[name] for name in BENCHMARKS[args.benchmark]]
    if args.scenario == "all":
        return [SCENARIOS[name] for name in sorted(SCENARIOS)]
    if args.scenario not in SCENARIOS:
        available = ", ".join(sorted(SCENARIOS))
        raise ProbeError(f"Unknown scenario '{args.scenario}'. Available: {available}")
    return [SCENARIOS[args.scenario]]


def _write_report(output_path: Path, report: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _summarize_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    total_tools = 0
    expected = 0
    expected_hits = 0
    stream_errors = 0
    by_power: dict[str, int] = {}
    for report in reports:
        stream = report.get("stream") or {}
        tool_calls = stream.get("tool_calls") or []
        total_tools += len(tool_calls)
        if report.get("did_expect_power_fire") is not None:
            expected += 1
            if report.get("did_expect_power_fire"):
                expected_hits += 1
        if stream.get("stream_error"):
            stream_errors += 1
        for evt in stream.get("power_events") or []:
            power_id = str(evt.get("id") or "")
            if not power_id:
                continue
            by_power[power_id] = by_power.get(power_id, 0) + 1
    return {
        "scenario_count": len(reports),
        "expected_activation_hit_rate": (
            round(expected_hits / expected, 3) if expected else None
        ),
        "avg_tool_calls": round(total_tools / len(reports), 2) if reports else 0.0,
        "stream_errors": stream_errors,
        "power_event_counts": by_power,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live Augmentum Powers probe.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Augmentum base URL, e.g. http://localhost:6100")
    parser.add_argument("--model", default=os.environ.get("AUGMENTUM_MODEL", ""), help="Model to use for /api/chat coder turns")
    parser.add_argument("--username", default="", help="Login username. Falls back to AUGMENTUM_USERNAME.")
    parser.add_argument("--password", default="", help="Login password. Falls back to AUGMENTUM_PASSWORD.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Per-request timeout in seconds.")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification for HTTPS.")
    parser.add_argument("--workspace-prefix", default="powers-probe", help="Prefix for disposable workspace IDs.")
    parser.add_argument("--workspace-ready-timeout", type=int, default=240, help="Workspace readiness timeout in seconds.")
    parser.add_argument("--scenario", default="contract-keeper", help="Built-in scenario slug or 'all'.")
    parser.add_argument("--benchmark", default="", help="Named benchmark suite to run.")
    parser.add_argument("--prompt", default="", help="Ad-hoc prompt. Overrides --scenario when provided.")
    parser.add_argument("--slug", default="", help="Slug label for --prompt reports.")
    parser.add_argument("--git-url", default="", help="Optional git repository URL to clone into the probe workspace.")
    parser.add_argument("--git-branch", default="", help="Optional git branch to clone.")
    parser.add_argument("--manual-power", default="", help="Manually activate this Power for the run.")
    parser.add_argument("--expect-activation", default="", help="Expected power activation id for ad-hoc prompt mode.")
    parser.add_argument("--enable-power", action="append", default=[], help="Power id to enable before running. Repeatable.")
    parser.add_argument("--disable-power", action="append", default=[], help="Power id to disable before running. Repeatable.")
    parser.add_argument("--publish-ports", action="store_true", help="Create probe workspaces with published dev ports.")
    parser.add_argument("--keep-workspace", action="store_true", help="Keep created workspaces instead of deleting them after the run.")
    parser.add_argument("--output", default="", help="Write single report JSON to this path.")
    parser.add_argument("--output-dir", default="", help="Write one JSON report per scenario into this directory.")
    parser.add_argument("--list-scenarios", action="store_true", help="Print built-in scenario descriptions and exit.")
    parser.add_argument("--list-powers-only", action="store_true", help="Login if needed, print /api/powers, and exit.")
    return parser


def _print_scenarios() -> None:
    for name in sorted(SCENARIOS):
        scenario = SCENARIOS[name]
        manual = f" manual={scenario.manual_power}" if scenario.manual_power else ""
        expect = f" expect={scenario.expect_activation}" if scenario.expect_activation else ""
        print(f"{scenario.slug}: {scenario.description}{manual}{expect}")
    if BENCHMARKS:
        print("")
        for name in sorted(BENCHMARKS):
            print(f"[benchmark] {name}: {', '.join(BENCHMARKS[name])}")


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.list_scenarios:
        _print_scenarios()
        return 0

    client = PowerProbeClient(
        base_url=args.base_url,
        verify=not args.insecure,
        timeout=args.timeout,
    )

    try:
        status = client.auth_status()
        auth_required = not status.get("authenticated", False)
        username, password = _load_credentials(args)
        if auth_required and username and password:
            client.login(username, password)
            status = client.auth_status()
        elif auth_required and not status.get("authenticated", False):
            raise ProbeError(
                "Server requires authentication. Pass --username/--password or set "
                "AUGMENTUM_USERNAME and AUGMENTUM_PASSWORD."
            )

        if args.list_powers_only:
            payload = client.list_powers()
            print(json.dumps(payload, indent=2))
            return 0

        if not args.model:
            raise ProbeError("Model is required. Pass --model or set AUGMENTUM_MODEL.")

        scenarios = _resolve_scenarios(args)
        reports: list[dict[str, Any]] = []
        for power_id in args.enable_power:
            client.set_enabled(power_id, True)
        for power_id in args.disable_power:
            client.set_enabled(power_id, False)

        for scenario in scenarios:
            workspace_info = client.create_workspace(
                name=f"{args.workspace_prefix}-{scenario.slug}",
                publish_ports=args.publish_ports,
                git_url=args.git_url,
                git_branch=args.git_branch,
            )
            workspace_id = str(workspace_info["id"])
            readiness = client.wait_workspace_ready(
                workspace_id,
                timeout_s=args.workspace_ready_timeout,
            )

            try:
                powers_before: dict[str, Any] = {"powers": []}
                active_before: dict[str, Any] = {"active": None}
                active_after: dict[str, Any] = {"active": None}
                stream = StreamSummary()
                started = time.monotonic()

                try:
                    client.clear_active_power(workspace_id=workspace_id)
                    manual_power = args.manual_power or scenario.manual_power
                    if manual_power:
                        client.activate_power(manual_power, workspace_id=workspace_id)
                except Exception as exc:
                    stream.stream_error = f"workspace setup failed: {exc}"

                try:
                    powers_before = client.list_powers(workspace_id=workspace_id)
                except Exception as exc:
                    stream.stream_error = (
                        f"{stream.stream_error}; " if stream.stream_error else ""
                    ) + f"list_powers failed: {exc}"

                try:
                    active_before = client.get_active_power(workspace_id=workspace_id)
                except Exception as exc:
                    active_before = {"active": None, "error": str(exc)}
                    stream.stream_error = (
                        f"{stream.stream_error}; " if stream.stream_error else ""
                    ) + f"active_before failed: {exc}"

                try:
                    live_stream = client.stream_coder_turn(
                        workspace_id=workspace_id,
                        model=args.model,
                        prompt=scenario.prompt,
                    )
                    merged_error = stream.stream_error
                    if live_stream.stream_error:
                        merged_error = (
                            f"{merged_error}; " if merged_error else ""
                        ) + live_stream.stream_error
                    stream = live_stream
                    stream.stream_error = merged_error
                except Exception as exc:
                    stream.stream_error = (
                        f"{stream.stream_error}; " if stream.stream_error else ""
                    ) + f"stream failed: {exc}"
                duration_s = time.monotonic() - started

                try:
                    active_after = client.get_active_power(workspace_id=workspace_id)
                except Exception as exc:
                    active_after = {"active": None, "error": str(exc)}
                    stream.stream_error = (
                        f"{stream.stream_error}; " if stream.stream_error else ""
                    ) + f"active_after failed: {exc}"

                report = _scenario_report(
                    scenario=scenario,
                    workspace_id=workspace_id,
                    workspace_info=workspace_info,
                    readiness=readiness,
                    model=args.model,
                    powers_before=powers_before,
                    active_before=active_before,
                    active_after=active_after,
                    stream=stream,
                    duration_s=duration_s,
                )
                reports.append(report)

                if args.output_dir:
                    _write_report(Path(args.output_dir) / f"{scenario.slug}.json", report)
            finally:
                if not args.keep_workspace:
                    try:
                        client.delete_workspace(workspace_id)
                    except Exception:
                        pass

        if args.output:
            payload: dict[str, Any]
            if len(reports) == 1:
                payload = reports[0]
            else:
                payload = {"summary": _summarize_reports(reports), "reports": reports}
            _write_report(Path(args.output), payload)

        if len(reports) > 1:
            print(json.dumps({"summary": _summarize_reports(reports), "reports": reports}, indent=2))
        else:
            print(json.dumps(reports[0], indent=2))
        return 0
    except ProbeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
