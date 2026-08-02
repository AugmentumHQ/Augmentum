"""Live Cardsmith audit harness.

Runs 5 representative Cardsmith scenarios end-to-end against the running
Augmentum container, captures per-turn transcripts + state snapshots +
final card data, then generates a markdown summary flagging anomalies.

# Why a separate harness vs. pytest tests

Unit tests cover protocol mechanics (parser, output mapper, scratchpad)
deterministically. This harness is the OTHER half — it validates the
Cardsmith works end-to-end with REAL LLM calls. Outputs are saved so
you (or I) can review what the model actually produced for each scenario,
not just whether the code paths are sound.

# Usage

    # One-time: log in via the UI and copy the session cookie value, OR
    # set AUGMENTUM_USERNAME + AUGMENTUM_PASSWORD env vars.
    export AUGMENTUM_USERNAME=admin
    export AUGMENTUM_PASSWORD=...

    # Run all 5 scenarios:
    python tests/live/cardsmith_audit.py

    # Run a single scenario:
    python tests/live/cardsmith_audit.py --scenario single_describe

    # Custom base URL:
    python tests/live/cardsmith_audit.py --url http://localhost:6100

# Output

Results land in ``tests/live/cardsmith_audits/<timestamp>/`` with one
sub-directory per scenario:

    transcript.md      — human-readable conversation log
    state_per_turn.json — committed fields + scratchpad snapshots per turn
    final_card.json    — what got persisted to ui_characters
    sse_log.jsonl      — every SSE event received (for low-level debugging)

A top-level ``summary.md`` flags anomalies across all scenarios.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Force UTF-8 stdout/stderr — Windows defaults to cp1252 which can't render
# em-dashes / arrows / bullets that show up in scenario replies + transcripts.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import httpx

# ── Scenario definitions ──────────────────────────────────────────────────

@dataclass
class Scenario:
    """One Cardsmith conversation to drive end-to-end.

    user_replies are scripted — the harness sends them in order whenever
    the Cardsmith finishes a turn. Most replies are short affirmations;
    the model reads them as "advance to the next question."
    """

    name: str
    card_type: str  # single | ensemble | world_rpg
    source: str  # describe | wiki | blank
    seed_prompt: str
    wiki_url: str = ""  # required when source=wiki
    user_replies: list[str] = field(default_factory=list)
    max_turns: int = 14
    # Soft expectations for the audit reporter
    expect_lorebook_min: int = 0
    expect_members_min: int = 0


SCENARIOS: list[Scenario] = [
    Scenario(
        name="single_describe",
        card_type="single",
        source="describe",
        seed_prompt="A reclusive cyberpunk medic hiding from her old crew. Soft for strays, deadly with a scalpel.",
        user_replies=[
            "Lyra Vex.",
            "Looks great. Maybe make her a bit younger — late twenties.",
            "Yes, that captures her.",
            "She carries her dead sister's compass everywhere.",
            "Yes those examples work.",
            "She's a stranger to {{user}} — they meet in a rain-streaked diner at 2am.",
            "Skip alternates for now.",
            "Skip the extras.",
            "Save it.",
        ],
    ),
    Scenario(
        name="single_wiki",
        card_type="single",
        source="wiki",
        wiki_url="https://naruto.fandom.com/wiki/Sasuke_Uchiha",
        seed_prompt="I want him pre-defection — still on Team 7 but already starting to crack.",
        user_replies=[
            "Yes, canon-faithful with the pre-defection twist.",
            "That looks right. Continue.",
            "Yes the personality matches.",
            "Skip the extra depth pass.",
            "Yes those examples work.",
            "Rival to {{user}} — they're both genin on Team 7, training partners.",
            "Skip alternates.",
            "Skip the extras.",
            "Save it.",
        ],
    ),
    Scenario(
        name="ensemble_describe",
        card_type="ensemble",
        source="describe",
        seed_prompt="An adventuring party of 4 — a paladin leader, sneaky rogue, cynical cleric, naive ranger.",
        user_replies=[
            "The Cinderhalls Crew.",
            "Names: Kira (paladin lead), Jek (rogue), Marn (cleric), Tess (ranger).",
            "Kira leads but Marn's cynicism keeps her honest. Jek and Tess argue constantly. There's tension between Kira and Jek over a past betrayal.",
            "Kira: tall, scarred face, mid-30s, plate armor. Pragmatic. Speaks in clipped imperatives.",
            "Jek: short, wiry, hood pulled low. Sarcastic, secretive. Always counting coin.",
            "Marn: heavy-set, grey-streaked beard. Speaks slowly. Has seen too much war.",
            "Tess: lithe, freckled, late teens. Bright-eyed but quick to anger.",
            "Kira-Jek: trust 0.3, affection -0.2, tension 0.7, label 'unresolved past'. Marn-Kira: trust 0.8, affection 0.6, tension 0.1. Jek-Tess: bickering siblings.",
            "They meet {{user}} at a wayside inn after a job gone wrong.",
            "Yes those examples work.",
            "llm_decide for the speaker.",
            "Skip the extras.",
            "Save it.",
        ],
        max_turns=18,
        expect_members_min=3,
    ),
    Scenario(
        name="world_wiki_fandom",
        card_type="world_rpg",
        source="wiki",
        wiki_url="https://tbate.fandom.com/wiki/Arthur_Leywin",
        seed_prompt="Build me a TBATE RPG centered on Xyrus Academy era — student protagonist.",
        user_replies=[
            "Yes, Xyrus Academy era is the right scope.",
            "Pull the magic system articles next: Mana, Aether, and the Lance Order.",
            "Fetch Sapin Kingdom and Elenoir Kingdom too.",
            "Looks good — make those into lorebook entries.",
            "Yes a narrator in third-person past tense, detached, observational.",
            "{{user}} is a transfer student arriving at Xyrus mid-semester.",
            "Skip alternates.",
            "Skip extras.",
            "Save it.",
        ],
        max_turns=14,
        expect_lorebook_min=5,
    ),
    Scenario(
        name="world_wiki_wikipedia",
        card_type="world_rpg",
        source="wiki",
        wiki_url="https://en.wikipedia.org/wiki/Middle-earth",
        seed_prompt="A Middle-earth campaign set during the Third Age, post-Smaug, pre-Council of Elrond.",
        user_replies=[
            "Yes, Third Age post-Smaug is the right window.",
            "Fetch the Shire and Rohan articles.",
            "Pull Gandalf and Aragorn too.",
            "Build the lorebook — focus on places + key NPCs.",
            "An impersonal omniscient narrator, no GM persona.",
            "{{user}} is a hobbit traveler from Bree, drawn into events.",
            "Skip alternates.",
            "Skip extras.",
            "Save it.",
        ],
        max_turns=14,
        expect_lorebook_min=4,
    ),
]


# ── Auth ──────────────────────────────────────────────────────────────────

async def login(client: httpx.AsyncClient, base_url: str) -> str:
    """Authenticate against /api/auth/login and return the session cookie value."""
    username = os.environ.get("AUGMENTUM_USERNAME") or os.environ.get("AUGMENTUM_USER")
    password = os.environ.get("AUGMENTUM_PASSWORD") or os.environ.get("AUGMENTUM_PASS")
    if not username or not password:
        raise RuntimeError(
            "Set AUGMENTUM_USERNAME + AUGMENTUM_PASSWORD env vars before running.\n"
            "These are the credentials you use to log in via the Augmentum UI."
        )

    resp = await client.post(
        f"{base_url}/api/auth/login",
        json={"username": username, "password": password},
        timeout=10.0,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Login failed: HTTP {resp.status_code} — {resp.text[:200]}"
        )
    # The session cookie is set by the server. httpx.AsyncClient persists
    # cookies across requests automatically.
    if "augmentum_session" not in client.cookies:
        raise RuntimeError("Login succeeded but no session cookie set")
    return client.cookies["augmentum_session"]


# ── Per-turn driver ───────────────────────────────────────────────────────

@dataclass
class TurnRecord:
    turn_num: int
    user_message: str
    assistant_text: str = ""
    field_emissions: list[dict[str, Any]] = field(default_factory=list)
    fetching_events: list[dict[str, Any]] = field(default_factory=list)
    fetched_count: int = 0
    finalized_event: dict[str, Any] | None = None
    error: str = ""
    sse_events: list[dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0


async def run_turn(
    client: httpx.AsyncClient,
    base_url: str,
    session_id: str,
    user_message: str,
    turn_num: int,
    *,
    model: str = "",
) -> TurnRecord:
    """Drive one /turn POST, consume the SSE stream, return a TurnRecord."""
    record = TurnRecord(turn_num=turn_num, user_message=user_message)
    start = time.time()
    text_buffer: list[str] = []
    body = {"session_id": session_id, "user_message": user_message}
    if model:
        body["model"] = model
    try:
        async with client.stream(
            "POST",
            f"{base_url}/api/characters/cardsmith/turn",
            json=body,
            timeout=httpx.Timeout(180.0),
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                record.error = f"HTTP {resp.status_code}: {body.decode('utf-8', errors='replace')[:300]}"
                return record

            buffer = ""
            async for chunk in resp.aiter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    raw, buffer = buffer.split("\n\n", 1)
                    line = raw.strip()
                    if not line.startswith("data:"):
                        continue
                    payload_str = line[len("data:"):].strip()
                    if payload_str == "[DONE]":
                        continue
                    try:
                        payload = json.loads(payload_str)
                    except json.JSONDecodeError:
                        continue
                    record.sse_events.append(payload)
                    kind = payload.get("type")
                    if kind == "delta":
                        text_buffer.append(payload.get("text", ""))
                    elif kind == "field":
                        record.field_emissions.append(
                            {"path": payload.get("path"), "value": payload.get("value")}
                        )
                    elif kind == "fetching":
                        record.fetching_events.append(payload)
                    elif kind == "fetched":
                        record.fetched_count += int(payload.get("count") or 0)
                    elif kind == "finalized":
                        record.finalized_event = payload
                    elif kind == "error":
                        record.error = payload.get("error", "")
    except httpx.HTTPError as exc:
        record.error = f"HTTP error: {type(exc).__name__}: {exc!r}"
    except Exception as exc:
        record.error = f"unexpected: {type(exc).__name__}: {exc!r}"

    record.assistant_text = "".join(text_buffer).strip()
    record.duration_seconds = time.time() - start
    return record


# ── Session state probing ─────────────────────────────────────────────────

async def fetch_saved_card(
    client: httpx.AsyncClient, base_url: str, char_id: str,
) -> dict[str, Any]:
    """Pull the saved character's full record."""
    resp = await client.get(
        f"{base_url}/api/characters/{char_id}", timeout=10.0,
    )
    if resp.status_code != 200:
        return {"_error": f"HTTP {resp.status_code}"}
    return resp.json()


# ── Scenario runner ───────────────────────────────────────────────────────

@dataclass
class ScenarioResult:
    scenario: Scenario
    session_id: str = ""
    turns: list[TurnRecord] = field(default_factory=list)
    final_card: dict[str, Any] = field(default_factory=dict)
    char_id: str = ""
    finalized: bool = False
    fatal_error: str = ""


async def run_scenario(
    client: httpx.AsyncClient,
    base_url: str,
    scenario: Scenario,
    *,
    model: str = "",
) -> ScenarioResult:
    result = ScenarioResult(scenario=scenario)

    # 1. Start session
    start_body = {
        "card_type": scenario.card_type,
        "source": scenario.source,
        "seed_prompt": scenario.seed_prompt,
    }
    if scenario.source == "wiki":
        start_body["wiki_url"] = scenario.wiki_url

    try:
        resp = await client.post(
            f"{base_url}/api/characters/cardsmith/start",
            json=start_body,
            timeout=60.0,
        )
        if resp.status_code != 200:
            result.fatal_error = f"start failed: HTTP {resp.status_code} — {resp.text[:200]}"
            return result
        result.session_id = resp.json()["session_id"]
    except httpx.HTTPError as exc:
        result.fatal_error = f"start http error: {type(exc).__name__}: {exc!r}"
        return result
    except Exception as exc:
        result.fatal_error = f"start unexpected: {type(exc).__name__}: {exc!r}"
        return result

    # 2. Drive turns
    reply_idx = 0
    for turn_num in range(1, scenario.max_turns + 1):
        # Pick the user reply: empty for first turn (Cardsmith opens), else
        # the next scripted reply, else a generic affirmation if we've run out.
        if turn_num == 1:
            user_msg = ""
        elif reply_idx < len(scenario.user_replies):
            user_msg = scenario.user_replies[reply_idx]
            reply_idx += 1
        else:
            user_msg = "Sounds good, continue."

        rec = await run_turn(
            client, base_url, result.session_id, user_msg, turn_num,
            model=model,
        )
        result.turns.append(rec)

        if rec.error:
            print(f"  ! turn {turn_num} error: {rec.error[:120]}")
            break
        if rec.finalized_event:
            result.finalized = True
            result.char_id = rec.finalized_event.get("char_id", "")
            break

        print(
            f"  [ok] turn {turn_num:>2}  ({rec.duration_seconds:>5.1f}s)  "
            f"{len(rec.field_emissions)} field commits"
            + (f" · fetched {rec.fetched_count}" if rec.fetched_count else "")
        )

    # 3. If we ran out of turns without a [CARDSMITH_DONE], force finalize.
    if not result.finalized and not result.fatal_error and result.session_id:
        try:
            resp = await client.post(
                f"{base_url}/api/characters/cardsmith/finalize",
                json={"session_id": result.session_id},
                timeout=120.0,  # allows recovery extraction
            )
            if resp.status_code == 200:
                result.finalized = True
                result.char_id = resp.json().get("char_id", "")
            else:
                result.fatal_error = (
                    f"forced finalize failed: HTTP {resp.status_code} — {resp.text[:200]}"
                )
        except httpx.HTTPError as exc:
            result.fatal_error = (
                f"finalize http error: {type(exc).__name__}: {exc!r}"
            )
        except Exception as exc:
            result.fatal_error = (
                f"finalize unexpected: {type(exc).__name__}: {exc!r}"
            )

    # 4. Pull the saved card if we have an id.
    if result.char_id:
        result.final_card = await fetch_saved_card(client, base_url, result.char_id)

    return result


# ── Audit / report rendering ──────────────────────────────────────────────

def render_transcript(result: ScenarioResult) -> str:
    """Markdown transcript — one section per turn."""
    sc = result.scenario
    lines = [
        f"# {sc.name}",
        "",
        f"- **card_type**: `{sc.card_type}`",
        f"- **source**: `{sc.source}`",
        f"- **seed_prompt**: {sc.seed_prompt!r}",
    ]
    if sc.wiki_url:
        lines.append(f"- **wiki_url**: {sc.wiki_url}")
    lines.extend([
        f"- **session_id**: `{result.session_id}`",
        f"- **finalized**: {result.finalized}",
        f"- **char_id**: `{result.char_id or '—'}`",
        f"- **total turns**: {len(result.turns)}",
        "",
    ])

    if result.fatal_error:
        lines.append(f"## ⚠️ Fatal error\n\n{result.fatal_error}\n")

    for rec in result.turns:
        lines.append(f"## Turn {rec.turn_num} ({rec.duration_seconds:.1f}s)")
        lines.append("")
        if rec.user_message:
            lines.append(f"**user**: {rec.user_message}")
            lines.append("")
        if rec.assistant_text:
            lines.append("**Cardsmith**:")
            lines.append("")
            for chunk in rec.assistant_text.split("\n"):
                lines.append("> " + chunk if chunk else ">")
            lines.append("")
        if rec.field_emissions:
            lines.append("### Field commits this turn")
            lines.append("")
            for emission in rec.field_emissions:
                value = emission.get("value", "")
                if isinstance(value, str) and len(value) > 120:
                    value = value[:120] + "…"
                lines.append(f"- `{emission['path']}` → `{value}`")
            lines.append("")
        if rec.fetching_events:
            for ev in rec.fetching_events:
                targets = ev.get("targets", [])
                lines.append(
                    f"### Fetched ({rec.fetched_count}) — requested: {', '.join(targets[:5])}"
                )
                lines.append("")
        if rec.error:
            lines.append(f"### ⚠️ Error\n\n{rec.error}\n")

    return "\n".join(lines)


def audit_anomalies(result: ScenarioResult) -> list[str]:
    """Per-scenario anomaly checks. Returns a list of human-readable warnings."""
    anomalies: list[str] = []
    sc = result.scenario

    if result.fatal_error:
        anomalies.append(f"Fatal: {result.fatal_error}")
        return anomalies

    if not result.finalized:
        anomalies.append("Did NOT finalize within max_turns + forced finalize")

    if not result.char_id or not result.final_card:
        anomalies.append("No final card retrieved")
        return anomalies

    data = result.final_card
    name = data.get("name", "")
    if not name or name == "New Character":
        anomalies.append("Card has default/empty name")

    # Critical fields by card type.
    desc = data.get("description", "")
    if not desc.strip():
        anomalies.append("description is empty")
    elif len(desc) < 80:
        anomalies.append(f"description suspiciously short ({len(desc)} chars)")

    greeting = data.get("greeting", "")
    if not greeting.strip():
        anomalies.append("greeting is empty")

    if sc.card_type == "ensemble":
        members = data.get("extensions", {}).get("augmentum", {}).get("cardsmith", {}).get("members", [])
        if len(members) < sc.expect_members_min:
            anomalies.append(
                f"only {len(members)} ensemble members (expected ≥ {sc.expect_members_min})"
            )

    lorebook = data.get("lorebook") or []
    if sc.expect_lorebook_min and len(lorebook) < sc.expect_lorebook_min:
        anomalies.append(
            f"only {len(lorebook)} lorebook entries (expected ≥ {sc.expect_lorebook_min})"
        )

    # Recovery signal: did the model fail the inline-tag protocol enough that
    # finalize had to run recovery extraction? We can detect this by looking at
    # whether assistant turns had zero field emissions but description wound up
    # populated anyway.
    turns_with_text = [t for t in result.turns if t.assistant_text]
    turns_with_emissions = [t for t in turns_with_text if t.field_emissions]
    if turns_with_text and len(turns_with_emissions) < len(turns_with_text) // 2:
        anomalies.append(
            f"only {len(turns_with_emissions)}/{len(turns_with_text)} model turns "
            "emitted field commits — protocol drift likely"
        )

    return anomalies


def render_summary(results: list[ScenarioResult], output_dir: Path) -> str:
    lines = [
        "# Cardsmith Audit Summary",
        "",
        f"_Generated {datetime.now().isoformat(timespec='seconds')}_",
        "",
        "| Scenario | Type | Source | Turns | Finalized | Lorebook | Members | Anomalies |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        sc = r.scenario
        anomalies = audit_anomalies(r)
        lb = len((r.final_card or {}).get("lorebook") or [])
        members = len(
            (r.final_card or {}).get("extensions", {})
            .get("augmentum", {}).get("cardsmith", {}).get("members", [])
        )
        anomaly_summary = "[ok] clean" if not anomalies else f"⚠️ {len(anomalies)}"
        lines.append(
            f"| `{sc.name}` | {sc.card_type} | {sc.source} | {len(r.turns)} | "
            f"{'[ok]' if r.finalized else '[fail]'} | {lb} | {members or '—'} | {anomaly_summary} |"
        )

    lines.append("")
    lines.append("## Anomaly detail")
    lines.append("")
    for r in results:
        anomalies = audit_anomalies(r)
        if not anomalies:
            continue
        lines.append(f"### `{r.scenario.name}`")
        lines.append("")
        for a in anomalies:
            lines.append(f"- {a}")
        lines.append("")

    lines.append("## Files")
    lines.append("")
    for r in results:
        lines.append(f"- `{r.scenario.name}/transcript.md`")
        lines.append(f"- `{r.scenario.name}/state_per_turn.json`")
        lines.append(f"- `{r.scenario.name}/final_card.json`")

    return "\n".join(lines)


# ── Persistence ───────────────────────────────────────────────────────────

def save_scenario_artifacts(result: ScenarioResult, scenario_dir: Path) -> None:
    scenario_dir.mkdir(parents=True, exist_ok=True)

    # Transcript
    (scenario_dir / "transcript.md").write_text(
        render_transcript(result), encoding="utf-8",
    )

    # Per-turn state
    state_per_turn = []
    for rec in result.turns:
        rec_dict = asdict(rec)
        # Drop the noisy SSE log from the per-turn JSON; live in sse_log.jsonl
        rec_dict.pop("sse_events", None)
        state_per_turn.append(rec_dict)
    (scenario_dir / "state_per_turn.json").write_text(
        json.dumps(state_per_turn, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # Final card
    if result.final_card:
        (scenario_dir / "final_card.json").write_text(
            json.dumps(result.final_card, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Raw SSE log (one event per line)
    with (scenario_dir / "sse_log.jsonl").open("w", encoding="utf-8") as f:
        for rec in result.turns:
            for ev in rec.sse_events:
                f.write(
                    json.dumps({"turn": rec.turn_num, **ev}, ensure_ascii=False) + "\n"
                )


# ── CLI entry ─────────────────────────────────────────────────────────────

async def main() -> int:
    parser = argparse.ArgumentParser(description="Live Cardsmith audit harness")
    parser.add_argument("--url", default="http://localhost:6100", help="Augmentum base URL")
    parser.add_argument(
        "--scenario", help="Run only the named scenario (default: run all)",
    )
    parser.add_argument(
        "--model", default="",
        help="Override the model used for /turn (e.g. 'qwen3-4b'). "
             "Default: server-resolved utility model.",
    )
    parser.add_argument(
        "--output", default="tests/live/cardsmith_audits",
        help="Output base directory",
    )
    args = parser.parse_args()

    selected = SCENARIOS
    if args.scenario:
        selected = [s for s in SCENARIOS if s.name == args.scenario]
        if not selected:
            print(f"No scenario named '{args.scenario}'. Available:")
            for s in SCENARIOS:
                print(f"  - {s.name}")
            return 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")
    print()

    # Origin header matches the request Host so the server's CSRF middleware
    # waves us through. Without this, every state-changing POST gets 403'd.
    headers = {"Origin": args.url.rstrip("/"), "Referer": args.url.rstrip("/") + "/"}

    async with httpx.AsyncClient(follow_redirects=True, headers=headers) as client:
        try:
            await login(client, args.url)
            print("[ok] Logged in\n")
        except Exception as exc:
            print(f"[fail] Login failed: {exc}")
            return 1

        results: list[ScenarioResult] = []
        if args.model:
            print(f"Model override: {args.model}\n")
        for sc in selected:
            print(f"== {sc.name} ({sc.card_type} / {sc.source}) ==")
            try:
                result = await run_scenario(client, args.url, sc, model=args.model)
            except Exception as exc:
                print(f"  [fail] Unexpected: {exc}")
                result = ScenarioResult(scenario=sc, fatal_error=str(exc))
            save_scenario_artifacts(result, output_dir / sc.name)
            results.append(result)
            print()

        # Summary
        summary = render_summary(results, output_dir)
        (output_dir / "summary.md").write_text(summary, encoding="utf-8")
        print("======================================")
        print(summary)
        print()
        print(f"Full results: {output_dir}")

    # Exit non-zero if any scenario had a fatal error.
    fatal_count = sum(1 for r in results if r.fatal_error)
    return 1 if fatal_count else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
