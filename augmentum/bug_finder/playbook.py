"""Seeded hunting playbook — a static, shipped corpus of per-vuln-class
detection knowledge.

Augmentum's bug_finder already has a strong *verification* core (repro
confirmation + cross-run + cross-family voting) and a *self-learned*
pattern memory (``patterns.py``) — but that memory cold-starts EMPTY on
a fresh repo, so on first contact the detector hunts only from the
model's latent knowledge. This module is the missing piece: a curated,
disclosure-informed playbook (the idea borrowed from Claude-BugHunter's
pattern library) that primes the planner with class-specific "where to
look / how to confirm / common false positives / what the fix restores"
BEFORE the workspace has any learned history.

Adapted to our reality: this is a WHITE-BOX code auditor, not a
black-box pentest. So entries describe source→sink code shapes and
confirmation by reading + tracing (not live payloads), and the
``common_fps`` lists exist to suppress noise the way our verifier
otherwise has to — and the way our own audits this session kept
catching by hand (a sink that isn't reachable, an identifier that's
already quoted, a value that's actually a constant).

Keyed by ``findings.ClaimSignature`` so it lines up with the taxonomy
the detector + dedup already use. Selection is TARGETED: only the
classes a given codebase's risk surfaces actually expose are injected,
so a small repo doesn't pay for cloud-IAM prose it has no surface for.

Composition: the planner sees this seeded brief ALONGSIDE the
self-learned ``patterns`` brief — priors-on-first-contact plus
compounding memory. It is deliberately NOT fed to the detector (same
rule as ``patterns``: detection reasons from chunk evidence, not from
priors, so a planted expectation can't manufacture a finding).
"""
from __future__ import annotations

from dataclasses import dataclass

from augmentum.bug_finder.findings import ClaimSignature


@dataclass(frozen=True)
class PlaybookEntry:
    """One vuln-class hunting card."""

    signature: str                  # ClaimSignature value
    title: str
    where_to_look: tuple[str, ...]  # source→sink code shapes
    confirm: tuple[str, ...]        # how to confirm exploitability by reading/tracing
    common_fps: tuple[str, ...]     # FP shapes to rule out (noise control)
    invariant: str                  # what a correct fix restores
    # Comprehender risk_surface ``name`` values this class lives behind.
    surface_tags: tuple[str, ...] = ()
    # Languages this card applies to; ("*",) = any.
    languages: tuple[str, ...] = ("*",)
    # Universal high-value class — included even on cold-start (no risk
    # surfaces known yet) because it's worth checking on essentially any
    # server codebase.
    universal: bool = False


PLAYBOOK: tuple[PlaybookEntry, ...] = (
    PlaybookEntry(
        signature=ClaimSignature.INJECTION.value,
        title="Injection (SQL / command / template / path)",
        where_to_look=(
            "Strings built with f-string / % / + / .format() that flow into "
            "a SQL execute(), a shell, a template render, or a filesystem path, "
            "where ANY interpolated part traces to a request body/query/header, "
            "a path param, uploaded-file content, or a DB row that itself came "
            "from user input.",
            "subprocess(..., shell=True) / os.system / os.popen with a "
            "non-constant argument.",
            "open()/Path joins built from user-supplied names (path traversal).",
            "render_template_string / eval / exec on a non-constant string.",
        ),
        confirm=(
            "Trace the interpolated value to a concrete untrusted source; "
            "confirm there is NO parameterization (`?` binds), quoting helper, "
            "or allowlist between source and sink.",
            "For SQL identifiers (table/column) that can't bind, confirm they "
            "are NOT passed through a quote_ident()/allowlist.",
        ),
        common_fps=(
            "Interpolated value is a hardcoded constant, enum, or frozenset.",
            "Identifier comes from sqlite_master/PRAGMA on a SERVER-controlled "
            "DB (and even then should be quoted — flag low, not high).",
            "Query already parameterizes the user value; the f-string only "
            "injects an identifier that IS quoted/allowlisted.",
            "ORM/query-builder binding the value.",
        ),
        invariant=(
            "Untrusted values bind as parameters, never string-built; "
            "identifiers go through a quote/allowlist helper; shell calls use "
            "argument lists, not shell=True."
        ),
        surface_tags=("http_routes", "websocket_handlers", "upload_endpoints",
                      "rpc_handlers", "cli"),
        universal=True,
    ),
    PlaybookEntry(
        signature=ClaimSignature.AUTH_BYPASS.value,
        title="Auth bypass / broken access control / IDOR",
        where_to_look=(
            "Handlers that read request.scope['user'] / session / request.state "
            "without a gating middleware or auth decorator above them.",
            "Object lookups by id with NO owner/user_id filter on a "
            "user-scoped table (IDOR) — read OR write.",
            "Identity (user_id / role / is_admin) taken from the request BODY "
            "or a client header instead of the auth scope.",
            "Permission checks that can be skipped: an early return before the "
            "check, a missing else, a check whose failure path still proceeds.",
        ),
        confirm=(
            "Trace whether an unauthenticated caller — or an authenticated "
            "DIFFERENT user — can reach the sink. Use the middleware-chain / "
            "decorator / who-calls tools to rule out invisible gating.",
            "For IDOR: confirm the query lacks the authenticated-owner filter "
            "on a table that is per-user.",
        ),
        common_fps=(
            "A gating middleware runs first (check the chain — auth is often "
            "invisible from the handler body).",
            "An @require_auth-style decorator applies above the function.",
            "The id is already narrowed to the caller upstream.",
        ),
        invariant=(
            "Every user-scoped read/write filters by the authenticated owner; "
            "identity comes from the auth scope, never the request body; auth "
            "runs before the handler."
        ),
        surface_tags=("http_routes", "websocket_handlers", "rpc_handlers"),
        universal=True,
    ),
    PlaybookEntry(
        signature=ClaimSignature.MISSING_VALIDATION.value,
        title="Untrusted input → unsafe sink (SSRF / XXE / deserialization / upload)",
        where_to_look=(
            "Outbound HTTP / file / DB request whose HOST or path is built "
            "from user input without an allowlist (SSRF).",
            "pickle.load / yaml.load(Loader=unsafe) / marshal / eval on bytes "
            "that can be attacker-supplied (deserialization RCE).",
            "XML parsed with external entities / DTDs enabled (XXE).",
            "File uploads accepted without type/size/destination-path checks.",
            "HTTP redirects to a user-supplied URL (open redirect).",
        ),
        confirm=(
            "Confirm the source is untrusted AND reaches the dangerous sink "
            "with no validating gate between them.",
            "SSRF: can the user control the HOST (not just a path segment)?",
            "Deserialization: are the bytes attacker-supplied AND the loader "
            "unsafe? A safe loader (json, yaml.safe_load) is not this bug.",
        ),
        common_fps=(
            "Input is validated/allowlisted at the boundary upstream.",
            "Dangerous loader is gated behind a default-OFF flag.",
            "Fetch is restricted to a fixed/allowlisted host.",
            "Parser is hardened (external entities disabled).",
        ),
        invariant=(
            "Untrusted input is validated/allowlisted at the boundary; unsafe "
            "loaders never touch untrusted bytes; SSRF-prone fetches go through "
            "an allowlist / SSRF-safe client."
        ),
        surface_tags=("http_routes", "upload_endpoints", "deserialize_sinks",
                      "external_api_clients", "file_ingest"),
        universal=True,
    ),
    PlaybookEntry(
        signature=ClaimSignature.RACE.value,
        title="Race condition / TOCTOU on shared state",
        where_to_look=(
            "Read-modify-write of shared state (a counter/balance, an in-memory "
            "dict, a JSON column) without a lock or atomic statement.",
            "Check-then-act: a guard query (e.g. 'is it available?') followed "
            "by an action, with an await between them.",
            "Comments asserting a 'single-X-at-a-time contract' enforced "
            "somewhere ELSE rather than at this site.",
        ),
        confirm=(
            "Two concurrent callers can interleave between the check and the "
            "act on the SAME key/row; the state is process-global or a DB row "
            "mutated by read-then-write.",
        ),
        common_fps=(
            "Protected by an asyncio.Lock / DB transaction.",
            "Atomic SQL (conditional UPDATE ... WHERE, json_patch, "
            "counter = counter + ?).",
            "Single-writer by construction (only one task touches it).",
        ),
        invariant=(
            "Read-modify-write is atomic (a lock or a conditional/atomic SQL "
            "UPDATE); a guard and its action share one transaction."
        ),
        surface_tags=("http_routes", "background_jobs", "websocket_handlers"),
    ),
    PlaybookEntry(
        signature=ClaimSignature.RESOURCE_LEAK.value,
        title="Resource leak / unbounded growth",
        where_to_look=(
            "File / socket / DB cursor / lock acquired without a `with` or "
            "try/finally — leaks on the error path.",
            "Fire-and-forget tasks created without being tracked/awaited.",
            "Caches/dicts keyed by an unbounded dimension (per-session, "
            "per-user, per-url) with no eviction.",
            "Connections held across an await inside a pooled context.",
        ),
        confirm=(
            "The resource leaks on a realistic error/early-return path, not "
            "just the happy path; OR the growth dimension is genuinely "
            "unbounded over the process lifetime.",
        ),
        common_fps=(
            "Managed by a context manager / closed in finally.",
            "Bounded by an LRU or TTL eviction.",
            "Cardinality is naturally small (fixed vocabulary).",
        ),
        invariant=(
            "Resources are acquired in a context manager or closed in finally; "
            "background tasks are tracked; caches are bounded."
        ),
        surface_tags=("http_routes", "background_jobs"),
    ),
    PlaybookEntry(
        signature=ClaimSignature.LOGIC_ERROR.value,
        title="Security-relevant logic error (fail-open / inverted gate)",
        where_to_look=(
            "A security gate that fails OPEN: an except that swallows the "
            "check and proceeds, a default that allows, a missing else.",
            "Inverted or off-by-one conditions on a security/bounds check.",
            "A flag or early-return that silently disables a check.",
        ),
        confirm=(
            "The wrong branch is reachable AND has a security/data "
            "consequence; the fail-open path is hit on a realistic error.",
        ),
        common_fps=(
            "The 'wrong' branch is intentional and safe.",
            "The default is fail-CLOSED (deny on error).",
        ),
        invariant=(
            "Security gates fail closed; defaults deny; bounds/conditions are "
            "correct."
        ),
        surface_tags=("http_routes", "auth"),
    ),
)


_BY_SIGNATURE = {e.signature: e for e in PLAYBOOK}


def select_playbook(
    *,
    risk_surface_names: tuple[str, ...] = (),
    signatures_seen: tuple[str, ...] = (),
    languages: tuple[str, ...] = (),
    max_entries: int = 4,
) -> list[PlaybookEntry]:
    """Pick the most relevant cards for THIS codebase — targeted, not a dump.

    Scoring (higher = more relevant):
      +3  this class already recurs in the workspace's learned memory
          (``signatures_seen``) — double down on what's actually here.
      +2  this class's surface_tags intersect the codebase's risk surfaces
          (the comprehender said this attack surface exists).
      +1  universal high-value class — keeps cold-start runs (no risk
          surfaces yet) seeded with the essentials.

    Language filter: an entry tagged to specific languages is dropped when
    we know the repo's languages and none match. Entries tagged ``("*",)``
    always survive.

    Returns up to ``max_entries`` entries, highest score first, stable on
    ties (corpus order). Empty only if everything scored 0 — which can't
    happen while any universal entry exists.
    """
    seen = set(signatures_seen)
    surfaces = set(risk_surface_names)
    langs = {x.lower() for x in languages}

    scored: list[tuple[int, int, PlaybookEntry]] = []
    for idx, e in enumerate(PLAYBOOK):
        if langs and e.languages != ("*",) and not (set(e.languages) & langs):
            continue
        score = 0
        if e.signature in seen:
            score += 3
        if surfaces and (set(e.surface_tags) & surfaces):
            score += 2
        if e.universal:
            score += 1
        if score == 0:
            continue
        scored.append((-score, idx, e))  # -score for ascending sort = desc score

    scored.sort()
    return [e for _, _, e in scored[:max_entries]]


def render_playbook_brief(entries: list[PlaybookEntry]) -> str:
    """Compact, prompt-friendly hunting brief. Empty string when no
    entries (caller passes the prompt through unmodified)."""
    if not entries:
        return ""
    out: list[str] = []
    out.append("## Targeted hunting playbook (seeded priors)")
    out.append("")
    out.append(
        "Class-specific guidance for the vulnerability classes this "
        "codebase's surfaces expose. Use it to focus the survey and to seed "
        "investigator threads — NOT as a finding by itself; the detector "
        "still reasons from chunk evidence and the verifier still needs a "
        "real repro."
    )
    for e in entries:
        out.append("")
        out.append(f"### {e.title}  (`{e.signature}`)")
        out.append("**Where to look:**")
        out.extend(f"  - {x}" for x in e.where_to_look)
        out.append("**Confirm (don't ship without this):**")
        out.extend(f"  - {x}" for x in e.confirm)
        out.append("**Common false positives — rule these out:**")
        out.extend(f"  - {x}" for x in e.common_fps)
        out.append(f"**A correct fix restores:** {e.invariant}")
    return "\n".join(out)


__all__ = [
    "PLAYBOOK",
    "PlaybookEntry",
    "render_playbook_brief",
    "select_playbook",
]
