"""Debt-paydown triage — turning audit findings into self-edit targets.

The debt-paydown loop is the first dogfood for self-editing because audit findings
have everything the doctrine wants: a mechanical oracle already exists (the
finding disappears + no regression, via the audit baseline-delta), the units are
small and independent, the stakes are low, and the score compounds.

But **not all debt is mechanical**, and treating it as if it were is how the loop
Goodharts. So this module's job is *triage*: split findings into

  * ``mechanical`` — a clear, smallest-unit fix the audit can confirm
    (silent-catch→log, stray console.log, dead CSS, a blocking call off the loop,
    a missing test). These can run the auto-lane.
  * ``structural`` — needs a human decision the agent must not make alone: an
    orphaned endpoint might want a feature OR removal; a setting might want a UI;
    **a missing-CSS class is taste** (the canonical ``human_required`` case the
    whole verifier exists to respect); security/schema findings are red-tier.

The ``objective`` is a scoped instruction the *agent* then interprets with full
intelligence against the live scanners — it points at the finding class, it does
not parse it. Mechanical targets feed the orchestrator; structural ones are
*proposed* to the human (P3), never auto-fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from augmentum.selfedit.scanners import AuditReport

# How a fix is confirmed (which oracle proves the intent was met).
CONFIRM_SCANNER = "scanner"   # the finding disappears + audit shows no regression
CONFIRM_TEST = "test"         # a new test exercises + passes
CONFIRM_HUMAN = "human"       # only a person can say it's right

KIND_MECHANICAL = "mechanical"
KIND_STRUCTURAL = "structural"


@dataclass(frozen=True)
class DebtRule:
    kind: str
    title: str
    objective: str
    confirms_via: str
    note: str = ""


# (scanner, metric) → how to treat it. Only catalogued findings are actionable;
# an unknown metric is never blindly "fixed" (silence is safer than a wrong fix).
_CATALOG: dict[tuple[str, str], DebtRule] = {
    # --- mechanical (auto-lane): clear, smallest-unit, audit-confirmable -----
    ("code_quality", "silent_catches"): DebtRule(
        KIND_MECHANICAL, "Silent exception on a save/load path",
        "Find one silent `except: pass` / `contextlib.suppress(Exception)` on a "
        "save or load path and replace it with a `log.warning` that surfaces the "
        "failure (per the project rule against silent catches). One occurrence only.",
        CONFIRM_SCANNER),
    ("code_quality", "console_log"): DebtRule(
        KIND_MECHANICAL, "Stray console.log in the frontend",
        "Remove one stray `console.log` left in the frontend JS.", CONFIRM_SCANNER),
    ("code_quality", "dead_css"): DebtRule(
        KIND_MECHANICAL, "Unused CSS rule",
        "Remove one CSS rule whose class has no reference in any JS/HTML "
        "(confirmed dead). One rule only.", CONFIRM_SCANNER),
    ("code_quality", "ws_gaps"): DebtRule(
        KIND_MECHANICAL, "WebSocket contract gap",
        "Close one WS contract gap — a message kind one side sends but the other "
        "never handles (or vice versa). Wire the missing handler/emit so the "
        "frontend↔backend WS contract matches. One gap.", CONFIRM_SCANNER),
    ("code_quality", "mixed_errors"): DebtRule(
        KIND_MECHANICAL, "Mixed error-handling pattern",
        "Normalize one inconsistent error path to the project convention "
        "(raise-or-return + a `log.warning`, not a swallowed/duplicated shape). "
        "One occurrence.", CONFIRM_SCANNER),
    ("async_blocking", "errors"): DebtRule(
        KIND_MECHANICAL, "Blocking call inside async def",
        "Move one blocking call (`time.sleep`/`requests.*`/`subprocess.run`) that "
        "runs directly inside an `async def` off the event loop via "
        "`asyncio.to_thread`/`ctx.run_in_thread`, or switch to the async API.",
        CONFIRM_SCANNER),
    ("async_blocking", "warnings"): DebtRule(
        KIND_MECHANICAL, "Advisory event-loop blocker",
        "Offload one advisory loop-blocker (a `subprocess.Popen`, or a synchronous "
        "`EmbeddingService.embed_*` call inside `async def`) onto a thread.",
        CONFIRM_SCANNER),
    ("runtime", "errors"): DebtRule(
        KIND_MECHANICAL, "Runtime bug pattern (error)",
        "Fix one runtime bug-pattern flagged as an ERROR (an empty model dict, an "
        "unhandled `fetch`, a silent except on a critical path). One occurrence; "
        "the scanner confirms it cleared.", CONFIRM_SCANNER),
    ("wiring", "errors"): DebtRule(
        KIND_MECHANICAL, "Hard wiring break",
        "Fix one hard wiring break — a registered route/setting whose handler or a "
        "required layer is missing (the wiring validator flags it as an error).",
        CONFIRM_SCANNER),
    ("doc_facts", "doc_inaccuracies"): DebtRule(
        KIND_MECHANICAL, "Doc fact drifted from reality",
        "Correct one fact-fenced doc claim that no longer matches the code "
        "(run `refresh_docs.py --apply`, or fix the value). The doc-fact check "
        "confirms it.", CONFIRM_SCANNER),
    ("exceptions", "stale_entries"): DebtRule(
        KIND_MECHANICAL, "Stale security-exception entry",
        "Remove one `security_exceptions.json` entry that references a file no "
        "longer in the repo (audit-infrastructure rot).", CONFIRM_SCANNER),
    ("coverage", "coverage_gaps"): DebtRule(
        KIND_MECHANICAL, "Untested module/route",
        "Add a focused pytest for one currently-untested module or route. The test "
        "must actually exercise the code and pass — not a placeholder.",
        CONFIRM_TEST),

    # --- structural (propose to human): needs a decision the agent can't make -
    ("dead_code", "orphaned_endpoints"): DebtRule(
        KIND_STRUCTURAL, "Orphaned endpoint (no caller)",
        "An endpoint has no frontend caller. Decide whether to wire a UI/caller or "
        "remove it — a product decision, not an automatic fix.",
        CONFIRM_HUMAN, note="may be intentional (API for external/mobile use)"),
    ("code_quality", "missing_css"): DebtRule(
        KIND_STRUCTURAL, "Missing CSS for a referenced class",
        "JS references a CSS class with no rule. Adding the styling is a visual "
        "design call (taste) — propose, don't auto-apply.",
        CONFIRM_HUMAN, note="the canonical human_required case — styling is taste"),
    ("wiring", "warnings"): DebtRule(
        KIND_STRUCTURAL, "Setting/route wired in only some layers",
        "A setting or route is wired in some but not all 4 layers. Confirm whether "
        "it actually needs the missing layer (many settings legitimately need no "
        "UI) before adding it.",
        CONFIRM_HUMAN, note="0/N fully-wired is partly a strict definition, not debt"),
    ("dead_code", "ghost_calls"): DebtRule(
        KIND_STRUCTURAL, "Ghost call (frontend → missing route)",
        "The frontend calls an endpoint that has no backend route — a silent UI "
        "bug. Decide: fix the URL, add the route, or it's a URL-matcher false "
        "positive to suppress. A judgment call.",
        CONFIRM_HUMAN, note="weight 1.0 — a real user-facing break when not an FP"),
    ("dead_code", "dependency_drift"): DebtRule(
        KIND_STRUCTURAL, "Dependency drift (declared vs used)",
        "Declared dependencies drift from what's actually imported — decide whether "
        "to add a missing one or drop an unused one. A dependency decision.",
        CONFIRM_HUMAN),
    ("code_quality", "model_map_misuse"): DebtRule(
        KIND_STRUCTURAL, "_model_map misused",
        "A model-map/role registry is used incorrectly — resolving it needs knowing "
        "the right model/role for the call site. Review before changing.",
        CONFIRM_HUMAN),
    ("registry", "drift"): DebtRule(
        KIND_STRUCTURAL, "Registry/manifest drift",
        "A registry declaration drifted from the historical registered set — "
        "reconcile deliberately (a stale or renamed entry, not an auto-fix).",
        CONFIRM_HUMAN),
    ("security", "low"): DebtRule(
        KIND_STRUCTURAL, "Low-severity security finding",
        "A potential low-severity security finding — route to human security "
        "review; never auto-edit security-sensitive code.",
        CONFIRM_HUMAN, note="red-tier"),
    ("security", "critical"): DebtRule(
        KIND_STRUCTURAL, "CRITICAL security finding",
        "A critical-severity security finding — stop and review immediately; never "
        "auto-edit security-sensitive code.",
        CONFIRM_HUMAN, note="red-tier — highest priority"),
    ("security", "high"): DebtRule(
        KIND_STRUCTURAL, "High-severity security finding",
        "A high-severity security finding — human security review before any change.",
        CONFIRM_HUMAN, note="red-tier"),
    ("security", "medium"): DebtRule(
        KIND_STRUCTURAL, "Medium-severity security finding",
        "A medium-severity security finding — human security review before any "
        "change; never auto-edit security-sensitive code.",
        CONFIRM_HUMAN, note="red-tier — higher priority than low"),
    ("red_team", "total"): DebtRule(
        KIND_STRUCTURAL, "Adversarial finding (red-team)",
        "An adversarial finding (SQL injection / XSS / data-isolation / token "
        "exposure). Human security review — the highest-stakes class, red-tier.",
        CONFIRM_HUMAN, note="red-tier — adversarial; never auto-attempted"),
    ("deps", "vulnerabilities"): DebtRule(
        KIND_STRUCTURAL, "Dependency CVE",
        "A known CVE in a dependency — needs a version-bump decision plus a test "
        "pass; never auto-bumped blind (a bump can break the app).",
        CONFIRM_HUMAN),
    ("db_safety", "warnings"): DebtRule(
        KIND_STRUCTURAL, "SQLite footgun in a migration/DB path",
        "A SQLite safety warning (AUTOINCREMENT, non-idempotent CREATE, unbounded "
        "DELETE, …). Schema changes are human-reviewed (corruption is the one "
        "class never auto-attempted).",
        CONFIRM_HUMAN, note="red-tier — schema corruption is irreversible"),
    ("db_safety", "errors"): DebtRule(
        KIND_STRUCTURAL, "SQLite error in a migration/DB path",
        "An ERROR-level SQLite footgun — the schema-corruption class. Human review "
        "only (the one class never auto-attempted).",
        CONFIRM_HUMAN, note="red-tier — irreversible if wrong"),
    ("code_quality", "tech_debt"): DebtRule(
        KIND_STRUCTURAL, "TODO/FIXME/HACK marker",
        "A tech-debt marker (TODO/FIXME/HACK). Review whether it's still relevant "
        "and what resolving it entails — a judgment call.",
        CONFIRM_HUMAN),
}


# Informational/aggregate metrics that aren't debt to surface.
_NON_DEBT_METRICS = frozenset({
    "modules_covered", "modules_total", "routes_covered", "routes_total",
    "registered", "score", "total",
})
# Name fragments that mark an UNCATALOGUED metric as a real problem count worth
# surfacing — so a NEW scanner/metric auto-appears in needs-you (safely structural)
# without manual cataloguing. Curating it later promotes it to a precise rule.
_PROBLEM_SHAPE = ("error", "warning", "gap", "issue", "drift", "misuse", "stale",
                  "vulnerab", "leak", "violation", "orphan", "dead", "missing",
                  "silent", "unhandled", "ghost", "contention", "lock", "skip",
                  "slow", "block", "fail", "mismatch", "unused", "duplicate")


def _looks_like_problem(metric: str) -> bool:
    low = metric.lower()
    return any(s in low for s in _PROBLEM_SHAPE)


def _humanize(s: str) -> str:
    s = s.replace("_", " ").strip()
    return s[:1].upper() + s[1:] if s else s


@dataclass
class DebtTarget:
    scanner: str
    metric: str
    count: int
    kind: str
    title: str
    objective: str
    confirms_via: str
    note: str = ""
    discovered: bool = False    # True = auto-surfaced (uncatalogued), not yet curated
    origin: str = "audit"       # "audit" (a scanner finding) | "demand" (lived user friction)

    def to_dict(self) -> dict:
        return {
            "scanner": self.scanner, "metric": self.metric, "count": self.count,
            "kind": self.kind, "title": self.title, "objective": self.objective,
            "confirms_via": self.confirms_via, "note": self.note,
            "discovered": self.discovered, "origin": self.origin,
        }


def select_debt_targets(
    report: AuditReport, *, kinds: tuple[str, ...] | None = None,
) -> list[DebtTarget]:
    """All actionable debt targets from an audit, mechanical-first then by count.

    Returns EVERY catalogued finding with a positive count (no silent cap — the
    caller decides how many to act on and should log what it defers). ``kinds``
    filters to e.g. only ``("mechanical",)`` for the auto-lane."""
    targets: list[DebtTarget] = []
    for scanner, metrics in (report.metrics or {}).items():
        if not isinstance(metrics, dict):
            continue
        for metric, count in metrics.items():
            if not isinstance(count, int | float) or count <= 0:
                continue
            rule = _CATALOG.get((scanner, metric))
            if rule is None:
                # Auto-surface a NEW, uncatalogued problem metric so it can't hide
                # as the app grows scanners — always STRUCTURAL (safe; an unknown
                # metric is never auto-fixable). Curating it later refines it.
                if metric in _NON_DEBT_METRICS or not _looks_like_problem(metric):
                    continue
                if kinds and KIND_STRUCTURAL not in kinds:
                    continue
                targets.append(DebtTarget(
                    scanner=scanner, metric=metric, count=int(count), kind=KIND_STRUCTURAL,
                    title=f"{_humanize(scanner)}: {_humanize(metric)}",
                    objective=(f"A newly-flagged '{scanner}.{metric}' ({int(count)}) that "
                               "isn't curated yet — review what it means and how to resolve."),
                    confirms_via=CONFIRM_HUMAN,
                    note="auto-surfaced (uncatalogued) — review, then curate it",
                    discovered=True))
                continue
            if kinds and rule.kind not in kinds:
                continue
            targets.append(DebtTarget(
                scanner=scanner, metric=metric, count=int(count), kind=rule.kind,
                title=rule.title, objective=rule.objective,
                confirms_via=rule.confirms_via, note=rule.note,
            ))
    # mechanical first (auto-lane), then biggest piles first within a kind
    targets.sort(key=lambda t: (t.kind != KIND_MECHANICAL, -t.count))
    return targets


def next_mechanical_objective(report: AuditReport) -> DebtTarget | None:
    """The single highest-value auto-lane target — the loop's pick for one
    candidate self-edit. ``None`` when there's no mechanical debt to clear."""
    mech = select_debt_targets(report, kinds=(KIND_MECHANICAL,))
    return mech[0] if mech else None


@dataclass
class DebtTriage:
    mechanical: list[DebtTarget] = field(default_factory=list)
    structural: list[DebtTarget] = field(default_factory=list)

    @property
    def mechanical_count(self) -> int:
        return sum(t.count for t in self.mechanical)

    @property
    def structural_count(self) -> int:
        return sum(t.count for t in self.structural)

    def to_dict(self) -> dict:
        return {
            "mechanical": [t.to_dict() for t in self.mechanical],
            "structural": [t.to_dict() for t in self.structural],
            "mechanical_count": self.mechanical_count,
            "structural_count": self.structural_count,
        }


def triage(report: AuditReport) -> DebtTriage:
    """Split the audit's debt into the auto-lane vs the propose-to-human lane."""
    all_targets = select_debt_targets(report)
    return DebtTriage(
        mechanical=[t for t in all_targets if t.kind == KIND_MECHANICAL],
        structural=[t for t in all_targets if t.kind == KIND_STRUCTURAL],
    )
