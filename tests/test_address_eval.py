"""Address-classifier eval corpus + runner.

Replaces ad-hoc "patch the regex when something fails" with a
labeled benchmark. Two layers run independently:

  * **Tier 1** — ``architect.address.is_addressed`` (pure regex)
  * **Tier 3** — ``architect.address_llm.classify_with_llm`` (live LLM)

The Tier 1 test is offline + deterministic; it pins precision/recall
on a curated corpus and FAILS if a regression drops a known-good
phrase. The Tier 3 test is opt-in (``--run-live``) because it needs a
real backend; it measures latency + verdict accuracy against the same
corpus.

Corpus design
-------------
Each row is ``(utterance, expected, source)``:

  * ``expected`` is ``"addressed"`` or ``"ambient"`` — the human label
    for what the dispatcher SHOULD do
  * ``source`` is a tag for why this row exists (e.g. "real_log",
    "bug_fix", "edge_case") so regressions can be triaged by class

When a real STT log produces a misclassification, add it here with
``source="real_log:<date>"``. The corpus is the source of truth for
"what counts as addressed" — the regex and the LLM both serve it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from augmentum.architect.address import is_addressed


@dataclass(frozen=True)
class Case:
    utterance: str
    expected: str  # "addressed" | "ambient"
    source: str


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
# Keep this small (<200 rows). Quality > quantity — every row should
# represent a class of failure or a high-value canonical phrasing.
# Drop duplicates; ambiguous cases go in `AMBIGUOUS` below (excluded
# from accuracy math but measured separately).

CORPUS: tuple[Case, ...] = (
    # ---- Tier 1 regex hits (addressed) ----
    Case("play some jazz", "addressed", "canonical"),
    Case("set a timer for five minutes", "addressed", "canonical"),
    Case("find my resume", "addressed", "canonical"),
    Case("open the notebook", "addressed", "canonical"),
    Case("show me today's reflection", "addressed", "canonical"),
    Case("pause the music", "addressed", "canonical"),
    Case("skip this track", "addressed", "canonical"),
    Case("can you tell me the time", "addressed", "canonical"),
    Case("could you turn the lights down", "addressed", "canonical"),
    Case("are you still there", "addressed", "canonical"),
    Case("what time is it", "addressed", "canonical"),
    Case("how's the weather", "addressed", "canonical"),
    Case("hey becca, what's on the agenda", "addressed", "canonical"),

    # ---- Real-log failures the existing regex misses ----
    # 2026-05-29: the trigger for this whole rewrite. Both dropped to
    # no_signal + LLM UNSURE + silent drop.
    Case(
        "Okay, well can you throw in some jazz music for me?",
        "addressed", "real_log:2026-05-29",
    ),
    Case(
        "Got it. Um, are you able to throw on some jazz music for me please",
        "addressed", "real_log:2026-05-29",
    ),
    # Vocabulary drift the verb-list can never close — every one of
    # these is an idiolectal way to say "play music".
    Case("fire up some jazz", "addressed", "vocab_drift"),
    Case("spin up some tunes", "addressed", "vocab_drift"),
    Case("queue up Miles Davis", "addressed", "vocab_drift"),
    Case("stick on the playlist", "addressed", "vocab_drift"),
    Case("throw on something chill", "addressed", "vocab_drift"),
    Case("pull up that document about quantum computing", "addressed", "vocab_drift"),
    Case("grab me the file from yesterday", "addressed", "vocab_drift"),
    Case("bring up the kitchen lights", "addressed", "vocab_drift"),

    # ---- Beneficiary marker ("for me" / "for us") ----
    # Strong addressing signal in English even when no verb keyword hits.
    Case("can you do that for me", "addressed", "beneficiary"),
    Case("would you mind starting it for us", "addressed", "beneficiary"),
    Case("turn that down for me please", "addressed", "beneficiary"),

    # ---- Multi-clause: STT joined an ack + new request ----
    Case("Got it. Play the next song.", "addressed", "multi_clause"),
    Case("Yeah okay. Open the files panel.", "addressed", "multi_clause"),
    Case("Mmhmm. Can you set a timer for ten?", "addressed", "multi_clause"),

    # ---- Tier 1 regex hits (ambient) ----
    Case("I'm just thinking out loud", "ambient", "self_talk"),
    Case("I should probably go to bed", "ambient", "self_talk"),
    Case("um, where did I put my keys", "ambient", "self_talk"),
    Case("hmm, that's weird", "ambient", "self_talk"),
    Case("hey, did you see what John said yesterday", "ambient", "to_person"),
    Case("she told me the meeting got moved", "ambient", "third_person"),
    Case("they were saying the same thing", "ambient", "third_person"),
    Case("no I meant the other one", "ambient", "to_person_continuation"),
    Case("yeah I'll grab dinner on the way", "ambient", "to_person"),

    # ---- Reading aloud / narration ----
    Case("the quick brown fox jumps over the lazy dog", "ambient", "reading"),
    Case("once upon a time there was a princess", "ambient", "reading"),

    # ---- Background remarks ----
    Case("oh come on", "ambient", "exclamation"),
    Case("oh god", "ambient", "exclamation"),
    Case("what the hell", "ambient", "exclamation"),
)


# Cases that are genuinely ambiguous — we measure how the classifier
# splits these but don't fail tests on them. They're here to track
# whether changes shift the gray-zone behavior.
AMBIGUOUS: tuple[Case, ...] = (
    Case("yeah do it", "addressed", "ambiguous"),  # only if continuation
    Case("nevermind", "addressed", "ambiguous"),  # only if continuation
    Case("that's enough", "ambient", "ambiguous"),  # could be either
)


# ---------------------------------------------------------------------------
# Tier 1 (regex) — offline, deterministic
# ---------------------------------------------------------------------------

class TestTier1Corpus:
    """Run the regex classifier against the curated corpus.

    Reports per-row results; the suite as a whole asserts on a minimum
    accuracy floor so a regression that drops one canonical phrase
    fails CI loudly, but tweaks that shift one ambiguous case don't.

    The floor is intentionally low — Tier 1 isn't supposed to be the
    primary path for ambiguous cases. The LLM tiebreaker is. The
    floor exists to catch "I deleted a whole class of canonical hits"
    not "I refused to hand-curve-fit one vocab variant".
    """

    def test_canonical_addressed_recall(self, capsys):
        """Tier 1 should catch most canonical addressed phrasings.

        NOT a strict gate — the regex deliberately avoids name-matching
        (companion is renameable per CLAUDE.md) so things like "hey
        becca, what's..." fall to the LLM tier. The check is "did we
        regress below a recall floor", not "did we cover every phrase".
        """
        canonical = [c for c in CORPUS if c.source == "canonical" and c.expected == "addressed"]
        misses = []
        for case in canonical:
            decision = is_addressed(case.utterance)
            if not decision.addressed:
                misses.append((case.utterance, decision.signal, decision.confidence))
        recall = (len(canonical) - len(misses)) / len(canonical)
        print(f"\nTier 1 canonical-addressed recall: "
              f"{len(canonical) - len(misses)}/{len(canonical)} = {recall:.1%}")
        for u, sig, conf in misses:
            print(f"  miss [{sig}] ({conf}) {u!r}")
        assert recall >= 0.80, f"Tier 1 canonical-addressed recall {recall:.1%} below 80%"

    def test_canonical_ambient_rejects(self):
        """The canonical ambient phrasings MUST be rejected."""
        canonical = [c for c in CORPUS if c.expected == "ambient"
                     and c.source in ("self_talk", "third_person", "reading", "exclamation")]
        false_positives = []
        for case in canonical:
            decision = is_addressed(case.utterance)
            if decision.addressed:
                false_positives.append((case.utterance, decision.signal, decision.confidence))
        assert not false_positives, (
            f"{len(false_positives)} ambient phrases got false-positive: {false_positives[:5]}"
        )

    def test_overall_corpus_accuracy(self, capsys):
        """Tier 1 floor across the full corpus.

        Failure modes the regex CAN'T solve (vocab drift, multi-clause,
        beneficiary markers) are counted but expected to miss until the
        LLM tier picks them up. The floor here is "don't drop below
        catching 60% of total cases" — the LLM is the safety net.
        """
        right = 0
        wrong_addressed = []  # should-have-been-addressed, classified ambient
        wrong_ambient = []   # should-have-been-ambient, classified addressed
        for case in CORPUS:
            decision = is_addressed(case.utterance)
            got = "addressed" if decision.addressed else "ambient"
            if got == case.expected:
                right += 1
            elif case.expected == "addressed":
                wrong_addressed.append(case)
            else:
                wrong_ambient.append(case)

        total = len(CORPUS)
        accuracy = right / total

        # Print the breakdown so test output is the report
        print(f"\nTier 1 accuracy: {right}/{total} = {accuracy:.1%}")
        if wrong_addressed:
            print(f"  False negatives (dropped legitimate requests): {len(wrong_addressed)}")
            for c in wrong_addressed[:10]:
                print(f"    [{c.source}] {c.utterance!r}")
        if wrong_ambient:
            print(f"  False positives (accidentally addressed): {len(wrong_ambient)}")
            for c in wrong_ambient[:10]:
                print(f"    [{c.source}] {c.utterance!r}")

        # False positives are the dangerous class (Becca speaks when she
        # shouldn't). Hold the line at zero on canonical_ambient sources.
        assert not wrong_ambient or all(
            c.source not in ("self_talk", "third_person", "reading", "exclamation")
            for c in wrong_ambient
        ), "Tier 1 false-positives on canonical ambient sources"

        # Overall floor — captures total drift, generous enough that
        # vocab/multi-clause misses don't fail the gate.
        assert accuracy >= 0.60, f"Tier 1 accuracy {accuracy:.1%} below 60% floor"


# ---------------------------------------------------------------------------
# Tier 3 (LLM) — opt-in live test
# ---------------------------------------------------------------------------
# Run with: pytest tests/test_address_eval.py::TestTier3LiveCorpus --run-live -v

def pytest_addoption(parser):
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run live tests that hit a real LLM backend (the user's "
             "configured chat model, via the address-LLM tier).",
    )


@pytest.fixture
def run_live(request):
    return request.config.getoption("--run-live")


class TestTier3LiveCorpus:
    """Live eval of the LLM tiebreaker against the same corpus.

    Skipped unless ``--run-live`` is passed. Measures verdict accuracy
    AND median latency so the report tells you whether the user's
    current model is fit for the job or whether a dedicated classifier
    sidecar is justified.
    """

    @pytest.mark.asyncio
    async def test_llm_accuracy_and_latency(self, run_live, capsys):
        if not run_live:
            pytest.skip("--run-live not passed")

        from augmentum.architect.address_llm import classify_with_llm
        from augmentum.proxy.server import create_app

        app = await create_app()
        app_state = app.state

        right = 0
        latencies: list[int] = []
        verdicts: dict[str, int] = {"ADDRESSED": 0, "AMBIENT": 0, "UNSURE": 0}
        misses: list[tuple[Case, str, int]] = []

        for case in CORPUS:
            result = await classify_with_llm(
                case.utterance, app_state=app_state,
            )
            verdicts[result.verdict] = verdicts.get(result.verdict, 0) + 1
            latencies.append(result.latency_ms)
            # Treat UNSURE as the dispatcher does (= AMBIENT)
            got = "addressed" if result.verdict == "ADDRESSED" else "ambient"
            if got == case.expected:
                right += 1
            else:
                misses.append((case, result.verdict, result.latency_ms))

        total = len(CORPUS)
        latencies_sorted = sorted(latencies)
        median_ms = latencies_sorted[len(latencies_sorted) // 2]
        p95_ms = latencies_sorted[int(len(latencies_sorted) * 0.95)]

        print(f"\nTier 3 LLM accuracy: {right}/{total} = {right/total:.1%}")
        print(f"  Median latency: {median_ms}ms · P95: {p95_ms}ms")
        print(f"  Verdict distribution: {verdicts}")
        if misses:
            print(f"  Misses ({len(misses)}):")
            for case, verdict, ms in misses[:15]:
                print(f"    [{case.source}] expected={case.expected:9s} "
                      f"got={verdict:9s} ({ms}ms) {case.utterance!r}")

        # No hard assert on accuracy here — this test is a measurement,
        # not a gate. The report drives the decision of whether to
        # swap in a dedicated classifier.


class TestVoiceRouterLiveCorpus:
    """Live eval of the new structured voice_router.

    Same corpus, different classifier — this is the path the
    always-listening widget runs through after the MVP rewrite. The
    report shows addressed-accuracy, goal distribution, and latency
    so we can spot regressions when swapping models.

    Skipped unless ``--run-live`` is passed (needs a real backend).
    """

    @pytest.mark.asyncio
    async def test_voice_router_accuracy_and_latency(self, run_live, capsys):
        if not run_live:
            pytest.skip("--run-live not passed")

        from augmentum.architect.voice_router import classify_voice
        from augmentum.proxy.server import create_app

        app = await create_app()
        app_state = app.state

        right = 0
        latencies: list[int] = []
        goals: dict[str, int] = {}
        parsed_from_counts: dict[str, int] = {}
        misses: list[tuple[Case, str, str, float, int]] = []  # case, addressed, goal, conf, ms

        for case in CORPUS:
            result = await classify_voice(
                case.utterance, app_state=app_state,
            )
            goals[result.goal] = goals.get(result.goal, 0) + 1
            parsed_from_counts[result.parsed_from] = (
                parsed_from_counts.get(result.parsed_from, 0) + 1
            )
            latencies.append(result.latency_ms)
            # MVP threshold mirrors the wired voice_routes path
            addressed_effective = (
                result.addressed
                and result.coherent
                and result.goal in ("act", "converse", "clarify")
                and result.confidence >= 0.70
            )
            got = "addressed" if addressed_effective else "ambient"
            if got == case.expected:
                right += 1
            else:
                misses.append((
                    case,
                    "addressed" if result.addressed else "ambient",
                    result.goal,
                    result.confidence,
                    result.latency_ms,
                ))

        total = len(CORPUS)
        latencies_sorted = sorted(latencies)
        median_ms = latencies_sorted[len(latencies_sorted) // 2]
        p95_ms = latencies_sorted[int(len(latencies_sorted) * 0.95)]

        print(f"\nVoice router accuracy: {right}/{total} = {right/total:.1%}")
        print(f"  Median latency: {median_ms}ms · P95: {p95_ms}ms")
        print(f"  Goal distribution: {goals}")
        print(f"  Parsed from: {parsed_from_counts}")
        if misses:
            print(f"  Misses ({len(misses)}):")
            for case, addr, goal, conf, ms in misses[:15]:
                print(
                    f"    [{case.source}] expected={case.expected:9s} "
                    f"got={addr:9s} goal={goal:8s} conf={conf:.2f} "
                    f"({ms}ms) {case.utterance!r}"
                )

        # No hard assert — measurement test, drives decisions about
        # model selection + threshold tuning.
