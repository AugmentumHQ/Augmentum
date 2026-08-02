"""Memory extraction pipeline — extracts facts from conversations asynchronously."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from augmentum.config import settings
from augmentum.memory.models import ExtractedFact, MemoryType
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.memory.store import MemoryStore
    from augmentum.models.base import ModelBackend

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Heuristic extraction patterns
# ---------------------------------------------------------------------------

# Phrases that indicate conversational filler, not real identity/preference facts.
_SKIP_PHRASES: frozenset[str] = frozenset({
    "a little", "a bit", "a lot", "little", "bit",
    "not sure", "just", "also",
    "happy", "sad", "confused", "curious", "wondering", "sorry",
    "glad", "afraid", "sure", "fine", "okay", "here", "there",
    "ready", "trying", "going",
})

# Each pattern: (compiled_regex, memory_type, importance, confidence, is_explicit)
_PATTERNS: list[tuple[re.Pattern, MemoryType, float, float, bool]] = [
    # Identity statements (min capture 5 chars to avoid junk)
    (re.compile(r"\b(?:i am|i'm)\s+(?:a|an)\s+(.{5,60}?)(?:\.|,|!|\?|$)", re.I),
     MemoryType.FACT, 0.85, 0.8, False),
    # Name declarations
    (re.compile(r"\bmy name is\s+(.{2,40}?)(?:\.|,|!|\?|$)", re.I),
     MemoryType.FACT, 0.95, 0.9, False),
    # You can call me
    (re.compile(r"\b(?:call me|i go by)\s+(.{2,30}?)(?:\.|,|!|\?|$)", re.I),
     MemoryType.FACT, 0.9, 0.85, False),
    # Preferences
    (re.compile(r"\bi (?:prefer|like|love|enjoy|want)\s+(.{3,80}?)(?:\.|,|!|\?|$)", re.I),
     MemoryType.PREFERENCE, 0.7, 0.75, False),
    # Dislikes
    (re.compile(r"\bi (?:don'?t like|hate|dislike|avoid)\s+(.{3,80}?)(?:\.|,|!|\?|$)", re.I),
     MemoryType.PREFERENCE, 0.7, 0.75, False),
    # Explicit remember instructions
    (re.compile(r"\b(?:remember|note) (?:that |this:?\s*)(.{5,200}?)(?:\.|!|\?|$)", re.I),
     MemoryType.FACT, 0.95, 0.95, True),
    # Always/never instructions
    (re.compile(r"\b(?:always|never)\s+(.{5,100}?)(?:\.|,|!|\?|$)", re.I),
     MemoryType.PREFERENCE, 0.9, 0.85, False),
    # Work/occupation
    (re.compile(r"\bi (?:work|worked) (?:at|for|in|as)\s+(.{3,60}?)(?:\.|,|!|\?|$)", re.I),
     MemoryType.FACT, 0.85, 0.8, False),
    # Location
    (re.compile(r"\bi (?:live|am based|reside) (?:in|at|near)\s+(.{3,60}?)(?:\.|,|!|\?|$)", re.I),
     MemoryType.FACT, 0.8, 0.8, False),
    # Expertise
    (re.compile(r"\bi (?:specialize|specialise|am expert|am experienced) in\s+(.{3,80}?)(?:\.|,|!|\?|$)", re.I),
     MemoryType.FACT, 0.85, 0.8, False),
    # Using/working with (reduced importance/confidence — often noise)
    (re.compile(r"\bi (?:use|am using|work with)\s+(.{3,60}?)(?:\s+(?:for|at|in)\b|\.|,|!|\?|$)", re.I),
     MemoryType.PREFERENCE, 0.4, 0.5, False),
]

# Targets that indicate a task/feature request, not a personal preference.
_REQUEST_OBJECT_PATTERN = re.compile(
    r"\b(?:a |an |the |this |that |my )?"
    r"(?:function|method|class|component|page|button|modal|toggle|endpoint|"
    r"script|test|fix|bug|feature|implementation|refactor|variable|"
    r"config|setting|option|field|column|table|route|handler|service)\b",
    re.I,
)

# Linguistic pattern for task instructions about a deliverable.
# Catches "I want the report to be detailed", "I need it to include citations",
# "to create/write/make a summary of X".
_TASK_INSTRUCTION_PATTERN = re.compile(
    r"(?:"
    r"(?:the|this|that|it)\s+(?:to\s+\w+|should|needs?\s+to|must|has\s+to)"
    r"|"
    r"to\s+(?:create|write|make|build|generate|produce|draft|prepare|design|compile)\s+(?:a|an|the)\b"
    r"|"
    r"(?:make|keep)\s+it\s+\w+"
    r"|"
    r"(?:need|require|want)\s+(?:the|this|that|it)\s+"
    r")",
    re.I,
)


# ---------------------------------------------------------------------------
# Post-extraction validation gate
# ---------------------------------------------------------------------------

# First-person pronouns/patterns that signal the user is talking about themselves
_FIRST_PERSON_RE = re.compile(
    r"\b(?:I|my|me|I'm|I've|mine|we|our)\b", re.IGNORECASE,
)

# Question-start words
_QUESTION_START_RE = re.compile(
    r"^(?:what|how|why|when|where|who|which|can|could|would|do|does|is|are|will|should)\b",
    re.IGNORECASE,
)

# Explicit preference verbs — used to exempt preference statements from question rejection
_PREFERENCE_VERB_RE = re.compile(
    r"\bI\s+(?:like|prefer|love|hate|enjoy|dislike|avoid)\b", re.IGNORECASE,
)

# Imperative info-requests — "tell me about X", "look up Y", "who is Z".
# These ask ABOUT a topic; the topic is NOT a fact about the user, even
# though "tell ME" trips the loose first-person regex. ``_QUESTION_START_RE``
# only catches interrogatives, so request imperatives slipped the gate and
# let article/character content (e.g. a looked-up "Rayen") be stored as
# "I am Rayen". This closes that hole alongside the prompt rule.
_REQUEST_START_RE = re.compile(
    r"^\s*(?:tell|show|give|explain|describe|teach|find|search|look(?:\s+up)?|"
    r"list|summari[sz]e|define|name|recommend|suggest|google|fetch|get)\b",
    re.IGNORECASE,
)

# GENUINE first-person self-disclosure — first-person SUBJECT or possessive,
# deliberately EXCLUDING bare "me" (so "tell me about X" / "help me" don't
# count). This is what must anchor a fact about the user; the LLM phrasing
# a fabricated fact in the first person ("I am Rayen") must NOT bypass it.
_SELF_DISCLOSURE_RE = re.compile(
    r"\b(?:I'm|I am|I've|I have|I'd|I'll|my|mine|myself)\b"
    r"|\bI\s+(?:like|love|hate|prefer|enjoy|dislike|avoid|live|lived|work|"
    r"worked|own|grew|moved|study|studied|use|speak|need|want|do|don't|"
    r"was|were|had|come|came|born|married|have)\b",
    re.IGNORECASE,
)

# Proper noun heuristic: capitalized word that is NOT at the start of a sentence
_PROPER_NOUN_RE = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{1,}")

# Number pattern (digits, optionally with decimal)
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

# Quoted term pattern
_QUOTED_RE = re.compile(r"""["'""''].+?["'""'']""")

# ---------------------------------------------------------------------------
# Anti-projection: artifact-subject detection
# ---------------------------------------------------------------------------
# Generic nouns that name a work-in-progress the user is describing rather
# than a property of the user themselves. When the fact's subject matches
# one of these AND the predicate describes the artifact's behaviour AND no
# identity-grounding language is present, we treat the fact as an artifact
# spec and drop it. See _is_artifact_description below for the three-signal
# heuristic and `memory_anti_projection_enabled` for the kill switch.

# Determined form: "the game", "my app", "this build", etc.
_ARTIFACT_SUBJECT_RE = re.compile(
    r"\b(?:the|this|that|a|an|my|our|their)\s+"
    r"(?:game|games|project|app|application|build|website|site|"
    r"page|script|code|codebase|repo|repository|module|"
    r"component|feature|story|chapter|level|world|character|"
    r"protagonist|antagonist|enemy|enemies|obstacle|obstacles|"
    r"player|players|user|users|menu|button|UI|interface|design|"
    r"prototype|MVP|demo|deployment|server|endpoint|API|database|"
    r"table|column|migration|bot|workflow|presentation|document|"
    r"essay|paper|report|novel|book|scene|track|song)\b",
    re.IGNORECASE,
)

# Bare-plural / proper-noun form at sentence start: "Walls kill the player",
# "Obstacles spawn every 30s". Common in bug reports and game-mechanic
# descriptions where the LLM extractor drops the determiner. Anchored at
# string start (or after a leading "Wants"/"Requires"-style prefix that
# the extractor sometimes adds) so we don't false-positive on artifact
# nouns appearing mid-sentence in legitimate user facts.
_ARTIFACT_BARE_PLURAL_RE = re.compile(
    r"^\s*(?:walls?|obstacles?|enemies|enemy|players?|levels?|"
    r"characters?|protagonists?|antagonists?|menus?|buttons?|"
    r"scripts?|tests?|migrations?|endpoints?)\b",
    re.IGNORECASE,
)

# Property predicates — describe the artifact's behavior/state/requirements.
# "Property" here means "intrinsic attribute of the thing" rather than
# "what the user did with the thing". Tuned to the patterns seen in the
# memory dump (must/should/needs/runs/kills/loads/is broken/is visible).
_ARTIFACT_PREDICATE_RE = re.compile(
    r"\b(?:must|should|needs?\s+to|has\s+to|requires?|runs?\s+on|"
    r"loads?|renders?|displays?|shows?|kills?|hits?|spawns?|"
    r"breaks?|crashes?|fails?|is\s+(?:visible|invisible|broken|"
    r"working|deployed|exposed|missing)|uses?\s+(?:port|url|api|database))\b",
    re.IGNORECASE,
)

# Identity-grounding language — first-person verb-of-being / verb-of-doing
# that anchors the fact to the user. If present, even a sentence mentioning
# an artifact noun is a legitimate user fact ("I work on indie games",
# "I'm building Augmentum, an AI proxy"). The "User " prefix covers the
# extractor emitting third-person reformulations.
_IDENTITY_GROUNDING_RE = re.compile(
    r"\b(?:I\s+(?:am|'m|work|live|do|like|prefer|love|hate|enjoy|"
    r"dislike|build|create|use|study|run|own|teach|write|design|"
    r"specialise|specialize)|"
    r"my\s+(?:name|background|job|role|wife|husband|partner|"
    r"family|kid|cat|dog|pet)|"
    r"User\s+(?:is|works|lives|prefers|loves|hates|enjoys|"
    r"builds|creates|uses|owns|teaches|writes|designs|runs|has))\b",
    re.IGNORECASE,
)


def _is_artifact_description(content: str) -> bool:
    """True if the content describes an artifact (work-in-progress) rather than the user.

    Three-signal heuristic — conservative on purpose. All three must align:
      1. Artifact subject (the/this/my + generic artifact noun)
      2. Property predicate (must/runs/kills/is visible/...)
      3. NO identity-grounding language (I am, I work, my name, User does ...)

    Single-signal false positives kept the rule from misfiring on legitimate
    facts like "I work on indie games" (has artifact noun but identity-grounded)
    or "User is excited" (has 'User' but no artifact + no property predicate).

    Explicit "remember X" facts skip this check entirely — see _validate_fact.
    """
    has_artifact_subject = (
        bool(_ARTIFACT_SUBJECT_RE.search(content))
        or bool(_ARTIFACT_BARE_PLURAL_RE.search(content))
    )
    if not has_artifact_subject:
        return False
    if not _ARTIFACT_PREDICATE_RE.search(content):
        return False
    # No identity-grounding language = the fact is purely about the artifact.
    return not _IDENTITY_GROUNDING_RE.search(content)


def _validate_fact(fact: ExtractedFact, user_messages: list[str]) -> bool:
    """Post-extraction validation gate.

    Checks that the fact is specific, grounded in user self-disclosure,
    and not inferred from a task instruction.
    Returns True if the fact passes all checks.
    """
    content = fact.content

    # 1. Length check: 10-300 chars (relaxed from 15-200)
    if len(content) < 10 or len(content) > 300:
        log.info("validate_reject_length", content=content[:60], length=len(content))
        return False

    # 2. Specificity: must contain a proper noun, number, quoted term,
    #    first-person reference, or have moderate importance
    has_proper_noun = bool(_PROPER_NOUN_RE.search(content))
    has_number = bool(_NUMBER_RE.search(content))
    has_quoted = bool(_QUOTED_RE.search(content))
    has_first_person = bool(_FIRST_PERSON_RE.search(content))
    if not (has_proper_noun or has_number or has_quoted
            or has_first_person
            or fact.is_explicit or fact.importance >= 0.5):
        log.debug("validate_reject_specificity", content=content[:60])
        return False

    # 3. First-person anchor: at least one user message must contain
    #    first-person language near the topic words.
    #    Skip this check if the fact itself contains first-person language
    #    (the LLM already judged it as user self-disclosure).
    # Strip surrounding punctuation so "Rayen," matches "Rayen" in a message.
    topic_words = {
        re.sub(r"[^\w]+", "", w).lower()
        for w in content.split()
        if len(re.sub(r"[^\w]+", "", w)) > 3
    }
    found_anchor = False
    anchor_message = ""
    if not has_first_person:
        for msg in user_messages:
            msg_lower = msg.lower()
            topic_match_pos = -1
            for tw in topic_words:
                pos = msg_lower.find(tw)
                if pos >= 0:
                    topic_match_pos = pos
                    break
            if topic_match_pos < 0:
                continue
            start = max(0, topic_match_pos - 200)
            end = min(len(msg), topic_match_pos + 200)
            window = msg[start:end]
            if _FIRST_PERSON_RE.search(window):
                found_anchor = True
                anchor_message = msg
                break

        if not found_anchor:
            log.debug("validate_reject_no_first_person", content=content[:60])
            return False
    else:
        # The fact is phrased first-person, but that's the LLM's framing —
        # it can fabricate "I am X" from a looked-up article. Require real
        # grounding: at least one user message must actually mention the
        # topic. No mention at all → the fact isn't about the user. Reject.
        for msg in user_messages:
            msg_lower = msg.lower()
            for tw in topic_words:
                if tw and tw in msg_lower:
                    anchor_message = msg
                    break
            if anchor_message:
                break
        if not anchor_message:
            log.debug("validate_reject_first_person_ungrounded", content=content[:60])
            return False
        found_anchor = True

    # 4. Not a question inference: if the anchor message is a question,
    #    still allow if the FACT contains first-person self-disclosure.
    #    "I live in <city>, what are some restaurants?" — the location
    #    is a real fact even though the message is a question.
    if anchor_message:
        stripped = anchor_message.strip()
        is_question = (
            stripped.endswith("?")
            or bool(_QUESTION_START_RE.match(stripped))
            or bool(_REQUEST_START_RE.match(stripped))
        )
        if is_question:
            # The anchor is a question or an info-request ("tell me about
            # Rayen"). Only keep the fact if the USER'S OWN message contains
            # genuine self-disclosure — NOT the LLM's first-person framing of
            # the answer. Trusting ``_FIRST_PERSON_RE.search(content)`` was
            # the bug: a model that read a looked-up article and wrote
            # "I am Rayen" passed the gate. Ground in the user's words.
            has_self_disclosure = bool(
                _SELF_DISCLOSURE_RE.search(anchor_message)
                or _PREFERENCE_VERB_RE.search(anchor_message)
            )
            if not has_self_disclosure:
                log.debug("validate_reject_question_inference", content=content[:60])
                return False

    # 5. Evidence validation: if the fact has evidence, fuzzy-match against
    #    user messages.  Threshold lowered to 0.3 because LLMs paraphrase.
    evidence = fact.evidence or (fact.source_context.get("evidence", "") if fact.source_context else "")
    if evidence:
        best_ratio = 0.0
        for msg in user_messages:
            ev_len = len(evidence)
            for start_pos in range(0, max(1, len(msg) - ev_len + 1), ev_len // 2 or 1):
                window = msg[start_pos:start_pos + ev_len + 50]
                ratio = SequenceMatcher(None, evidence.lower(), window.lower()).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                if best_ratio > 0.3:
                    break
            if best_ratio > 0.3:
                break
        if best_ratio <= 0.3:
            log.debug(
                "validate_reject_evidence_mismatch",
                content=content[:60],
                evidence=evidence[:60],
                best_ratio=round(best_ratio, 2),
            )
            return False

    # 6. Task instruction filter: reject task instructions
    if not fact.is_explicit:
        if _REQUEST_OBJECT_PATTERN.search(content):
            log.debug("validate_reject_request_object", content=content[:60])
            return False
        if _TASK_INSTRUCTION_PATTERN.search(content):
            log.debug("validate_reject_task_instruction", content=content[:60])
            return False

    # 7. Anti-projection: reject facts describing an artifact rather than
    #    the user. These come from coder/builder turns where the user is
    #    describing the thing they're making — "the game must run on a web
    #    server" is a project spec, not a user identity fact. Cross-context
    #    durability is the test: in an unrelated future conversation, would
    #    this fact help the assistant respond? Artifact specs fail that test.
    #
    #    Logged at INFO so users can audit what got filtered (the user's
    #    stated concern: "wrong settings can manipulate the model in unseen
    #    ways since the user doesn't know which memories get used at one
    #    time"). Drops must be observable, not silent.
    #
    #    Explicit "remember X" facts bypass this gate via the is_explicit
    #    check — users can always force-save artifact context with explicit
    #    phrasing. The whole rule can be disabled via
    #    memory_anti_projection_enabled for domains where artifact facts ARE
    #    user-relevant at scale (novelist, game designer).
    if (
        not fact.is_explicit
        and settings.memory_anti_projection_enabled
        and _is_artifact_description(content)
    ):
        log.info(
            "validate_reject_artifact_projection",
            content=content[:120],
            importance=round(fact.importance, 2),
        )
        return False

    return True


# ---------------------------------------------------------------------------
# Pre-filter: skip messages unlikely to contain extractable facts
# ---------------------------------------------------------------------------

# Greetings and conversational noise — no facts to extract
_GREETING_PATTERNS: frozenset[str] = frozenset({
    "hi", "hey", "hello", "howdy", "yo", "sup", "hiya",
    "good morning", "good afternoon", "good evening", "good night",
    "thanks", "thank you", "thx", "ty", "cheers",
    "bye", "goodbye", "see you", "later", "cya",
    "ok", "okay", "sure", "yeah", "yep", "nope", "no", "yes",
    "got it", "understood", "makes sense", "cool", "nice", "great",
    "lol", "lmao", "haha", "heh", "wow", "hmm", "ah", "oh",
    "please", "pls", "np", "no problem", "you're welcome",
    "continue", "go on", "go ahead", "next", "more",
})

# First-person pronouns/verbs that signal self-disclosure (worth extracting)
_SELF_DISCLOSURE_PATTERN = re.compile(
    r"\b(?:i am|i'm|my name|i work|i live|i use|i prefer|i like|i love|"
    r"i hate|i need|i have|i've|i was|i used to|i always|i never|"
    r"i specialize|i enjoy|remember that|note that|call me|"
    r"i go by|i studied|i moved|i started|i built|i created)\b",
    re.IGNORECASE,
)

# Pure question patterns (asking the AI, not self-disclosing)
_PURE_QUESTION_PATTERN = re.compile(
    r"^(?:what|how|why|when|where|who|which|can you|could you|would you|"
    r"do you|does|is it|are there|tell me|explain|show me|help me|"
    r"give me|list|describe|compare|summarize|write|generate|create|make)\b",
    re.IGNORECASE,
)


def should_extract(user_message: str) -> bool:
    """Lightweight pre-filter: should this message enter the extraction pipeline?

    Returns False for messages extremely unlikely to contain user facts:
    - Very short messages (< 10 chars)
    - Pure greetings / conversational filler
    - Pure questions with no self-disclosure
    - Code-only messages (no natural language)

    This is intentionally conservative — borderline messages pass through
    and the LLM handles nuanced filtering via few-shot examples.
    """
    text = user_message.strip()

    # Very short messages — no room for meaningful facts
    if len(text) < 10:
        return False

    # Exact match against greeting/filler phrases (case-insensitive, stripped punctuation)
    normalized = text.lower().rstrip(".,!?…")
    if normalized in _GREETING_PATTERNS:
        return False

    # If message contains self-disclosure patterns, always extract
    if _SELF_DISCLOSURE_PATTERN.search(text):
        return True

    # Pure questions with no first-person content — skip
    if _PURE_QUESTION_PATTERN.match(text) and not re.search(r"\bi\b", text, re.I):
        return False

    # Code-only: if >70% of the message is code blocks, skip
    code_blocks = re.findall(r"```[\s\S]*?```", text)
    if code_blocks:
        code_chars = sum(len(b) for b in code_blocks)
        if code_chars / len(text) > 0.7:
            return False

    # Default: let it through (conservative)
    return True


def heuristic_extract(user_message: str) -> list[ExtractedFact]:
    """Extract facts from user message using regex patterns.

    Only processes the user message (not assistant response) to capture
    what the user says about themselves.
    """
    facts: list[ExtractedFact] = []
    seen_contents: set[str] = set()

    for pattern, mem_type, importance, confidence, is_explicit in _PATTERNS:
        for match in pattern.finditer(user_message):
            # Build the fact content from the full match context
            full_match = match.group(0).strip().rstrip(".,!?")
            if not full_match or full_match.lower() in seen_contents:
                continue

            # Skip very short or garbage matches
            captured = match.group(1).strip() if match.lastindex else full_match
            if len(captured) < 3:
                continue

            # Quality gate: skip conversational filler phrases
            captured_lower = captured.lower()
            if any(captured_lower.startswith(skip) for skip in _SKIP_PHRASES):
                continue

            # Quality gate: skip task instructions for ALL types.
            # "I want to create a report" and "I need it to be detailed" are
            # task parameters, not lasting facts about the user.
            if not is_explicit:
                if _REQUEST_OBJECT_PATTERN.search(captured):
                    continue
                if _TASK_INSTRUCTION_PATTERN.search(full_match):
                    continue

            seen_contents.add(full_match.lower())
            facts.append(ExtractedFact(
                content=full_match,
                type=mem_type,
                importance=importance,
                confidence=confidence,
                is_explicit=is_explicit,
                source_context={},
            ))

    return facts


# ---------------------------------------------------------------------------
# Post-extraction dedup — cosine similarity on embeddings
# ---------------------------------------------------------------------------

# Threshold for considering two extracted facts as duplicates.
# Lower than the store's 0.92 dedup threshold since these are from the same
# extraction batch and more likely to be near-duplicates.
_EXTRACTION_DEDUP_THRESHOLD = 0.80


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _deduplicate_facts(
    facts: list[ExtractedFact],
    threshold: float = _EXTRACTION_DEDUP_THRESHOLD,
) -> list[ExtractedFact]:
    """Remove near-duplicate facts from an extraction batch.

    Embeds all facts, computes pairwise cosine similarity, and drops
    the lower-importance duplicate when similarity exceeds threshold.

    Async because ``EmbeddingService.embed`` runs ONNX inference (CPU-bound,
    ~1s for a handful of facts). Calling it directly on the event loop blocks
    every other coroutine — including auth/sync requests from other clients —
    for the duration. Offload to a thread instead.
    """
    if len(facts) <= 1:
        return facts

    try:
        import asyncio
        from augmentum.memory.embeddings import EmbeddingService

        texts = [f.content for f in facts]
        embeddings = await asyncio.to_thread(EmbeddingService.embed, texts)
    except Exception:
        log.debug("dedup_embedding_failed", exc_info=True)
        return facts  # Can't embed — return all facts unchanged

    # Mark indices to drop
    drop: set[int] = set()
    for i in range(len(facts)):
        if i in drop:
            continue
        for j in range(i + 1, len(facts)):
            if j in drop:
                continue
            sim = _cosine_similarity(embeddings[i], embeddings[j])
            if sim >= threshold:
                # Drop the one with lower importance; on tie, drop the later one
                if facts[i].importance >= facts[j].importance:
                    drop.add(j)
                else:
                    drop.add(i)
                    break  # i is dropped, stop comparing it

    if drop:
        kept = [f for idx, f in enumerate(facts) if idx not in drop]
        log.info(
            "extraction_dedup",
            original=len(facts),
            kept=len(kept),
            dropped=len(drop),
        )
        return kept

    return facts


async def extract_and_store(
    session_id: str,
    user_id: str,
    user_message: str,
    assistant_response: str,
    store: MemoryStore,
    scope: str | None = None,
) -> int:
    """Extract facts from a conversation turn and store them.

    Runs asynchronously after response streaming completes. Returns count of stored facts.
    """
    try:
        facts = heuristic_extract(user_message)
        if not facts:
            return 0

        stored = 0
        for fact in facts:
            fact.source_context = {
                "session_id": session_id,
                "extraction": "heuristic",
            }
            await store.store_fact(
                fact, user_id=user_id, session_id=session_id,
                is_explicit=fact.is_explicit, scope=scope,
            )
            stored += 1

        if stored:
            log.info("memory_extracted", count=stored, session_id=session_id)
        return stored
    except Exception:
        log.debug("memory_extraction_failed", session_id=session_id, exc_info=True)
        return 0


# ---------------------------------------------------------------------------
# Explicit-only extraction (always runs, zero-latency)
# ---------------------------------------------------------------------------

# Indices of the explicit-remember patterns in _PATTERNS
_EXPLICIT_PATTERNS: list[tuple[re.Pattern, MemoryType, float, float, bool]] = [
    p for p in _PATTERNS if p[4]  # is_explicit == True
]


def _extract_explicit_only(user_message: str) -> list[ExtractedFact]:
    """Extract only explicit "remember X" / "note that X" facts.

    Always runs, even when LLM extraction is enabled — zero-latency path.

    Rejects captures whose payload is a 2nd-person instruction to the
    assistant ("remember that you should X", "note that you couldn't Y").
    Those are operator directives, not personal facts — storing them as
    "facts about the user" pollutes recall across all domains equally.
    Generic guard: the rule is grammatical, not vocabulary-based.
    """
    facts: list[ExtractedFact] = []
    seen: set[str] = set()

    for pattern, mem_type, importance, confidence, _is_explicit in _EXPLICIT_PATTERNS:
        for match in pattern.finditer(user_message):
            full_match = match.group(0).strip().rstrip(".,!?")
            if not full_match or full_match.lower() in seen:
                continue
            captured = match.group(1).strip() if match.lastindex else full_match
            if len(captured) < 3:
                continue
            if _is_assistant_directive(captured):
                continue
            seen.add(full_match.lower())
            facts.append(ExtractedFact(
                content=full_match,
                type=mem_type,
                importance=importance,
                confidence=confidence,
                is_explicit=True,
                source_context={},
            ))

    return facts


_ASSISTANT_DIRECTIVE_RE = re.compile(
    r"^\s*(?:you|your|you're|you've|you'll|you'd)\b",
    re.IGNORECASE,
)


def _is_assistant_directive(captured: str) -> bool:
    """True if the captured 'remember X' payload addresses the assistant.

    "you couldn't verify externally" / "you should X" / "your search missed Y"
    — these are meta-instructions to the model, not user facts. Generic across
    all users because the signal is the 2nd-person pronoun at the head of the
    payload, not any domain-specific vocabulary.
    """
    return bool(_ASSISTANT_DIRECTIVE_RE.match(captured))


# ---------------------------------------------------------------------------
# Smart extraction orchestrator (LLM-first with fallback)
# ---------------------------------------------------------------------------


async def smart_extract_and_store(
    session_id: str,
    user_id: str,
    user_message: str,
    assistant_response: str,
    store: MemoryStore,
    scope: str | None = None,
    backend: ModelBackend | None = None,
    model: str | None = None,
    mode: str = "passthrough",
) -> int:
    """Smart extraction for a single message pair (legacy API).

    Delegates to batch_extract_and_store with a single pair.
    Returns count only (not details) for backward compatibility.
    """
    result = await batch_extract_and_store(
        session_id=session_id,
        user_id=user_id,
        pairs=[(user_message, assistant_response)],
        store=store,
        scope=scope,
        backend=backend,
        model=model,
        mode=mode,
    )
    return result[0] if isinstance(result, tuple) else result


async def batch_extract_and_store(
    session_id: str,
    user_id: str,
    pairs: list[tuple[str, str]],
    store: MemoryStore,
    scope: str | None = None,
    backend: ModelBackend | None = None,
    model: str | None = None,
    mode: str = "passthrough",
) -> int | tuple[int, list[dict]]:
    """Batch extraction: processes multiple message pairs in one LLM call.

    Fallback chain:
    1. Run heuristic patterns across all user messages (fallback).
    2. Fetch top-10 existing memories similar to the batch content.
    3. If LLM extraction enabled and backend available, run batch LLM extraction
       with existing_memories for reconciliation.
    4. Post-extraction validation gate on ALL facts.
    5. Handle UPDATE actions (update existing memory) vs ADD actions (store new).
    6. Store ADD results normally.

    Returns count of stored facts.
    """
    try:
        all_facts: list[ExtractedFact] = []
        seen_contents: set[str] = set()
        user_messages = [user_msg for user_msg, _asst in pairs]

        # Step 1: Run heuristic patterns across all user messages
        heuristic_facts: list[ExtractedFact] = []
        for user_msg, _asst in pairs:
            for f in heuristic_extract(user_msg):
                if f.content.lower() not in seen_contents:
                    seen_contents.add(f.content.lower())
                    heuristic_facts.append(f)

        # Step 2: Fetch existing memories similar to the batch content
        existing_memories: list[dict] = []
        try:
            query_parts: list[str] = []
            chars_used = 0
            for msg in user_messages:
                remaining = 500 - chars_used
                if remaining <= 0:
                    break
                chunk = msg[:remaining]
                query_parts.append(chunk)
                chars_used += len(chunk)
            query_text = " ".join(query_parts)

            if query_text.strip():
                recalled = await store.recall(
                    query_text, user_id=user_id, limit=10,
                )
                existing_memories = [
                    {
                        "id": m.id,
                        "content": m.content,
                        "type": m.memory_type.value if hasattr(m.memory_type, "value") else str(m.memory_type),
                        "importance": m.importance,
                    }
                    for m in recalled
                ]
                if existing_memories:
                    log.debug(
                        "extraction_recalled_existing",
                        count=len(existing_memories),
                        session_id=session_id,
                    )

                # Shadow-touch PROVISIONAL memories: if topics overlap with
                # what's being discussed, increment access_count so they can
                # promote to ACTIVE (proving the topic is recurring, not
                # noise). The store owns both metrics — cosine when
                # sqlite-vec is loaded, LIKE-keyword fallback otherwise —
                # so this layer just hands over the text and the keyword
                # backstop and lets the store decide. See
                # docs/superpowers/specs/2026-05-31-memory-establishment-rebalance.md.
                try:
                    fallback_keywords: list[str] = []
                    for kw in query_text.split()[:5]:
                        kw_clean = kw.replace("%", "").replace("_", "").strip()
                        if len(kw_clean) >= 4 and kw_clean.isalnum():
                            fallback_keywords.append(kw_clean)
                    await store.shadow_touch_provisional(
                        query_text, user_id,
                        keyword_fallback=fallback_keywords,
                    )
                except Exception as exc:
                    log.debug(
                        "extractor_access_count_bump_skipped",
                        session_id=session_id,
                        error=str(exc),
                    )
        except Exception:
            log.warning("extraction_recall_failed", session_id=session_id, exc_info=True)

        # Step 3: LLM extraction (batch) with existing_memories for reconciliation
        llm_facts: list[ExtractedFact] = []
        if settings.memory_llm_extraction_enabled and backend is not None:
            from augmentum.memory.llm_extractor import llm_extract_batch

            effective_model = model or ""
            llm_facts = await llm_extract_batch(
                pairs, backend, effective_model, mode=mode,
                existing_memories=existing_memories,
            )
            log.info(
                "llm_extraction_result",
                llm_facts=len(llm_facts),
                session_id=session_id,
                model=effective_model,
            )
            for f in llm_facts:
                if f.content.lower() not in seen_contents:
                    seen_contents.add(f.content.lower())
                    all_facts.append(f)

        # Step 3b: Add heuristic facts only if LLM found nothing
        if not llm_facts:
            all_facts.extend(heuristic_facts)
        else:
            # LLM found something — only add explicit heuristic facts
            all_facts.extend(f for f in heuristic_facts if f.is_explicit)

        # Step 4: Post-extraction validation gate
        pre_validation = len(all_facts)
        all_facts = [f for f in all_facts if _validate_fact(f, user_messages)]
        if pre_validation > 0 and len(all_facts) < pre_validation:
            log.info(
                "extraction_validation_filtered",
                before=pre_validation,
                after=len(all_facts),
                session_id=session_id,
            )

        # Step 4.5: Post-extraction dedup — drop near-duplicate facts via embedding similarity
        if len(all_facts) > 1:
            all_facts = await _deduplicate_facts(all_facts)

        if not all_facts:
            return 0

        # Step 5 & 6: Handle UPDATE vs ADD actions, then store.
        #
        # Wrap the storage loop in a single transaction. Without this,
        # each per-fact store_fact/update_content takes its own writer
        # lock and fsyncs — a 5-fact batch becomes 5 sequential commits
        # which is the single largest source of SQLite write contention
        # in the system. With batch_write(), the same 5 facts share one
        # BEGIN…COMMIT. The nested store/update calls detect the active
        # batch via a ContextVar and skip their own lock acquisition +
        # commit; concurrent writers in other asyncio tasks are
        # unaffected.
        stored = 0
        updated = 0
        stored_details: list[dict] = []  # for notification queue

        async with store.batch_write():
            for fact in all_facts:
                if "session_id" not in fact.source_context:
                    fact.source_context["session_id"] = session_id

                # Check for UPDATE action from LLM reconciliation
                action = fact.source_context.get("action", "add")
                target_id = fact.source_context.get("target_memory_id", "")

                if action == "update" and target_id:
                    try:
                        success = await store.update_content(
                            memory_id=target_id,
                            new_content=fact.content,
                            new_importance=fact.importance,
                            user_id=user_id,
                        )
                        if success:
                            updated += 1
                            stored += 1
                            stored_details.append({
                                "id": target_id,
                                "content": fact.content,
                                "evidence": fact.evidence,
                                "type": fact.type.value if hasattr(fact.type, "value") else str(fact.type),
                                "confidence": fact.confidence,
                                "action": "update",
                            })
                            continue
                        log.debug(
                            "update_fallback_to_add",
                            target_id=target_id,
                            content=fact.content[:60],
                        )
                    except Exception:
                        log.warning(
                            "memory_update_failed",
                            target_id=target_id,
                            exc_info=True,
                        )

                # Default: store as new memory (ADD)
                mem_id = await store.store_fact(
                    fact, user_id=user_id, session_id=session_id,
                    is_explicit=fact.is_explicit, scope=scope,
                )
                stored += 1
                stored_details.append({
                    "id": mem_id,
                    "content": fact.content,
                    "evidence": fact.evidence,
                    "type": fact.type.value if hasattr(fact.type, "value") else str(fact.type),
                    "confidence": fact.confidence,
                    "action": "add",
                })

        if stored:
            log.info(
                "batch_memory_extracted",
                count=stored,
                updated=updated,
                session_id=session_id,
                turns=len(pairs),
                llm=len(llm_facts),
            )
        return stored, stored_details
    except Exception:
        log.warning("batch_extraction_failed", session_id=session_id, exc_info=True)
        return 0
