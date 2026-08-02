"""Sprint 1 reliability fixes — regression pins (audit 2026-06-17).

Pure/fast assertions guarding the contracts the Sprint 1 changes
established. DB-backed behaviors (wondering local-day cap, today
wondering-ref resolution) are covered by their existing suites; these
pin the logic that previously drifted silently.
"""
from __future__ import annotations

# ── NSFW matcher: phrase pass + conservative-FP property ──────────────

def test_nsfw_phrase_match():
    from augmentum.discovery.safety import is_nsfw_text
    assert is_nsfw_text("barely legal teens")
    assert is_nsfw_text("barely-legal")          # separator-normalized
    assert is_nsfw_text("BARELY LEGAL")          # case-insensitive
    assert is_nsfw_text("jail bait")


def test_nsfw_token_still_matches():
    from augmentum.discovery.safety import is_nsfw_text
    assert is_nsfw_text("free porn videos")
    assert is_nsfw_text("X — Pornhub")      # site stem token


def test_nsfw_false_positives_preserved():
    from augmentum.discovery.safety import is_nsfw_text
    for s in (
        "sex education",
        "naked truth",
        "the analysis was thorough",
        "hot take on the news",
        "adult learner programs",
        "legal barely passed the bar",   # words present but not the phrase
    ):
        assert not is_nsfw_text(s), s


def test_nsfw_empty_input():
    from augmentum.discovery.safety import is_nsfw_text
    assert not is_nsfw_text("")
    assert not is_nsfw_text(None)


# ── Curator safety block: subdomain suffix + domain-from-url ──────────

def test_curator_blocks_subdomain_of_blocked_host():
    from augmentum.companion_runtime import curator
    blk = next(iter(curator._ADULT_DOMAIN_BLOCKLIST))
    assert curator._curator_safety_block({"domain": blk}, "news") is not None
    assert curator._curator_safety_block({"domain": f"m.{blk}"}, "news") is not None
    assert curator._curator_safety_block({"domain": f"cdn.{blk}"}, "news") is not None


def test_curator_domain_suffix_no_overmatch():
    from augmentum.companion_runtime import curator
    blk = next(iter(curator._ADULT_DOMAIN_BLOCKLIST))
    # notpornhub.com must NOT match pornhub.com
    assert curator._curator_safety_block({"domain": f"not{blk}"}, "news") is None


def test_curator_derives_domain_from_url():
    from augmentum.companion_runtime import curator
    blk = next(iter(curator._ADULT_DOMAIN_BLOCKLIST))
    # No explicit domain key — must be derived from the URL host.
    assert curator._curator_safety_block({"url": f"https://{blk}/some/path"}, "news") is not None


def test_curator_clean_rec_passes():
    from augmentum.companion_runtime import curator
    assert curator._curator_safety_block(
        {"domain": "example.com", "url": "https://example.com/a", "title": "A neutral article"},
        "technology news",
    ) is None


# ── Dismiss signal contract (the worse-than-reported boost bug) ───────

def test_dismiss_maps_to_negative_kind():
    from augmentum.companion_runtime.feedback import KIND_WEIGHTS
    from augmentum.proxy.companion_routes import _NOTE_FEEDBACK_KINDS
    assert _NOTE_FEEDBACK_KINDS["dismiss"] == "dismissed"
    # The whole point: dismiss must be a NEGATIVE signal, not the +0.2
    # "acknowledged" the UI was posting before the fix.
    assert KIND_WEIGHTS["dismissed"] < 0
    assert KIND_WEIGHTS["acknowledged"] > 0


# ── Settings restore registration ─────────────────────────────────────

def test_pad_emit_registered_for_restore():
    from augmentum.proxy.config_routes import _TOOL_SETTINGS
    # Has a config.py default but previously no validator entry, so it
    # reverted to its default on every restart.
    assert "companion_pad_emit_enabled" in _TOOL_SETTINGS


# ── Honest-gate: consolidate() no longer claims success ───────────────

def test_consolidate_reports_not_wired():
    import asyncio

    from augmentum.companion.companion import Companion

    # consolidate() does no DB work; a bare instance is enough to assert
    # the contract is honest (ok is False, not a no-op success).
    comp = Companion.__new__(Companion)
    result = asyncio.run(comp.consolidate())
    assert result["ok"] is False
