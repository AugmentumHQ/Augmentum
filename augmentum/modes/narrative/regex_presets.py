"""Built-in regex preset packs for narrative mode.

Organized into tiers based on common RP platform patterns (SillyTavern,
RisuAI, KoboldAI community).  Each pack is a list of RegexScript-compatible
dicts that can be bulk-installed via the API.

Tiers:
  1. Essential Cleanup  — whitespace, quote normalization, punctuation cleanup
  2. Formatting         — OOC removal, markdown cleanup, header stripping
  3. Anti-Slop          — overused AI phrases replaced with {{random:...}} alternatives
  4. Style & Immersion  — thought formatting, censorship bypass, meta-text removal

Pattern notes:
- Inline flags `(?i)` (case-insensitive) and `(?m)` (multiline anchors) are
  used explicitly on every pattern that needs them. This keeps the transformer
  free of global flags that would change semantics of user-authored scripts.
- Replacements use `\\1` for backreferences (not `$1`).
- Anti-slop patterns use `{{random:a,b,c}}` macros expanded per match by
  regex_transformer._sub_with_random.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Tier 1: Essential Cleanup
# ---------------------------------------------------------------------------

_CLEANUP_PACK: list[dict] = [
    {
        "name": "Collapse multiple spaces",
        "find_regex": r"  +",
        "replace_string": " ",
        "placement": "output",
        "order_num": 10,
    },
    {
        "name": "Collapse triple+ newlines",
        "find_regex": r"\n{3,}",
        "replace_string": "\n\n",
        "placement": "output",
        "order_num": 11,
    },
    {
        "name": "Normalize curly double quotes",
        "find_regex": r"[\u201c\u201d]",
        "replace_string": '"',
        "placement": "output",
        "order_num": 12,
    },
    {
        "name": "Normalize curly single quotes",
        "find_regex": r"[\u2018\u2019]",
        "replace_string": "'",
        "placement": "output",
        "order_num": 13,
    },
    {
        # Collapse ..., !!!, ??? etc. (3+ of the same terminal mark) down to 2.
        "name": "Fix repeated punctuation",
        "find_regex": r"([.!?])\1{2,}",
        "replace_string": r"\1\1",
        "placement": "output",
        "order_num": 14,
    },
    {
        # Normalize em-dash spacing only. En-dash (–) is intentionally left
        # alone — it carries range semantics (e.g. "2020–2025") that would
        # be destroyed by promoting it to an em-dash.
        "name": "Fix em-dash spacing",
        "find_regex": r"\s*—\s*",
        "replace_string": "—",
        "placement": "output",
        "order_num": 15,
    },
    {
        # (?m) makes ^/$ match every line, not just start/end of the whole
        # response. Without it this only trimmed the first and last line.
        "name": "Remove per-line trailing whitespace",
        "find_regex": r"(?m)[ \t]+$",
        "replace_string": "",
        "placement": "output",
        "order_num": 16,
    },
    {
        "name": "Remove per-line leading whitespace",
        "find_regex": r"(?m)^[ \t]+",
        "replace_string": "",
        "placement": "output",
        "order_num": 17,
    },
]

# ---------------------------------------------------------------------------
# Tier 2: Formatting
# ---------------------------------------------------------------------------

_FORMATTING_PACK: list[dict] = [
    {
        # (?is): case-insensitive + dot matches newline (OOC blocks can wrap).
        "name": "Remove OOC blocks",
        "find_regex": r"(?is)\(OOC:.*?\)|\[OOC:.*?\]",
        "replace_string": "",
        "placement": "output",
        "order_num": 20,
    },
    {
        "name": "Remove author notes",
        "find_regex": r"(?is)\[(?:Author'?s?\s*Note|A/?N):.*?\]",
        "replace_string": "",
        "placement": "output",
        "order_num": 21,
    },
    {
        "name": "Strip code blocks",
        "find_regex": r"```[\s\S]*?```",
        "replace_string": "",
        "placement": "output",
        "order_num": 22,
    },
    {
        # (?s) lets bold wrap lines in rare cases. Backref \1 preserves the
        # inner text so only the ** markers are stripped.
        "name": "Strip bold formatting",
        "find_regex": r"(?s)\*\*(.+?)\*\*",
        "replace_string": r"\1",
        "placement": "output",
        "order_num": 23,
    },
    {
        # (?m) — match headers on any line, not only when the entire
        # response is a single header line.
        "name": "Remove markdown headers",
        "find_regex": r"(?m)^#{1,6}\s+.*$",
        "replace_string": "",
        "placement": "output",
        "order_num": 24,
    },
    {
        "name": "Clean empty parentheticals",
        "find_regex": r"\(\s*\)|\[\s*\]",
        "replace_string": "",
        "placement": "output",
        "order_num": 25,
    },
    {
        "name": "Normalize ellipsis",
        "find_regex": r"\.{4,}",
        "replace_string": "...",
        "placement": "output",
        "order_num": 26,
    },
]

# ---------------------------------------------------------------------------
# Tier 3: Anti-Slop (overused AI phrases with random alternatives)
#
# All pronoun patterns use (?i) so sentence-starting capitalized pronouns
# ("Her breath hitched.") match the same as mid-sentence lowercase. Without
# the flag about half of real-world slop slips past untouched.
# ---------------------------------------------------------------------------

_ANTI_SLOP_PACK: list[dict] = [
    # -- Physical reactions --
    {
        "name": "Anti-slop: breath hitching",
        "find_regex": r"(?i)(?:her|his|their)\s+breath\s+hitch(?:es|ed|ing)?",
        "replace_string": "{{random:a sharp intake of breath,a sudden stillness,a quiet gasp,a catch in the throat,a trembling exhale}}",
        "placement": "output",
        "order_num": 30,
    },
    {
        "name": "Anti-slop: shiver down spine",
        "find_regex": r"(?i)(?:a\s+)?shivers?\s+(?:runs?|ran|running|sent|sends?)\s+down\s+(?:her|his|their)\s+spine",
        "replace_string": "{{random:a prickling awareness,goosebumps rising,a tingling sensation,a sudden chill,electricity across the skin}}",
        "placement": "output",
        "order_num": 31,
    },
    {
        "name": "Anti-slop: lets out a breath",
        "find_regex": r"(?i)lets?\s+out\s+a\s+breath\s+(?:she|he|they)\s+didn'?t\s+(?:know|realize)(?:\s+(?:she|he|they)\s+(?:was|were)\s+holding)?",
        "replace_string": "{{random:finally exhales,releases the tension in a slow breath,breathes again,feels the tightness in the chest ease}}",
        "placement": "output",
        "order_num": 32,
    },
    {
        "name": "Anti-slop: heart skips a beat",
        "find_regex": r"(?i)(?:her|his|their)\s+heart\s+skip(?:s|ped)?\s+a\s+beat",
        "replace_string": "{{random:a flutter of surprise passes through,a jolt of recognition hits,a sudden warmth blooms in,a startled pause catches}}",
        "placement": "output",
        "order_num": 33,
    },
    # -- Emotional descriptors --
    {
        "name": "Anti-slop: palpable tension",
        "find_regex": r"(?i)(?:the\s+)?(?:palpable|thick)\s+tension(?:\s+in\s+the\s+(?:air|room))?",
        "replace_string": "{{random:the charged silence,the weight of the moment,the unspoken undercurrent,the heavy quiet}}",
        "placement": "output",
        "order_num": 34,
    },
    {
        "name": "Anti-slop: eyes darkened with",
        "find_regex": r"(?i)(?:her|his|their)\s+eyes\s+darken(?:s|ed)?\s+with",
        "replace_string": "{{random:a shift in expression betrays,intensity flickers behind their gaze showing,a deepening look reveals,something unreadable crosses their features as}}",
        "placement": "output",
        "order_num": 35,
    },
    {
        "name": "Anti-slop: couldn't help but",
        "find_regex": r"(?i)(?:she|he|they)\s+couldn'?t\s+help\s+but",
        "replace_string": "{{random:found themselves compelled to,instinctively began to,almost involuntarily started to,without thinking began to}}",
        "placement": "output",
        "order_num": 36,
    },
    {
        "name": "Anti-slop: a mix of emotions",
        "find_regex": r"(?i)a\s+(?:mix|mixture|whirlwind|storm|maelstrom)\s+of\s+emotions?",
        "replace_string": "{{random:conflicting feelings,tangled thoughts,warring impulses,an unfamiliar uncertainty}}",
        "placement": "output",
        "order_num": 37,
    },
    # -- Overused actions --
    {
        "name": "Anti-slop: bit her/his lip",
        "find_regex": r"(?i)bit\s+(?:her|his|their)\s+(?:lower\s+)?lip",
        "replace_string": "{{random:pressed their lips together,tightened their jaw,clenched their teeth,swallowed hard,looked away}}",
        "placement": "output",
        "order_num": 38,
    },
    {
        "name": "Anti-slop: tucked hair behind ear",
        "find_regex": r"(?i)tuck(?:s|ed|ing)?\s+(?:a\s+strand\s+of\s+)?(?:her|his|their)\s+hair\s+behind\s+(?:her|his|their)\s+ear",
        "replace_string": "{{random:fidgets with a strand of hair,brushes hair from their face,absently twists a lock of hair,runs fingers through their hair}}",
        "placement": "output",
        "order_num": 39,
    },
    {
        # The verb usage dominates in RP — previous replacements were noun
        # phrases, which produced ungrammatical output ("He smirks" → "He
        # a crooked smile"). All alternatives here are verb-form phrases.
        "name": "Anti-slop: smirk/smirks",
        "find_regex": r"(?i)\bsmirks?\b",
        "replace_string": "{{random:grins crookedly,flashes a wry grin,gives a knowing look,arches a brow with amusement,tilts a half-smile}}",
        "placement": "output",
        "order_num": 40,
    },
    {
        "name": "Anti-slop: orbs (eyes)",
        "find_regex": r"(?i)\borbs\b",
        "replace_string": "eyes",
        "placement": "output",
        "order_num": 41,
    },
    # -- Narrative clichés --
    {
        "name": "Anti-slop: silence stretches",
        "find_regex": r"(?i)(?:the\s+)?silence\s+stretch(?:es|ed)(?:\s+between\s+them)?",
        "replace_string": "{{random:a quiet moment passes,neither speaks for a beat,the pause lingers,a wordless understanding settles}}",
        "placement": "output",
        "order_num": 42,
    },
    {
        "name": "Anti-slop: unspoken understanding",
        "find_regex": r"(?i)(?:an?\s+)?unspoken\s+understanding",
        "replace_string": "{{random:a shared glance,a mutual recognition,a silent acknowledgment,something understood without words}}",
        "placement": "output",
        "order_num": 43,
    },
    {
        "name": "Anti-slop: the weight of the world",
        "find_regex": r"(?i)the\s+weight\s+of\s+the\s+world",
        "replace_string": "{{random:an invisible burden,an exhausting pressure,the crushing responsibility,everything bearing down}}",
        "placement": "output",
        "order_num": 44,
    },
    {
        "name": "Anti-slop: sends shivers",
        "find_regex": r"(?i)sends?\s+shivers?\s+(?:down|through)",
        "replace_string": "{{random:resonates through,echoes in,stirs something within,reverberates across}}",
        "placement": "output",
        "order_num": 45,
    },
    {
        "name": "Anti-slop: a single tear",
        "find_regex": r"(?i)a\s+single\s+tear\s+(?:rolls?|fell|falls?|slides?|trickles?)\s+down(?:\s+(?:her|his|their)\s+(?:cheek|face))?",
        "replace_string": "{{random:their eyes glisten,moisture wells up in their eyes,tears threaten at the corners of their eyes,their vision blurs momentarily}}",
        "placement": "output",
        "order_num": 46,
    },
    {
        "name": "Anti-slop: ministrations",
        "find_regex": r"(?i)\bministrations\b",
        "replace_string": "{{random:attention,touch,efforts,care,movements}}",
        "placement": "output",
        "order_num": 47,
    },
    {
        "name": "Anti-slop: delicate dance",
        "find_regex": r"(?i)(?:a\s+)?(?:delicate|intricate)\s+dance\s+(?:of|between)",
        "replace_string": "{{random:a careful balance of,an interplay of,a push and pull of,a negotiation between}}",
        "placement": "output",
        "order_num": 48,
    },
]

# ---------------------------------------------------------------------------
# Tier 4: Style & Immersion
# ---------------------------------------------------------------------------

_STYLE_PACK: list[dict] = [
    {
        # Wrap quoted thoughts (curly single-quoted text followed by a
        # "she thought" attribution) in italics. Restricted to smart quotes
        # \u2018...\u2019 so straight-apostrophe contractions (don't, it's)
        # can never match as a delimiter.
        "name": "Italicize thoughts",
        "find_regex": r"(?i)\u2018([^\u2019\n]{5,}?)\u2019\s*(?:she|he|they)\s+(?:thinks?|thought|muses?|mused|wonders?|wondered|considers?|considered)",
        "replace_string": r"*\1*",
        "placement": "output",
        "order_num": 50,
    },
    {
        "name": "Strip AI refusal phrases",
        "find_regex": r"(?i)I\s+(?:cannot|can'?t|shouldn'?t|won'?t)\s+(?:write|generate|create|produce|depict|continue)\s+(?:that|this|such|explicit|violent|sexual)[^.!?]*[.!?]",
        "replace_string": "",
        "placement": "output",
        "order_num": 51,
    },
    {
        "name": "Remove consent/content disclaimers",
        "find_regex": r"(?is)\[?(?:Content\s+(?:Warning|Note)|(?:Author'?s?\s+)?Note:?\s*(?:This|The\s+following|All)[^.!?]*?(?:consensual|fictional|appropriate)[^.!?]*?)\]?\.?\s*",
        "replace_string": "",
        "placement": "output",
        "order_num": 52,
    },
    {
        "name": "Remove safety disclaimers",
        "find_regex": r"(?i)(?:Please\s+)?(?:Note|Remember)\s*(?::|that)\s*(?:this\s+is|all|these|in\s+real)[^.!?]*?(?:fictional|consent|boundaries|real\s+life)[^.!?]*?[.!?]",
        "replace_string": "",
        "placement": "output",
        "order_num": 53,
    },
    {
        # (?im) — match roleplay/scene headers on any line. Allows the colon
        # and closing `**` in either order so both `**Scene: Continuation**`
        # and `**Scene Continuation:**` match.
        "name": "Remove roleplay headers",
        "find_regex": r"(?im)^(?:---\s*)?(?:\*\*)?(?:Roleplay|RP|Scene|Story)\s*(?:Continuation|Response|Continue[ds]?):?(?:\*\*)?(?:\s*---)?\s*$",
        "replace_string": "",
        "placement": "output",
        "order_num": 54,
    },
    {
        # Padding phrases that usually appear at the *start* of paragraphs
        # to burn tokens. Anchor to paragraph start (?m) so it can't strip
        # mid-sentence fragments like "She knew that as the story continues".
        "name": "Remove length padding phrases",
        "find_regex": r"(?im)(?:^|\n)\s*As\s+(?:the\s+)?(?:scene|story|narrative|conversation)\s+(?:continues|unfolds|progresses|develops),?\s*",
        "replace_string": "\n",
        "placement": "output",
        "order_num": 55,
    },
    {
        "name": "Clean trailing AI meta",
        "find_regex": r"(?is)\n+(?:---\n)?(?:\*\*)?(?:Word\s+Count|Character\s+Count|Note|End\s+of).*$",
        "replace_string": "",
        "placement": "output",
        "order_num": 56,
    },
]

# ---------------------------------------------------------------------------
# Pack registry
# ---------------------------------------------------------------------------

PRESET_PACKS: dict[str, dict] = {
    "cleanup": {
        "name": "Essential Cleanup",
        "description": "Whitespace normalization, quote fixes, punctuation collapse, em-dash spacing.",
        "tier": 1,
        "count": len(_CLEANUP_PACK),
        "scripts": _CLEANUP_PACK,
    },
    "formatting": {
        "name": "Formatting",
        "description": "Remove OOC blocks, author notes, code blocks, bold formatting, markdown headers.",
        "tier": 2,
        "count": len(_FORMATTING_PACK),
        "scripts": _FORMATTING_PACK,
    },
    "anti_slop": {
        "name": "Anti-Slop",
        "description": "Replace overused AI phrases (breath hitching, shivers, orbs, etc.) with random alternatives. Requires {{random:...}} macro.",
        "tier": 3,
        "count": len(_ANTI_SLOP_PACK),
        "scripts": _ANTI_SLOP_PACK,
    },
    "style": {
        "name": "Style & Immersion",
        "description": "Thought formatting, refusal/disclaimer removal, meta-text cleanup for immersive RP.",
        "tier": 4,
        "count": len(_STYLE_PACK),
        "scripts": _STYLE_PACK,
    },
}
