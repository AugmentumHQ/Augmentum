"""Act-gap corrective escalation + roster anchor age gate (2026-06-11)."""
from __future__ import annotations


def test_corrective_line_renders_only_with_streak():
    from augmentum.companion_runtime.prompt_compose import _tool_roster_block
    tools = [{"name": "web.search", "description": "Search the web.",
              "args_hint": 'query="..."'}]
    plain = _tool_roster_block(tools, act_mode=True, act_corrective=False)
    escalated = _tool_roster_block(tools, act_mode=True, act_corrective=True)
    assert "MUST include its tag" in plain
    assert "still waiting" not in plain
    assert "still waiting" in escalated
    assert "Never claim you lack a capability" in escalated.replace("never claim", "Never claim")


def test_corrective_never_renders_outside_act_mode():
    from augmentum.companion_runtime.prompt_compose import _tool_roster_block
    tools = [{"name": "web.search", "description": "Search.", "args_hint": ""}]
    block = _tool_roster_block(tools, act_mode=False, act_corrective=True)
    assert "still waiting" not in block


def test_delivery_policy_teaches_layers_and_composition():
    # The grammar header is the ONE place the gather-vs-show rule and
    # the told-AND-shown composition live — per-tool descriptions are
    # trimmed to ~60 chars in the roster, so the rule can't ride them.
    from augmentum.companion_runtime.prompt_compose import _TOOL_GRAMMAR_HEADER
    assert "silently" in _TOOL_GRAMMAR_HEADER
    assert "on the user's screen" in _TOOL_GRAMMAR_HEADER.replace(
        "ONLY", "only",
    ).lower() or "the user's screen" in _TOOL_GRAMMAR_HEADER
    assert "told AND shown" in _TOOL_GRAMMAR_HEADER
    assert "gather silently first" in _TOOL_GRAMMAR_HEADER


def test_web_tools_first_sentences_carry_layer_keyword():
    # The headless pair reach the model via native FC schemas (full
    # description), the surface verb via roster lines that keep only
    # the first ~60 chars — in BOTH channels the delivery kind must
    # ride the first sentence (see _tool_roster_block CONVENTION).
    from augmentum.tools.web import WebTool
    from augmentum.tools.web_search import WebSearchTool

    for cls, keyword in ((WebSearchTool, "silent"), (WebTool, "silent")):
        inst = object.__new__(cls)  # description is state-free
        head = inst.description.split(".")[0][:80].lower()
        assert keyword in head, f"{cls.__name__} lost {keyword!r}: {head!r}"

    import augmentum.architect.primitives.web_search  # noqa: F401 — registers
    from augmentum.intent.registry import REGISTRY
    action = REGISTRY.get("web.search")
    assert action is not None
    head = action.summary.split(".")[0][:60].lower()
    assert "screen" in head, f"web.search lost 'screen' in trim window: {head!r}"


def test_web_search_default_is_headless_not_screen():
    """A plain 'search / look up / google X' should reach the HEADLESS
    web_search tool (gather + answer in words), not the screen verb.

    Pins the 2026-06-18 rebias: web.search's examples used to be the
    everyday phrasings ('google python decorators'), so the roster's
    example-relevance ranker pulled the SCREEN verb to the top for
    ordinary look-ups — she'd open a browser and guess a query instead
    of answering. The headless tool must advertise itself as the
    DEFAULT, and the screen verb must stay the EXCEPTION with
    SHOW-shaped examples only.
    """
    import augmentum.architect.primitives.web_search  # noqa: F401 — registers
    from augmentum.intent.registry import REGISTRY
    from augmentum.tools.web_search import WebSearchTool

    # Headless tool advertises itself as the default look-up path.
    desc = object.__new__(WebSearchTool).description.lower()
    assert "default" in desc, f"web_search lost its DEFAULT framing: {desc!r}"

    action = REGISTRY.get("web.search")
    summary = action.summary.lower()
    assert "exception" in summary or "not the default" in summary, (
        f"web.search must mark itself the exception: {action.summary!r}"
    )
    assert "web_search" in action.summary, (
        "web.search should redirect plain look-ups to web_search"
    )

    # No example may be a bare everyday look-up opener — those belong to
    # the headless default. Each must carry an explicit SHOW cue.
    bare_openers = ("search for ", "look up ", "google ", "find info on ")
    show_cues = ("show", "pull up", "open", "look", "results", "browse", "see")
    for ex in action.examples or []:
        low = ex.lower()
        assert not any(low.startswith(op) for op in bare_openers), (
            f"web.search example {ex!r} is an everyday look-up — it belongs "
            "on the headless web_search default; keep examples SHOW-shaped"
        )
        assert any(cue in low for cue in show_cues), (
            f"web.search example {ex!r} lacks an explicit SHOW cue"
        )


def test_web_search_query_arg_teaches_searxng_design():
    """The model DESIGNS the query — the arg schema must teach
    SearXNG-effective keyword craft, not just say 'The search query'.

    Pins the 2026-06-18 fix for 'it opens a browser and guesses a query':
    query construction was undocumented, so weak models dumped the user's
    raw sentence into a keyword engine. The schema now teaches: keywords
    not questions, drop filler, use operators (site:, quotes).
    """
    from augmentum.tools.web_search import WebSearchTool

    schema = object.__new__(WebSearchTool).input_schema
    desc = schema["properties"]["query"]["description"].lower()
    assert "keyword" in desc, f"query arg lost keyword guidance: {desc!r}"
    assert "site:" in desc, f"query arg lost operator guidance: {desc!r}"
    assert any(
        cue in desc for cue in ("verbatim", "filler", "not the user's sentence")
    ), f"query arg must steer away from raw sentences: {desc!r}"

    # Small models (E4B) get the keyword nudge via the model_hint channel
    # that _inject_tool_schemas appends to the description — no second
    # model call, the guidance rides the tool call it makes anyway.
    hint = object.__new__(WebSearchTool).model_hint.lower()
    assert "keyword" in hint, f"web_search lost its small-model hint: {hint!r}"
