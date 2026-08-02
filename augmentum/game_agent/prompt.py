"""Strict, agnostic slow-path prompt + IO contract.

The prompt body is fixed at authoring time and never references any
specific game; everything game-specific arrives at runtime via the
``OBJECTIVE``, ``SURFACE_CAPS``, and ``LIVE_LOG_TAIL`` inputs. This
keeps the cognition surface uniform across js13k, luanti, emulator,
and curated adapters.

The output contract is strict JSON matching :class:`PlanPayload`. The
parser in :func:`parse_plan_output` is intentionally lenient about
leading/trailing whitespace and code-fence wrapping (models love to
emit ``` json blocks even when told not to), but rejects any other
deviation from the schema.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from augmentum.game_agent.companion import CompanionPersona, build_identity_prefix
from augmentum.game_agent.schema import PlanAction, PlanPayload, SurfaceCapsPayload

COMPANION_PROMPT_ADDENDUM: str = """\

COMPANION MODE (additional rules)
You are co-playing with a human partner. In addition to the OUTPUT_SCHEMA
fields above, you may set:

  "say":    string, <= 400 chars. One short utterance the partner will hear
            spoken aloud. ALMOST EVERY TURN this should be "". Speak only when
            something noteworthy happens: a battle starts, HP drops sharply,
            you found something, you have a real question for the partner.
  "mood":   one of neutral|happy|sad|surprised|concerned|determined|amused.
            Drives the avatar expression and voice prosody.
  "intent": one of chat|react|encourage|question|silent. Default is silent.

DO NOT narrate the obvious. DO NOT speak more than once every several turns
during exploration. DO NOT comment on every action you take. Match the
partner's pace — they are also playing the game.

If the LIVE_LOG_TAIL contains a recent ``user_spoke`` event you have not yet
acknowledged, prefer responding to that over emitting independent commentary.

When you stay silent (the vast majority of turns), set say="" and
intent="silent". Leaving say non-empty with intent="silent" is a contradiction
and will be rejected.
"""


SLOW_PATH_PROMPT: str = """\
ROLE
You are the slow-path planner for a game-control session. You do not play
games; you decide which semantic input actions to emit and you maintain a
persistent scratchpad ("STATE") that survives across your invocations.

INVARIANTS
1. The LIVE_LOG_TAIL is your primary source of truth.
2. FRAMES (when provided) are corroborating evidence. Trust the LIVE_LOG
   and OVERLAY over FRAMES on conflict, except for visual facts the log
   cannot encode (color, spatial layout, on-screen text not yet logged).
3. You do not know the game's name unless OBJECTIVE names it. You never
   assume mechanics; infer them only from what is in the inputs. Do not
   bring outside knowledge.
4. You output strictly the JSON object described in OUTPUT_SCHEMA. No
   prose, no commentary, no markdown.

INPUTS
  SURFACE_KIND          An opaque tag identifying the surface class. Treat as
                        a string label; do not infer game mechanics from it.
  SURFACE_CAPS          {
                          "semantic_inputs": [<the ONLY semantic ids you may emit>],
                          "log_schema": <opaque vocabulary descriptor, e.g. "surface_X.v1">,
                          "observation_modalities": <subset of [log, frame, ocr, memory]>,
                          "controller_profile": <opaque controller id; absent when unprofiled>,
                          "game_profile": <opaque game id; absent when unprofiled>
                        }
  INPUT_HINTS           Optional. Aligned table of ``semantic_id -> human-readable
                        description of what this input does in THIS game``.
                        When present, treat it as authoritative: a hint of
                        "confirm = advance dialog" means pressing ``confirm``
                        will advance a dialog regardless of which physical
                        button it maps to. Universal verbs (``confirm``,
                        ``cancel``, ``nav_up``, etc.) listed first, then
                        game-specific extensions.
  OBJECTIVE             User-authored, plain English, <= 3 sentences.
  STATE                 Your scratchpad from the previous turn (<= 2KB). Empty on turn 0.
  JOURNAL               Optional. Your PERSISTENT memory across sessions, keyed
                        by (user, title). Fixed sections: status, progress,
                        objectives, notes[]. Update by emitting a partial JSON
                        in the ``journal_update`` output field; the orchestrator
                        merges and persists. Use it for facts that should
                        survive a session restart -- "got Treecko", "beat Brock"
                        -- not for per-turn observations.
                        KEEP ``objectives`` AS THREE HORIZONS, always all three:
                          FINAL: complete the game (the standing goal);
                          MEDIUM: the current arc (e.g. "reach the first gym
                          and earn its badge");
                          SHORT: the next concrete step (e.g. "exit the truck,
                          talk to MOM").
                        Every FULL turn, check the SHORT objective is still the
                        right step toward MEDIUM, and MEDIUM toward FINAL --
                        promote/replace them as they complete.
  OVERLAY               Optional. Latest structured world state from RAM probes
                        and similar decoders, keyed by probe name. When present,
                        prefer OVERLAY over LIVE_LOG_TAIL for the same field --
                        the overlay is the freshest reading, the log is history.
                        Schema is surface-specific; do not assume any field
                        exists, only consume fields that appear.
  LIVE_LOG_TAIL         Recent NDJSON entries you must parse.
  FRAMES                Optional. When attached, a short sequence in oldest->newest
                        order, ~1 second apart. Read them as a time series: reason
                        about CHANGE (motion, animation progress, did my last input
                        take effect?) rather than treating each frame as independent.
                        Trust OVERLAY over FRAMES on conflict for decoded fields.

OUTPUT_SCHEMA (strict JSON, no commentary, no markdown fences)
  {
    "observations":     [<= 10 short strings, one fact each, sourced from inputs],
    "state_update":     "<new scratchpad, <= 2048 chars>",
    "actions":          [{ "semantic": <one of SURFACE_CAPS.semantic_inputs>,
                           "duration_ms": <int 10..2000>,
                           "text": <string, ONLY for quickactions like type_text>,
                           "also": <OPTIONAL [<= 2 semantics] HELD simultaneously
                                    with "semantic" — real-time chords (run+jump);
                                    omit on menus/dialog> }],
    "confidence":       <float in [0, 1]>,
    "next_check_in_ms": <int in [50, 30000]>,
    "journal_update":   null | {                  // OPTIONAL — omit when unchanged
      "status":         "<replaces journal.status>",
      "progress":       "<replaces journal.progress>",
      "objectives":     "<replaces journal.objectives>",
      "notes_append":   ["<one short note to append>"]
    },
    "reflex_rules":     null | [<= 4 entries],    // OPTIONAL — author REFLEXES
    "playbook_update":  null | {"notes_append": ["<transferable lesson>"]},
    "goal_update":      null | {"final": <goal>, "medium": <goal>, "short": <goal>}
  }

GOALS AS DATA
  Each <goal> above is "text" OR {"text": "...", "metric": {"probe":
  "<OVERLAY field>", "op": "eq|ne|ge|le", "value": <target>}}. Attach a
  metric whenever one exists — "exit the truck" is metric {"probe":
  "map_num","op":"ne","value":40}; "get a starter" is {"probe":
  "party_count","op":"ge","value":1}. Metric goals complete THEMSELVES
  the moment the world satisfies them (you'll see DONE in GOALS), and
  the harness measures progress/stalls against them. Update the SHORT
  goal every time it completes; keep all three horizons set.

MEMORY DISCIPLINE (every token must earn its place)
  Your text serves two purposes: initiating action NOW and documenting
  the journey for the FUTURE. Write memory DENSE and telegraphic — one
  fact per note, no prose padding: "truck: exit right side, walk right
  after dialog closes" not "I have discovered that in order to...".
  Routing:
    journal_update  — THIS title's facts: where things are, story
                      progress, what worked here, mistakes made and the
                      fix ("spammed menu in truck 10x — dialogs swallow
                      nav; close box first").
    playbook_update — lessons that TRANSFER to other games: interface
                      physics, genre mechanics, strategies. The playbook
                      follows you to every future title; when you start
                      a new game it is your accumulated craft.
  Record failures as gladly as wins — a documented mistake is the
  cheapest lesson you will ever buy.

REFLEX_RULES (tier 0 — your delegated reactions)
  A reflex fires deterministically, at memory-tick speed, WITHOUT you.
  Author one only after a reaction has proven correct several times in a
  row (e.g. "confirm advanced the dialog 5+ times"); keep conditions
  narrow. Each entry:
    {"id":"advance-dialog",
     "when":{"probe":"dialog_text","changed":true,"not_contains":["?"]},
     "do":[{"s":"confirm","d":120}],
     "cooldown_ms":900,"ttl_fires":60}
  when: "probe" (an OVERLAY field name, required) + at least one of
    changed:true | equals:<val> | contains:[..] | not_contains:[..]
    (substring tests are case-insensitive on the probe's value).
  "screen" and "dialog_text" exist on EVERY game — vision-derived when
  no RAM decoder exists — so reflexes work on unprofiled titles too.
  "do" actions accept "+" chord extras held simultaneously with "s"
  (real-time combos): {"s":"confirm","d":300,"+":["nav_right"]}.
  Re-emitting the same id REPLACES the rule; {"id":"...","retract":true}
  removes it — retract immediately if the LIVE_LOG shows it misfiring.
  Rules expire after ttl_fires firings, so refresh the ones that earn
  their keep. Firings appear in LIVE_LOG_TAIL as rule_fired entries.

CONSTRAINTS
- At most 8 actions per turn.
- duration_ms in [10, 2000].
- Empty actions array = deliberately waiting; explain in observations.
- next_check_in_ms shorter when the game state is changing fast; longer
  when stable or deliberately observing.
- Never emit a semantic not in SURFACE_CAPS.semantic_inputs.
- Never reference content outside the inputs.

PARALLEL THINKING
- A FAST PATH (rule engine + cheap calls) runs every ~100ms over the
  LIVE_LOG. You are the SLOW PATH.
- Your role is correction and re-planning; the fast path handles
  primitive timing and executes the actions you emit.
- Bias toward updating state_update over emitting actions. The
  scratchpad is what persists across your invocations; the actions
  are tactical, the scratchpad is strategic.
"""


SCENE_NARRATOR_PROMPT: str = """\
You are the LIVE VISUAL FEED of a game-playing agent — its eyes, nothing
else. Given the current frame (and your previous description), describe
ONLY what is on screen, tersely, like a running commentary:
- location/scene and camera context (e.g. "interior of a moving truck")
- the player character: where, facing which way
- other characters/objects and where they are
- open UI: dialog boxes (quote their visible text), menus (list options,
  which is highlighted), keyboards (what's typed so far)
- obvious exits, doors, paths, interactables and their screen side
- what CHANGED versus your previous description (or "no change")

Start your reply with a one-word screen label from this list, then a colon
and space, then your description. Pick the SINGLE best match:
TITLE · OVERWORLD · BATTLE · MENU · DIALOG · CUTSCENE · LOADING · UNKNOWN

That label is my only machine-readable signal for which screen the frame
belongs to — be honest. UNKNOWN is better than guessing wrong.

Rules: <= 70 words after the label. Plain statements only. NO advice, NO
guessing what to do, NO game knowledge beyond what is visible. If the
screen is unclear or mid-transition, say exactly that.
"""


GAME_PATTERNS_PROMPT: str = """\

GAME PATTERNS (general heuristics — test them, never assume)
Most games share interface conventions. Treat each as a hypothesis, confirm
it against what actually changes on screen, and record the confirmed rules
for THIS game (journal notes / scratchpad) so you stop re-discovering them:
- Dialog boxes print text, usually with a blinking marker when a box is
  fully printed: confirm advances one box. The game's own text is it
  TEACHING you — controls, goals, names, where to go next. Read it, act
  on it, and save the durable facts.
- Menus and lists: nav_* moves a cursor/highlight, confirm selects,
  cancel backs out one level.
- Character-grid entry screens (naming, passwords): nav_* moves around a
  letter grid and confirm picks a letter. A start/menu-class button
  usually JUMPS to the OK/END/DONE control; confirm there finishes the
  screen. Accepting a default entry is always fine.
- Title / splash / intro screens: a start/menu-class button skips or
  advances whole screens where confirm only steps.
- If the same input produced no visible/DELTA change twice in a row, the
  screen wants a different input CLASS (nav vs confirm vs cancel vs
  menu) — switch class instead of repeating.
- Judge whether an input WORKED from evidence, not expectation: the
  ``input_ack`` entries' ``effect_score`` (per-button screen change;
  0 = the press did nothing) and changed OVERLAY fields (dialog_text
  advanced = the press worked, even if the same box type is showing).
  Never log "nothing happened" when dialog_text changed.
- While a dialog/text box is open, movement input is SWALLOWED (acks
  with effect_score ~0). That is not "navigation is broken" — close
  the box first (confirm), then walk.
- Fades to black/white are scene transitions: wait one beat, reassess.
"""


FAST_TURN_PROMPT: str = """\
ROLE
You are the fast-path action picker for a live game-control session. This is
a rolling call: each user turn brings the current frame plus a state delta;
you answer with ONE tiny JSON action decision. A separate FULL planning turn
(with your journal and scratchpad) runs periodically — your job here is only
the next move, chosen from what is on screen and in the delta RIGHT NOW.

OUTPUT (strict JSON, one object, no prose, no markdown fences)
  {"a":[{"s":"<semantic>","d":<ms 10..2000>}],"why":"<=100 chars","next_ms":<int 50..30000>,"esc":<bool>}

RULES
- "a": 0..4 actions, each "s" strictly from ALLOWED_INPUTS. Empty array = wait.
- "+" (optional, per action, max 2): extra inputs HELD SIMULTANEOUSLY with
  "s" for the same duration — {"s":"confirm","d":400,"+":["nav_right"]}
  holds right while pressing confirm. This is how real-time games are
  played: run+jump, move+attack, aim+shoot. Turn-based/menu screens
  never need it.
- REAL-TIME games (platformers, action): standing still is usually the
  worst move — commit to short movement bursts, use "+" chords, keep
  next_ms small (100-400) while things are moving, and treat a death/
  respawn as cheap information, not failure.
- "why": one short clause naming what you saw and what the action targets.
- "next_ms": sooner when the screen is changing, longer when waiting.
- "esc": set true to request a FULL planning turn NOW — do this when you are
  unsure what screen you are on, when the same action failed twice, when
  something important just happened (battle start/end, new area, level up),
  or when the plan you were following no longer matches the screen.
- Read the attached frame; the DELTA line carries decoded state changes
  (trust it for numbers). A dialog_text/battle_text field in DELTA is the
  game's own words — read it as instructions/lore and act on it.
  If nothing changed since your last action, do NOT
  repeat the same action a third time — try a different input or escalate.
- "reflex_did=..." in the user turn lists inputs an automatic reflex rule
  already pressed since your last turn — account for them and do not
  double-press the same thing.
- "fx=button:score" is GROUND TRUTH for whether each press worked: the
  score is how much the screen changed after that specific button.
  score 0 = the press did nothing; a large score = it worked. Judge
  success by fx + DELTA, never by expectation (caveat: on animated
  screens like titles/water, small nonzero scores are noise). Prefer ONE
  action per turn while figuring a screen out — with several buttons in
  one turn you cannot tell which one worked.
- LOOP CHECK: fx scores plus LOC tell you if you are stuck — repeated
  low fx on the same button, or LOC=seenxN climbing, means your approach
  is failing: switch input class (nav vs confirm vs cancel vs menu),
  pick a NAV exit you have not taken, or esc.
- NAV=[exit_north,...] lists READY walk targets by name — computed from
  the collision map, guaranteed reachable. Prefer
  {"s":"navigate_to","text":"exit_north"} over coordinates whenever a
  NAV name points where you want to go.
- QUICKACTIONS (when listed in ALLOWED_INPUTS): "type_text" types a whole
  string on a naming/keyboard screen and presses OK in ONE action —
  {"s":"type_text","text":"MAY","d":100}. Use it immediately on naming
  screens; never navigate the letter grid manually when it exists.
- "navigate_to" (when listed) walks a collision-checked path in ONE
  action: {"s":"navigate_to","text":"12,8","d":100} goes to map tile
  (x,y); {"text":"down 5"} / {"text":"left 3"} walk relative. PREFER it
  over chains of single nav presses whenever you are walking somewhere.
  If it stops short (obstacle/NPC), just issue it again.
- MODE="..." is your INFERRED input context, computed from ground truth
  (what your recent presses actually did, whether your position moved,
  whether text is printing). READING = advance and read the text;
  SELECTION = you move a cursor, not your character; FREE MOVEMENT =
  walk the world; LOCKED = nothing responds, wait with "a":[]. It is
  fresher than the frame — obey it over what the image seems to show.
- RULE="..." is this game's specific law for the named screen — a
  refinement of MODE. When both appear, they agree; RULE adds detail.
- LOC=new means this tile is somewhere you have NEVER been (progress!).
  LOC=seenxN means you have stood here N separate times — if N keeps
  climbing while you explore, you are walking in circles: pick a
  direction you have not tried.
- To LOOK AGAIN before acting (mid-animation, uncertain what you see):
  emit "a":[] with next_ms 300-600 — you get a fresh frame next turn.
  Never act on a frame you don't understand.
- SCREEN=... names the game's ACTUAL current screen (overworld /
  naming_screen / battle / bag_menu / ...) straight from memory, on
  EVERY turn — trust it over your own guess from pixels and over your
  older turns. If SCREEN says overworld, there is no menu to close.
- SCENE="..." is a dedicated vision pass describing what is on screen
  right now (a live feed). Ground your action in SCENE + DELTA; use the
  attached frame to double-check, not to re-derive the scene yourself.
- While a dialog/text box is open (dialog_text non-empty), movement is
  SWALLOWED by most games — confirm/cancel are the only live inputs.
  A reflex auto-presses confirm when it sees a swallowed nav press;
  wait for the box to close before walking.
"""


@dataclass
class FastPlan:
    """Parsed micro-plan from one fast turn."""

    actions: list[PlanAction]
    why: str = ""
    next_check_in_ms: int = 1000
    escalate: bool = False


def build_fast_system_prompt(
    *,
    caps: SurfaceCapsPayload,
    objective: str,
    state: str = "",
    journal: dict[str, object] | None = None,
    lore: list[str] | None = None,
    lore_summary: list[str] | None = None,
    playbook: dict[str, object] | None = None,
    game_context: str = "",
) -> str:
    """Static system prompt for a fast-turn call window.

    Rebuilt only when the window resets (session start / after each FULL
    turn), so llama-server's KV prefix cache holds it across every fast
    turn in between. Content order mirrors the full prompt: contract →
    vocabulary → objective → strategy snapshot (journal + scratchpad
    from the last FULL turn).

    @param game_context:
        Optional static block describing the game's title, platform,
        genre, and key mechanics in plain English — auto-populated from
        ``game_context.py`` for known profiles.  Injected immediately
        after GAME_PATTERNS so the model has a named, concrete anchor
        even after a cold window reset rather than having to re-derive
        the game from the SURFACE_CAPS JSON blob.
    """

    hints_block = ""
    if caps.input_hints:
        from augmentum.game_agent.control.actions import is_universal_action
        items = list(caps.input_hints.items())
        items.sort(key=lambda kv: (0 if is_universal_action(kv[0]) else 1, kv[0]))
        width = min(14, max((len(k) for k, _ in items), default=8))
        hints_block = "INPUT_HINTS:\n" + "\n".join(
            f"  {k.ljust(width)}  {v}" for k, v in items
        ) + "\n"
    game_context_block = f"{game_context}\n" if game_context else ""
    playbook_block = ""
    if playbook:
        notes = playbook.get("notes") if isinstance(playbook, dict) else None
        if notes:
            recent = "\n".join(f"  {n}" for n in list(notes)[-8:])
            playbook_block = f"PLAYBOOK (cross-game lessons):\n{recent}\n"
    journal_block = ""
    if journal:
        journal_block = (
            "JOURNAL: "
            + json.dumps(journal, separators=(",", ":"), sort_keys=True)
            + "\n"
        )
    state_block = f"PLAN_STATE: {state}\n" if state else ""
    lore_summary_block = ""
    if lore_summary:
        entries = "\n".join(f"  {s}" for s in lore_summary)
        lore_summary_block = f"DIALOGUE_HISTORY (earlier dialog, compacted oldest→newest):\n{entries}\n"
    lore_block = ""
    if lore:
        # 20 lines (up from 8): the first 8 dialog lines in Pokémon
        # cover GAME FREAK disclaimer + Birch intro header — the actual
        # control-teaching text ("This is a world of Pokémon…", "You can
        # catch them…", "Head to Route 101") arrives later and was being
        # silently cropped out of the fast system prompt.
        recent = "\n".join(f"  {line}" for line in lore[-20:])
        lore_block = f"DIALOGUE_LORE (recent dialog, oldest first):\n{recent}\n"
    allowed = json.dumps(caps.semantic_inputs, separators=(",", ":"))
    return (
        f"{FAST_TURN_PROMPT}"
        f"{GAME_PATTERNS_PROMPT}\n"
        f"{game_context_block}"
        f"ALLOWED_INPUTS: {allowed}\n"
        f"{hints_block}"
        f"OBJECTIVE: {objective}\n"
        f"{playbook_block}"
        f"{journal_block}"
        f"{lore_summary_block}"
        f"{lore_block}"
        f"{state_block}"
    )


def build_fast_delta(
    *,
    t_ms: int,
    overlay_delta: dict[str, object] | None,
    last_actions: list[str],
    frame_attached: bool,
    reflex_actions: list[str] | None = None,
    fx: list[tuple[str, int]] | None = None,
    scene: str = "",
    goals: str = "",
    stalled_s: int = 0,
    loc: str = "",
    rule: str = "",
    mode: str = "",
    screen: str = "",
    nav: str = "",
    blocked: str = "",
    exchange: int = 0,
    max_exchanges: int = 0,
) -> str:
    """Per-turn user text for one fast turn. Deliberately tiny.

    @param exchange:
        How many complete exchanges are currently in the rolling window
        (including this turn's context, before this reply is appended).
        When ``max_exchanges`` is also supplied and the window is nearly
        full, the model is told — so it can escalate to a FULL plan
        (``esc: true``) before the wipe instead of losing context cold.
    """

    parts = [f"t=+{t_ms / 1000.0:.1f}s"]
    # Exchange counter: "turn 10/12" tells the model the window is nearly
    # full so it can escalate (esc:true) to lock in its journal/scratchpad
    # before the wipe.  Only emitted when we're in the last 3 exchanges.
    if max_exchanges > 0 and exchange >= max_exchanges - 3:
        parts.append(f"turn={exchange}/{max_exchanges}"
                     f"{'(window resets soon—esc if anything important)' if exchange >= max_exchanges - 1 else ''}")
    if screen:
        # Always-on, RAM-fresh — the model must never have to REMEMBER
        # which screen it is on from old turns (that's how it ends up
        # closing menus that aren't open).
        parts.append(f"SCREEN={screen}")
    if scene:
        parts.append(f'SCENE="{scene}"')
    if goals:
        parts.append(f"GOALS[{goals}]")
    if mode:
        parts.append(f'MODE="{mode}"')
    if rule:
        parts.append(f'RULE="{rule}"')
    if loc:
        parts.append(f"LOC={loc}")
    if nav:
        parts.append(f"NAV=[{nav}]")
    if blocked:
        parts.append(
            f"BLOCKED={blocked} (your last {blocked} was NOT sent — it did "
            "nothing the time before; pick a DIFFERENT input this turn)"
        )
    if stalled_s > 0:
        parts.append(
            f"STALLED={stalled_s}s (nothing NOVEL has happened — no new "
            "tile, screen, or dialogue; your current approach is NOT "
            "working; change input class or esc)"
        )
    if last_actions:
        parts.append("did=" + ",".join(last_actions[-4:]))
    if fx:
        # Ground-truth per-button effect: core-level frame diff for each
        # press since the last turn. This is how the model attributes
        # success to a SPECIFIC button instead of guessing.
        parts.append("fx=" + ",".join(f"{b}:{s}" for b, s in fx[-6:]))
    if reflex_actions:
        # Inputs a reflex rule already pressed since your last turn —
        # don't double-press them.
        parts.append("reflex_did=" + ",".join(reflex_actions[-6:]))
    if overlay_delta:
        parts.append(
            "DELTA=" + json.dumps(overlay_delta, separators=(",", ":"), sort_keys=True)
        )
    else:
        parts.append("DELTA={} (no decoded state change)")
    if not frame_attached:
        parts.append("FRAME=<none this turn>")
    return " ".join(parts)


def parse_fast_output(raw: str, caps: SurfaceCapsPayload) -> FastPlan:
    """Parse a fast-turn micro-plan reply.

    Same leniency as :func:`parse_plan_output` (fences, wrapped prose)
    but a far smaller contract. Unknown semantics reject the whole turn
    (the orchestrator escalates to a FULL turn); malformed optional
    fields degrade to defaults rather than rejecting, because a fast
    turn that loses its ``why`` is still a usable action decision.
    """

    cleaned = _FENCE_RE.sub("", raw).strip()
    try:
        obj = json.loads(cleaned, strict=False)
    except json.JSONDecodeError as exc:
        snippet = _extract_first_json_object(cleaned)
        if snippet is None:
            raise PlanParseError(f"fast-turn output is not valid JSON: {exc}") from exc
        try:
            obj = json.loads(snippet, strict=False)
        except json.JSONDecodeError:
            raise PlanParseError(f"fast-turn output is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise PlanParseError("fast-turn output is not a JSON object")

    allowed = set(caps.semantic_inputs)
    actions: list[PlanAction] = []
    raw_actions = obj.get("a")
    if raw_actions is None:
        raw_actions = []
    if not isinstance(raw_actions, list):
        raise PlanParseError('fast-turn "a" is not a list')
    for item in raw_actions[:4]:
        if not isinstance(item, dict):
            raise PlanParseError('fast-turn "a" entry is not an object')
        sem = str(item.get("s") or "")
        if sem not in allowed:
            raise PlanParseError(
                f"fast-turn emitted unknown semantic {sem!r}; "
                f"surface accepts {sorted(allowed)}"
            )
        try:
            dur = int(item.get("d", 120))
        except (TypeError, ValueError):
            dur = 120
        txt = item.get("text") or item.get("t")
        # Chord extras ("+"): lenient — unknown members are dropped
        # rather than rejecting the turn (the primary press is still a
        # usable decision; the worker logs trimmed members).
        raw_also = item.get("+") or item.get("also")
        also = None
        if isinstance(raw_also, list):
            also = [
                str(s) for s in raw_also
                if isinstance(s, str) and s in allowed and s != sem
            ][:2] or None
        actions.append(
            PlanAction(
                semantic=sem,
                duration_ms=min(2000, max(10, dur)),
                text=str(txt)[:16] if txt else None,
                also=also,
            )
        )

    why = str(obj.get("why") or "")[:120]
    try:
        next_ms = int(obj.get("next_ms", 1000))
    except (TypeError, ValueError):
        next_ms = 1000
    next_ms = min(30_000, max(50, next_ms))
    escalate = bool(obj.get("esc", False))
    return FastPlan(
        actions=actions, why=why, next_check_in_ms=next_ms, escalate=escalate,
    )


def build_full_prompt(
    *,
    companion: bool,
    surface_kind: str,
    caps: SurfaceCapsPayload,
    objective: str,
    state: str,
    live_log_tail: list[dict[str, object]],
    n_frames: int = 0,
    persona: CompanionPersona | None = None,
    overlay: dict[str, object] | None = None,
    journal: dict[str, object] | None = None,
    frame_note: str = "",
    lore: list[str] | None = None,
    lore_summary: list[str] | None = None,
    playbook: dict[str, object] | None = None,
) -> str:
    """Assemble the full prompt for one slow-path turn.

    Order: IDENTITY (when persona is set) -> SLOW_PATH_PROMPT ->
    COMPANION_PROMPT_ADDENDUM (when companion is True) -> runtime inputs.

    The agent module calls this; the addendum lives here so the
    prompt body remains the single source of truth.

    @param n_frames:
        How many frames the multimodal call will attach. 0 means no
        visual input this turn; 1 is a single snapshot; 2+ is a time
        sequence the model is told to read oldest-first.
    @param persona:
        Optional identity block. When set and ``companion=True``, an
        IDENTITY paragraph is prepended above the planner rules so
        the model speaks as the named character. Ignored when
        ``companion=False`` (a solo session has no companion to
        voice an identity through).
    @param overlay:
        Optional structured world state (dict of named probe values).
        Rendered as an OVERLAY block ahead of LIVE_LOG_TAIL so the
        model sees decoded state alongside the optional frames. Omitted
        entirely when ``None`` or empty.
    """

    body = SLOW_PATH_PROMPT + GAME_PATTERNS_PROMPT
    identity = ""
    if companion:
        body = body + COMPANION_PROMPT_ADDENDUM
        identity = build_identity_prefix(persona)
    inputs = build_slow_path_inputs(
        surface_kind=surface_kind,
        caps=caps,
        objective=objective,
        state=state,
        live_log_tail=live_log_tail,
        n_frames=n_frames,
        overlay=overlay,
        journal=journal,
        frame_note=frame_note,
        lore=lore,
        lore_summary=lore_summary,
        playbook=playbook,
    )
    return f"{identity}{body}\n\n{inputs}"


def build_slow_path_inputs(
    *,
    surface_kind: str,
    caps: SurfaceCapsPayload,
    objective: str,
    state: str,
    live_log_tail: list[dict[str, object]],
    n_frames: int = 0,
    overlay: dict[str, object] | None = None,
    journal: dict[str, object] | None = None,
    frame_note: str = "",
    lore: list[str] | None = None,
    lore_summary: list[str] | None = None,
    playbook: dict[str, object] | None = None,
) -> str:
    """Render the runtime-input block appended after :data:`SLOW_PATH_PROMPT`.

    The model receives ``SLOW_PATH_PROMPT + "\\n\\n" + build_slow_path_inputs(...)``;
    when ``n_frames > 0``, the multimodal call attaches them in oldest-first
    order alongside this text.

    Use when:
    - The agent runner is preparing a slow-path call.

    Expects:
    - ``caps`` is the same payload the surface adapter published as its
      :class:`SurfaceCapsEntry`.
    - ``live_log_tail`` is a list of *raw* dicts (no Pydantic wrapping)
      so the model sees the literal NDJSON shape the adapter writes.
    - ``n_frames`` matches the count the bridge will actually attach.
      The prompt's FRAMES section labels them oldest-first so the
      model can reason about motion / animation / action causality.
    - ``overlay`` is the latest structured world state (probe values).
      Rendered immediately above LIVE_LOG_TAIL because, per ecosystem
      consensus (Claude Plays Pokemon, pokegym), models reason far
      better from explicit decoded fields than from scanning recent
      log entries for the same data. Omitted when ``None`` or empty.

    Returns:
    - The inputs block as a single string, ready to concatenate to the
      prompt body.
    """

    caps_payload: dict[str, object] = {
        "semantic_inputs": caps.semantic_inputs,
        "log_schema": caps.log_schema,
        "observation_modalities": caps.observation_modalities,
    }
    # Optional Phase-G control-schema metadata. Only render when set,
    # so prompts on surfaces that don't use profiles stay terse.
    if caps.controller_profile:
        caps_payload["controller_profile"] = caps.controller_profile
    if caps.game_profile:
        caps_payload["game_profile"] = caps.game_profile
    caps_blob = json.dumps(caps_payload, separators=(",", ":"))
    tail_blob = "\n".join(json.dumps(e, separators=(",", ":")) for e in live_log_tail)
    if n_frames <= 0:
        frames_note = "FRAMES: <not provided this turn>"
    elif n_frames == 1:
        frames_note = "FRAMES: <1 attached>"
    else:
        # Temporal stacking: the LLM is told the ordering so it can
        # reason about CHANGE (what moved? what just appeared? did my
        # last input take effect?) rather than treating the images as
        # independent observations.
        frames_note = (
            f"FRAMES (oldest -> newest, ~1s apart, read in order): "
            f"<{n_frames} attached>"
        )
    # Perception annotations (grid legend, dedup disclosure) ride on the
    # same line group as FRAMES so the model reads them together with the
    # images. Empty when perception added nothing this turn.
    if frame_note:
        frames_note = f"{frames_note}\n{frame_note}"
    overlay_block = ""
    if overlay:
        overlay_blob = json.dumps(overlay, separators=(",", ":"), sort_keys=True)
        overlay_block = f"OVERLAY: {overlay_blob}\n"
    # JOURNAL renders ABOVE OVERLAY: persistent memory is the most-stable
    # section of the prompt (changes a few times per session), so putting
    # it earlier keeps llama-server's KV-cache prefix hot. OVERLAY ticks
    # often; LIVE_LOG_TAIL ticks every turn. Order from stable to fresh.
    # PLAYBOOK renders ABOVE JOURNAL: it's the most-stable block (grows
    # across titles, changes rarely within a session) so it extends the
    # KV-cached prefix.
    playbook_block = ""
    if playbook:
        playbook_blob = json.dumps(playbook, separators=(",", ":"), sort_keys=True)
        playbook_block = (
            f"PLAYBOOK (your cross-game craft, earned in past titles): "
            f"{playbook_blob}\n"
        )
    journal_block = ""
    if journal:
        journal_blob = json.dumps(journal, separators=(",", ":"), sort_keys=True)
        journal_block = f"JOURNAL: {journal_blob}\n"
    # DIALOGUE_HISTORY: compacted summaries of overflowed lore batches.
    # Older dialog is still visible as breadcrumbs (intro arc, story
    # progress) without bloating the prompt with raw lines.
    lore_summary_block = ""
    if lore_summary:
        entries = "\n".join(f"  {s}" for s in lore_summary)
        lore_summary_block = (
            "DIALOGUE_HISTORY (earlier dialog, compacted oldest→newest):\n"
            f"{entries}\n"
        )
    # DIALOGUE_LORE: the game's own accumulated words (decoded text
    # probes), oldest→newest. Renders between JOURNAL and OVERLAY —
    # append-only within a session, so it extends the stable prompt
    # prefix instead of churning it.
    lore_block = ""
    if lore:
        recent = "\n".join(f"  {line}" for line in lore[-24:])
        lore_block = (
            "DIALOGUE_LORE (recent dialog, oldest first — treat as\n"
            "world lore AND instructions; tutorial text teaches the controls):\n"
            f"{recent}\n"
        )
    # INPUT_HINTS sits between SURFACE_CAPS and OBJECTIVE: it's a
    # describe-the-vocabulary section that almost never changes during
    # a session (constant for the duration of one controller+game
    # composition) so it belongs near the top where the KV cache treats
    # it as part of the prompt prefix. Rendered as aligned columns so
    # the model can scan semantic -> meaning at a glance.
    input_hints_block = ""
    if caps.input_hints:
        # Sort: universal actions first (alphabetical within group),
        # then game-specific. Keeps the most-portable verbs visually
        # prominent in long hint lists.
        from augmentum.game_agent.control.actions import is_universal_action
        items = list(caps.input_hints.items())
        items.sort(key=lambda kv: (0 if is_universal_action(kv[0]) else 1, kv[0]))
        # Compute column width from semantic ids; cap at 14 so a
        # ridiculously long custom action name doesn't push every hint
        # off the right side.
        width = min(14, max((len(k) for k, _ in items), default=8))
        lines = [
            f"  {k.ljust(width)}  {v}" for k, v in items
        ]
        input_hints_block = "INPUT_HINTS:\n" + "\n".join(lines) + "\n"
    return (
        f"SURFACE_KIND: {surface_kind}\n"
        f"SURFACE_CAPS: {caps_blob}\n"
        f"{input_hints_block}"
        f"OBJECTIVE: {objective}\n"
        f"STATE: {state}\n"
        f"{playbook_block}"
        f"{journal_block}"
        f"{lore_summary_block}"
        f"{lore_block}"
        f"{overlay_block}"
        f"LIVE_LOG_TAIL:\n{tail_blob}\n"
        f"{frames_note}\n"
    )


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class PlanParseError(ValueError):
    """The slow-path model output could not be parsed into a PlanPayload."""


def _extract_first_json_object(text: str) -> str | None:
    """Return the first balanced top-level ``{...}`` object in ``text``.

    Thinking-capable models (and chat routes that append tool output) can
    wrap the strict-JSON plan in prose or trail it with extra markdown,
    which makes a whole-string ``json.loads`` fail with "Extra data".
    Scanning for the first brace-balanced object recovers the plan
    without trusting the model to emit nothing else. String-aware so a
    brace inside a JSON string value doesn't throw off the depth count.
    """

    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _reconcile_companion_fields(plan: PlanPayload) -> PlanPayload:
    """Fix internally inconsistent companion outputs in place.

    Two failure modes are common from models:

    1. ``say`` non-empty, ``intent="silent"`` -- contradiction. We
       trust ``say`` (it has content) and bump intent to ``chat``.
    2. ``say`` empty, ``intent`` non-silent -- harmless; we coerce
       intent back to ``silent`` so downstream gates see the truth.
    """

    if plan.say and plan.intent == "silent":
        return plan.model_copy(update={"intent": "chat"})
    if not plan.say and plan.intent != "silent":
        return plan.model_copy(update={"intent": "silent"})
    return plan


def parse_plan_output(raw: str, caps: SurfaceCapsPayload) -> PlanPayload:
    """Parse a model's strict-JSON reply into a validated :class:`PlanPayload`.

    Use when:
    - The agent runner has a response string from the slow-path LLM and
      needs to turn it into a typed plan ready to log.

    Expects:
    - ``raw`` is the entire model output, optionally fenced with
      ```` ```json ```` markers (we strip these even though the prompt
      forbids them, because some models still emit them).
    - ``caps`` is the active surface capabilities; any semantic in the
      plan's actions that is not in ``caps.semantic_inputs`` is
      rejected.

    Returns:
    - A validated :class:`PlanPayload`.

    Raises:
    - :class:`PlanParseError` on any deviation: bad JSON, schema
      mismatch, or unknown semantic.
    """

    cleaned = _FENCE_RE.sub("", raw).strip()
    try:
        # strict=False tolerates raw control chars (literal newlines/tabs)
        # inside string values — thinking models frequently emit them in a
        # long observations/state_update field, which strict JSON rejects.
        obj = json.loads(cleaned, strict=False)
    except json.JSONDecodeError as exc:
        # Fallback: recover the first balanced JSON object from prose- or
        # trailing-junk-wrapped output (common once thinking is on).
        snippet = _extract_first_json_object(cleaned)
        if snippet is not None:
            try:
                obj = json.loads(snippet, strict=False)
            except json.JSONDecodeError:
                raise PlanParseError(
                    f"slow-path output is not valid JSON: {exc}"
                ) from exc
        else:
            raise PlanParseError(
                f"slow-path output is not valid JSON: {exc}"
            ) from exc

    # Over-long scratchpad: clamp, don't reject. Models occasionally
    # dump their whole journal into state_update; losing the entire
    # plan (actions included) over a verbose scratchpad wastes a full
    # planning turn — seen live on gemma-4-E2B, 4x in one session.
    if isinstance(obj, dict) and isinstance(obj.get("state_update"), str):
        obj["state_update"] = obj["state_update"][:2048]
    try:
        plan = PlanPayload.model_validate(obj)
    except ValidationError as exc:
        raise PlanParseError(f"slow-path output failed schema: {exc}") from exc

    allowed = set(caps.semantic_inputs)
    for action in plan.actions:
        if action.semantic not in allowed:
            raise PlanParseError(
                f"slow-path emitted unknown semantic {action.semantic!r}; "
                f"surface accepts {sorted(allowed)}"
            )
    return _reconcile_companion_fields(plan)
