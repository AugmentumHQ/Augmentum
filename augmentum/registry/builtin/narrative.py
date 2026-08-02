"""Narrative-mode settings — full user-tunable surface migrated into
the declarative substrate.
"""

from __future__ import annotations

from augmentum.registry.registry import SettingsRegistry
from augmentum.registry.settings import Setting

_NARRATIVE_TAG = ("narrative",)
_NARRATIVE_ADV = ("narrative", "advanced")


def register(r: SettingsRegistry) -> None:
    # ---- Memory / ledger / archive ----
    r.register(
        Setting(
            key="narrative_memory_mode",
            kind="enum",
            default="standard",
            label="Long-term memory style",
            description=(
                "How the long-term memory summary is written. 'lite' = terse "
                "bullets; 'standard' = narrative prose that reads back as "
                "story continuity."
            ),
            section="narrative.memory",
            enum_values=("lite", "standard"),
            max_length=16,
            tags=_NARRATIVE_TAG,
            voice_aliases=("memory mode", "memory style"),
        )
    )
    r.register(
        Setting(
            key="narrative_smart_retrieval",
            kind="bool",
            default=True,
            label="Smart memory retrieval",
            description=(
                "Use vector + keyword hybrid retrieval for memory pulls. "
                "Disabling falls back to a flat recency window."
            ),
            section="narrative.memory",
            tags=_NARRATIVE_TAG,
        )
    )
    r.register(
        Setting(
            key="narrative_smart_retrieval_count",
            kind="int",
            default=5,
            label="Smart-retrieval memory count",
            description=(
                "How many memory fragments the smart-retrieval path injects "
                "per turn. Higher = more grounding, more tokens."
            ),
            section="narrative.memory",
            min_value=1,
            max_value=20,
            tags=_NARRATIVE_TAG,
        )
    )
    r.register(
        Setting(
            key="narrative_memory_state_enabled",
            kind="bool",
            default=True,
            label="STATE layer",
            description=(
                "Enable the STATE snapshot layer (current scene + active "
                "entities + present mood). Disable for narrative-as-chat."
            ),
            section="narrative.memory",
            tags=_NARRATIVE_TAG,
        )
    )
    r.register(
        Setting(
            key="narrative_memory_ledger_enabled",
            kind="bool",
            default=True,
            label="LEDGER layer",
            description=(
                "Enable the LEDGER summary layer (consolidated mid-term "
                "memory bridging recent turns into the archive)."
            ),
            section="narrative.memory",
            tags=_NARRATIVE_TAG,
        )
    )
    r.register(
        Setting(
            key="narrative_memory_ledger_ceiling",
            kind="int",
            default=60,
            label="LEDGER ceiling",
            description=(
                "Maximum LEDGER entries before compaction triggers. Higher "
                "= more memory continuity but more tokens spent."
            ),
            section="narrative.memory",
            min_value=0,
            max_value=500,
            tags=_NARRATIVE_TAG,
        )
    )
    r.register(
        Setting(
            key="narrative_memory_compaction_enabled",
            kind="bool",
            default=True,
            label="LEDGER compaction",
            description=(
                "Auto-compact LEDGER entries when ceiling is hit. Disabling "
                "lets the LEDGER grow unbounded — only for short sessions."
            ),
            section="narrative.memory",
            tags=_NARRATIVE_ADV,
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="narrative_memory_continuous_archive",
            kind="bool",
            default=True,
            label="Continuous ARCHIVE",
            description=(
                "Stream older LEDGER entries into the embedded ARCHIVE as "
                "they age out. Disabling truncates instead of archiving."
            ),
            section="narrative.memory",
            tags=_NARRATIVE_ADV,
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="narrative_archive_min_messages",
            kind="int",
            default=75,
            label="ARCHIVE minimum messages",
            description=(
                "Minimum messages a session needs before ARCHIVE materializes. "
                "Avoids archiving brief one-shot scenes."
            ),
            section="narrative.memory",
            min_value=0,
            max_value=500,
            tags=_NARRATIVE_ADV,
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="narrative_memory_max_words",
            kind="int",
            default=0,
            label="Memory summary max words",
            description=(
                "Word budget for the LEDGER memory summary. 0 = unbounded "
                "(falls back to token cap). Tighter caps force terser prose."
            ),
            section="narrative.memory",
            min_value=0,
            max_value=20000,
            tags=_NARRATIVE_ADV,
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="narrative_memory_max_tokens",
            kind="int",
            default=0,
            label="Memory summary max tokens",
            description=(
                "Token budget for the LEDGER memory summary. 0 = unbounded."
            ),
            section="narrative.memory",
            min_value=0,
            max_value=2000,
            tags=_NARRATIVE_ADV,
            advanced=True,
        )
    )
    # Historical inconsistency preserved: config.py ships default=4 but
    # the runtime validator floors at 5. The UI rejects user attempts to
    # set < 5; the initial value remains 4 until the user changes it.
    # Documented here so future migrations don't 'fix' the gap.
    r.register(
        Setting(
            key="narrative_memory_interval",
            kind="int",
            default=4,
            label="Memory refresh interval",
            description=(
                "Turns between LEDGER updates. Higher = less frequent (cheaper) "
                "but staler memory; lower = fresher but more tokens spent. "
                "Validator floor is 5; ships at 4 for legacy reasons."
            ),
            section="narrative.memory",
            min_value=4,  # tightened from historical 5 to honor the shipped default
            max_value=50,
            tags=_NARRATIVE_ADV,
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="narrative_memory_prompt",
            kind="str",
            default="",
            label="Custom LTM prompt",
            description=(
                "Override the system prompt used for long-term memory "
                "summarization. Empty = use the built-in template."
            ),
            section="narrative.memory",
            max_length=2000,
            tags=_NARRATIVE_ADV,
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="narrative_memory_model",
            kind="str",
            default="",
            label="LTM extraction model",
            description=(
                "Model used for LEDGER summarization calls. Empty = default "
                "backend model. Use a smaller model here to cut tokens."
            ),
            section="narrative.memory",
            max_length=256,
            tags=_NARRATIVE_ADV,
            advanced=True,
        )
    )

    # ---- Extraction / facts ----
    r.register(
        Setting(
            key="narrative_llm_extraction",
            kind="bool",
            default=True,
            label="LLM fact extraction",
            description=(
                "Use the LLM to extract structured facts (entities, plot "
                "threads, contradictions) from narrative turns. Disabling "
                "falls back to regex-based extraction."
            ),
            section="narrative.extraction",
            tags=_NARRATIVE_TAG,
        )
    )
    r.register(
        Setting(
            key="narrative_extraction_interval",
            kind="int",
            default=5,
            label="Extraction interval",
            description=(
                "Turns between fact-extraction passes. Higher = cheaper but "
                "may miss facts; lower = more thorough but more tokens."
            ),
            section="narrative.extraction",
            min_value=1,
            max_value=20,
            tags=_NARRATIVE_ADV,
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="narrative_extraction_model",
            kind="str",
            default="",
            label="Extraction model",
            description=(
                "Model used for fact-extraction calls. Empty = default backend."
            ),
            section="narrative.extraction",
            max_length=256,
            tags=_NARRATIVE_ADV,
            advanced=True,
        )
    )

    # ---- Recall tools ----
    r.register(
        Setting(
            key="narrative_recall_tools_enabled",
            kind="bool",
            default=False,
            label="LLM recall tools",
            description=(
                "Expose memory-recall lookups as LLM-callable tools so the "
                "model can ask for relevant memories explicitly during a turn."
            ),
            section="narrative.recall",
            tags=_NARRATIVE_TAG,
        )
    )
    r.register(
        Setting(
            key="narrative_recall_tools_max_iters",
            kind="int",
            default=3,
            label="Recall tool max iterations",
            description=(
                "How many recall-tool calls the model can make per turn before "
                "the loop is forced to converge."
            ),
            section="narrative.recall",
            min_value=1,
            max_value=10,
            tags=_NARRATIVE_ADV,
            advanced=True,
        )
    )

    # ---- Lorebook tools ----
    r.register(
        Setting(
            key="narrative_lorebook_tools_enabled",
            kind="bool",
            default=False,
            label="LLM lorebook tools",
            description=(
                "Expose lorebook search/create/update/delete as LLM-callable "
                "tools so the model can look up and author world info entries "
                "during narrative turns."
            ),
            section="narrative.recall",
            tags=_NARRATIVE_TAG,
        )
    )
    r.register(
        Setting(
            key="narrative_lorebook_native_tools_enabled",
            kind="bool",
            default=True,
            label="Native lorebook grounding tools",
            description=(
                "Expose the native lorebook.check / lorebook.create verbs "
                "(F1/F5) so the model grounds descriptions in established "
                "lore mid-scene and records newly-established detail as "
                "session-scoped lore (never modifies the source card). "
                "These are the tools the companion training data uses; "
                "default-on so a trained model's calls reach real handlers."
            ),
            section="narrative.recall",
            tags=_NARRATIVE_TAG,
        )
    )
    r.register(
        Setting(
            key="narrative_world_systems_enabled",
            kind="bool",
            default=True,
            label="World systems (card-declared)",
            description=(
                "Honor a character card's extensions.world_system manifest "
                "(trackers, tables, dice, status sheet). Cards without a "
                "manifest are unaffected; turn this off to play a manifest "
                "card as pure prose."
            ),
            section="narrative.recall",
            tags=_NARRATIVE_TAG,
        )
    )

    # ---- Context budgeting ----
    r.register(
        Setting(
            key="narrative_context_budget",
            kind="int",
            default=0,
            label="Soft context budget",
            description=(
                "Soft token budget for narrative context (memory + lorebook + "
                "scene). 0 = no soft cap. Cuts the lowest-priority sections first."
            ),
            section="narrative.context",
            min_value=0,
            max_value=128000,
            tags=_NARRATIVE_ADV,
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="narrative_context_limit",
            kind="int",
            default=0,
            label="Hard context limit",
            description=(
                "Hard token cap for narrative context. 0 = no hard cap. "
                "Overrides budget when set."
            ),
            section="narrative.context",
            min_value=0,
            max_value=500000,
            tags=_NARRATIVE_ADV,
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="narrative_scene_context_rounds",
            kind="int",
            default=3,
            label="Recent-turns scene window",
            description=(
                "How many of the most-recent turns get re-included verbatim "
                "in the next prompt. Higher = more local continuity, more tokens."
            ),
            section="narrative.context",
            min_value=1,
            max_value=10,
            tags=_NARRATIVE_TAG,
        )
    )
    r.register(
        Setting(
            key="narrative_request_log_limit",
            kind="int",
            default=10,
            label="Inspector log depth",
            description=(
                "How many recent narrative prompts the inspector retains for "
                "debugging. Cheap; safe to leave at default."
            ),
            section="narrative.context",
            min_value=5,
            max_value=50,
            tags=_NARRATIVE_ADV,
            advanced=True,
        )
    )

    # ---- Backgrounds / scene images ----
    r.register(
        Setting(
            key="narrative_auto_background",
            kind="bool",
            default=False,
            label="Auto-generate scene backgrounds",
            description=(
                "After each major scene shift, generate a background image "
                "via the configured image model. Adds latency + GPU cost."
            ),
            section="narrative.scene",
            tags=_NARRATIVE_TAG,
        )
    )
    r.register(
        Setting(
            key="narrative_auto_background_interval",
            kind="int",
            default=4,
            label="Background generation interval",
            description=(
                "Minimum turns between auto-background generations. Higher = "
                "fewer regenerations, less cost."
            ),
            section="narrative.scene",
            min_value=1,
            max_value=20,
            tags=_NARRATIVE_TAG,
        )
    )
    r.register(
        Setting(
            key="narrative_auto_bg_distiller_model",
            kind="str",
            default="",
            label="Background prompt distiller",
            description=(
                "Model used to distill a background image prompt from the "
                "current scene. Empty = default backend model."
            ),
            section="narrative.scene",
            max_length=256,
            tags=_NARRATIVE_ADV,
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="narrative_auto_bg_image_model",
            kind="str",
            default="",
            label="Background image model",
            description=(
                "Image model used to generate scene backgrounds. Empty = "
                "default image backend."
            ),
            section="narrative.scene",
            max_length=256,
            tags=_NARRATIVE_ADV,
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="narrative_scene_image_model",
            kind="str",
            default="",
            label="In-scene image model",
            description=(
                "Image model invoked for /v scene illustrations during "
                "narrative mode. Empty = default image backend."
            ),
            section="narrative.scene",
            max_length=256,
            tags=_NARRATIVE_TAG,
        )
    )
    r.register(
        Setting(
            key="narrative_scene_distiller_model",
            kind="str",
            default="",
            label="In-scene prompt distiller",
            description=(
                "Model used to write image prompts for /v scene illustrations. "
                "Empty = default backend."
            ),
            section="narrative.scene",
            max_length=256,
            tags=_NARRATIVE_ADV,
            advanced=True,
        )
    )

    # ---- Translation ----
    r.register(
        Setting(
            key="narrative_translate_auto_save",
            kind="bool",
            default=True,
            label="Auto-save translated cards",
            description=(
                "When a character card translation completes, save it to the "
                "card immediately instead of asking."
            ),
            section="narrative.translate",
            tags=_NARRATIVE_TAG,
        )
    )
    r.register(
        Setting(
            key="narrative_translate_default_language",
            kind="str",
            default="English",
            label="Default translate language",
            description=(
                "Target language for the 'Translate card' action. Free-form "
                "language name (e.g. 'English', 'Spanish', 'Japanese')."
            ),
            section="narrative.translate",
            max_length=64,
            tags=_NARRATIVE_TAG,
        )
    )
