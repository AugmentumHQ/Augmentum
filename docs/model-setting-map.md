# Model Setting Map

Current runtime map of model-related settings and where they are actually used.

## Source Of Truth

- Settings are declared in `augmentum/config.py`.
- Free-form model settings are accepted and persisted through `PUT /api/config/tools` in `augmentum/proxy/config_routes.py`.
- Current UI writers:
  - `ui/scripts/settings.js`
    - `syncToolSettingsToBackend()` writes most model settings, including `utility_model`, `classifier_model`, and feature-specific overrides.
    - `pushPrimaryChatModel()` mirrors the active header chat model into `primary_chat_model`.
  - `ui/scripts/models.js`
    - `persistRoleSettings()` writes `utility_model` and `classifier_model` from the Devices and Models manager.

## Core Role Resolver

`augmentum/models/provider_registry.py:613` resolves role-based text models in this order:

1. Explicit per-feature override
2. `classifier_model` when the requested role is `"classifier"`
3. `utility_model` when the requested role is `"classifier"` or `"utility"`
4. `primary_chat_model`
5. The first model on the default backend

Important current reality:

- `utility_model` is live and used in many places.
- `classifier_model` is effectively dormant right now.
- There are no current callers of `resolve_model_for_role("classifier", ...)`.
- The request classifier in `augmentum/classifier/router.py` is heuristic/rule-based and does not use `classifier_model`.

## Core Settings

### `primary_chat_model`

Origin:

- Set by the active chat model picker in the header via `ui/scripts/settings.js:7041`.
- Stored only as a server-side mirror of what the user is actively chatting with.

Used by:

- Blank-model resolution in `augmentum/models/provider_registry.py` for API/tool
  paths that call `resolve_backend_for_model("")`.
- `augmentum/models/provider_registry.py:654` as the fallback behind `utility_model` / `classifier_model`.
- `augmentum/tools/artifact_ebook.py:1068` to route ebook art-planning calls to the same backend as the user's active chat model.

Notes:

- This is not the same thing as the current request's `model` field.
- It matters most when a background/internal task says "Auto - use Primary".

### `utility_model`

Origin:

- Set in Settings and in Devices and Models.

Used by:

- Any caller that resolves the `"utility"` role and does not provide its own override.

Current utility-role callers:

- `augmentum/proxy/ui_routes.py:123` chat title generation
- `augmentum/proxy/ui_routes.py:532` character portrait prompt generation
- `augmentum/proxy/ui_routes.py:681` character/persona field enhancement
- `augmentum/proxy/ui_routes.py:1018` character card translation
- `augmentum/proxy/browse_routes.py:5801` page AI actions
- `augmentum/proxy/files_routes.py:2064` file summarization / enrichment
- `augmentum/proxy/note_intelligence_routes.py:176` note rewrite/expand/AI actions
- `augmentum/proxy/image_routes.py:165` prompt enhancement / negative prompt LLM resolution
- `augmentum/proxy/image_routes.py:229` narrative scene image handler bootstrap
- `augmentum/proxy/image_routes.py:336` visual trait extraction
- `augmentum/proxy/server.py:683` queue-side prompt condensation for image jobs
- `augmentum/image/distiller.py:971` image distillation when no distiller model is explicitly set
- `augmentum/memory/integration.py:300` batch memory extraction
- `augmentum/memory/integration.py:450` document RAG query-analysis backend resolution
- `augmentum/memory/integration.py:1107` reflection generation
- `augmentum/memory/core_profile.py:274` core profile synthesis
- `augmentum/memory/compactor.py:64` memory compaction loop
- `augmentum/modes/narrative/handler.py:983` narrative extraction
- `augmentum/dream/engine.py:353` dream generation
- `augmentum/dream/portrait.py:360` dream portrait generation

Notes:

- For many of the routes above, `utility_model` only matters when the feature-specific override is empty.
- If a feature has its own model setting, that setting wins before `utility_model` is consulted.

### `classifier_model`

Origin:

- Set in Settings and in Devices and Models.

Used by:

- Currently not used by any runtime caller.

Notes:

- The fallback logic exists in `augmentum/models/provider_registry.py:643`, but nothing currently requests the `"classifier"` role.
- The mode classifier in `augmentum/classifier/router.py` is not LLM-backed, so changing `classifier_model` does not affect chat routing today.

### Default text-model fallback

Origin:

- Not a dedicated setting field.
- This is the final fallback inside `resolve_model_for_role()` when no override, role model, or primary chat model is available.
- It is also the last resort for blank direct-model resolution after trying `primary_chat_model`.

Used by:

- Any role-based caller that falls all the way through the chain.

Notes:

- `augmentum/models/provider_registry.py:667` logs a soft warning when this fallback lands on a very small model.

## Feature Overrides And Defaults

These settings sit above or beside the core `utility -> primary -> default backend` chain.

| Setting | Fallback behavior | Current runtime use |
| --- | --- | --- |
| `ghost_text_model` | Falls back to the current chat model | `ui/scripts/workspace.js:840` |
| `memory_llm_extraction_model` | If set, it overrides memory-related utility tasks; otherwise those tasks fall back through `utility_model -> primary_chat_model -> default backend` | `augmentum/memory/integration.py:300`, `augmentum/memory/integration.py:1107`, `augmentum/memory/core_profile.py:274`, `augmentum/memory/compactor.py:64` |
| `document_rag_query_analysis_model` | Falls back to `memory_llm_extraction_model`, then the normal utility chain | `augmentum/memory/integration.py:448` |
| `image_prompt_condense_model` | Falls back through the normal utility chain | `augmentum/proxy/image_routes.py:165`, `augmentum/proxy/image_routes.py:336`, `augmentum/image/distiller.py:971`, `augmentum/proxy/server.py:683`, `augmentum/tools/image_generation.py:203` |
| `narrative_extraction_model` | Falls back through the normal utility chain | `augmentum/modes/narrative/handler.py:983` |
| `narrative_memory_model` | Uses that exact model/backend when set; otherwise uses the narrative handler's active chat model/backend | `augmentum/modes/narrative/handler.py:687`, `augmentum/modes/narrative/handler.py:763` |
| `narrative_scene_distiller_model` | Direct override for scene distillation; if empty, downstream distillation falls back to `image_prompt_condense_model` and then the utility chain | `augmentum/modes/narrative/handler.py:1534`, `augmentum/image/distiller.py:971` |
| `narrative_scene_image_model` | Falls back to `image_default_model` after explicit per-request, card, and UI panel choices | `augmentum/modes/narrative/handler.py:1438` |
| `narrative_auto_bg_distiller_model` | Overrides the scene distiller model for auto-background generation | `augmentum/modes/narrative/handler.py:1049` |
| `narrative_auto_bg_image_model` | Overrides the scene/global image model for auto-background generation | `augmentum/modes/narrative/handler.py:1045` |
| `dream_model` | Falls back through the utility chain | `augmentum/dream/engine.py:353` |
| `dream_portrait_model` | Falls back to `dream_model`, then through the utility chain | `augmentum/dream/portrait.py:360` |
| `uarf_verify_model` | Only used when a reasoning step asks for model override `"verify"`; otherwise the step keeps its default model | `augmentum/reasoning/executor.py:69` |
| `image_default_model` | Global image-generation default when no request/UI-specific image model is provided | `augmentum/tools/image_generation.py:181`, `augmentum/proxy/image_routes.py:455`, `augmentum/proxy/image_routes.py:598`, `augmentum/proxy/image_routes.py:682`, `augmentum/proxy/openai_routes.py:536`, `augmentum/proxy/openai_routes.py:639`, `augmentum/modes/v_command.py:94`, `augmentum/proxy/artifact_routes.py:1708` |
| `agentic_image_model` | Overrides `image_default_model` for agentic image tool flows | `augmentum/tools/image_generation.py:171` |
| `stt_default_model` | Provider-specific default model hint for STT integrations | `augmentum/proxy/server.py:985` |

## Practical Summary

- If you set `utility_model`, it will affect a lot of internal helper features, but only when those features do not already have a more specific model override.
- If you set `classifier_model`, nothing user-facing changes today.
- If you want a feature to stop following `utility_model`, give that feature its own explicit model setting.
- If neither explicit feature settings nor `utility_model` are set, background tasks usually land on `primary_chat_model`.
- If `primary_chat_model` is also missing, Augmentum falls through to the first model on the default backend.
