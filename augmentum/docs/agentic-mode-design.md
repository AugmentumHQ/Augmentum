# Agentic Mode — Design Document

## Overview

A fourth processing mode alongside Passthrough, Analytical, and Narrative.
Where Analytical enhances a *single response* with structured reasoning,
Agentic mode handles *multi-step autonomous tasks* that produce deliverables
(documents, presentations, reports, storybooks, code projects).

The user says: "Research climate change and create a presentation."
The agent plans, researches, drafts, generates images, assembles a PPTX,
and hands the user a download link — all while streaming live progress.

---

## Architecture Principles

1. **No VM.** Use structured artifact tools + existing sandboxed executor.
2. **Flow-native.** Agentic tasks run as reasoning flows with new step roles.
3. **Autonomy dial.** Per-flow setting controlling how much the agent asks vs. acts.
4. **Plan as attention anchor.** A running plan (todo.md pattern) is re-injected
   into every step to keep the model focused during long chains.
5. **Checkpoint persistence.** Task state survives page refreshes and disconnects.
6. **Graceful degradation.** Works with 7B models (structured tools) and shines
   with 70B+ (flexible executor-based generation).

---

## Integration Points

### 1. Mode System

```
augmentum/classifier/router.py
  Mode enum += AGENTIC = "agentic"
  MODE_PREFIXES += "g/": Mode.AGENTIC
  MODE_MAP += "agentic": Mode.AGENTIC
```

Classification: Agentic mode is **never auto-detected** — always explicit.
Users select it via the mode badge, model prefix `g/`, or header override.
This prevents accidental long-running autonomous tasks.

### 2. Handler

```
augmentum/modes/agentic/
  __init__.py
  handler.py        — AgenticHandler(ModeHandler)
  planner.py        — Plan decomposition + todo.md management
  task_state.py     — Persistent task state (checkpoints)
  autonomy.py       — Autonomy level enforcement
```

`AgenticHandler.handle_stream()` flow:

```
1. Parse user intent → determine if this is a new task or continuation
2. If new task:
   a. Resolve flow (agentic template or user-customized)
   b. Create TaskState record in SQLite
   c. Run planning step → decompose into sub-tasks
   d. Stream plan to user for approval (if autonomy level requires it)
3. Execute steps sequentially:
   a. For each sub-task in plan:
      - Inject current plan state as context (attention anchor)
      - Execute step (research / draft / create / etc.)
      - Update plan status (mark complete, note issues)
      - Save checkpoint to SQLite
      - Stream progress to UI
   b. If step requires approval (autonomy dial):
      - Pause execution, yield approval-request chunk
      - Wait for user response in next message
      - Resume from checkpoint
4. Final step: deliver artifacts (file links in response)
```

### 3. New Step Roles

Extend `FlowStep.role` with agentic-specific roles:

| Role | Behavior |
|------|----------|
| `plan` | Decompose task into sub-steps, output structured plan. Re-injected as context in all subsequent steps via `{plan}` variable. Updated after each step completes. |
| `draft` | Generate content for a section/slide/chapter. Output stored as artifact fragment. |
| `create` | Call artifact tools to produce files. Can also call python_exec for custom generation. |
| `illustrate` | Generate image prompts, call image pipeline, collect results. |
| `review` | Review assembled output, flag issues, loop back if needed. |
| `deliver` | Package artifacts, generate download links, present to user. |

Existing roles (`classify`, `search`, `analyze`, `verify`, `respond`) work unchanged.

### 4. Plan System (Attention Anchor)

```
augmentum/reasoning/variables.py
  StepContext += plan: str          # Current plan markdown
  resolve_variables() += {plan}     # New template variable
```

The plan is a markdown checklist maintained in `StepContext`:

```markdown
## Task: Create climate change presentation

- [x] 1. Research current climate data (3 sources found)
- [x] 2. Outline slide structure (12 slides planned)
- [ ] 3. Draft slide content ← CURRENT
- [ ] 4. Generate illustrations for key slides
- [ ] 5. Assemble PPTX file
- [ ] 6. Review and deliver

Notes:
- User wants dark theme
- Focus on actionable solutions, not just doom
```

The `plan` role step generates this. After each subsequent step, the handler
updates checkboxes and appends notes. The full plan is injected via `{plan}`
at the END of every step's user message (recency bias = attention anchor).

### 5. Autonomy Dial

```
augmentum/reasoning/models.py
  ReasoningFlow += autonomy_level: int = 2  # 1-4
```

| Level | Name | Behavior |
|-------|------|----------|
| 1 | Suggest | Agent proposes plan, waits for approval at every step |
| 2 | Ask (default) | Agent executes, pauses before high-impact actions (file creation, >3 tool calls) |
| 3 | Inform | Agent executes everything, reports what it did |
| 4 | Autonomous | Agent runs to completion, user sees final result |

Enforcement in handler:

```python
if flow.autonomy_level <= 2 and step.role == "create":
    yield approval_request_chunk(step_name, plan_summary)
    return  # Pause — next user message resumes
```

The UI renders approval requests as interactive cards with Approve/Modify/Skip buttons.

### 6. Artifact Tools

New tool category: `ToolCategory.ARTIFACT = "artifact"`

```
augmentum/tools/
  artifact_document.py   — DocumentTool: markdown → PDF/DOCX
  artifact_presentation.py — PresentationTool: structured slides → PPTX
  artifact_spreadsheet.py — SpreadsheetTool: tabular data → XLSX
```

#### DocumentTool

```python
class DocumentTool(Tool):
    name = "create_document"
    category = ToolCategory.ARTIFACT
    input_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "format": {"type": "string", "enum": ["pdf", "docx", "md"]},
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "heading": {"type": "string"},
                        "level": {"type": "integer", "default": 1},
                        "body": {"type": "string"},
                        "image_url": {"type": "string"},
                    }
                }
            }
        },
        "required": ["title", "sections"]
    }
```

Tool writes file to `artifacts/` directory, returns artifact ID + path.

#### PresentationTool

```python
class PresentationTool(Tool):
    name = "create_presentation"
    category = ToolCategory.ARTIFACT
    input_schema = {
        "properties": {
            "title": {"type": "string"},
            "theme": {"type": "string", "enum": ["dark", "light", "corporate", "creative"]},
            "slides": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "layout": {"type": "string", "enum": [
                            "title", "section", "bullets", "two_column",
                            "image_left", "image_right", "full_image", "quote",
                            "comparison", "blank"
                        ]},
                        "title": {"type": "string"},
                        "subtitle": {"type": "string"},
                        "body": {"type": "string"},
                        "bullets": {"type": "array", "items": {"type": "string"}},
                        "image_prompt": {"type": "string"},
                        "speaker_notes": {"type": "string"},
                    }
                }
            }
        },
        "required": ["title", "slides"]
    }
```

When a slide has `image_prompt`, the tool calls the image pipeline to generate
an illustration and embeds it in the slide.

### 7. Artifact Storage & Delivery

```
augmentum/state/migrations/012_artifacts.sql

CREATE TABLE artifacts (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    session_id TEXT NOT NULL DEFAULT '',
    filename TEXT NOT NULL,
    format TEXT NOT NULL,          -- pdf, docx, pptx, xlsx, png, md
    size_bytes INTEGER DEFAULT 0,
    path TEXT NOT NULL,            -- filesystem path
    created_at TEXT DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'     -- JSON: slide count, page count, etc.
);
```

```
augmentum/proxy/artifact_routes.py

GET  /api/artifacts                    — list artifacts for session
GET  /api/artifacts/{id}               — artifact metadata
GET  /api/artifacts/{id}/download      — serve file (Content-Disposition: attachment)
DELETE /api/artifacts/{id}             — delete artifact and file
```

Artifact files stored in `data/artifacts/{task_id}/{filename}`.

### 8. Task State Persistence (Checkpoints)

```
augmentum/state/migrations/012_artifacts.sql (same migration)

CREATE TABLE agentic_tasks (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL DEFAULT '',
    flow_id TEXT,
    status TEXT NOT NULL DEFAULT 'planning',  -- planning, running, paused, completed, failed
    autonomy_level INTEGER DEFAULT 2,
    plan_md TEXT DEFAULT '',                   -- current plan markdown
    current_step INTEGER DEFAULT 0,
    total_steps INTEGER DEFAULT 0,
    step_outputs TEXT DEFAULT '{}',            -- JSON: step_name → output
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    completed_at TEXT,
    error TEXT
);
```

The handler saves state after each step. If the user disconnects:
- Next message to the same session checks for incomplete tasks
- Offers to resume: "You have an unfinished task: 'Climate presentation' (step 3/6). Resume?"
- Resume loads checkpoint and continues from saved state

### 9. Agentic Flow Templates

```
augmentum/reasoning/templates.py += agentic_report_flow(), agentic_presentation_flow()
```

#### Report Flow

```
Steps:
  1. Plan (plan role)         — decompose into research topics + sections
  2. Research (search role)   — web search for each topic
  3. Outline (analyze role)   — structure the report
  4. Draft (draft role)       — write each section, gate: iterate per section
  5. Review (review role)     — check accuracy, coherence, completeness
  6. Create (create role)     — call create_document tool
  7. Deliver (deliver role)   — present download link + summary
```

#### Presentation Flow

```
Steps:
  1. Plan (plan role)         — decompose into slide topics
  2. Research (search role)   — gather data for content slides
  3. Structure (analyze role) — define slide order + layouts
  4. Draft (draft role)       — write content for each slide
  5. Illustrate (illustrate)  — generate images for visual slides
  6. Create (create role)     — call create_presentation tool
  7. Review (review role)     — check slide quality
  8. Deliver (deliver role)   — present download link + summary
```

#### Storybook Flow

```
Steps:
  1. Plan (plan role)         — outline chapters, characters, settings
  2. Write (draft role)       — generate prose per chapter
  3. Illustrate (illustrate)  — generate images per scene
  4. Create (create role)     — assemble illustrated PDF
  5. Review (review role)     — check narrative coherence
  6. Deliver (deliver role)   — present download link
```

### 10. UI Integration

#### Mode Badge
Add "Agentic" option to mode selector with a new color variable `--mode-agentic`.

#### Inspector Panel — Task View
When in agentic mode, the inspector shows:
- **Plan checklist** — live-updating with checkmarks as steps complete
- **Autonomy level** — visual indicator (1-4 dots or a slider)
- **Artifacts** — list of generated files with download buttons
- **Stats** — elapsed time, steps completed, tools used

#### Chat — Approval Cards
When autonomy level requires approval, render an interactive card:
```
┌─────────────────────────────────────────┐
│ ⚡ Agent wants to: Create presentation  │
│                                         │
│ 12 slides, dark theme, 4 images         │
│ Estimated: ~2 minutes                   │
│                                         │
│  [Approve]  [Modify]  [Skip]            │
└─────────────────────────────────────────┘
```

#### Chat — Artifact Cards
When artifacts are delivered, render a download card:
```
┌─────────────────────────────────────────┐
│ 📄 Climate_Change_Report.pptx           │
│    12 slides · 4 images · 2.3 MB        │
│                                         │
│  [Download]  [Preview]                  │
└─────────────────────────────────────────┘
```

### 11. New Config Settings

```python
# config.py
AUGMENTUM_AGENTIC_ENABLED: bool = True
AUGMENTUM_AGENTIC_MAX_STEPS: int = 20          # safety limit
AUGMENTUM_AGENTIC_DEFAULT_AUTONOMY: int = 2    # 1-4
AUGMENTUM_AGENTIC_ARTIFACT_DIR: str = "data/artifacts"
AUGMENTUM_AGENTIC_MAX_ARTIFACT_SIZE_MB: int = 50
AUGMENTUM_AGENTIC_CHECKPOINT_ENABLED: bool = True
```

### 12. New Dependencies

```toml
# pyproject.toml
python-pptx = ">=1.0.0"       # PPTX generation
python-docx = ">=1.0.0"       # DOCX generation
weasyprint = ">=62.0"          # HTML/CSS → PDF (or reportlab as fallback)
openpyxl = ">=3.1.0"           # XLSX generation
```

WeasyPrint is preferred over reportlab for PDF because it renders markdown → HTML → PDF
with CSS styling support. Fallback: use the sandboxed executor with reportlab if
WeasyPrint's system dependencies (GTK/Pango) are too heavy for the Docker image.

---

## Implementation Phases

### Phase A — Foundation (artifact tools + delivery)
- [ ] Migration 012: artifacts + agentic_tasks tables
- [ ] `ToolCategory.ARTIFACT` in base.py
- [ ] `DocumentTool` (markdown → PDF via weasyprint or executor)
- [ ] `PresentationTool` (structured → PPTX via python-pptx)
- [ ] Artifact storage manager (save/list/delete/serve)
- [ ] `/api/artifacts/*` routes
- [ ] Tests for tools + routes

### Phase B — Mode + Handler
- [ ] `Mode.AGENTIC` in classifier
- [ ] `AgenticHandler` — flow execution with plan management
- [ ] Plan step role + `{plan}` variable
- [ ] Task state persistence (checkpoints)
- [ ] Resume-from-checkpoint logic
- [ ] Handler factory wiring
- [ ] Tests for handler + checkpoints

### Phase C — Autonomy + Approval
- [ ] `autonomy_level` field on ReasoningFlow
- [ ] Approval request stream chunks
- [ ] Approval response parsing (next user message)
- [ ] Pause/resume mechanics
- [ ] Tests for autonomy levels

### Phase D — Flow Templates
- [ ] `agentic_report_flow()` template
- [ ] `agentic_presentation_flow()` template
- [ ] `agentic_storybook_flow()` template (ties into image pipeline)
- [ ] Image pipeline integration for illustrate role
- [ ] Update seed_builtins

### Phase E — UI
- [ ] Agentic mode in mode selector
- [ ] Task view in inspector panel (plan + artifacts + stats)
- [ ] Approval cards in chat
- [ ] Artifact download cards in chat
- [ ] Autonomy dial in flow editor
- [ ] Task resume prompt on reconnect

### Phase F — Polish
- [ ] SpreadsheetTool (XLSX)
- [ ] Parallel sub-task support (optional)
- [ ] Observational memory integration
- [ ] Safety limits (max steps, max file size, timeout)
- [ ] Comprehensive test suite

---

## File Map (new files)

```
augmentum/
  modes/agentic/
    __init__.py
    handler.py              — AgenticHandler
    planner.py              — Plan decomposition + todo.md
    task_state.py           — TaskState model + SQLite persistence
    autonomy.py             — Autonomy level enforcement
  tools/
    artifact_document.py    — DocumentTool
    artifact_presentation.py — PresentationTool
    artifact_spreadsheet.py — SpreadsheetTool
    artifact_storage.py     — ArtifactStore (save/list/serve/delete)
  proxy/
    artifact_routes.py      — REST API for artifacts
  state/migrations/
    012_artifacts.sql       — artifacts + agentic_tasks tables
```

---

## Data Flow

```
User: "Create a presentation about renewable energy"
  │
  ├── Classifier: mode = agentic (explicit selection)
  │
  ├── HandlerFactory → AgenticHandler
  │     │
  │     ├── Resolve flow: "Presentation" template
  │     ├── Create TaskState in SQLite
  │     │
  │     ├── Step 1: Plan
  │     │   └── LLM → structured plan (8 slides)
  │     │   └── yield phase_chunk(plan, step=1/7, status=complete)
  │     │
  │     ├── Step 2: Research
  │     │   └── web_search × 3 queries
  │     │   └── web_fetch × 5 sources
  │     │   └── yield phase_chunk + content_chunks
  │     │
  │     ├── Step 3: Structure
  │     │   └── LLM → slide order + layouts
  │     │
  │     ├── Step 4: Draft
  │     │   └── LLM → content per slide (iterative)
  │     │
  │     ├── Step 5: Illustrate
  │     │   └── For slides with image_prompt:
  │     │       └── image_pipeline.generate()
  │     │       └── Store image as artifact
  │     │
  │     ├── Step 6: Create (autonomy check)
  │     │   ├── If level ≤ 2: yield approval_request → pause
  │     │   │   └── User approves → resume
  │     │   └── PresentationTool.execute(slides=[...])
  │     │       └── python-pptx → .pptx file
  │     │       └── ArtifactStore.save() → artifact record
  │     │
  │     ├── Step 7: Deliver
  │     │   └── yield content_chunk with artifact card markdown
  │     │   └── Update TaskState: status=completed
  │     │
  │     └── Done
  │
  └── UI: Shows plan progress + artifact download card
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Runaway execution (infinite loops) | Max steps config (default 20), per-step timeout |
| Large file generation | Max artifact size config (default 50MB) |
| Model generates bad plans | Review step catches issues, backtracking supported |
| User disconnects mid-task | Checkpoint persistence, resume on reconnect |
| Small models can't handle complex plans | Structured artifact tools work with any model size; only the planning step needs quality |
| WeasyPrint system deps too heavy | Fallback: use sandboxed executor with reportlab, or markdown output |
| Concurrent agentic tasks | One active task per session (queue additional) |
