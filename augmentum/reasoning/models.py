"""Pydantic models for reasoning flows and steps."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# Valid step roles accepted by the executor.
VALID_ROLES = frozenset({
    "classify", "search", "analyze", "verify", "respond",
    "plan", "draft", "create", "illustrate", "review", "deliver",
    "transform",
})

# Valid complexity gate values.
VALID_COMPLEXITIES = frozenset({"simple", "moderate", "complex"})

# Sentinel string values for FlowStep.tool_choice. Anything else is
# interpreted as a specific tool name and translated by the executor
# into the provider-shaped tool_choice payload.
_TOOL_CHOICE_SENTINELS = frozenset({"", "auto", "required", "none"})


class FlowStep(BaseModel):
    """A single step in a reasoning flow."""

    id: str = ""
    flow_id: str = ""
    sort_order: int = 0
    name: str = ""
    system_prompt: str = ""
    user_template: str = ""
    role: str = "analyze"
    tool_categories: list[str] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)
    complexity_gate: list[str] = Field(default_factory=list)
    stream_to_user: bool = False
    output_cap: int = 800
    enabled: bool = True
    model_override: str = ""
    # Per-step tool_choice override. Empty string = let the model decide
    # (current default). "auto" / "required" / "none" pass straight through
    # to the provider. Any other non-empty value is treated as a specific
    # tool name and translated by the executor into the provider's pinned-
    # tool payload shape ({"type":"function", "function":{"name":...}} for
    # OpenAI-compat; ``_translate_tool_choice`` handles Anthropic).
    # No-op when the step has no tools resolved.
    tool_choice: str = ""

    @field_validator("role")
    @classmethod
    def _validate_role(cls, v: str) -> str:
        if v and v not in VALID_ROLES:
            raise ValueError(
                f"Invalid step role '{v}'. Must be one of: {', '.join(sorted(VALID_ROLES))}"
            )
        return v

    @field_validator("complexity_gate")
    @classmethod
    def _validate_complexity_gate(cls, v: list[str]) -> list[str]:
        for item in v:
            if item not in VALID_COMPLEXITIES:
                raise ValueError(
                    f"Invalid complexity gate '{item}'. Must be one of: {', '.join(sorted(VALID_COMPLEXITIES))}"
                )
        return v

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if v and len(v) > 200:
            raise ValueError("Step name must be at most 200 characters")
        return v

    @field_validator("system_prompt", "user_template")
    @classmethod
    def _validate_prompt_length(cls, v: str) -> str:
        # 100KB limit to prevent accidental mega-prompts
        if len(v) > 100_000:
            raise ValueError("Prompt text must be at most 100,000 characters")
        return v

    @field_validator("tool_choice")
    @classmethod
    def _validate_tool_choice(cls, v: str) -> str:
        # Length bound — a "specific tool name" is the only non-sentinel
        # case and tool names in the registry are well under 200 chars.
        if len(v) > 200:
            raise ValueError("tool_choice must be at most 200 characters")
        return v


class ReasoningFlow(BaseModel):
    """A named reasoning pipeline with ordered steps."""

    id: str = ""
    name: str = ""
    description: str = ""
    icon: str = ""
    version: int = 1
    is_default: bool = False
    is_builtin: bool = False
    auto_select: bool = True
    trigger_domains: list[str] = Field(default_factory=list)
    trigger_keywords: list[str] = Field(default_factory=list)
    pinned_models: list[str] = Field(default_factory=list)
    auto_search: bool = True
    max_tool_calls_per_step: int = 3
    autonomy_level: int = 2  # 1=suggest, 2=ask, 3=inform, 4=autonomous
    escalation_flow: str = ""  # Name of flow to suggest when uncertainty detected
    kind: str = "workflow"     # "workflow" = fixed DAG of steps, "dynamic" = ReAct-style agent loop over the full tool registry
    steps: list[FlowStep] = Field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


class FlowCreateRequest(BaseModel):
    """Request to create a new flow."""

    name: str = Field(..., min_length=1, max_length=200)
    description: str = ""
    icon: str = ""
    template: str = ""  # template name to seed from
    auto_select: bool = True
    trigger_domains: list[str] = Field(default_factory=list)
    trigger_keywords: list[str] = Field(default_factory=list)
    pinned_models: list[str] = Field(default_factory=list)
    auto_search: bool = True
    max_tool_calls_per_step: int = 3
    autonomy_level: int = 2
    steps: list[FlowStep] = Field(default_factory=list)


class FlowUpdateRequest(BaseModel):
    """Request to update a flow."""

    name: str | None = None
    description: str | None = None
    icon: str | None = None
    auto_select: bool | None = None
    trigger_domains: list[str] | None = None
    trigger_keywords: list[str] | None = None
    pinned_models: list[str] | None = None
    auto_search: bool | None = None
    max_tool_calls_per_step: int | None = None
    autonomy_level: int | None = None
    steps: list[FlowStep] | None = None
