# Reasoning & Agent Orchestration Systems: Engineering Reality (2025-2026)

Research compiled March 2026. Focused on practical architectural details, empirical evidence, and what actually works in production.

---

## 1. OpenAI Reasoning Models (o1, o3, o4-mini)

### Architecture: Hidden Chain-of-Thought via RL

OpenAI's o-series models are standard transformer LLMs trained via **large-scale reinforcement learning** to produce an internal "private chain of thought" before generating a visible answer. The key insight: they are NOT multi-agent systems or multi-call pipelines. They are single models that have learned to reason in a single forward pass (though with many more output tokens).

**Training approach:**
- Base model (GPT-4 class) is fine-tuned via RL to produce chain-of-thought reasoning
- RL reward signal is based on answer correctness (rule-based, not neural reward models)
- The model learns to generate hundreds to thousands of hidden "reasoning tokens" before the visible answer
- These reasoning tokens are generated but NOT shown to the user (hidden CoT)

**Inference-time compute scaling:**
- More reasoning tokens = better answers, following a log-linear scaling curve
- The `reasoning_effort` parameter (low/medium/high) controls how many reasoning tokens the model generates
- `max_completion_tokens` caps visible output tokens but does NOT constrain reasoning tokens
- OpenAI recommends reserving at least 25,000 tokens for reasoning+output when starting

**The `reasoning_effort` parameter in practice:**
- Low: comparable to o1-mini performance, ~60% cost of non-reasoning models
- Medium: comparable to o1 performance (default)
- High: outperforms both o1 and o1-mini, cost roughly equivalent to non-reasoning models
- Moving from low to high raises accuracy by 10-30% on hard benchmarks (AIME, GPQA Diamond, Codeforces)
- GPT-5.2 adds `none` and `xhigh` levels

**o4-mini vs o3 (April 2025):**
- o4-mini at ~10x cheaper than o3
- o4-mini matches o3 "medium effort" on most benchmarks
- AIME 2025: o4-mini 92.7% vs o3 88.9%
- Codeforces: o4-mini 2719 vs o3 2706 Elo
- For most production use cases, o4-mini is the sweet spot

**Streaming behavior:**
- Streaming for o3 is limited access only
- Reasoning tokens are hidden; you see a delay, then the visible answer streams
- `reasoning_tokens` count is reported in `completion_tokens_details` in the response

**Tool use:**
- o3/o4-mini were trained via RL to reason about WHEN to use tools, not just how
- They can integrate images into their chain of thought (multimodal reasoning)
- Tools are called as part of the reasoning process, not as a separate phase

**Key API details:**
- Use `developer` messages instead of `system` messages (system messages are treated as developer messages)
- `max_completion_tokens` is the only supported output limit parameter
- The 80% price cut in June 2025 (o3: $10/$40 → $2/$8 per million tokens) made reasoning models production-viable

### What This Means for Augmentum

OpenAI's approach is fundamentally different from UARF's multi-phase pipeline. The reasoning happens inside a single model call, trained via RL. The key advantage: no inter-call latency, no context loss between phases. The disadvantage: it's a black box you can't inspect or steer mid-reasoning.

---

## 2. Claude Extended Thinking

### Architecture: Visible Reasoning Blocks

Unlike OpenAI's hidden CoT, Anthropic makes thinking visible (or at least summarized). Extended thinking is an integrated capability, not a separate model.

**Core mechanism:**
- When enabled, Claude generates `thinking` content blocks before `text` content blocks
- Response structure: `[thinking_block, text_block]`
- Each thinking block includes a `signature` field for verification/integrity
- Claude 4+ models return **summarized** thinking (condensed for latency); Claude 3.7 Sonnet returns full thinking

**Budget control:**
- `budget_tokens` parameter sets max thinking tokens per turn
- Budget must be less than `max_tokens` (exception: interleaved thinking can exceed)
- You're billed for FULL thinking tokens generated, not the summarized output returned
- Larger budgets improve quality but Claude may not use entire allocation (especially >32k)
- **Deprecated on Opus 4.6**: Use adaptive thinking with `effort` parameter instead

**Adaptive thinking (Opus 4.6+):**
- Model decides when to use extended thinking based on task difficulty
- Automatically enabled interleaved thinking
- No beta header needed
- Similar to OpenAI's reasoning_effort but the model decides dynamically

**Interleaved thinking (critical for tool use):**
- Without it: one thinking block at start, then tool calls with no thinking between them
- With it: thinking blocks between each tool call, allowing reasoning about intermediate results
- This is architecturally significant: the model can reason ABOUT tool results before deciding next action

**Tool use constraints:**
- Only `tool_choice: "auto"` or `"none"` supported (can't force specific tools)
- Must pass thinking blocks back in multi-turn conversations (complete, unmodified)
- Cannot toggle thinking on/off mid-turn; entire assistant turn operates in single mode
- If mid-turn conflict occurs, API automatically disables thinking for that request

**Streaming behavior:**
- SSE streaming with `thinking_delta` events
- Thinking content may arrive in larger chunks (batch processing)
- Possible delays between streaming events
- Text arrives after complete thinking block

**Prompt caching interaction:**
- Thinking blocks are removed from context but still cached
- Changing budget_tokens breaks cache
- System prompts remain cached despite thinking changes
- Creates a real tradeoff: preserve thinking for reasoning continuity vs. token consumption

### Key Insight: Single-Model vs Multi-Phase

Anthropic explicitly states their philosophy: "reasoning should be an integrated capability of frontier models rather than a separate model entirely, just as humans use a single brain for both quick responses and deep reflection." This is a direct contrast to multi-phase pipelines like UARF.

---

## 3. Microsoft AutoGen / Magentic-One / Agent Framework

### The Five Orchestration Patterns (Azure Architecture Center, Feb 2026)

Microsoft formalized five distinct patterns. This is the most rigorous taxonomy available.

**1. Sequential Orchestration**
- Linear pipeline; each agent processes previous agent's output
- Deterministic, predefined order
- Best for: step-by-step refinement with clear stage dependencies
- Failure mode: errors in early stages propagate; no parallelism; no backtracking

**2. Concurrent Orchestration**
- Parallel; agents work independently on same input
- Best for: independent analysis from multiple perspectives; latency-sensitive scenarios
- Failure mode: requires conflict resolution when results contradict; resource-intensive

**3. Group Chat Orchestration**
- Agents contribute to a shared conversational thread
- Chat manager controls turn order
- Best for: consensus-building, iterative maker-checker validation
- Failure mode: conversation loops; difficult to control with >3 agents
- **Maker-Checker Loop** is a specific sub-pattern: one agent creates, another evaluates, cycle until approval or iteration cap
  - Also known as: evaluator-optimizer, generator-verifier, critic loop, reflection loop
  - Requires clear acceptance criteria and iteration cap to prevent infinite refinement

**4. Handoff Orchestration**
- Dynamic delegation; one active agent at a time
- Agents decide when to transfer control based on context
- Best for: tasks where the right specialist emerges during processing
- Failure mode: infinite handoff loops; unpredictable routing paths

**5. Magentic Orchestration (Magentic-One)**
- Plan-build-execute with adaptive task ledger
- Manager agent builds and refines plan dynamically
- Most complex and most flexible pattern
- Best for: open-ended problems with no predetermined solution path
- Failure mode: slow to converge; stalls on ambiguous goals

### Magentic-One Architecture Detail

The Orchestrator agent maintains two key data structures:

**Task Ledger:**
- High-level goals and subgoals
- Resolution approach plan
- Facts gathered and educated guesses
- Dynamically built and refined during execution

**Progress Ledger:**
- Self-reflection on task progress at each step
- Checks whether task is completed
- If not complete, assigns subtask to specialized agent
- Updated after each agent completes its subtask

**Agent roster:** Orchestrator + WebSurfer + FileSurfer + Coder + Computer Terminal

**Key implementation detail:** Magentic-One is model-agnostic. Different agents can use different models (heterogeneous model assignment for cost optimization).

### Production Guidance from Microsoft

Critical advice from the Azure Architecture Center:

- "Start with the right level of complexity" — use the simplest pattern that works
- Single agent with multiple tools should be tried BEFORE multi-agent
- "Decision-making and flow-control overhead often exceed the benefits of breaking the task into multiple agents"
- Context windows grow rapidly in multi-agent systems; use compaction between agents
- Implement timeout, retry, circuit breaker patterns
- "Validate agent output before passing it to the next agent. Low-confidence, malformed, or off-topic responses can cascade through a pipeline"
- "Assign each agent a model that matches the complexity of its task. Not every agent requires the most capable model."
- Limit group chat to 3 or fewer agents

---

## 4. LangGraph / LangChain Agent Patterns

### State of Agent Engineering (LangChain Survey, Dec 2025)

1,340 respondents. Key findings:

- **57.3%** now have agents in production (up from 51% previous year)
- **Quality is the #1 production killer** (32% cite it as top barrier)
- **Latency is #2 challenge** (20%) — reflects multi-step quality vs. speed tradeoff
- Larger organizations (10k+) have 67% agents in production
- Top use case: customer service (26.5%), research/data analysis (24.4%)

### ReAct Pattern (Reasoning + Acting)

The most widely used pattern in practice.

**How it works:**
1. Model receives input + observation history
2. Model generates a "thought" (reasoning about what to do)
3. Model generates an "action" (tool call)
4. System executes action, returns "observation"
5. Repeat until model generates final answer

**Strengths:**
- Simple to implement
- Works well for tasks with clear tool boundaries
- Natural fit for function-calling models

**Failure modes:**
- Loops: model repeats same action expecting different results
- Context bloat: observations accumulate and exceed context window
- Tool selection errors: model picks wrong tool, cascading failures
- No lookahead: purely reactive, doesn't plan ahead

### Plan-and-Execute Pattern

**How it works:**
1. Planner agent creates multi-step plan upfront
2. Executor agent executes each step sequentially
3. Planner can optionally be re-invoked to revise plan based on execution results

**Strengths:**
- Faster execution: sub-tasks can use lighter-weight LLM or no LLM at all
- Better for complex multi-step tasks
- Plan is inspectable before execution

**Failure modes:**
- Plan quality depends entirely on initial understanding of task
- Re-planning is expensive (full LLM call)
- Rigid: doesn't adapt well to unexpected results mid-execution

### Reflection / Self-Critique Pattern

**How it works:**
1. Generator produces initial output
2. Reflector critiques the output
3. Generator revises based on critique
4. Repeat until quality threshold or iteration limit

**Critical evidence:** The ICLR paper "Large Language Models Cannot Self-Correct Reasoning Yet" showed that LLMs struggle to self-correct without external feedback, and performance sometimes DEGRADES after self-correction. The observed improvement in multi-pass approaches is "not attributed to self-correction, but rather to self-consistency" (majority voting over multiple attempts).

**What actually works for verification:**
- External tool-based verification (run code, check math with SymPy, etc.)
- Confidence-weighted self-consistency (CISC) — 46% reduction in computational cost vs naive majority voting
- Process reward models (PRMs) for step-by-step verification
- Multi-agent debate significantly UNDERPERFORMS simple self-consistency with majority voting

### LATS (Language Agent Tree Search)

Unifies reasoning, planning, and reflection. Uses tree search over possible action sequences with a value function to evaluate intermediate states. Most sophisticated but also most expensive approach.

---

## 5. Coding Agents (Claude Code, Aider, OpenCode)

### Claude Code Architecture

**Core: Single-threaded master loop (codenamed "nO")**

Operational flow:
1. User input arrives
2. Model analyzes and decides on actions
3. If tools needed, they're called
4. Results feed back to model
5. Cycle continues until final answer emerges
6. Control returns to user

**Three blended phases:** gather context → take action → verify results

**Key implementation details:**
- Context compression triggers at ~92% context window usage ("Compressor wU2")
- Summarizes conversation and moves important info to long-term storage
- Long-term storage is a simple Markdown document (project memory)
- 99.9th percentile turn duration: ~45 minutes (up from ~25 min in Oct 2025)
- Asks yes/no before applying diffs or running test suites (human-in-the-loop)

**Subagent system:**
- Claude can spawn subagents for subtasks
- Subagents have restricted tool access (e.g., read-only agents can't Edit/Write)
- Different system prompts per subagent type
- Tool list is a subset of main agent's tools
- Agent teams: lead AI spawns teammate agents, each with own context window, coordinate via message passing (shipped Feb 2026)

**No explicit phase pipeline.** Claude Code does NOT use UARF-style phases. It's a simple loop: think → act → observe → repeat. The model's own reasoning (via extended thinking) replaces structured multi-phase orchestration.

### Aider Architecture

**Repository map system:**
- Uses tree-sitter to parse source code into ASTs
- Extracts function, class, variable definitions from all files
- Builds a NetworkX MultiDiGraph of file relationships (files as nodes, dependencies as edges)
- Ranks with PageRank (personalized based on conversation context)
- Selects most relevant definitions within token budget (default 1k tokens for repo map)

**Edit formats:**
- Search/replace blocks (replaced earlier "edit block" format — reduced malformed edits)
- Whole-file replacement for smaller files
- Git-native: every AI edit is a git commit with descriptive message

**Context management:**
- Conversation context persisted across restarts via compressed git graph representation
- Preserves code evolution history, not just transcripts
- `/map-tokens` flag controls repo map size

**No reasoning pipeline.** Aider is also a single-call-per-turn architecture. The model receives repo map + conversation history + user request, generates edits in one pass.

### OpenCode Architecture

- Two switchable agents: `build` (full access) and `plan` (read-only analysis)
- Supports `/thinking-tokens` and `/reasoning-effort` commands for fine-grained reasoning control
- Native LSP integration: auto-detects Language Server for type checking and cross-file awareness
- Provider-independent: 75+ LLM providers
- Agent teams with mixed models from different providers (something Claude Code can't do)
- Tab to toggle between build and plan modes

### Key Pattern: None Use Explicit Phase Pipelines

All three major coding agents use a simple agent loop, NOT multi-phase reasoning:
1. Receive input
2. Think (single model call, possibly with extended thinking)
3. Act (tool calls)
4. Observe results
5. Repeat

The "phases" emerge from the model's own reasoning, not from external orchestration.

---

## 6. Open-Source Reasoning Models (DeepSeek-R1, QwQ, Qwen3)

### DeepSeek-R1 Training Pipeline

Four-stage process:
1. **Cold-start SFT**: Fine-tune base model on thousands of long CoT examples
2. **Reasoning-oriented RL**: Using Group Relative Policy Optimization (GRPO) with rule-based rewards
3. **Rejection sampling + dual-domain SFT**: 600k reasoning + 200k general examples
4. **Secondary RL**: Combining rule-based and preference rewards for all scenarios

**GRPO key innovation:** Eliminates need for separate critic model by estimating baselines from group scores. Uses rule-based rewards (not neural reward models) to prevent reward hacking.

**Emergent reasoning (DeepSeek-R1-Zero, pure RL without SFT):**
- AIME 2024: improved from 15.6% to 71.0%
- Spontaneously learned reflection and alternative-approach exploration
- "Aha moment" where model learned to reconsider problems
- Limitation: poor formatting, language mixing (addressed by cold-start data)

### The Critical Distillation Finding

**Distillation from large reasoning models into small models outperforms RL training on small models directly.**

Direct comparison (Qwen-32B-Base):
- DeepSeek-R1-Zero-Qwen-32B (10,000+ RL steps): achieved QwQ-32B-Preview parity
- DeepSeek-R1-Distill-Qwen-32B (distilled from R1): outperformed on ALL benchmarks

Distilled model benchmarks:
| Model | AIME 2024 | MATH-500 | LiveCodeBench |
|-------|-----------|----------|---------------|
| R1-Distill-Qwen-7B | 55.5% | — | — |
| R1-Distill-Qwen-14B | > QwQ-32B-Preview | — | — |
| R1-Distill-Qwen-32B | 72.6% | 94.3% | 57.2% |
| R1-Distill-Llama-70B | 70.0% | 86.7% (maj) | — |

**Implication:** If you're working with small models (<70B), distilling from a large reasoning model is more effective than trying to train reasoning via RL.

### Qwen3 Architecture (April 2025)

**Hybrid reasoning model:** Can switch between thinking mode (extended CoT) and non-thinking mode (fast response).

Training pipeline:
1. Long CoT cold start
2. Reasoning-based RL
3. Thinking mode fusion (learning to switch between modes)
4. General RL

**Model lineup:**
- Dense: 0.6B, 1.7B, 4B, 8B, 14B, 32B
- MoE: 30B (3B active), 235B (22B active)
- All Apache 2.0 licensed
- 36 trillion training tokens, 119 languages

**Qwen3-235B-A22B:** ~89.2% on AIME 2025, outperforming many proprietary models while activating only 22B parameters per token.

### QwQ-32B

Dense 32B reasoning model. Trained with RL on reasoning tasks. Competitive with models 2-3x its size on reasoning benchmarks. Available on Hugging Face.

---

## 7. What Actually Works: Empirical Evidence

### Test-Time Compute Scaling

**Key paper:** "Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters" (ICLR 2025)

Findings:
- Using a smaller model + generating more tokens often outperforms a larger model at fixed compute budget
- Two mechanisms: (1) search against process-based verifier reward models (PRMs), (2) adaptive distribution updating
- Compute-optimal strategy improves efficiency 4x compared to naive best-of-N
- **No single test-time scaling strategy universally dominates**
- Optimal strategy varies by problem difficulty
- For a given model type, optimal TTS performance scales monotonically with compute budget

**DeepPrune finding (Tsinghua):** Over 80% of parallel reasoning paths converge to the same answer. DeepPrune cuts computation by ~80% while keeping accuracy within ~3 percentage points on AIME and GPQA.

### Extended Thinking vs Multi-Phase Pipelines

**Evidence favoring single-call extended thinking:**
- Accuracy increases logarithmically with thinking token budget
- No inter-call context loss
- No prompt engineering between phases
- All major coding agents use this approach, not multi-phase
- Anthropic and OpenAI both moved toward integrated reasoning, not pipelines

**Evidence for limitations:**
- Forcing models to think beyond natural limits causes accuracy DECLINE
- Even best models plateau in state-tracking after 6-8 reasoning steps
- State-of-the-art MLLMs only achieve 25-29% on multi-step visual reasoning (GPT-4o, Qwen2.5-VL-72B)

**The uncomfortable truth for multi-phase pipelines like UARF:**
The empirical evidence strongly suggests that for capable models (>30B), single-call extended thinking outperforms orchestrated multi-call reasoning. The main value of multi-phase approaches is with smaller models that can't do extended reasoning in a single call, or when you need inspectable/steerable intermediate results.

### Self-Correction and Verification

**What doesn't work:**
- Naive "check your work" prompting (ICLR: "LLMs Cannot Self-Correct Reasoning Yet")
- Multi-agent debate (underperforms simple majority voting)
- Asking the same model to verify its own answer without external input
- Self-correction with smaller models often degrades performance

**What does work:**
- External tool-based verification (code execution, math solvers, fact checking)
- Confidence-weighted self-consistency (CISC): 46% compute reduction vs naive majority voting
- Process Reward Models (PRMs): reward each reasoning step, not just final answer
- Best-of-N with verifier: generate N answers, use verifier to pick best
- Compute-optimal scaling: adaptively allocate compute based on problem difficulty

**For small models specifically:**
- Small models need strong external verifiers to self-correct effectively
- Distillation from large reasoning models works better than RL for adding reasoning capability
- Self-consistency with 10 samples + confidence weighting is cost-effective

### Benchmark Leaderboard (Early 2026)

| Benchmark | Best Score | Model |
|-----------|-----------|-------|
| GPQA Diamond | 93.2% | GPT-5.2 Pro |
| MATH-500 | 97.3% | DeepSeek-R1 |
| AIME 2025 | 92.7% | o4-mini |
| ARC-AGI-2 | 45.1% | Gemini 3 Pro |
| Codeforces | 2719 Elo | o4-mini |
| AIME 2025 (open) | ~89.2% | Qwen3-235B-A22B |

---

## 8. Latency vs Quality in Production

### The "Long Wait" Problem

Reasoning models consume 5-100x more tokens than standard models. Even with price cuts, complex queries take 10-100x longer.

**Production approaches:**

1. **Streaming reasoning tokens** (Claude's approach): Show the thinking process as it happens. Reduces perceived wait time. Thinking arrives in chunks, then visible answer streams normally.

2. **Hidden reasoning with progress indicators** (OpenAI's approach): User sees a "thinking..." indicator with elapsed time. No visibility into reasoning process.

3. **Adaptive reasoning depth** (emerging consensus): Route easy questions to fast path, hard questions to deep reasoning. This is what Claude's adaptive thinking and OpenAI's reasoning_effort do.

4. **Thinking budget control** (NVIDIA NIM): `max_thinking_tokens` parameter forces early exit from reasoning. Grants 10% grace period for sentence completion. Currently incompatible with SGLang.

### Latency Data

- Latency is the #2 production challenge (20% of respondents in LangChain survey)
- 99.9th percentile Claude Code turn: ~45 minutes (for complex multi-file operations)
- Simple queries on reasoning models: 2-10 seconds
- Complex reasoning: 30-120+ seconds
- Multi-agent orchestrations: multiply by number of agent hops

### Inference Infrastructure

- SGLang and LMDeploy: ~16,200 tokens/sec on H100 (fastest as of early 2026)
- vLLM: ~12,500 tokens/sec on H100 (29% slower)
- NVIDIA Dynamo: distributed inference framework for reasoning models at scale
- NVIDIA Nemotron Nano 2: produces thinking tokens 6x faster than comparable models

### What Production Systems Actually Do

From Anthropic's engineering guide on long-running agents:
- Track progress in structured artifacts (`claude-progress.txt`)
- Commit work with descriptive messages (git as checkpoint)
- Work on one feature at a time (prevents context exhaustion)
- Use browser automation for verification (curl commands miss end-to-end failures)
- Environment setup scripts for clean state enforcement
- Session initialization: read progress files, git logs, run health checks

---

## Summary: Implications for Augmentum's UARF Pipeline

### Where UARF Aligns with Industry

1. **Maker-Checker pattern** (APPLY → VERIFY → backtrack) is a recognized, validated pattern
2. **Tool integration in reasoning** is universal across all systems
3. **Classifier/routing** is the handoff pattern Microsoft recommends
4. **External verification** (SymPy, code execution) is what actually works

### Where UARF Diverges from Industry Consensus

1. **Multi-phase pipeline vs single-call reasoning**: Every major system (OpenAI, Claude, coding agents) uses single-model reasoning, not orchestrated phases. The phases in UARF (ASSESS → IDENTIFY → RELEVANT → APPLY → VERIFY → CONCLUDE) are each separate LLM calls, which loses context and adds latency. Modern reasoning models do this internally.

2. **Explicit decomposition vs emergent reasoning**: UARF decomposes problems explicitly. Modern reasoning models decompose implicitly via CoT trained through RL.

3. **Verification approach**: UARF's VERIFY phase asks the same model to check its work. Evidence says this doesn't work well without external grounding. The external tools (SymPy, code exec) are the parts that actually add value.

### What the Evidence Suggests Augmentum Should Consider

1. **For capable models (>30B)**: Route to passthrough with extended thinking enabled. The model's internal reasoning will outperform UARF's multi-phase pipeline.

2. **For small models (<14B)**: UARF's multi-phase approach may still add value, especially the external tool verification. But consider distilling rather than orchestrating.

3. **The real value add**: Tool integration, external verification, and context management — not the multi-phase reasoning pipeline itself.

4. **Adaptive complexity**: Match Augmentum's approach to the model's capability. If the model has native reasoning (R1 distills, Qwen3, etc.), let it reason. If it doesn't, UARF phases can help.

---

## Sources

### OpenAI Reasoning Models
- [Introducing OpenAI o3 and o4-mini](https://openai.com/index/introducing-o3-and-o4-mini/)
- [Reasoning models - OpenAI API](https://developers.openai.com/api/docs/guides/reasoning/)
- [o3 and o4-mini System Card](https://cdn.openai.com/pdf/2221c875-02dc-4789-800b-e7758f3722c1/o3-and-o4-mini-system-card.pdf)
- [How Reasoning Models Actually Work Under the Hood](https://www.letsdatascience.com/blog/reasoning-models-how-ai-learned-to-think-step-by-step)
- [OpenAI for Developers 2025](https://developers.openai.com/blog/openai-for-developers-2025/)
- [OpenAI o3 - Wikipedia](https://en.wikipedia.org/wiki/OpenAI_o3)
- [The Reasoning Revolution](https://www.financialcontent.com/article/tokenring-2026-1-1-the-reasoning-revolution-how-openais-o3-series-and-the-rise-of-inference-scaling-redefined-artificial-intelligence)

### Claude Extended Thinking
- [Claude's extended thinking](https://www.anthropic.com/news/visible-extended-thinking)
- [Building with extended thinking - Claude Docs](https://platform.claude.com/docs/en/build-with-claude/extended-thinking)
- [Claude Opus 4.6 announcement](https://www.marktechpost.com/2026/02/05/anthropic-releases-claude-opus-4-6-with-1m-context-agentic-coding-adaptive-reasoning-controls-and-expanded-safety-tooling-capabilities/)
- [Claude 3.7 Sonnet announcement](https://www.anthropic.com/news/claude-3-7-sonnet)

### Microsoft AutoGen / Agent Framework
- [AI Agent Orchestration Patterns - Azure Architecture Center](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns)
- [Magentic-One - AutoGen](https://microsoft.github.io/autogen/stable//user-guide/agentchat-user-guide/magentic-one.html)
- [Magentic-One: A Generalist Multi-Agent System](https://www.microsoft.com/en-us/research/articles/magentic-one-a-generalist-multi-agent-system-for-solving-complex-tasks/)
- [Microsoft Agent Framework Migration Guide](https://learn.microsoft.com/en-us/agent-framework/migration-guide/from-autogen/)

### LangGraph / LangChain
- [State of Agent Engineering - LangChain](https://www.langchain.com/state-of-agent-engineering)
- [Plan-and-Execute Agents - LangChain Blog](https://blog.langchain.com/planning-agents/)
- [Reflection Agents - LangChain Blog](https://blog.langchain.com/reflection-agents/)
- [ReAct vs Plan-and-Execute Comparison](https://dev.to/jamesli/react-vs-plan-and-execute-a-practical-comparison-of-llm-agent-patterns-4gh9)

### Coding Agents
- [How Claude Code works](https://code.claude.com/docs/en/how-claude-code-works)
- [Tracing Claude Code's LLM Traffic](https://medium.com/@georgesung/tracing-claude-codes-llm-traffic-agentic-loop-sub-agents-tool-use-prompts-7796941806f5)
- [Claude Code: Behind the Scenes](https://blog.promptlayer.com/claude-code-behind-the-scenes-of-the-master-agent-loop/)
- [Effective Harnesses for Long-Running Agents - Anthropic](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [Building Agent Teams in OpenCode](https://dev.to/uenyioha/porting-claude-codes-agent-teams-to-opencode-4hol)
- [Aider Repository Map](https://aider.chat/docs/repomap.html)
- [Claude Code vs Gemini CLI vs OpenCode vs Aider](https://sanj.dev/post/comparing-ai-cli-coding-assistants)

### DeepSeek-R1 / Qwen
- [DeepSeek-R1 Paper](https://arxiv.org/html/2501.12948v1)
- [DeepSeek-R1 GitHub](https://github.com/deepseek-ai/DeepSeek-R1)
- [Complete Guide to DeepSeek Models](https://www.bentoml.com/blog/the-complete-guide-to-deepseek-models-from-v3-to-r1-and-beyond)
- [Qwen3: Think Deeper, Act Faster](https://qwenlm.github.io/blog/qwen3/)
- [Top 10 Open-Source Reasoning Models 2026](https://www.clarifai.com/blog/top-10-open-source-reasoning-models-in-2026)

### Test-Time Compute & Verification
- [Scaling LLM Test-Time Compute Optimally](https://openreview.net/forum?id=4FWAwZtd2n)
- [Inference Scaling Laws](https://arxiv.org/abs/2408.00724)
- [The Art of Scaling Test-Time Compute](https://arxiv.org/abs/2512.02008)
- [Inference-Time Scaling for Complex Tasks - Microsoft Research](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/03/Inference-Time-Scaling-for-Complex-Tasks-Where-We-Stand-and-What-Lies-Ahead-2.pdf)
- [Large Language Models Cannot Self-Correct Reasoning Yet (ICLR)](https://openreview.net/forum?id=IkmD3fKBPQ)
- [Training Language Models to Self-Correct (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/871ac99fdc5282d0301934d23945ebaa-Paper-Conference.pdf)
- [Confidence Improves Self-Consistency in LLMs](https://arxiv.org/pdf/2502.06233)
- [When Can LLMs Actually Correct Their Own Mistakes?](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00713/125177/)
- [Does Thinking More Always Help?](https://arxiv.org/html/2506.04210v1)

### Production & Infrastructure
- [AI Agents in Production 2025 - Cleanlab](https://cleanlab.ai/ai-agents-in-production-2025/)
- [Thinking Budget Control - NVIDIA NIM](https://docs.nvidia.com/nim/large-language-models/latest/thinking-budget-control.html)
- [NVIDIA Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [vLLM vs SGLang vs LMDeploy](https://blog.premai.io/vllm-vs-sglang-vs-lmdeploy-fastest-llm-inference-engine-in-2026/)
- [2025 LLM Review](https://atoms.dev/blog/2025-llm-review-gpt-5-2-gemini-3-pro-claude-4-5)
