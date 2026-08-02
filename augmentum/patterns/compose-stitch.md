---
name: compose-stitch
purpose: Chain pattern A's output into pattern B; verify the shapes match.
cadence: on-demand
voice: system
inputs: [pattern_a_name, pattern_b_name, optional_intermediate_transform]
output: [composed invocation, or adapter pattern if shapes mismatch]
tags: [meta, composition]
---

User wants pattern A's output fed into pattern B. Verify the shape matches (B's expected input fits A's output schema). If they don't match, propose a minimal adapter pattern in the middle. Return the composed invocation. Refuse to chain patterns whose outputs are user-facing prose into patterns expecting structured input — that's where stitching usually breaks.
