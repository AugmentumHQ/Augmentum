---
name: improve-prompt
purpose: Diagnose and rewrite a user's underperforming prompt without flattening their voice.
cadence: on-demand
voice: system
inputs: [users_current_prompt, what_they_wanted]
output: [diagnosis, rewritten prompt in their voice, one line on what to test against]
tags: [meta, fabric-port]
---

User has a prompt that's not getting them what they want. Diagnose: (1) what's underspecified, (2) what's over-specified and constraining the model, (3) what implicit assumption isn't stated. Rewrite to a tighter version with structured output sections. Keep their voice — don't replace their style with corporate-prompt-engineer style. End with one line on what to test against.
