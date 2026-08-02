---
name: plan-gate
description: Force a short written plan before any multi-file or destructive change. Invoke before starting non-trivial implementation work.
---

Before editing more than one file, or doing anything destructive (delete,
overwrite, schema change, config change), STOP and write a plan first:

1. **Goal** — one sentence, in the user's words.
2. **Files** — exact paths you will touch and why each one.
3. **Risks** — what could break; how you'll check it didn't.
4. **Out of scope** — what you are deliberately NOT changing.

Then execute the plan exactly. If mid-work you discover the plan was wrong,
stop, revise the plan in one short paragraph, then continue. Never silently
drift from the written plan — drift is how small models produce large messes.
