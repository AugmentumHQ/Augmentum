---
name: scope-fence
description: Execute precisely what was asked, nothing more. Invoke when a request is narrow or when you feel the urge to refactor/improve surrounding code.
---

Do exactly what was requested. Before adding ANYTHING not explicitly asked
for (refactors, renames, extra features, formatting sweeps, dependency
bumps, "while I'm here" fixes), apply this test:

- Would the change happen anyway if the user's request didn't exist? → out.
- Is it required for the requested change to work? → in.
- Is it a genuine bug you found? → mention it in your reply, do NOT fix it
  unless asked.

Diff discipline: the final diff should read as one idea. If you can't
describe the whole diff in one sentence, you've exceeded scope — revert
the extras.
