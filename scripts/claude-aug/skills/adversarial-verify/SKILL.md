---
name: adversarial-verify
description: Self-refute before delivering. Invoke before presenting any final answer, fix, or claim of completion.
---

Before saying "done" or presenting a conclusion, attack your own work:

1. Re-read the ORIGINAL request. List each explicit requirement and mark
   it satisfied/unsatisfied with evidence (a file path, a command output,
   a quote — not your memory of doing it).
2. Try to refute your main claim. What input, edge case, or environment
   would break it? If you can't rule it out, test it now:
   - Code: run it (Bash) or execute the logic via `mcp__atp__python_exec`.
   - Math/logic: check with `mcp__atp__math_verify` or `mcp__atp__calculator`.
   - Factual claims: verify with `mcp__atp__web_search` or `mcp__atp__wikipedia`.
   - Multi-claim answers: run `mcp__atp__consistency_check` on your summary.
   - **Reproduction** (bugs, builds, installs): prefer `mcp__atp__sandbox_shell`
     over `mcp__atp__python_exec`. A real Docker container beats model recall —
     it has a filesystem, package manager, and state that survives across calls.
     `python_exec` is stateless and stdlib-only; sandbox_shell can `pip install`,
     `git clone`, and verify a fix in the same environment the user will run it in.
3. If verification fails, fix it BEFORE replying — never present known-broken
   work with a caveat.
4. In your reply, state what you verified and how, in one line.

Tools are cheaper than your own confidence. When a tool can check it, the
tool decides — not you.
