## Summary

What does this PR do and why? (Link the issue it closes, if any: `Closes #123`.)

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor / cleanup (no behavior change)
- [ ] Docs
- [ ] Build / CI / tooling
- [ ] Breaking change

## Test plan

How did you verify this works? Commands run, manual steps, screenshots for UI changes.

## Checklist

- [ ] `python .claude/skills/augmentum-dev/scripts/audit.py --smoke` passes (`--with-tests` if I touched logic)
- [ ] `ruff check augmentum/ tests/` is clean
- [ ] New/changed settings touch all four layers (config.py, config_routes.py, server.py restore map, settings.js) — see CONTRIBUTING.md
- [ ] New user-data tables have a `user_id` column; CRUD functions accept and scope by `user_id`
- [ ] New routes are registered in `server.py` and pass `validate_wiring.py`
- [ ] New migration follows the `CREATE TABLE IF NOT EXISTS` / try-wrapped `ALTER` pattern
- [ ] `${...}` in `innerHTML` template literals goes through `escapeHtml()`
- [ ] If this adds audit findings, the baseline is bumped in this PR with an explanation
- [ ] Commits follow `type(scope): subject`, one commit per coherent change
