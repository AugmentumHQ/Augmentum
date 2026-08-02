---
name: Performance Profiler
description: >
  Focused performance investigation for slow builds, tests, routes, UI flows,
  memory growth, CPU spikes, and repeated agent-loop stalls.
kind: verifier
activation_policy: controller
activation_windows:
  - pre_plan
  - verify_failed
  - pre_finish
modes:
  - coder
triggers:
  - performance
  - slow
  - latency
  - memory leak
  - cpu
  - benchmark
preferred_tools:
  - shell_read
  - shell_exec
  - test_run
  - service_logs
  - service_probe
verification_recipe:
  - Capture a baseline command, trace, or timing before changing code.
  - Change one suspected bottleneck at a time.
  - Re-run the same measurement and report before/after.
memory_writes:
  - category: failure
    key: recurring_performance_bottleneck
success_criteria:
  - Baseline and post-change evidence are both recorded.
  - The final answer names any remaining measurement gap.
tags:
  - performance
  - profiling
  - verification
---

# Performance Profiler

Use this Power when speed or resource behavior is the task. Subjective
impressions ("feels slow") don't survive review — concrete numbers do.
The discipline: measure before you change anything, measure again the
same way after, report the delta.

## Workflow

1. **Establish the question** in one sentence: "is the /api/foo
   endpoint slower than 200ms p99?" — not "let me look at perf".
2. **Capture baseline** with a repeatable command. Examples:
   - HTTP latency: `hey -n 100 -c 4 http://localhost:8000/api/foo`
     or `wrk -t2 -c10 -d30s http://...`
   - Test wall time: `time pytest tests/test_foo.py`
   - Build time: `time npm run build`
   - Memory: `docker stats --no-stream <container>` or
     `cat /proc/<pid>/status | grep -i vmrss`
   - CPU during workload: `top -b -n 1 -p <pid>`
3. **Record the baseline** via `observe(category="build", fact="…")`
   or in the chat so the post-change measurement has something to
   compare to.
4. **Change ONE thing at a time**. Cumulative changes destroy
   attribution.
5. **Re-measure** with the exact same command. Same machine, same
   load, same warm-up state.
6. **Report the delta** in both absolute and percentage form:
   "p99 latency 218ms → 142ms (-35%)". Never just "looks faster".

## When to spawn a subagent

If the perf investigation requires reading 5+ files to find a
hotspot, spawn:
`task_dispatch(role="explore", prompt="find every site that calls expensive_op")`
and let the subagent return a list. Don't blow your context budget
on wide grep.

## Measurement gotchas

- **Cold vs warm**: JIT, file-system cache, DB query plan cache —
  all skew the first measurement. Discard the first N runs.
- **Noisy neighbors**: another container hitting the same disk/CPU
  will hide your win. `docker stats` once before measuring.
- **Statistical noise**: a single measurement is meaningless. Run
  3-5 times and report median + range.
- **Premature optimization**: micro-optimizing a 100µs function
  that runs once per request is wasted effort if the bottleneck is
  a 50ms DB query.

## What's allowed without measurement

Trivial refactors that don't change asymptotics (rename, extract,
dedupe) need no perf check. But anything touching: a hot loop, an
N×M nested iteration, a DB query plan, a network round-trip pattern,
or memory growth — measure.

## Guardrails

- Don't trust output from `time` if the workload isn't deterministic;
  use a benchmarking tool that warms up + reports stats.
- Don't optimize against a profiler the prod environment doesn't
  have; production characteristics may differ.
- If you can't measure (no benchmark harness, can't reproduce the
  load), say so and propose adding one rather than guessing.

## Good outputs

- "Baseline: p99 218ms over 1000 requests via `hey -n 1000 -c 4`.
  Post-change: p99 142ms (-35%). The win came from replacing
  N+1 query with a join in `users_store.list_with_roles`."
- "Build time 92s → 78s after parallelising the asset pipeline.
  Recorded build observation so future sessions don't accidentally
  serialize it again."
- "Could not measure — no benchmark for the dispatch queue exists.
  Proposed adding `scripts/bench_dispatch.py` first, then revisit."

