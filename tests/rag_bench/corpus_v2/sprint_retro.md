# Sprint 14 Retrospective

**Sprint dates:** Feb 24 – Mar 7, 2026 (10 working days)
**Retro date:** Mar 10, 2026, 2:00 PM PT (async notes compiled by Alex)
**Attendees:** Alex (backend), Priya (frontend), Jordan (QA), Sam (DevOps)
**Facilitator:** Jordan

---

## Velocity

| Metric | Value |
|---|---|
| Points planned | 42 |
| Points completed | 34 |
| Completion rate | 81% |
| Carry-over | 8 pts (3 stories) |

**Carry-over stories:**
- AUTH-204: Token refresh race condition fix (5 pts) — blocked by prod incident investigation
- DASH-118: Dashboard filter persistence (2 pts) — Priya deprioritized due to outage recovery
- OPS-77: Prometheus alerting rule tuning (1 pt) — Sam ran out of cycle

Velocity trending: Sprint 12 = 38 pts, Sprint 13 = 41 pts, Sprint 14 = 34 pts. Downward spike attributable to auth outage (see below) — not structural.

---

## What Went Well

### DB Migration: PostgreSQL → Aurora Serverless v2

The planned database migration from self-hosted PostgreSQL 14 to AWS Aurora Serverless v2 (PostgreSQL-compatible) completed successfully with **zero data loss and under 4 minutes downtime** (RTO target was 15 min). Highlights:

- Sam's blue/green switchover script worked cleanly on first attempt in prod
- Pre-migration validation queries confirmed row counts matched across 47 tables
- Aurora's autoscaling ACU configuration held up under immediate post-cutover load spike (2× normal QPS for ~8 min)
- Rollback plan was tested and rehearsed the prior Friday — knowing we had a clean path out reduced anxiety significantly

Shoutout to Sam for the weekend dry-run. That prep made the difference.

### CI/CD Pipeline: 98% Green Rate

For the 10-day sprint window, the main branch CI pipeline ran 214 times and failed on 4 runs (all immediately fixed). That's a 98.1% green rate vs. 91% in Sprint 13. Contributing factors:

- Priya's fix for flaky CSS snapshot tests (replaced pixel-diff with structural assertions)
- Jordan's updated Docker layer caching config reduced avg build time from 6m40s → 4m12s
- New `lint-on-push` step catches import errors before integration tests run — caught 2 issues early

### Async Service Decomposition Progress

The notifications service was successfully extracted into its own FastAPI microservice (NOTIF-01 through NOTIF-05, 12 pts). It now runs as an independent container with its own DB schema and queue consumer. This was the largest single architectural item since Sprint 9. No regressions in downstream consumers.

---

## What Didn't Go Well

### Auth Service Outage — March 2, ~11:15 AM – 2:18 PM PT (3 hours 3 minutes)

**Root cause:** Connection pool exhaustion on the auth service DB layer. A deploy of AUTH-198 (session expiry batch job) introduced a code path that opened connections without releasing them under certain error conditions. Under normal load the pool recovers, but a traffic spike from a scheduled email campaign (200% of typical login rate for ~15 min) caused pool slots to saturate, after which auth service began returning 503s.

**Impact:**
- 3 hours 3 minutes of degraded/unavailable user-facing authentication
- ~1,200 unique users affected (based on error log analysis)
- No data loss; all sessions valid after recovery
- Postmortem doc: [AUTH-OUTAGE-20260302.md](../postmortems/AUTH-OUTAGE-20260302.md) *(link internal only)*

**What slowed recovery:**
- Connection pool metrics were not in the existing alerting dashboard — took 35 minutes to identify root cause
- On-call rotation had Alex as primary but he was mid-deploy on an unrelated service; rotation coverage gap
- No circuit breaker between API gateway and auth service; cascade to API layer was worse than it needed to be

**Immediate mitigations already applied:**
- Sam patched the connection pool config (max_overflow=5 → 20, pool_timeout=10 → 30)
- Alex reverted AUTH-198; fix being reworked as AUTH-204 (in carry-over)

### Flaky E2E Tests Still a Problem

The Cypress E2E suite had **11 flaky test failures** in Sprint 14, same order of magnitude as Sprint 13. Root causes are varied but the main patterns Jordan identified:

1. **Timing issues (6 failures):** `cy.wait(500)` hard-waits don't survive slow CI environments. Need `cy.intercept` + alias pattern.
2. **State bleed (3 failures):** Tests that delete test users don't handle the case where a prior test already deleted the user. Teardown needs to be idempotent.
3. **Element selector brittleness (2 failures):** Two tests using auto-generated CSS class names that change on rebuild.

Flaky tests erode trust in the suite. Jordan tracking in QA-TECH-44.

### Sprint Planning Overcommitment

We planned 42 points against a 10-day sprint. Historical adjusted velocity (after removing sprint 10 anomaly) is 36.5 pts. We've planned at or above capacity 3 sprints in a row. Product pressure is a factor but we need to hold the line on capacity.

---

## Action Items

| # | Owner | Description | Due Date | Priority |
|---|---|---|---|---|
| AI-14-01 | Alex | Fix connection pooling in auth service (AUTH-204); add pool exhaustion to alerting dashboard | Mar 14, 2026 | Critical |
| AI-14-02 | Priya | Refactor dashboard FilterPanel component: extract state into useReducer-equivalent pattern; reduce prop drilling | Mar 21, 2026 | High |
| AI-14-03 | Jordan | Add retry logic and intercept patterns to E2E suite; fix 6 timing-related failures in QA-TECH-44 | Mar 21, 2026 | High |
| AI-14-04 | Sam | Add circuit breaker (resilience4j or equivalent) between API gateway and auth service | Mar 28, 2026 | High |
| AI-14-05 | Alex | Document on-call rotation coverage; identify and fill rotation gaps for upcoming sprint | Mar 14, 2026 | Medium |
| AI-14-06 | Priya | Submit DASH-118 carry-over (filter persistence) in Sprint 15 planning as top-priority story | Sprint 15 planning | Medium |
| AI-14-07 | Jordan | Present flaky test taxonomy and resolution plan to team in Sprint 15 kickoff | Mar 17, 2026 | Medium |
| AI-14-08 | Sam | Review Prometheus alerting rules with Alex; add DB connection pool, queue depth, and auth error rate alerts | Mar 21, 2026 | Medium |
| AI-14-09 | All | In Sprint 15 planning: enforce 35-point cap; push back on scope add if it would exceed this | Sprint 15 planning | Process |

---

## Tech Debt Review

**Current backlog:** 14 items tagged `tech-debt` across JIRA boards
**Oldest item:** TD-003 — Logging standardization (structured JSON logging across all services) — **opened 6 months ago** (Sept 2025)

| Item | Age | Estimated Effort | Impact if Unaddressed |
|---|---|---|---|
| TD-003: Logging standardization | 6 months | 5 pts | Ops observability, incident response |
| TD-007: Remove deprecated `/v1/users/search` endpoint | 4 months | 2 pts | Security surface, confusion |
| TD-009: Upgrade SQLAlchemy 1.4 → 2.0 in legacy services | 4 months | 8 pts | Python 3.12 compatibility blocker |
| TD-012: Extract inline SQL queries from notification service | 2 months | 3 pts | Maintainability |
| TD-013: Replace homegrown JWT library with python-jose | 2 months | 3 pts | Security |
| TD-014: Consolidate 3 separate Redis clients into shared pool | 1 month | 2 pts | Resource efficiency |

**Team agreement:** Dedicate 15% of each sprint to tech debt items (≈5–6 pts per sprint at current velocity). Alex to create TD cleanup stories for Sprint 15. SQLAlchemy upgrade (TD-009) needs to ship before Aug 2026 or it blocks Python 3.12 upgrade.

---

## Next Sprint Goals (Sprint 15: Mar 10 – Mar 21, 2026)

**Primary objectives:**
1. **User notification service** — end-to-end email + in-app notification delivery for the new subscription tier (NOTIF-06 through NOTIF-11; ~14 pts). Backend API + Priya to wire frontend notification bell component.
2. **Performance benchmarks** — establish baseline p50/p95/p99 latency benchmarks for all API endpoints using k6 load testing. Sam to configure k6 in CI; target: benchmark on every merge to main. (~8 pts)
3. **AUTH-204 carry-over** — Alex to deliver connection pooling fix + alerting. Non-negotiable given the outage. (5 pts)

**Capacity:** Team at full strength. Tentative planning: 32 pts (conservative; allows room for carry-over and debt items).

---

## Team Health

Quick poll (anonymous, 1–5):

| Question | Avg Score |
|---|---|
| Clarity of sprint goals | 3.8 |
| Feeling of accomplishment | 3.2 (lower than usual; outage impact) |
| Process satisfaction | 3.5 |
| Workload manageability | 2.9 ← flag |
| Team communication quality | 4.2 |

**Workload flag (2.9/5):** This is the second sprint with workload below 3. Jordan noted that incident response is eating into planned work without adjustment to sprint commitment. Agreed: incidents > 2 hours = remove 2 pts from active sprint. Will trial in Sprint 15.

---

*Notes compiled by Alex. Review and corrections welcome in #engineering-retro Slack thread before EOD March 11.*
