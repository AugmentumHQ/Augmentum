# Meeting Notes — Product & Engineering Sync

**Date**: March 15, 2026
**Time**: 2:00 PM – 4:35 PM PST
**Location**: Conf Room B (Hybrid — David dialed in remotely)
**Facilitator**: Sarah
**Notes**: Lisa

---

## Attendees

- **Sarah Chen** — Product
- **David Kumar** — Engineering (remote)
- **Lisa Park** — Design
- **Marcus Webb** — Sales

*Apologies*: Raj (DevOps — on PTO), Fiona (Legal — scheduling conflict)

---

## Agenda Item 1: Q1 Pricing Review

Sarah opened with the Q1 numbers. MRR currently at $284K, up 11% QoQ but 4pts below the plan number. Enterprise tier performing well — 3 new logos in Feb, avg ACV $38K. Pro tier is the problem. Churn up to 6.2% monthly from 4.8% in Q4.

**Sarah's hypothesis**: Pro tier is underpriced for what it delivers, but also missing features that justify the sticker. Two things happening at once.

**Numbers on the table**:
- Pro tier today: $49/mo per seat
- Proposed new price: $59/mo per seat
- Growth tier (new): $29/mo per seat (stripped-down, aimed at freelancers)
- Enterprise: no change (custom contracts anyway)

Marcus flagged that the Sales team has been eating it on the Pro renewals. "We've had 3 customers this quarter explicitly mention that Driftboard undercut us on price by like 20%." He said the $59 move needs to be paired with a clear feature narrative — sales can't defend a price increase on vibes.

David raised margin. At $49 infrastructure costs are ~$14/seat, so ~71% margin. "If we go to $59 and add AI features (see Agenda 3), infra costs probably go to $18–20/seat. Margin compression from 71% to ~66%. That's still fine but we should model it." He shared a quick back-of-envelope in the Slack thread.

Lisa asked whether anyone had talked to users. "We keep making pricing decisions without UX data." She mentioned she ran an informal Typeform last month — 23 responses, not statistically meaningful but directionally: users who churned cited "didn't use enough features to justify cost," not price specifically.

**Discussion got a bit heated** — Sarah and Marcus disagreed on timing. Marcus wants to hold pricing flat until Q2 feature drop. Sarah says waiting costs us the margin improvement for 3 months and doesn't solve the churn signal.

**Outcome**: Tentative agreement to go to $59/mo for Pro starting April 1. But gated on: (1) updated feature comparison page live before price change, (2) existing customers get 60-day notice and locked-in $49 rate for 6 months. Marcus will draft the customer communication. Sarah owns the feature page update working with Lisa.

**Open question**: Should we grandfather annual Pro subscribers indefinitely or just 6 months? Didn't resolve — Fiona needs to weigh in on contract implications. Sarah to ping Fiona async.

---

## Agenda Item 2: Technical Debt Sprint

David came prepared with a doc (linked in Notion: "Tech Debt Audit March 2026"). He proposed a dedicated 2-week sprint starting April 6, pulling 4 engineers off roadmap work. Sarah's initial reaction: "That's a lot of velocity to sacrifice." David: "The auth stuff is a real liability. We had 2 near-misses in Feb."

**David's priority list** (roughly in order):

1. **Authentication refactor** — Current setup is a mess of JWT + session cookies + a legacy OAuth2 flow from 2022 that nobody fully understands. Plan: consolidate to stateless JWT only, deprecate session cookies, add refresh token rotation. Estimated 3–4 days of eng time. Risk: any third-party integration using the old OAuth flow breaks — David has a list of 7 integrations to audit.

2. **Database migration: PostgreSQL → CockroachDB** — This one is the biggie. Current Postgres setup on RDS is getting expensive ($2,100/mo) and we've had 2 unplanned outages this quarter due to connection pool exhaustion. David is proposing CockroachDB as the replacement for horizontal scalability. Estimated 8–10 days. "This is a phased migration, not a big bang. Shadow writes first." Sarah asked about data consistency during migration — David said "we accept eventual consistency for analytics tables, strong consistency for transactional tables. CockroachDB gives us serializable isolation by default."

   Marcus: "Will this affect the Salesforce integration?" David: "The integration talks to our API not the DB directly so no, but the latency profile might change slightly."

3. **Logging cleanup** — Currently logging 3x too much to CloudWatch, burning ~$800/mo in log ingestion fees. Quick win, maybe half a day. David wants to do this regardless of whether the sprint happens.

4. **Dependency updates** — 14 packages flagged as CVE-bearing in the Feb security scan. None critical, but 3 are high severity. David: "We need to stop deferring these."

Lisa raised a concern: if auth is changing, design needs to update the login flow and onboarding screens. "Can you loop me in before you start the implementation so I can do the design work in parallel?" David agreed to share the technical spec this week.

**Outcome**: Sprint approved in principle. Sarah agreed to carve out 4 engineers (David + 3 from his team) for 2 weeks starting April 6. Roadmap features moved to April 20 start. David to finalize sprint scope and share by March 20.

**Risk flag**: Marcus noted that a key enterprise prospect (can't name here, "the healthcare one") wants a security review done before they sign. David's auth refactor might actually help with that. Marcus to check if prospect would accept a "refactor in progress" status or needs completed work.

---

## Agenda Item 3: Feature Proposals

Three proposals on the table. Roughly 30 min discussion total — running short on time.

### 3a. AI-Powered Search (Lisa's proposal)

Lisa walked through a prototype she built in Figma. Core idea: instead of exact-match search, use semantic search so users can find tasks/docs with natural phrasing. "Right now if you type 'overdue invoices' it literally doesn't match anything because our tags say 'billing-delayed.'"

Tech implications — David: "We'd need to embed task titles and descriptions, store vectors, serve approximate nearest-neighbor queries. Our current stack doesn't have any of this. Ballpark: 2 weeks to build the embedding pipeline, 1 week for the search UI, ongoing infra cost probably $150–200/mo depending on corpus size." He mentioned Weaviate or Qdrant as candidate vector stores.

Sarah: strongly in favor. "This is exactly the kind of thing that justifies the Pro price increase narrative." Wants it in the Q2 roadmap.

Marcus agreed. "Sales has been asked about smart search by 4 enterprise prospects this year. It's on every competitor's homepage now."

**Outcome**: On the Q2 roadmap. David to spike the embedding approach and provide estimate by March 28. Lisa to flesh out full design spec.

### 3b. Bulk Export (Marcus's proposal)

Marcus: "The number one support ticket from enterprise customers is 'how do I export my data.' Right now we have CSV export on individual lists but nothing bulk." He showed a Zendesk report — 47 tickets in Q1 mentioning export, up from 28 in Q4.

What he's asking for: org-wide export of all tasks, comments, attachments as a ZIP. Scheduled exports via email. API-based export for technical customers.

David flagged this could be large-payload territory. "An org with 5 years of data could be generating a 500MB+ export. We'd need async job queue, not a synchronous request." He mentioned the executor service they already have and said it's probably not too bad to extend.

Sarah: "Can we do a v1 that's just ZIP export, no scheduling, and save the scheduled + API version for v2?" Marcus: fine with that.

**Outcome**: Bulk export v1 (ZIP download) added to Q2. Quick win, probably 3–4 days. David to scope.

### 3c. Mobile Push Notifications

Brief discussion — ran out of time. Sarah asked Marcus to write a 1-pager on the use case and user impact. "We've deferred this twice. I want a stronger case before we commit eng time." Tabled to next sync.

---

## Action Items

| # | Owner | Action | Due |
|---|-------|--------|-----|
| 1 | Marcus | Draft customer communication for Pro price change + 60-day notice | March 20 |
| 2 | Sarah | Update feature comparison page for Pro tier (working with Lisa) | March 27 |
| 3 | Sarah | Ping Fiona re: annual subscriber grandfathering question | March 17 |
| 4 | David | Share authentication refactor technical spec with Lisa | March 19 |
| 5 | David | Finalize tech debt sprint scope and share with team | March 20 |
| 6 | Marcus | Check with enterprise prospect on auth refactor timing vs. security review | March 18 |
| 7 | David | Spike AI search embedding approach, provide estimate | March 28 |
| 8 | Lisa | Full design spec for AI-powered search | March 28 |
| 9 | David | Scope bulk export v1 (async ZIP download) | March 22 |
| 10 | Marcus | Write 1-pager on mobile push notification use case | March 25 |

---

## Misc / Side Discussions

**On the AI search spike**: David mentioned he's been looking at LlamaIndex vs. a hand-rolled pipeline. "LlamaIndex is nice for prototyping but we'd probably want to own the chunking logic for production — document structure matters a lot for our use case." Lisa asked if the search results would show snippets or just titles. David: "Snippets for sure, that's table stakes. We'd highlight the matching phrases." Sarah: "Can we also search inside comments, not just task titles?" David: "Yes, but that meaningfully increases the corpus size. Let me factor that into the estimate."

**On the CockroachDB migration**: Marcus asked why not just upgrade Postgres to Aurora. David explained: "Aurora still has the single-writer bottleneck. We want to be able to scale writes horizontally when we get to enterprise-scale data volumes, and we're already seeing connection pool pressure. CockroachDB's distributed SQL lets us add nodes without a failover window." He acknowledged it's a bigger change than just swapping RDS flavors. "The migration tooling for CockroachDB has gotten a lot better — they have a `IMPORT` job that does shadow writes with conflict resolution. I'd rather do this now at 50K rows per table than when we're at 50M."

**On mobile push**: Lisa mentioned that from a design perspective push notifications are a lot of nuance — notification fatigue, permission flows, deep linking. "The 1-pager Marcus writes needs to answer: what exactly triggers a notification, what happens when you tap it, how do users opt out per-notification-type. Those decisions have real engineering implications." Marcus nodded. "Fair, I'll make it specific."

**Quick logistics**: Next three Sundays are blocked for data center maintenance window — David flagged this as relevant to migration timing. "Don't schedule the CockroachDB cutover on a Sunday, even though that's usually our preferred window." He'll coordinate with Raj when Raj is back from PTO.

---

## Parking Lot (not discussed this meeting)

- SSO/SAML for enterprise — on radar, no owner yet
- Dark mode (Design backlog, Lisa estimates 2 weeks)
- Time tracking integration (partnership discussion with Toggl, Sarah handling)
- Data residency / EU hosting — Fiona flagged as legal requirement for 2 prospective EU customers
- In-app changelog / release notes widget (Marketing request, not yet scoped)
- Webhook support for third-party integrations (came up on enterprise call last week — David: "2–3 weeks, low risk, high value for enterprise")

---

## Next Meeting

**Date**: March 22, 2026 — same time, same room
**Proposed agenda**: Tech debt sprint sign-off, Q2 roadmap finalization, pricing change approval

Marcus out March 24–28 (sales conference). Get sign-off from him by March 22 if decisions are needed.

---

*Notes captured by Lisa. Please flag corrections by EOD Monday. Full recording in Google Drive (internal).*
