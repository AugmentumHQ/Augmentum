---
name: agenda-prep
purpose: Pre-meeting prep grounded in memory of prior interactions with attendees.
cadence: event-triggered (pre-calendar-event) | on-demand
voice: system
inputs: [meeting_event, attendees, prior_conversations_from_memory, open_threads]
output: [2-4 lines context, 1 line goal, 1 line ask, landmines if any]
tags: [daily, meetings]
---

Meeting in N minutes. Pull from memory: who's attending, prior conversations with them, open threads from the last meeting (if any), commitments user made that are still unfulfilled. Output: 2-4 lines of context, one line on the goal-of-the-meeting per user's calendar/memory, one line on what they should ask. If there's a known landmine (unresolved disagreement, missed deadline), surface it.
