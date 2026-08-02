---
name: meeting-distill
purpose: Turn a transcript into decisions, actions, and subtle memory writes about attendees.
cadence: event-triggered (transcript save)
voice: system
inputs: [transcript, attendees]
output: [decisions, actions, open-questions, per-attendee memory writes]
tags: [meetings, memory-write]
---

From transcript: **decisions** (who decided what, when by), **actions** (assignee + deadline if stated, `[UNASSIGNED]` if not), **open questions** (parked items), **noted preferences/positions** worth adding to memory about each attendee (subtle — what they revealed about how they think, not what they said). Skip the chitchat. If a decision was implied but not made explicit, mark `[IMPLIED]` and quote the line.
