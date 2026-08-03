"""Narrative chunking strategy comparison test.

Tests how different chunking parameters affect RAG coherence for narrative
content specifically.  Creates multiple narrative documents at varying sizes
and styles, then ingests each under several chunking configs and measures
fact retrieval accuracy.

Run manually:

    .venv/Scripts/python tests/live_narrative_chunking_test.py [OPTIONS]

    --url URL           OpenAI-compatible base URL (default: http://localhost:1234/v1)
    --model NAME        Model to test (required — use whatever is loaded)
    --verbose / -v      Show injected context and model responses
    --timeout SECS      Per-call timeout (default: 120)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

if sys.platform == "win32" and "pytest" not in sys.modules:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from augmentum.documents.chunker import chunk_text, chunk_with_parents, extract_text
from augmentum.memory.embeddings import EmbeddingService
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Narrative documents — varied sizes and styles
# ---------------------------------------------------------------------------

NARRATIVE_DOCUMENTS = {

    # -----------------------------------------------------------------------
    # SHORT (~800 words) — Dense action scene, many specific details packed tight
    # -----------------------------------------------------------------------
    "ironvault_short.md": {
        "size_class": "short",
        "style": "action/heist",
        "content": """\
# The Ironvault Job

Kael Morrow checked his chrono: 03:47:12.  Exactly thirteen minutes until the
vault's temporal lock cycled from the Seventh Configuration to the First — a
42-second window where the magnetic tumblers aligned and the 900-kilogram door
could be moved by hand.

He crouched on the rooftop of the Mercantile Exchange, seven stories above
Coppergate Street.  Below, two Wardsmen patrolled the perimeter in a figure-eight
pattern, crossing paths every 90 seconds at the southeast corner.  Kael had
timed them for three nights straight.

"Fenwick, status," he whispered into the bone-conductor mic embedded behind
his left ear.

"Sewers are clear.  I've set the resonance charge on the gas main — junction
T-14, exactly 40 meters northwest of the vault basement.  Detonation will
knock out the acoustic sensors for roughly 8 seconds."  Fenwick's voice was
steady despite the waist-deep canal water.

"And Lira?"

"Already inside.  She replaced the night guard's thermos with the dosed one at
21:30.  Guard Hendrik Voss — yes, unfortunately a relation — consumed
approximately 200 milliliters of valerian-laced coffee at 01:15.  He's been
unconscious since 02:40, slumped in the monitoring alcove on sublevel two.
Lira re-looped the camera feeds at 03:00, sixteen cameras cycling a 4-minute
recorded segment."

The plan had cost them three months and 14,200 silver marks to assemble.
The Ironvault held the Castellan's private ledger — a leather-bound folio
containing the names of every noble house that had paid bribes to circumvent
the Grain Tariff of 1847.  Their client, a woman known only as "The Auditor,"
was paying 45,000 marks for its retrieval.

Kael uncoiled the 30-meter rappelling line — woven graphene, rated for 500
kilograms — and anchored it to the rooftop ventilation housing with a
Thornsby clamp rated for 12 kilonewtons of shear force.

At 03:59:58 he dropped.

The descent took 4.2 seconds.  He landed on the third-floor ledge, exactly
adjacent to the service corridor window that Lira had left unlatched.  Inside,
the corridor smelled of machine oil and old stone.  He counted doors: one,
two, three — the third door bore a brass plate reading "Archive Sublevel
Access" and required the seven-digit code that Fenwick had extracted from
the building superintendent's personal notebook: 4-7-1-9-0-3-8.

The stairs descended 14 meters to sublevel one, then another 9 meters to
sublevel two.  At the bottom, Guard Voss was indeed unconscious, his breath
slow and even.  The monitoring station showed sixteen camera feeds, all
displaying the same 4-minute loop Lira had installed.

The vault door was ahead: two meters tall, 1.4 meters wide, forged from
Valderian steel with a thickness of 23 centimeters.  The temporal lock's
display showed the Seventh Configuration, counting down.

03:59:54.  03:59:55.  03:59:56.

At 04:00:00, the tumblers clicked.  Kael gripped the recessed handle and
pulled.  The 900-kilogram door swung open on counterweighted hinges,
moving with surprising ease.

Inside: seven shelves of deposit boxes.  The Castellan's ledger was in box
number 714, third shelf, marked with a discreet wax seal bearing the crest
of House Aldenmere.  Kael opened it with the clay-cast key Lira had made
from the Castellan's ring impression.

The ledger was 340 pages, bound in oxblood leather with brass corner guards.
He photographed every page with the miniature camera — 12 seconds per page,
68 minutes total — then returned the ledger to its box.

At 05:12, Kael exited through the sewer junction where Fenwick waited with
a change of clothes and a bottle of whiskey.  The resonance charge had
detonated at exactly 04:00:00, masking the vault's brief opening from the
acoustic grid.

The Auditor received the photographs three days later at a dead drop in
the Millhaven post office, box 2207.  Payment arrived by wire transfer
within the hour: 45,000 marks, split three ways — 20,000 for Kael,
15,000 for Fenwick, 10,000 for Lira.
""",
        "questions": [
            {
                "query": "What is the seven-digit code to the Archive Sublevel Access?",
                "expected_facts": ["4-7-1-9-0-3-8", "4719038"],
                "id": "short_code",
            },
            {
                "query": "How much did the Ironvault job cost to assemble and what was the payout?",
                "expected_facts": ["14,200", "45,000"],
                "id": "short_money",
            },
            {
                "query": "What is the vault door made of and how thick is it?",
                "expected_facts": ["Valderian steel", "23 centimeters", "900-kilogram"],
                "id": "short_vault",
            },
            {
                "query": "Where was the dead drop and how was payment split?",
                "expected_facts": ["Millhaven", "2207", "20,000", "15,000", "10,000"],
                "id": "short_split",
            },
            {
                "query": "What did Fenwick set up in the sewers?",
                "expected_facts": ["resonance charge", "T-14", "40 meters", "acoustic sensors", "8 seconds"],
                "id": "short_fenwick",
            },
        ],
    },

    # -----------------------------------------------------------------------
    # MEDIUM (~1500 words) — Multi-scene chapter, dialogue + description mixed
    # -----------------------------------------------------------------------
    "thornfield_medium.md": {
        "size_class": "medium",
        "style": "mystery/investigation",
        "content": """\
# The Thornfield Disappearance — Chapter 12

## The Librarian's Testimony

Inspector Calloway sat across from Mirabel Whitstone in the reading room of
the Thornfield Municipal Library.  Mirabel was 67 years old, had worked at
the library for 34 years, and had the particular quality of noticing everything
while appearing to notice nothing.

"Mrs. Whitstone, you told my sergeant that you saw Aldous Crake on the evening
of November 14th.  Can you describe exactly what you observed?"

Mirabel set down her tea — Darjeeling, brewed at exactly 95 degrees Celsius
as she insisted — and straightened her reading glasses.  "Mr. Crake came in at
5:47 PM.  I know the exact time because the grandfather clock in the foyer
had just chimed the quarter-hour, and I looked up from reshelving the Dewey
940 section — European History."

"Was that unusual?"

"Entirely.  Mr. Crake had not visited the library in eleven years.  The last
time was March 2014, when he requested a copy of Hargreaves' 'Industrial
Metallurgy of the Pennines,' which we had to order through interlibrary loan
from the University of Sheffield.  He returned it 23 days overdue."

Calloway suppressed a smile.  "What did he do during this visit?"

"He went directly to the archive room — that's the climate-controlled section
in the basement, temperature maintained at 18 degrees Celsius, 45% relative
humidity.  He used his own library card, number 7734-A, which was technically
expired but I didn't stop him because the system still accepted it."

"How long was he there?"

"Precisely two hours and fourteen minutes.  He entered at 5:49 PM and emerged
at 8:03 PM.  I was about to close up — the library closes at 8:30 PM on
Thursdays — when he came back upstairs carrying a single sheet of paper."

"Did you see what was on it?"

"I couldn't read it from where I was standing, about four meters away, but I
could see it was a hand-drawn map.  Pencil lines, not printed.  He was holding
it very carefully, by the edges, like it was fragile or valuable.  He also had
a small brown envelope — approximately 15 by 10 centimeters — that he tucked
inside his coat pocket."

## The Archive Records

After Mirabel left, Calloway descended to the archive room with Sergeant
Patel.  The electronic access log confirmed Mirabel's account: card 7734-A
had been used at 17:49 on November 14th, accessing Cabinet 12, Drawer 7 —
the Thornfield Estate historical documents, 1780-1850.

"Cabinet 12, Drawer 7," Patel read from the log.  "That's the Pemberton
correspondence.  Letters between Sir Reginald Pemberton and his mining
engineer, a man named Josiah Blackwell."

Calloway opened the drawer.  Inside were 47 letters, each preserved in
individual acid-free sleeves, plus three folded maps.  He counted the maps:
one, two... only two.

"There should be three maps according to the 2019 inventory," Patel said,
checking her tablet.  "Map P-12-7-C is missing.  It's described as 'Survey
of Thornfield Mine tunnels, Level 3 and below, drawn by J. Blackwell,
dated April 1842.  Includes notation of sealed passages and water table
measurements.'"

"So Crake stole a mining map from 1842."

"It appears so, Inspector."

## The Mine Connection

Calloway stood at his office window, looking out at the rain.  On his desk
lay the case file for Aldous Crake: age 58, former mining consultant for
Northrock Resources Ltd (dissolved 2018), resident of 14 Blackthorn Lane,
Thornfield.  No criminal record.  One surviving relative — a niece, Daphne
Crake, age 31, who worked as a geological surveyor for the county council.

The timeline was becoming clearer:

- **November 14, 5:47 PM:** Crake visits library, steals Map P-12-7-C
- **November 14, 9:30 PM:** Crake's car (blue Volvo V60, plate NR67 WXH)
  captured by ANPR camera on the B6049 heading toward Thornfield Moor
- **November 14, 11:15 PM:** Last mobile phone ping, cell tower TF-7,
  covering the area around the abandoned Thornfield Mine entrance
- **November 15, 7:00 AM:** Crake fails to attend his regular breakfast
  at the Copper Kettle café (he had been a daily customer for 9 years,
  always ordering the full English with black pudding substituted for
  toast, and a pot of Assam tea)
- **November 15, 2:30 PM:** Niece Daphne reports him missing
- **November 16, 10:00 AM:** Crake's car found parked at the old mine
  access road, 400 meters from the sealed entrance, doors locked,
  keys in the glovebox, a half-eaten sandwich on the passenger seat

The mine itself had been closed since 1953 and sealed with concrete
in 1978.  But Calloway had learned something from the county archives
that morning: the 1978 sealing only covered the main entrance.
Blackwell's 1842 map — the very one Crake had stolen — showed three
additional access points: an air shaft 200 meters northeast (grid ref
SK 287 614), a drainage adit on the south face, and a "passage of
uncertain purpose" connecting to the cellar of what was now the
Thornfield Arms pub, 1.2 kilometers to the west.

Calloway picked up his phone and called Daphne Crake.  "Miss Crake, I
need to ask you something.  Did your uncle ever mention Josiah Blackwell?"

The silence on the line lasted exactly seven seconds before Daphne answered:
"Inspector, I think we need to talk in person.  There are things about
this family that I've been afraid to say."
""",
        "questions": [
            {
                "query": "What was Crake's library card number and when did he last visit?",
                "expected_facts": ["7734-A", "eleven years", "March 2014"],
                "id": "medium_card",
            },
            {
                "query": "What map did Crake steal and what did it show?",
                "expected_facts": ["P-12-7-C", "Josiah Blackwell", "1842", "sealed passages"],
                "id": "medium_map",
            },
            {
                "query": "Describe Crake's car and where it was found",
                "expected_facts": ["Volvo V60", "NR67 WXH", "400 meters", "keys in the glovebox"],
                "id": "medium_car",
            },
            {
                "query": "What are the three additional mine access points on Blackwell's map?",
                "expected_facts": ["air shaft", "SK 287 614", "drainage adit", "Thornfield Arms pub", "1.2 kilometers"],
                "id": "medium_mine",
            },
            {
                "query": "Describe Inspector Calloway's timeline of Crake's movements on November 14-16",
                "expected_facts": ["5:47 PM", "9:30 PM", "B6049", "cell tower TF-7", "Copper Kettle"],
                "id": "medium_timeline",
            },
            {
                "query": "What were the archive room conditions and what was in Cabinet 12 Drawer 7?",
                "expected_facts": ["18 degrees", "45%", "47 letters", "Pemberton", "Cabinet 12"],
                "id": "medium_archive",
            },
        ],
    },

    # -----------------------------------------------------------------------
    # LONG (~2500 words) — Epic fantasy, dense world-building, many names/places
    # -----------------------------------------------------------------------
    "solanthir_long.md": {
        "size_class": "long",
        "style": "epic fantasy",
        "content": """\
# The Siege of Solanthir — Book III, Chapter 21

## The Council of Twelve Banners

On the morning of the forty-third day of the siege, the twelve commanders
gathered in the war pavilion atop the Basalt Ridge, 300 meters above the
burning city of Solanthir.  General Arathan Vosk — a woman of sixty-one
years, veteran of the Meridian Campaigns and the architect of the Pincer
Doctrine that had won the Battle of Ironcross seven years prior — stood at
the sand table and began her assessment.

"The outer wall fell at dawn.  Solanthir's curtain wall was 14 meters high
and 3 meters thick at the base — respectable, but the sappers from the
Quarryborn Regiment tunneled beneath the southeast section over nineteen
days, removing approximately 2,400 cubic meters of limestone.  The collapse
opened a breach 40 meters wide."

She moved sand pieces.  "Commander Thessik holds the breach with 1,200
heavy infantry.  Behind him, Commander Yelara's 800 archers occupy the
grain warehouses along Candlemaker's Row.  The defenders have retreated
to the inner citadel — a much harder target."

Marshal Dren Kaelos, the supreme commander, leaned over the table.  He
was 74, the oldest serving officer in the Allied Host, and he had personally
negotiated the Twelve Banner Alliance that united the fractious southern
kingdoms against the Tyrant of Ashenmoor.  "What is the citadel's
strength?"

"The Solanthir Citadel sits on a granite plug, 60 meters above the city
floor.  Its walls are 8 meters thick — double-skinned basalt with a rubble
core.  There is only one gate, facing north, protected by three successive
portcullises and a murder corridor 25 meters long.  The garrison is
estimated at 3,400 soldiers under Castellan Brynn Harrowgate, plus
approximately 12,000 civilians who fled the lower city."

"Water?"

"Three deep wells fed by an underground aquifer.  Our engineers estimate
the aquifer supplies roughly 8,000 liters per day — sufficient for 15,000
people on half rations for approximately 90 days."

"Then we starve them," said Commander Ossek Draal, leader of the Steppe
Riders, impatiently.  Draal was 38, built like a siege tower himself, and
had little patience for prolonged campaigns.

"We cannot wait 90 days," Vosk replied.  "The Tyrant's relief army —
40,000 strong under General Marek Ironwell — departed Ashenmoor twelve
days ago.  At their forced-march pace of 28 kilometers per day, they
will arrive in 31 days."

## The Engineer's Proposal

Into the silence stepped a figure most of the commanders had never met:
Thessa Brightforge, Master Engineer of the Quarryborn Regiment, age 44,
who had designed the tunnel that brought down the outer wall.

"There is another way in," she said, unrolling a scroll — a survey map
she had personally drawn over the past three weeks from observations,
captured documents, and the testimony of a defector named Sergeant
Aldric Pell (formerly of Solanthir's 4th Gate Watch, who surrendered on
Day 27 in exchange for safe passage for his family of five to the
neutral port of Kelvenmere).

"Beneath the citadel, Solanthir's original builders constructed a
drainage culvert to prevent the granite plug from flooding during the
spring melt.  The culvert is 1.8 meters in diameter — just wide enough
for soldiers in single file without armor.  It runs 270 meters from
the citadel's foundation to an outflow point here—" she pointed to
the map "—in the Tanner's Quarter, which we already control."

"If they know about it, it'll be defended," Draal objected.

"Sergeant Pell says the current garrison does not know.  The culvert
was sealed during the renovation of 1687 and doesn't appear on any
map made after that date.  He only knew because his grandfather,
Aldric Pell Senior, was a stonemason who worked on a repair job
in 1952 and told the family about it."

Vosk studied the map.  "What are the risks?"

Brightforge was precise.  "Three.  First, structural integrity — the
culvert hasn't been inspected in 70 years.  I estimate a 15-20%
probability of partial collapse, which we can mitigate with portable
timber shoring.  Second, detection — we would need to clear the sealed
entrance quietly, which I estimate will take 6-8 hours of careful
masonry work.  Third, capacity — we can only move soldiers through at
a rate of approximately 40 per hour, meaning it would take 5 hours
to insert a force of 200."

"Two hundred against 3,400?" Draal scoffed.

"Two hundred who emerge inside the citadel walls at 3:00 AM, seize
the northern gate mechanism, and raise the portcullises for the main
assault force.  The heavy infantry can be at the gate in 4 minutes
from their current position at the breach."

## The Vote

Marshal Kaelos called the vote.  Under the Alliance Charter (drafted
at the Congress of Pellinor, ratified by all twelve kingdoms on the
Summer Solstice of Year 1491 of the Common Calendar), military
operations requiring more than 500 casualties in expected losses
needed a two-thirds majority of Banner commanders.

The vote was 9 to 3 in favor.  Commanders Draal, Velthis, and Arun
Caide voted against — Draal because he considered the plan too clever
by half, Velthis because she feared the defector was a plant, and
Caide because he had lost his son in a tunnel assault at the Battle
of Greyfen Pass and could not bring himself to order men underground
again.

Kaelos recorded the vote in the Alliance Ledger — Volume XII, page
847 — and signed it with the date: Day 43 of the Siege, Year 1494 CC.

## The Night Assault

At 01:30 AM on Day 44, Master Engineer Brightforge led a team of twelve
sappers to the sealed culvert entrance in the Tanner's Quarter.  They
worked by candlelight behind a screen of wet hides to dampen sound.
The mortar seal was 60 centimeters thick, and they removed it stone by
stone using hand chisels and wooden mallets — no metal-on-metal sounds
that might carry through the granite.

By 07:45 AM — six hours and fifteen minutes, within Brightforge's
estimate — the passage was open.  A scout, Private Maren Holloway
(age 19, the youngest and slenderest soldier available), crawled the
full 270 meters in 35 minutes and returned to report: "The culvert
is intact.  There is standing water — about 30 centimeters deep —
and two partial collapses requiring timber shoring, but it's passable."

The assault force of 200 volunteers, drawn from the Quarryborn Regiment
and Commander Thessik's heavy infantry, began entering at 09:00 PM.
Each soldier carried a short sword, a dagger, no armor, and a
stoppered clay pot of quicklime for use in close combat.  They moved
in groups of ten, with 15-minute intervals between groups.

At 02:47 AM on Day 45, the lead group — commanded by Lieutenant
Caelen Roth, age 27, who had volunteered despite a broken left
collarbone sustained in the breach assault — reached the culvert's
interior exit: a stone grate in the floor of the citadel's undercroft,
directly beneath the granary.

Roth's group silently lifted the grate, neutralized three sleeping
guards in the granary, and proceeded to the northern gate mechanism —
a system of counterweights and chains requiring exactly seven people
to operate, which Sergeant Pell had described in his debriefing.

At 03:14 AM, the three portcullises rose simultaneously.  Commander
Thessik's heavy infantry, 1,200 strong, charged through the murder
corridor.  Castellan Harrowgate, awakened by the alarm bells,
organized a defense of the inner keep but surrendered at 05:30 AM
when Vosk offered terms: the garrison would be disarmed and released,
and the 12,000 civilians would be unharmed.

The Siege of Solanthir was over.  Total Allied casualties: 47 dead,
183 wounded.  Citadel garrison casualties: 312 dead, 891 wounded.
Civilian casualties: none.

In the Alliance Ledger, Marshal Kaelos wrote: "Solanthir taken by
engineering and courage.  Brightforge commended.  Roth promoted to
Captain.  Pell granted citizenship of Kelvenmere as promised."
""",
        "questions": [
            {
                "query": "What are the dimensions and defenses of the Solanthir Citadel?",
                "expected_facts": ["60 meters", "8 meters thick", "three successive portcullises", "25 meters", "3,400"],
                "id": "long_citadel",
            },
            {
                "query": "Describe the drainage culvert plan — dimensions, length, and risks",
                "expected_facts": ["1.8 meters", "270 meters", "15-20%", "6-8 hours", "40 per hour"],
                "id": "long_culvert",
            },
            {
                "query": "What was the vote result and who voted against?",
                "expected_facts": ["9 to 3", "Draal", "Velthis", "Caide", "two-thirds"],
                "id": "long_vote",
            },
            {
                "query": "Who was Sergeant Pell and how did he know about the culvert?",
                "expected_facts": ["Aldric Pell", "4th Gate Watch", "Day 27", "Kelvenmere", "grandfather", "stonemason", "1952"],
                "id": "long_pell",
            },
            {
                "query": "Describe the night assault timeline from start to surrender",
                "expected_facts": ["01:30 AM", "09:00 PM", "02:47 AM", "03:14 AM", "05:30 AM"],
                "id": "long_timeline",
            },
            {
                "query": "What were the final casualties on both sides?",
                "expected_facts": ["47 dead", "183 wounded", "312 dead", "891 wounded", "none"],
                "id": "long_casualties",
            },
            {
                "query": "Who is Thessa Brightforge and what did she accomplish?",
                "expected_facts": ["Brightforge", "Master Engineer", "Quarryborn", "44"],
                "id": "long_brightforge",
            },
            {
                "query": "Who was Private Maren Holloway and what did she report?",
                "expected_facts": ["Maren Holloway", "19", "35 minutes", "30 centimeters", "standing water"],
                "id": "long_scout",
            },
        ],
    },

    # -----------------------------------------------------------------------
    # DIALOGUE-HEAVY (~1000 words) — Mostly dialogue with embedded facts
    # -----------------------------------------------------------------------
    "redlantern_dialogue.md": {
        "size_class": "medium",
        "style": "dialogue/interrogation",
        "content": """\
# The Red Lantern Transcript

## Audio Recording — Interview Room 3, Blackwater Station
## Date: 2025-09-17, 14:32 UTC
## Present: Detective Sergeant Yuki Tanaka (YT), Person of Interest: Elias Varga (EV)

**YT:** For the record, state your full name and date of birth.

**EV:** Elias Mikael Varga.  Born April 7th, 1989.  I'm 36 years old, Detective, and I've done nothing wrong.

**YT:** Nobody said you did.  We're here about the Red Lantern.  You're the owner, correct?

**EV:** Co-owner.  My business partner is Natasha Orlova.  We opened the Red Lantern on June 15th, 2021.  It's a cocktail bar at 224 Riverside Drive, licensed for 120 patrons, fully compliant with all fire codes.  I have the certificates if you—

**YT:** I'm not here about fire codes, Mr. Varga.  I'm here about the basement.

**EV:** [pause — 4 seconds]  What about the basement?

**YT:** Your lease agreement with the building owner, Greystone Properties Ltd, covers the ground floor and first floor only.  Lease number GS-2021-4478.  But last week, our forensic accountants flagged something interesting: your electricity consumption is 4.7 times the average for a venue of your size.  That's 23,000 kilowatt-hours per month versus a benchmark of 4,900.

**EV:** We have a large kitchen.  Commercial ovens—

**YT:** Your menu is cocktails and charcuterie boards.  You don't have commercial ovens.  You have a prep counter and a glass washer.  We checked.

**EV:** [pause — 7 seconds]

**YT:** Mr. Varga, we obtained a warrant — Magistrate Court Order MC-2025-11247 — and searched the premises at 06:00 this morning.  In the basement, which is accessed through a concealed door behind the wine rack on the north wall — a rack that slides on a motorized rail system, by the way, very well made — we found a cryptocurrency mining operation.

**EV:** I want a lawyer.

**YT:** That's your right.  But let me tell you what we found, so you can pass it along.  Forty-seven ASIC mining units — Bitmain Antminer S21 models, each drawing 3,500 watts.  That's 164,500 watts total, running 24/7, explaining your power bill of approximately £8,200 per month.  The units were connected to a mining pool — F2Pool, specifically — and the wallet address traces to a Cayman Islands shell company called Stormveil Holdings Ltd, incorporated on February 3rd, 2024, with registered agent Hendricks & Associates of George Town.

**EV:** I want a lawyer.

**YT:** You'll get one.  But here's what concerns us more than the mining.  In a locked cabinet beside the mining racks — a Fortress-brand steel cabinet, model FX-400, combination lock — we found 14 passports.  Fourteen.  Seven nationalities.  All with different names but four of them had your photograph, Mr. Varga.  Elias Varga, Austrian.  Erik Lindqvist, Swedish.  Marco Ferretti, Italian.  Dmitri Volkov, Russian.

**EV:** I want a lawyer.

**YT:** The other ten passports belong to individuals our colleagues in Interpol are very interested in.  Operation Crossfire — you may have heard of it on the news.  I'm going to pause this interview so you can contact legal representation.  Interview suspended at 14:47 UTC.

## Post-Interview Note (DS Tanaka, 15:30 UTC)

Varga requested solicitor Rachel Evershaw of Mercer & Hallam LLP (phone:
+44 20 7946 0823).  Evershaw arrived at 16:15 UTC.  In the resumed
interview (16:45-18:20 UTC), Varga claimed the mining operation was
Orlova's idea and that he had "no knowledge" of the passports.

Forensic analysis of the passport cabinet's fingerprints returned
matches for Varga (47 prints), Orlova (23 prints), and an
unidentified third party (8 prints, submitted to IDENT1 database,
pending).

Power company records (Thames Valley Energy, account TF-8849201)
confirm elevated consumption beginning September 2023 — approximately
6 months before Stormveil Holdings was incorporated.
""",
        "questions": [
            {
                "query": "What was the electricity consumption at the Red Lantern versus the benchmark?",
                "expected_facts": ["23,000", "4,900", "4.7 times"],
                "id": "dialogue_power",
            },
            {
                "query": "Describe the mining equipment found in the basement",
                "expected_facts": ["47", "Antminer S21", "3,500 watts", "164,500", "F2Pool"],
                "id": "dialogue_mining",
            },
            {
                "query": "What fake identities did Varga have?",
                "expected_facts": ["Erik Lindqvist", "Marco Ferretti", "Dmitri Volkov", "14 passports"],
                "id": "dialogue_passports",
            },
            {
                "query": "What is the Red Lantern's address and when did it open?",
                "expected_facts": ["224 Riverside", "June 15th, 2021", "Natasha Orlova"],
                "id": "dialogue_bar",
            },
            {
                "query": "What was behind the wine rack and how was the mining operation connected offshore?",
                "expected_facts": ["wine rack", "motorized rail", "Stormveil Holdings", "Cayman", "Hendricks"],
                "id": "dialogue_offshore",
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Chunking strategies to compare
# ---------------------------------------------------------------------------

CHUNKING_CONFIGS = {
    "tiny": {
        "label": "Tiny (500/100)",
        "child_size": 500,
        "parent_size": 1500,
        "overlap": 100,
    },
    "default": {
        "label": "Default (1500/200)",
        "child_size": 1500,
        "parent_size": 4000,
        "overlap": 200,
    },
    "large": {
        "label": "Large (3000/400)",
        "child_size": 3000,
        "parent_size": 6000,
        "overlap": 400,
    },
    "huge": {
        "label": "Huge (5000/500)",
        "child_size": 5000,
        "parent_size": 8000,
        "overlap": 500,
    },
    "high_overlap": {
        "label": "High Overlap (1500/600)",
        "child_size": 1500,
        "parent_size": 4000,
        "overlap": 600,
    },
    "no_parent": {
        "label": "No Parents (1500/200, child only)",
        "child_size": 1500,
        "parent_size": None,  # Skip parent chunks, return child content directly
        "overlap": 200,
    },
}


# ---------------------------------------------------------------------------
# Backend + Harness (reused from live_document_rag_test.py)
# ---------------------------------------------------------------------------

class LiveTestBackend:
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        import httpx
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout)

    async def warmup(self, model: str) -> bool:
        print(f"  Warming up {model}...", end=" ", flush=True)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Say OK"}],
            "stream": False,
            "max_tokens": 4,
        }
        for attempt in range(4):
            try:
                resp = await self._client.post("/chat/completions", json=payload)
                if resp.status_code == 200:
                    print("ready.")
                    return True
                print(f"loading ({attempt + 1}/4)...", end=" ", flush=True)
                await asyncio.sleep(15)
            except Exception:
                print(f"timeout ({attempt + 1}/4)...", end=" ", flush=True)
                await asyncio.sleep(15)
        print("failed.")
        return False

    async def chat(self, model: str, system: str, user: str, max_tokens: int = 512) -> tuple[str, float]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }
        start = time.monotonic()
        resp = await self._client.post("/chat/completions", json=payload)
        elapsed = (time.monotonic() - start) * 1000

        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        content = data["choices"][0]["message"].get("content") or ""
        return content, elapsed

    async def close(self) -> None:
        await self._client.aclose()


class ChunkingHarness:
    """Ingests a single document under a specific chunking config and runs queries."""

    def __init__(self) -> None:
        self._db_path = ""
        self._conn = None
        self._vec_enabled = False

    async def setup(self) -> None:
        import aiosqlite

        self._db_path = os.path.join(
            tempfile.gettempdir(), f"aug_chunk_test_{uuid.uuid4().hex[:8]}.db",
        )
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row

        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL DEFAULT 'default',
                filename TEXT NOT NULL,
                mime_type TEXT NOT NULL DEFAULT 'text/plain',
                file_size INTEGER DEFAULT 0,
                chunk_count INTEGER DEFAULT 0,
                scope TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS document_chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                page_num INTEGER,
                char_offset INTEGER DEFAULT 0,
                token_count INTEGER DEFAULT 0,
                embedding BLOB,
                parent_id TEXT REFERENCES document_chunks(id) ON DELETE SET NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON document_chunks(document_id, chunk_index);
            CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5(content, content_rowid='rowid');
            CREATE TRIGGER IF NOT EXISTS trg_ai AFTER INSERT ON document_chunks
                BEGIN INSERT INTO document_chunks_fts(rowid, content) VALUES (NEW.rowid, NEW.content); END;
            CREATE TRIGGER IF NOT EXISTS trg_ad AFTER DELETE ON document_chunks
                BEGIN INSERT INTO document_chunks_fts(document_chunks_fts, rowid, content)
                VALUES ('delete', OLD.rowid, OLD.content); END;
        """)

        try:
            await self._conn.enable_load_extension(True)
            try:
                import sqlite_vec
                await self._conn.load_extension(sqlite_vec.loadable_path())
                self._vec_enabled = True
            except Exception:
                for ext in ["vec0", "sqlite_vec", "vec"]:
                    try:
                        await self._conn.load_extension(ext)
                        self._vec_enabled = True
                        break
                    except Exception:
                        continue
        except Exception:
            pass

        if self._vec_enabled:
            try:
                await self._conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks_vec USING vec0(
                        chunk_id TEXT PRIMARY KEY,
                        embedding float[768] distance_metric=cosine
                    )
                """)
                await self._conn.commit()
            except Exception:
                self._vec_enabled = False

    async def clear(self) -> None:
        """Wipe all data for a fresh ingestion round by dropping and recreating tables."""
        # Drop everything and reinit — safest way to avoid FTS5/vec0 sync issues
        if self._vec_enabled:
            try:
                await self._conn.execute("DROP TABLE IF EXISTS doc_chunks_vec")
            except Exception:
                pass
        await self._conn.execute("DROP TRIGGER IF EXISTS trg_ai")
        await self._conn.execute("DROP TRIGGER IF EXISTS trg_ad")
        await self._conn.execute("DROP TABLE IF EXISTS document_chunks_fts")
        await self._conn.execute("DELETE FROM document_chunks")
        await self._conn.execute("DELETE FROM documents")
        await self._conn.commit()
        # Recreate FTS5 and triggers
        await self._conn.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5(content, content_rowid='rowid');
            CREATE TRIGGER IF NOT EXISTS trg_ai AFTER INSERT ON document_chunks
                BEGIN INSERT INTO document_chunks_fts(rowid, content) VALUES (NEW.rowid, NEW.content); END;
            CREATE TRIGGER IF NOT EXISTS trg_ad AFTER DELETE ON document_chunks
                BEGIN INSERT INTO document_chunks_fts(document_chunks_fts, rowid, content)
                VALUES ('delete', OLD.rowid, OLD.content); END;
        """)
        # Recreate vec0
        if self._vec_enabled:
            try:
                await self._conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks_vec USING vec0(
                        chunk_id TEXT PRIMARY KEY,
                        embedding float[768] distance_metric=cosine
                    )
                """)
                await self._conn.commit()
            except Exception:
                self._vec_enabled = False

    async def ingest(
        self,
        filename: str,
        content: str,
        child_size: int,
        parent_size: int | None,
        overlap: int,
    ) -> tuple[str, int, int]:
        """Ingest a document with specific chunking params. Returns (doc_id, child_count, parent_count)."""
        data = content.encode("utf-8")
        pages = extract_text(data, "text/markdown", filename)
        if not pages:
            raise ValueError(f"No text from {filename}")

        doc_id = uuid.uuid4().hex

        if parent_size is not None:
            child_chunks, parent_chunks = chunk_with_parents(
                pages, child_size=child_size, parent_size=parent_size,
                chunk_overlap=overlap, filename=filename,
            )
        else:
            child_chunks = chunk_text(
                pages, chunk_size=child_size, chunk_overlap=overlap, filename=filename,
            )
            parent_chunks = []

        await self._conn.execute(
            "INSERT INTO documents (id, user_id, filename, mime_type, file_size, chunk_count) "
            "VALUES (?, 'test', ?, 'text/markdown', ?, ?)",
            (doc_id, filename, len(data), len(child_chunks)),
        )

        # Store parents
        parent_id_map: dict[int, str] = {}
        for p in parent_chunks:
            pid = uuid.uuid4().hex
            parent_id_map[p.index] = pid
            await self._conn.execute(
                "INSERT INTO document_chunks (id, document_id, chunk_index, content, page_num, char_offset, token_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (pid, doc_id, -(p.index + 1), p.text, p.page_num, p.char_offset, len(p.text) // 4),
            )

        # Embed and store children
        embed_texts = [c.enriched_text or c.text for c in child_chunks]
        embeddings = await asyncio.to_thread(EmbeddingService.embed, embed_texts)

        for chunk, emb in zip(child_chunks, embeddings, strict=False):
            cid = uuid.uuid4().hex
            blob = EmbeddingService.to_blob(emb)
            parent_db_id = parent_id_map.get(chunk.parent_index) if chunk.parent_index is not None else None

            await self._conn.execute(
                "INSERT INTO document_chunks (id, document_id, chunk_index, content, page_num, char_offset, token_count, embedding, parent_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (cid, doc_id, chunk.index, chunk.text, chunk.page_num, chunk.char_offset, len(chunk.text) // 4, blob, parent_db_id),
            )
            if self._vec_enabled:
                try:
                    await self._conn.execute(
                        "INSERT INTO doc_chunks_vec (chunk_id, embedding) VALUES (?, ?)", (cid, blob),
                    )
                except Exception:
                    pass

        await self._conn.commit()
        return doc_id, len(child_chunks), len(parent_chunks)

    async def search_for_context(self, query: str, doc_id: str, limit: int = 3) -> str:
        """Search and build context string, returning parent content when available."""
        from augmentum.documents.store import DocumentStore

        class _Shim:
            def __init__(s, conn, vec):
                s.conn = conn
                s.vec_enabled = vec

        store = DocumentStore(_Shim(self._conn, self._vec_enabled))
        results = await store.search_for_recall(
            query, user_id="test", limit=limit, document_ids=[doc_id],
        )

        if not results:
            return "[document_context]\n(No results found.)"

        lines = ["[document_context]"]
        for r in results:
            lines.append(f"{r['source']} {r['content'][:800]}")
        return "\n".join(lines)

    async def teardown(self) -> None:
        if self._conn:
            await self._conn.close()
        if self._db_path and os.path.exists(self._db_path):
            try:
                os.unlink(self._db_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def _check_facts(response: str, expected: list[str]) -> tuple[list[str], list[str]]:
    rl = response.lower()
    found = [f for f in expected if f.lower() in rl]
    missing = [f for f in expected if f.lower() not in rl]
    return found, missing


@dataclass
class QuestionResult:
    question_id: str
    config_name: str
    doc_name: str
    facts_found: int
    facts_total: int
    elapsed_ms: float
    child_chunks: int
    parent_chunks: int
    passed: bool
    detail: str = ""


async def run_test(
    backend: LiveTestBackend,
    model: str,
    verbose: bool,
    timeout: float,
) -> list[QuestionResult]:

    all_results: list[QuestionResult] = []
    harness = ChunkingHarness()
    await harness.setup()

    total_configs = len(CHUNKING_CONFIGS)
    total_docs = len(NARRATIVE_DOCUMENTS)

    for cfg_idx, (cfg_name, cfg) in enumerate(CHUNKING_CONFIGS.items(), 1):
        print(f"\n{'=' * 60}")
        print(f"  Config {cfg_idx}/{total_configs}: {cfg['label']}")
        print(f"{'=' * 60}")

        for doc_name, doc_data in NARRATIVE_DOCUMENTS.items():
            await harness.clear()

            doc_id, n_child, n_parent = await harness.ingest(
                doc_name, doc_data["content"],
                child_size=cfg["child_size"],
                parent_size=cfg["parent_size"],
                overlap=cfg["overlap"],
            )
            print(f"\n  {doc_name} [{doc_data['size_class']}] -> {n_child} children, {n_parent} parents")

            for q in doc_data["questions"]:
                qid = q["id"]
                try:
                    print(f"    [{qid}] searching...", end="", flush=True)
                    context = await harness.search_for_context(q["query"], doc_id, limit=3)
                    print(" asking LLM...", end="", flush=True)

                    if verbose:
                        print(f"    Context ({len(context)} chars): {context[:200]}...")

                    system_prompt = (
                        f"{context}\n\n"
                        "You are a helpful assistant. Answer the user's question "
                        "using ONLY the information in [document_context] above. "
                        "Include specific numbers, names, and details from the text. "
                        "If the answer is not in the context, say 'Not in context.'"
                    )

                    response, elapsed = await asyncio.wait_for(
                        backend.chat(model, system_prompt, q["query"]),
                        timeout=timeout,
                    )

                    if verbose:
                        print(f"    Response: {response[:200]}")

                    found, missing = _check_facts(response, q["expected_facts"])
                    threshold = max(1, len(q["expected_facts"]) // 2)
                    passed = len(found) >= threshold

                    icon = "\u2713" if passed else "\u2717"
                    print(f"    {icon} {qid}: {len(found)}/{len(q['expected_facts'])} facts ({elapsed:.0f}ms)"
                          + (f" missing: {missing}" if missing else ""))

                    all_results.append(QuestionResult(
                        question_id=qid, config_name=cfg_name, doc_name=doc_name,
                        facts_found=len(found), facts_total=len(q["expected_facts"]),
                        elapsed_ms=elapsed, child_chunks=n_child, parent_chunks=n_parent,
                        passed=passed, detail=", ".join(missing) if missing else "",
                    ))

                except TimeoutError:
                    print(f"    \u2717 {qid}: TIMEOUT")
                    all_results.append(QuestionResult(
                        question_id=qid, config_name=cfg_name, doc_name=doc_name,
                        facts_found=0, facts_total=len(q["expected_facts"]),
                        elapsed_ms=0, child_chunks=n_child, parent_chunks=n_parent,
                        passed=False, detail="timeout",
                    ))
                except Exception as exc:
                    print(f"    \u2717 {qid}: {str(exc)[:100]}")
                    all_results.append(QuestionResult(
                        question_id=qid, config_name=cfg_name, doc_name=doc_name,
                        facts_found=0, facts_total=len(q["expected_facts"]),
                        elapsed_ms=0, child_chunks=n_child, parent_chunks=n_parent,
                        passed=False, detail=str(exc)[:100],
                    ))

    await harness.teardown()
    return all_results


def print_summary(results: list[QuestionResult], model: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"NARRATIVE CHUNKING COMPARISON — {model}")
    print(f"{'=' * 70}")

    # Per-config summary
    configs = sorted(set(r.config_name for r in results))
    docs = sorted(set(r.doc_name for r in results))

    # Header
    print(f"\n  {'Config':<25s}", end="")
    for doc in docs:
        short = doc.split(".")[0][:12]
        print(f" {short:>12s}", end="")
    print(f" {'TOTAL':>8s}")
    print(f"  {'-' * 25}", end="")
    for _ in docs:
        print(f" {'-' * 12}", end="")
    print(f" {'-' * 8}")

    best_config = ""
    best_score = -1

    for cfg in configs:
        cfg_results = [r for r in results if r.config_name == cfg]
        label = CHUNKING_CONFIGS[cfg]["label"]
        print(f"  {label:<25s}", end="")

        cfg_total_passed = 0
        cfg_total_q = 0

        for doc in docs:
            doc_results = [r for r in cfg_results if r.doc_name == doc]
            passed = sum(1 for r in doc_results if r.passed)
            total = len(doc_results)
            cfg_total_passed += passed
            cfg_total_q += total
            pct = (passed / total * 100) if total else 0
            print(f" {passed}/{total} ({pct:4.0f}%)", end="")

        overall_pct = (cfg_total_passed / cfg_total_q * 100) if cfg_total_q else 0
        print(f" {cfg_total_passed}/{cfg_total_q} ({overall_pct:.0f}%)")

        if overall_pct > best_score:
            best_score = overall_pct
            best_config = label

    print(f"\n  Best config: {best_config} ({best_score:.0f}%)")

    # Per-document size class analysis
    print("\n  --- By Document Size ---")
    for size in ["short", "medium", "long"]:
        size_docs = [d for d, dd in NARRATIVE_DOCUMENTS.items() if dd["size_class"] == size]
        for cfg in configs:
            r_set = [r for r in results if r.config_name == cfg and r.doc_name in size_docs]
            if r_set:
                passed = sum(1 for r in r_set if r.passed)
                total = len(r_set)
                chunks = r_set[0].child_chunks if r_set else 0
                label = CHUNKING_CONFIGS[cfg]["label"]
                print(f"    {size:>6s} | {label:<25s} | {passed}/{total} | {chunks} chunks")

    # Chunk count comparison
    print("\n  --- Chunk Counts ---")
    print(f"  {'Config':<25s}", end="")
    for doc in docs:
        short = doc.split(".")[0][:12]
        print(f" {short:>12s}", end="")
    print()
    for cfg in configs:
        label = CHUNKING_CONFIGS[cfg]["label"]
        print(f"  {label:<25s}", end="")
        for doc in docs:
            r_set = [r for r in results if r.config_name == cfg and r.doc_name == doc]
            if r_set:
                print(f" {r_set[0].child_chunks:>12d}", end="")
            else:
                print(f" {'?':>12s}", end="")
        print()

    print()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Narrative Chunking Strategy Comparison")
    parser.add_argument("--url", default="http://localhost:1234/v1")
    parser.add_argument("--model", required=True, help="Model name (must be loaded)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--json", action="store_true", help="Output raw results as JSON")
    args = parser.parse_args()

    print("=" * 60)
    print("NARRATIVE CHUNKING STRATEGY COMPARISON")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Documents: {len(NARRATIVE_DOCUMENTS)}")
    print(f"  Configs: {len(CHUNKING_CONFIGS)}")
    total_q = sum(len(d["questions"]) for d in NARRATIVE_DOCUMENTS.values())
    print(f"  Questions per config: {total_q}")
    print(f"  Total LLM calls: {total_q * len(CHUNKING_CONFIGS)}")

    backend = LiveTestBackend(args.url, timeout=args.timeout)

    if not await backend.warmup(args.model):
        print("Could not load model. Exiting.")
        return

    results = await run_test(backend, args.model, args.verbose, args.timeout)

    print_summary(results, args.model)

    if args.json:
        print("\n--- JSON ---")
        print(json.dumps([
            {
                "question_id": r.question_id,
                "config": r.config_name,
                "document": r.doc_name,
                "facts_found": r.facts_found,
                "facts_total": r.facts_total,
                "passed": r.passed,
                "elapsed_ms": r.elapsed_ms,
                "child_chunks": r.child_chunks,
                "parent_chunks": r.parent_chunks,
                "detail": r.detail,
            }
            for r in results
        ], indent=2))

    await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
