"""Live document RAG coherence test harness.

Tests the full document pipeline against real LLM backends (LM Studio,
Ollama, cloud APIs) to verify:

1. Context coherence — does the LLM use injected document facts correctly?
2. Negative control — does the LLM refuse/guess without context?
3. Retrieval precision — does vector search surface the right chunk?
4. Inject mode comparison — RAG (search) vs full injection
5. Cross-session isolation — documents bound per-session only
6. Multi-document routing — correct doc retrieved for each question
7. Cross-model transferability — same tests across loaded models

Run manually (not in regular pytest suite):

    .venv/Scripts/python tests/live_document_rag_test.py [OPTIONS]

    --url URL           OpenAI-compatible base URL (default: http://localhost:1234/v1)
    --model NAME        Test a single model instead of discovering all
    --ollama URL        Also test Ollama models at this URL
    --verbose / -v      Show full model outputs and injected context
    --json              Output results as JSON
    --timeout SECS      Per-call timeout (default: 90)
    --skip-negative     Skip negative control tests (faster)
    --skip-cross        Skip cross-session/cross-model tests (faster)
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
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Force UTF-8 stdout on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    Message,
    Usage,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Planted documents — synthetic facts no LLM could know from training
# ---------------------------------------------------------------------------

# Each document contains unique, verifiable facts with distinctive keywords
# that make retrieval validation unambiguous.

PLANTED_DOCUMENTS = {
    "project_zephyr_report.md": {
        "content": """\
# Project Zephyr — Q4 Status Report

## Executive Summary

Project Zephyr is Meridian Corp's internal initiative to develop a quantum-resistant
encryption protocol for satellite communication systems.  The project began on
2025-06-14 under the leadership of Dr. Elara Voss (Principal Cryptographer).

## Budget and Timeline

The total approved budget is $4.73 million across three fiscal years.  Phase 1
(lattice-based key exchange) was completed on 2025-11-30 at 12% under budget.
Phase 2 (post-quantum TLS handshake) is scheduled for completion by 2026-08-15.

The critical milestone "Orion Gate" — full protocol validation against NIST PQC
Round 4 candidates — is targeted for 2026-05-01.  Three external auditors
(Kuiper Security, Northvane Labs, and Stormgate Consulting) have been retained
for the independent review.

## Technical Details

The protocol uses a hybrid approach combining CRYSTALS-Kyber-1024 for key
encapsulation with a proprietary lattice construction called "FrostLattice-7"
developed by the Meridian Research Division.  FrostLattice-7 achieves a
security level of 2^192 with a handshake overhead of only 847 bytes — 23%
smaller than the nearest competitor (NTRUPrime-953).

Key exchange completes in 3.2 milliseconds on commodity ARM Cortex-A78 hardware.

## Team

- Dr. Elara Voss — Principal Cryptographer, project lead
- Marcus Chen-Watanabe — Protocol Engineer, TLS integration
- Priya Sundaram — Hardware Validation Lead
- Tomás Reyes-Björk — Satellite Systems Architect
""",
        "questions": [
            {
                "query": "What is the total approved budget for Project Zephyr?",
                "expected_facts": ["4.73 million", "$4.73"],
                "category": "exact_retrieval",
            },
            {
                "query": "What is FrostLattice-7 and what security level does it achieve?",
                "expected_facts": ["FrostLattice-7", "2^192", "lattice", "Meridian"],
                "category": "technical_detail",
            },
            {
                "query": "Who leads Project Zephyr and what is the Orion Gate milestone?",
                "expected_facts": ["Elara Voss", "Orion Gate", "2026-05-01", "NIST"],
                "category": "multi_fact",
            },
            {
                "query": "How does the handshake overhead compare to competitors?",
                "expected_facts": ["847 bytes", "23%", "NTRUPrime"],
                "category": "comparison",
            },
        ],
    },
    "velaris_species_catalog.md": {
        "content": """\
# Velaris Deep-Sea Species Catalog — Trench Survey 2025

## Overview

The Velaris Trench (11,847m depth, coordinates 34.7°S 178.2°W) was surveyed
by the autonomous submersible "Abyssal Pioneer IV" between March and September
2025.  A total of 14 previously undescribed species were documented.

## Notable Species

### Luminara crystallophora

A bioluminescent cnidarian found at 9,200-10,400m depth.  Its bell diameter
averages 34cm and produces a distinctive violet-amber pulsating glow at
a frequency of exactly 0.73 Hz.  The organism's mesoglea contains microscopic
calcium-fluorite crystals arranged in a Fibonacci spiral pattern, giving it
a prismatic appearance under ROV spotlights.

Tissue samples reveal an anomalous protein designated "velarin-9" with a
molecular weight of 47.3 kDa.  Velarin-9 is hypothesized to function as
both a structural scaffold and a piezoelectric transducer, converting
mechanical pressure waves into the bioluminescent signal.

### Ferrovermis magnetotacticus

A polychaete worm (avg. length 12.7cm) found exclusively in the
magnetite-rich sediment layer at 11,200-11,600m.  The species navigates
using internal chains of biogenic magnetite nanocrystals (50-80nm diameter)
aligned with the local magnetic field.  Gut content analysis shows a diet
of chemolithoautotrophic archaea (primarily Methanopyrus kandleri variants).

Population density: approximately 340 individuals per square meter in optimal
sediment patches.  Reproductive cycle appears tied to seasonal hydrothermal
vent activity, with egg masses deposited in 14-day intervals.

## Survey Equipment

Abyssal Pioneer IV specifications:
- Maximum operational depth: 12,500m
- Sampling arms: 2 titanium manipulators + 1 suction sampler
- Camera array: 8K stereo pair + UV/IR multispectral + 360° sonar
- Endurance: 72 hours continuous operation at maximum depth
""",
        "questions": [
            {
                "query": "What is Luminara crystallophora and at what frequency does it glow?",
                "expected_facts": ["Luminara crystallophora", "0.73 Hz", "bioluminescent", "cnidarian"],
                "category": "exact_retrieval",
            },
            {
                "query": "What protein was found in the Velaris Trench organisms and what does it do?",
                "expected_facts": ["velarin-9", "47.3 kDa", "piezoelectric"],
                "category": "technical_detail",
            },
            {
                "query": "Describe the Ferrovermis magnetotacticus navigation mechanism",
                "expected_facts": ["magnetite", "nanocrystals", "50-80nm", "magnetic field"],
                "category": "technical_detail",
            },
        ],
    },
    "helix_protocol_memo.md": {
        "content": """\
# INTERNAL MEMO — Helix Manufacturing Protocol v3.2

**From:** Director Kazuhiro Tanaka, Advanced Materials Division
**To:** Production Team Leads (Sites: Reykjavik, Kuala Lumpur, Valparaíso)
**Date:** 2025-10-28
**Classification:** Company Confidential

## Process Change: Annealing Temperature

Effective immediately, the annealing temperature for HX-grade titanium alloy
billets is changed from 1,680°C to 1,742°C (±3°C tolerance).  This 62-degree
increase was validated by the Reykjavik metallurgy lab over 847 test cycles
and results in a 14.3% improvement in grain boundary coherence.

The previous temperature (1,680°C) caused intermittent micro-void formation
at a rate of 2.1 per 10,000 billets.  The new temperature eliminates this
defect entirely based on destructive testing of 12,000 sample billets.

## New Catalyst: Iridium-Palladium Nanoparticle Blend

Replace the current platinum catalyst (Pt-7 grade) with the new IrPd-12
nanoparticle blend (60% iridium, 40% palladium by mass, particle diameter
15-22nm).  This reduces catalyst cost by 37% while improving reaction
selectivity from 94.2% to 98.7%.

Supplier: Aethon Nanomaterials GmbH (contract AET-2025-4419)
Lead time: 6 weeks from PO to delivery
Minimum order: 500g per site

## Safety Note

The higher annealing temperature requires upgrading the argon shielding flow
rate from 12 L/min to 18 L/min to prevent surface oxidation.  All three sites
must complete the gas system recalibration before the switchover date of
2025-12-01.
""",
        "questions": [
            {
                "query": "What is the new annealing temperature for HX-grade titanium billets?",
                "expected_facts": ["1,742", "1742", "62-degree", "62 degree"],
                "category": "exact_retrieval",
            },
            {
                "query": "What catalyst replaced Pt-7 and what improvement does it bring?",
                "expected_facts": ["IrPd-12", "iridium", "palladium", "98.7%"],
                "category": "multi_fact",
            },
        ],
    },

    # -----------------------------------------------------------------
    # NARRATIVE — fiction with distinctive world-building details
    # -----------------------------------------------------------------
    "ashenveil_chapter_7.md": {
        "content": """\
# The Ashenveil Chronicles — Chapter 7: The Cartographer's Descent

The city of Duskhollow hung from the underside of the Vanthar Shelf like a
chandelier of iron and phosphor-glass.  Seventeen thousand souls lived in its
inverted spires, their days measured not by sunlight — which never reached
this depth — but by the rhythmic dimming of the bioluminescent kelp forests
that clung to the shelf's underbelly.  The locals called each cycle a "bloom,"
and there were exactly 3.7 blooms to a surface-world day.

Maren Ashwick pressed her charcoal pencil against the vellum and sketched the
last quadrant of her map — the Gullet, a spiraling descent of 412 steps carved
into living coral, connecting Duskhollow's Market Ring to the Abyssal Observatory
below.  She had been commissioned by the Cartographers' Accord, a guild of
37 members who maintained the only accurate maps of the Undershelf territories.
The fee was 140 silver thalers, half paid upfront.

"The Gullet shifts," said Brevven, her lantern-bearer, a wiry man with
gill-scars behind both ears.  "Every ninety-one blooms, the coral grows and
the steps rearrange.  Your map will be wrong before the season turns."

Maren knew this.  It was why the Accord employed three full-time Drift
Surveyors — cartographers who did nothing but measure the coral's growth
rate of 2.3 centimeters per bloom and update the living maps accordingly.
The current Chief Drift Surveyor was an elderly woman named Thessaly Coralwight,
who had held the post for eleven years and claimed she could feel the coral's
movement through the soles of her bare feet.

At the base of the Gullet stood the Abyssal Observatory, a dome of fused
obsidian and leviathan bone, 28 meters in diameter.  Inside, three massive
lenses — ground from kraken-eye crystal by the Opticum monks of Mount
Pellucid — focused the deep-ocean phosphorescence into visible images of the
seafloor terrain up to 900 meters below.  The Observatory's director,
Hamund Voss (no relation to the surface family), had recently discovered
what he called "shadow currents" — invisible rivers of ultra-dense brine
flowing along the abyssal plain at speeds of exactly 4.1 knots, carrying
trace minerals that the kelp forests depended upon for their luminescence.

Maren added the Observatory to her map with a careful circle and the notation:
"Obs. — Voss.  Shadow currents confirmed.  Kelp dependency established."
""",
        "questions": [
            {
                "query": "How many steps are in the Gullet and what does it connect?",
                "expected_facts": ["412", "Gullet", "Market Ring", "Abyssal Observatory"],
                "category": "exact_retrieval",
            },
            {
                "query": "What are shadow currents and who discovered them?",
                "expected_facts": ["shadow currents", "Hamund Voss", "4.1 knots", "brine"],
                "category": "technical_detail",
            },
            {
                "query": "Describe the Cartographers' Accord and the role of Drift Surveyors",
                "expected_facts": ["Cartographers' Accord", "37 members", "Drift Surveyor", "Thessaly Coralwight"],
                "category": "multi_fact",
            },
            {
                "query": "How does the Abyssal Observatory work?",
                "expected_facts": ["obsidian", "leviathan bone", "28 meters", "kraken-eye crystal", "900 meters"],
                "category": "technical_detail",
            },
        ],
    },

    # -----------------------------------------------------------------
    # CODING — API design doc with specific signatures and behaviors
    # -----------------------------------------------------------------
    "vortex_stream_api.md": {
        "content": """\
# Vortex Stream Processing SDK — API Reference v2.4.0

## Overview

Vortex is an internal stream processing SDK for handling real-time telemetry
from Kestrel satellite constellation (48 satellites, LEO orbit, 1.2 TB/day
aggregate throughput).  It provides exactly-once delivery guarantees and
sub-millisecond windowed aggregation.

## Core Classes

### `VortexConsumer`

```python
class VortexConsumer:
    def __init__(
        self,
        cluster_uri: str,
        group_id: str,
        deserializer: Callable[[bytes], T] = json.loads,
        max_poll_bytes: int = 2_097_152,  # 2 MB
        heartbeat_interval_ms: int = 3000,
        session_timeout_ms: int = 45000,
    ) -> None: ...
```

**Important:** `max_poll_bytes` must be a power of 2.  Non-power-of-2 values
silently round down to the nearest power.  Default is 2 MB (2^21).

The consumer uses a proprietary back-pressure algorithm called "Tide Control"
which dynamically adjusts fetch rates based on a 3-second sliding window of
processing latency.  When p99 latency exceeds 847ms (the "storm threshold"),
Tide Control halves the fetch rate and emits a `TIDE_STORM` metric to the
monitoring pipeline.  Recovery occurs when p99 drops below 212ms (the
"calm threshold") for 5 consecutive windows.

### `VortexProducer`

```python
class VortexProducer:
    def __init__(
        self,
        cluster_uri: str,
        serializer: Callable[[T], bytes] = json.dumps_bytes,
        batch_size: int = 500,
        linger_ms: int = 50,
        compression: Literal["none", "zstd", "lz4"] = "zstd",
        idempotency_key_fn: Callable[[T], str] | None = None,
    ) -> None: ...
```

**Exactly-once semantics:** When `idempotency_key_fn` is provided, the producer
maintains a 256-entry LRU dedup cache per partition.  Duplicate keys within the
cache window are silently dropped.  The cache uses MurmurHash3-128 internally.

### `WindowAggregator`

```python
class WindowAggregator:
    def __init__(
        self,
        window_size_ms: int = 1000,
        slide_ms: int = 200,
        watermark_delay_ms: int = 5000,
        allowed_lateness_ms: int = 30000,
    ) -> None: ...

    def aggregate(
        self,
        key_fn: Callable[[T], str],
        reduce_fn: Callable[[list[T]], R],
        emit_fn: Callable[[str, R, WindowMetadata], None],
    ) -> None: ...
```

**Late data handling:** Events arriving after `watermark_delay_ms` but before
`allowed_lateness_ms` trigger a re-emission of the affected window with an
updated result.  Events after `allowed_lateness_ms` are routed to the dead
letter topic `{topic}.late.v2` with the header `X-Vortex-Late-By-Ms` set
to the actual lateness in milliseconds.

## Error Handling

All Vortex operations raise from a hierarchy rooted at `VortexError`:

- `VortexTimeoutError` — cluster unreachable after 3 retry attempts (exponential backoff: 100ms, 400ms, 1600ms)
- `VortexSerializationError` — deserializer/serializer raised; carries `.raw_bytes` attribute
- `VortexBackPressureError` — Tide Control rejected the operation; retry after `.retry_after_ms` milliseconds
- `VortexPartitionRebalanceError` — consumer group rebalance in progress; current batch is invalidated

## Configuration: Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VORTEX_CLUSTER_URI` | (required) | Comma-separated broker addresses |
| `VORTEX_TLS_CERT_PATH` | `/etc/vortex/client.pem` | mTLS client certificate |
| `VORTEX_METRICS_PORT` | `9147` | Prometheus metrics exporter port |
| `VORTEX_LOG_LEVEL` | `WARN` | SDK log level (TRACE/DEBUG/INFO/WARN/ERROR) |
| `VORTEX_TIDE_STORM_THRESHOLD_MS` | `847` | Tide Control storm trigger (p99 latency) |
| `VORTEX_TIDE_CALM_THRESHOLD_MS` | `212` | Tide Control calm recovery (p99 latency) |
""",
        "questions": [
            {
                "query": "What is the Tide Control storm threshold and how does back-pressure recovery work?",
                "expected_facts": ["847ms", "847", "Tide Control", "212ms", "212", "5 consecutive"],
                "category": "technical_detail",
            },
            {
                "query": "How does Vortex handle late data in the WindowAggregator?",
                "expected_facts": ["watermark_delay_ms", "allowed_lateness_ms", "dead letter", "X-Vortex-Late-By-Ms"],
                "category": "technical_detail",
            },
            {
                "query": "What is the idempotency mechanism in VortexProducer?",
                "expected_facts": ["idempotency_key_fn", "256-entry", "LRU", "MurmurHash3"],
                "category": "exact_retrieval",
            },
            {
                "query": "What errors can Vortex operations raise?",
                "expected_facts": ["VortexTimeoutError", "VortexBackPressureError", "retry_after_ms"],
                "category": "multi_fact",
            },
        ],
    },

    # -----------------------------------------------------------------
    # BUSINESS — financial report with specific numbers and strategy
    # -----------------------------------------------------------------
    "aurelia_q3_earnings.md": {
        "content": """\
# Aurelia Dynamics — Q3 FY2026 Earnings Report

**Ticker:** AUDY (NASDAQ)  |  **Report Date:** 2026-01-15  |  **Fiscal Year End:** March 31

## Financial Highlights

| Metric | Q3 FY2026 | Q3 FY2025 | YoY Change |
|--------|-----------|-----------|------------|
| Revenue | $287.4M | $231.9M | +23.9% |
| Gross Margin | 64.2% | 58.7% | +5.5pp |
| Operating Income | $47.3M | $29.1M | +62.5% |
| Net Income | $38.9M | $22.4M | +73.7% |
| EPS (diluted) | $1.47 | $0.86 | +70.9% |
| Free Cash Flow | $52.1M | $31.7M | +64.4% |

Total headcount reached 2,847 employees across 6 offices (headquarters in
Zürich, with regional offices in Austin TX, Bangalore, São Paulo, Helsinki,
and Shenzhen).

## Segment Performance

### Quantum Sensing Division (QSD) — 58% of revenue

Revenue: $166.7M (+31.2% YoY).  The division's flagship product, the
"Eigenstate 400" quantum magnetometer, shipped 1,247 units in Q3, up from
891 in the prior year.  Average selling price increased 7.3% to $112,400
per unit due to the new titanium-sapphire sensor housing upgrade (codename
"Project Garnet").

Backlog stands at $194.3M, representing approximately 3.5 quarters of
QSD revenue at current run rate.  The U.S. Department of Energy contract
(DE-SC0024-7718) for 340 units was the single largest order, valued at
$38.2M with delivery over 18 months.

### Industrial Photonics Division (IPD) — 42% of revenue

Revenue: $120.7M (+15.1% YoY).  Growth driven by the "Prism Ultra" laser
calibration system adopted by 14 new automotive OEM customers in Q3.
Recurring SaaS revenue (calibration-as-a-service) reached $23.4M,
representing 19.4% of IPD revenue, up from 11.2% a year ago.

Key win: Stellantis signed a 5-year enterprise agreement worth $67M for
Prism Ultra deployment across 23 manufacturing plants globally.

## Strategic Outlook

CEO Dr. Linnea Ekström announced three strategic priorities for FY2027:

1. **Photonic AI Accelerator ("Project Sable"):** $45M R&D investment over
   24 months to develop a photonic tensor processing unit targeting 2.4
   PFLOPS at 18 watts.  First silicon tape-out expected Q2 FY2027.

2. **Acquisition of NovaSpin GmbH:** Letter of intent signed for €73M
   all-cash acquisition of NovaSpin, a 42-person spin-qubit startup based
   in Dresden.  Expected to close by Q4 FY2026 pending regulatory approval
   from BaFin and CFIUS.

3. **APAC Expansion:** Opening a new 12,000 sqm R&D center in Yokohama,
   Japan (operational by September 2026), with initial staffing of 85
   engineers focused on automotive quantum sensing applications.

## Guidance

FY2026 full-year revenue guidance raised to $1.08B-$1.12B (prior: $1.02B-$1.06B).
Gross margin expected to expand to 65-66% in Q4 driven by Project Garnet
yield improvements.  The Board approved a $75M share buyback program
effective February 2026.
""",
        "questions": [
            {
                "query": "What was Aurelia Dynamics Q3 revenue and EPS?",
                "expected_facts": ["287.4", "$287.4M", "1.47", "$1.47"],
                "category": "exact_retrieval",
            },
            {
                "query": "What is Project Sable and what are its performance targets?",
                "expected_facts": ["Project Sable", "photonic", "$45M", "2.4 PFLOPS", "18 watts"],
                "category": "technical_detail",
            },
            {
                "query": "Describe the NovaSpin acquisition details",
                "expected_facts": ["NovaSpin", "73M", "Dresden", "spin-qubit", "CFIUS"],
                "category": "multi_fact",
            },
            {
                "query": "What is the Eigenstate 400 and how many units shipped?",
                "expected_facts": ["Eigenstate 400", "1,247", "quantum magnetometer", "$112,400"],
                "category": "comparison",
            },
            {
                "query": "What is the updated revenue guidance for FY2026?",
                "expected_facts": ["1.08B", "1.12B", "$75M", "buyback"],
                "category": "exact_retrieval",
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Backend wrapper (reused from live_model_test.py pattern)
# ---------------------------------------------------------------------------

class LiveTestBackend:
    """Minimal OpenAI-compatible backend for live testing."""

    def __init__(self, base_url: str, timeout: float = 90.0) -> None:
        import httpx
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout)

    async def warmup(self, model: str, max_attempts: int = 6, wait_secs: int = 15) -> bool:
        """Send a tiny request to force model loading, wait for it to be ready.

        LM Studio can take 30-90s to swap models depending on size and VRAM.
        """
        print(f"    Warming up {model}...", end=" ", flush=True)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Say OK"}],
            "stream": False,
            "max_tokens": 4,
        }
        for attempt in range(max_attempts):
            try:
                resp = await self._client.post("/chat/completions", json=payload)
                if resp.status_code == 200:
                    print("ready.")
                    return True
                # Parse error message for better diagnostics
                try:
                    err = resp.json().get("error", {}).get("message", "")
                except Exception:
                    err = resp.text[:100]
                if "canceled" in err.lower() or "loading" in err.lower():
                    print(f"loading ({attempt + 1}/{max_attempts})...", end=" ", flush=True)
                elif "unloaded" in err.lower():
                    print(f"unloaded, retrying ({attempt + 1}/{max_attempts})...", end=" ", flush=True)
                else:
                    print(f"error: {err[:80]} ({attempt + 1}/{max_attempts})...", end=" ", flush=True)
                await asyncio.sleep(wait_secs)
            except Exception:
                print(f"timeout ({attempt + 1}/{max_attempts})...", end=" ", flush=True)
                await asyncio.sleep(wait_secs)
        print("failed to load after all attempts.")
        return False

    async def chat(self, request: InternalChatRequest) -> InternalChatResponse:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in request.messages
            ],
            "stream": False,
            "temperature": 0.1,  # Low temp for deterministic fact recall
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        resp = await self._client.post("/chat/completions", json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Backend returned {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        choice = data["choices"][0]
        msg = choice["message"]

        return InternalChatResponse(
            message=Message(
                role=msg.get("role", "assistant"),
                content=msg.get("content") or "",
            ),
            model=data.get("model", request.model),
            finish_reason=choice.get("finish_reason"),
            usage=Usage(
                prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
                completion_tokens=data.get("usage", {}).get("completion_tokens", 0),
            ),
        )

    async def list_models(self) -> list[str]:
        resp = await self._client.get("/models")
        if resp.status_code != 200:
            return []
        data = resp.json()
        return [m["id"] for m in data.get("data", [])]

    async def close(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Document RAG harness — wires up real SQLite + embeddings + DocumentStore
# ---------------------------------------------------------------------------

class DocumentRAGHarness:
    """Sets up a real document store with SQLite, embeddings, and vec0."""

    def __init__(self) -> None:
        self._db_path: str = ""
        self._conn = None
        self._store = None
        self._doc_ids: dict[str, str] = {}  # filename -> doc_id

    async def setup(self) -> None:
        """Initialize a temporary SQLite database with all document migrations."""
        import aiosqlite

        self._db_path = os.path.join(
            tempfile.gettempdir(), f"augmentum_rag_test_{uuid.uuid4().hex[:8]}.db",
        )

        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row

        # Run minimal schema needed for document store
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
            CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);

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
            CREATE INDEX IF NOT EXISTS idx_chunks_document
                ON document_chunks(document_id, chunk_index);

            CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts
                USING fts5(content, content_rowid='rowid');

            CREATE TRIGGER IF NOT EXISTS trg_doc_chunks_ai
                AFTER INSERT ON document_chunks
                BEGIN
                    INSERT INTO document_chunks_fts(rowid, content)
                    VALUES (NEW.rowid, NEW.content);
                END;
            CREATE TRIGGER IF NOT EXISTS trg_doc_chunks_ad
                AFTER DELETE ON document_chunks
                BEGIN
                    INSERT INTO document_chunks_fts(document_chunks_fts, rowid, content)
                    VALUES ('delete', OLD.rowid, OLD.content);
                END;
            CREATE TRIGGER IF NOT EXISTS trg_doc_chunks_au
                AFTER UPDATE OF content ON document_chunks
                BEGIN
                    INSERT INTO document_chunks_fts(document_chunks_fts, rowid, content)
                    VALUES ('delete', OLD.rowid, OLD.content);
                    INSERT INTO document_chunks_fts(rowid, content)
                    VALUES (NEW.rowid, NEW.content);
                END;

            CREATE TABLE IF NOT EXISTS session_documents (
                session_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                inject_mode TEXT NOT NULL DEFAULT 'search',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (session_id, document_id)
            );
        """)

        # Try to set up vec0 table
        vec_enabled = False
        try:
            await self._conn.enable_load_extension(True)
            # Try common sqlite-vec extension paths
            for ext_name in ["vec0", "sqlite_vec", "vec"]:
                try:
                    await self._conn.load_extension(ext_name)
                    vec_enabled = True
                    break
                except Exception:
                    continue

            if not vec_enabled:
                # Try loading from Python package location
                try:
                    import sqlite_vec
                    await self._conn.load_extension(sqlite_vec.loadable_path())
                    vec_enabled = True
                except Exception:
                    pass
        except Exception:
            pass

        if vec_enabled:
            try:
                await self._conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks_vec USING vec0(
                        chunk_id TEXT PRIMARY KEY,
                        embedding float[768] distance_metric=cosine
                    )
                """)
                await self._conn.commit()
            except Exception:
                vec_enabled = False

        # Create a lightweight backend wrapper for DocumentStore
        self._backend_wrapper = _SQLiteBackendShim(self._conn, vec_enabled)

        from augmentum.documents.store import DocumentStore
        self._store = DocumentStore(self._backend_wrapper)

        status = "vec0 + FTS5" if vec_enabled else "FTS5 only (no vector search)"
        print(f"  Database: {self._db_path}")
        print(f"  Search backends: {status}")

    async def ingest_document(self, filename: str, content: str) -> str:
        """Ingest a planted document, return its doc_id."""
        data = content.encode("utf-8")
        result = await self._store.ingest(
            data, filename, "text/markdown", user_id="test_user",
        )
        doc_id = result["id"]
        self._doc_ids[filename] = doc_id
        return doc_id

    async def ingest_all(self) -> dict[str, str]:
        """Ingest all planted documents. Returns {filename: doc_id}."""
        for filename, doc_data in PLANTED_DOCUMENTS.items():
            doc_id = await self.ingest_document(filename, doc_data["content"])
            print(f"    Ingested {filename} -> {doc_id[:12]}... "
                  f"({len(doc_data['content'])} chars)")
        return dict(self._doc_ids)

    async def search(self, query: str, limit: int = 3, document_id: str | None = None) -> list[dict]:
        """Search document chunks."""
        return await self._store.search(
            query, user_id="test_user", limit=limit, document_id=document_id,
        )

    async def search_for_recall(
        self, query: str, limit: int = 3, document_ids: list[str] | None = None,
    ) -> list[dict]:
        """Search optimized for context injection (returns parent chunks)."""
        return await self._store.search_for_recall(
            query, user_id="test_user", limit=limit, document_ids=document_ids,
        )

    async def get_full_content(self, doc_id: str) -> dict | None:
        """Get the full content of a document."""
        return await self._store.get_full_content(doc_id)

    def build_context_prompt(
        self,
        doc_results: list[dict],
        full_docs: list[dict] | None = None,
    ) -> str:
        """Build the [document_context] block as the real injection pipeline does."""
        lines = ["[document_context]"]

        if full_docs:
            for doc in full_docs:
                lines.append(f"[Document: {doc['filename']}]\n{doc['content']}")

        for dr in doc_results:
            lines.append(f"{dr['source']} {dr['content'][:500]}")

        return "\n".join(lines)

    async def teardown(self) -> None:
        if self._conn:
            await self._conn.close()
        if self._db_path and os.path.exists(self._db_path):
            try:
                os.unlink(self._db_path)
            except OSError:
                pass


class _SQLiteBackendShim:
    """Minimal shim so DocumentStore thinks it has an SQLiteBackend."""

    def __init__(self, conn, vec_enabled: bool) -> None:
        self.conn = conn
        self.vec_enabled = vec_enabled


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    WARN = "WARN"


@dataclass
class TestResult:
    test_name: str
    model: str
    status: Status
    elapsed_ms: float = 0.0
    detail: str = ""
    category: str = ""
    raw_output: str = ""
    facts_found: list[str] = field(default_factory=list)
    facts_missing: list[str] = field(default_factory=list)


@dataclass
class ModelReport:
    model: str
    results: list[TestResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == Status.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == Status.FAIL)

    @property
    def warnings(self) -> int:
        return sum(1 for r in self.results if r.status == Status.WARN)

    @property
    def coherence_rate(self) -> float:
        coherence_tests = [r for r in self.results if r.category in (
            "exact_retrieval", "technical_detail", "multi_fact", "comparison",
        )]
        if not coherence_tests:
            return 0.0
        passed = sum(1 for r in coherence_tests if r.status == Status.PASS)
        return passed / len(coherence_tests) * 100


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

class LiveDocumentRAGTester:
    """Runs the full document RAG coherence test suite against a live backend."""

    def __init__(
        self,
        backend: LiveTestBackend,
        harness: DocumentRAGHarness,
        model: str,
        *,
        verbose: bool = False,
        timeout: float = 90.0,
        skip_negative: bool = False,
        skip_cross: bool = False,
    ) -> None:
        self.backend = backend
        self.harness = harness
        self.model = model
        self.verbose = verbose
        self.timeout = timeout
        self.skip_negative = skip_negative
        self.skip_cross = skip_cross
        self.results: list[TestResult] = []
        self._doc_ids: dict[str, str] = {}

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"    {msg}")

    def _add_result(self, **kwargs) -> None:
        r = TestResult(model=self.model, **kwargs)
        self.results.append(r)
        icon = {"PASS": "\u2713", "FAIL": "\u2717", "SKIP": "\u2192", "WARN": "\u26a0"}[r.status.value]
        detail = f" \u2014 {r.detail}" if r.detail else ""
        elapsed = f" ({r.elapsed_ms:.0f}ms)" if r.elapsed_ms else ""
        facts_info = ""
        if r.facts_found or r.facts_missing:
            facts_info = f" [{len(r.facts_found)}/{len(r.facts_found) + len(r.facts_missing)} facts]"
        print(f"  {icon} {r.test_name}{facts_info}{elapsed}{detail}")

    async def _ask_llm(self, system_content: str, user_query: str) -> tuple[str, float, Usage]:
        """Send a query with system context to the LLM."""
        messages = []
        if system_content:
            messages.append(Message(role="system", content=system_content))
        messages.append(Message(role="user", content=user_query))

        request = InternalChatRequest(
            model=self.model,
            messages=messages,
            stream=False,
            max_tokens=512,
        )

        start = time.monotonic()
        response = await asyncio.wait_for(
            self.backend.chat(request), timeout=self.timeout,
        )
        elapsed_ms = (time.monotonic() - start) * 1000

        return response.message.content, elapsed_ms, response.usage

    def _check_facts(self, response: str, expected_facts: list[str]) -> tuple[list[str], list[str]]:
        """Check which expected facts appear in the response."""
        response_lower = response.lower()
        found = []
        missing = []
        for fact in expected_facts:
            if fact.lower() in response_lower:
                found.append(fact)
            else:
                missing.append(fact)
        return found, missing

    # -------------------------------------------------------------------
    # Test 1: Positive coherence — does the LLM use the injected context?
    # -------------------------------------------------------------------

    async def test_positive_coherence_search_mode(self) -> None:
        """RAG search mode: retrieve chunks, inject, verify LLM uses them."""
        print("\n  --- Positive Coherence (search/RAG mode) ---")

        for filename, doc_data in PLANTED_DOCUMENTS.items():
            doc_id = self._doc_ids[filename]

            for q in doc_data["questions"]:
                test_name = f"coherence_search_{filename}_{q['category']}"

                try:
                    # Retrieve relevant chunks (mimics the real injection pipeline)
                    results = await self.harness.search_for_recall(
                        q["query"], limit=3, document_ids=[doc_id],
                    )

                    if not results:
                        self._add_result(
                            test_name=test_name, status=Status.WARN,
                            category=q["category"],
                            detail="No chunks retrieved — search returned empty",
                        )
                        continue

                    self._log(f"Retrieved {len(results)} chunks for: {q['query'][:60]}")
                    for r in results:
                        self._log(f"  Score: {r['score']:.4f} | {r['source']}")

                    # Build context exactly as the real pipeline does
                    context = self.harness.build_context_prompt(results)

                    system_prompt = (
                        f"{context}\n\n"
                        "You are a helpful assistant. Answer the user's question "
                        "using ONLY the information provided in the [document_context] above. "
                        "If the answer is not in the context, say 'I don't have that information.'"
                    )

                    response, elapsed, usage = await self._ask_llm(system_prompt, q["query"])

                    self._log(f"Response: {response[:200]}")

                    # Verify facts
                    found, missing = self._check_facts(response, q["expected_facts"])

                    # Pass if at least half the expected facts are present
                    # (LLMs paraphrase, so exact matching is generous)
                    threshold = max(1, len(q["expected_facts"]) // 2)
                    if len(found) >= threshold:
                        self._add_result(
                            test_name=test_name, status=Status.PASS,
                            elapsed_ms=elapsed, category=q["category"],
                            facts_found=found, facts_missing=missing,
                            raw_output=response,
                        )
                    else:
                        self._add_result(
                            test_name=test_name, status=Status.FAIL,
                            elapsed_ms=elapsed, category=q["category"],
                            facts_found=found, facts_missing=missing,
                            detail=f"Missing: {missing}",
                            raw_output=response,
                        )

                except TimeoutError:
                    self._add_result(
                        test_name=test_name, status=Status.FAIL,
                        category=q["category"], detail="Timeout",
                    )
                except Exception as exc:
                    self._add_result(
                        test_name=test_name, status=Status.FAIL,
                        category=q["category"], detail=str(exc)[:200],
                    )

    # -------------------------------------------------------------------
    # Test 2: Full injection mode — entire document in context
    # -------------------------------------------------------------------

    async def test_positive_coherence_full_mode(self) -> None:
        """Full injection: entire document content in system prompt."""
        print("\n  --- Positive Coherence (full injection mode) ---")

        for filename, doc_data in PLANTED_DOCUMENTS.items():
            doc_id = self._doc_ids[filename]

            # Pick one question per document (full mode is expensive)
            q = doc_data["questions"][0]
            test_name = f"coherence_full_{filename}_{q['category']}"

            try:
                full = await self.harness.get_full_content(doc_id)
                if not full:
                    self._add_result(
                        test_name=test_name, status=Status.WARN,
                        category=q["category"],
                        detail="get_full_content returned None",
                    )
                    continue

                context = f"[document_context]\n[Document: {full['filename']}]\n{full['content']}"

                system_prompt = (
                    f"{context}\n\n"
                    "You are a helpful assistant. Answer the user's question "
                    "using ONLY the information provided in the [document_context] above."
                )

                response, elapsed, usage = await self._ask_llm(system_prompt, q["query"])

                found, missing = self._check_facts(response, q["expected_facts"])

                threshold = max(1, len(q["expected_facts"]) // 2)
                if len(found) >= threshold:
                    self._add_result(
                        test_name=test_name, status=Status.PASS,
                        elapsed_ms=elapsed, category=q["category"],
                        facts_found=found, facts_missing=missing,
                        raw_output=response,
                    )
                else:
                    self._add_result(
                        test_name=test_name, status=Status.FAIL,
                        elapsed_ms=elapsed, category=q["category"],
                        facts_found=found, facts_missing=missing,
                        detail=f"Missing: {missing}",
                        raw_output=response,
                    )

            except TimeoutError:
                self._add_result(
                    test_name=test_name, status=Status.FAIL,
                    category=q["category"], detail="Timeout",
                )
            except Exception as exc:
                self._add_result(
                    test_name=test_name, status=Status.FAIL,
                    category=q["category"], detail=str(exc)[:200],
                )

    # -------------------------------------------------------------------
    # Test 3: Negative control — LLM should NOT know planted facts
    # -------------------------------------------------------------------

    async def test_negative_control(self) -> None:
        """Without context, the LLM should not know the planted facts."""
        if self.skip_negative:
            print("\n  --- Negative Control (skipped) ---")
            return

        print("\n  --- Negative Control (no context) ---")

        # Pick one distinctive question per document
        negative_cases = [
            (
                "project_zephyr_report.md",
                "What is FrostLattice-7?",
                ["FrostLattice-7", "2^192", "Meridian"],
            ),
            (
                "velaris_species_catalog.md",
                "What is velarin-9 protein?",
                ["velarin-9", "47.3 kDa", "piezoelectric"],
            ),
            (
                "helix_protocol_memo.md",
                "What temperature is used for HX-grade titanium annealing at Meridian Corp?",
                ["1,742", "1742"],
            ),
            (
                "ashenveil_chapter_7.md",
                "Who is Thessaly Coralwight and what does she do in Duskhollow?",
                ["Thessaly Coralwight", "Drift Surveyor", "Duskhollow"],
            ),
            (
                "vortex_stream_api.md",
                "What is the Vortex Tide Control storm threshold?",
                ["847ms", "Tide Control", "TIDE_STORM"],
            ),
            (
                "aurelia_q3_earnings.md",
                "What is Project Sable at Aurelia Dynamics?",
                ["Project Sable", "2.4 PFLOPS", "photonic tensor"],
            ),
        ]

        for filename, query, forbidden_facts in negative_cases:
            test_name = f"negative_control_{filename}"

            try:
                # No document context — bare query
                system_prompt = (
                    "You are a helpful assistant. If you don't know something, "
                    "say 'I don't have information about that.' Do not make up facts."
                )
                response, elapsed, usage = await self._ask_llm(system_prompt, query)

                self._log(f"No-context response: {response[:200]}")

                # Check if the LLM hallucinated our planted facts
                found, _ = self._check_facts(response, forbidden_facts)

                if not found:
                    self._add_result(
                        test_name=test_name, status=Status.PASS,
                        elapsed_ms=elapsed, category="negative_control",
                        detail="Model correctly did not know planted facts",
                        raw_output=response,
                    )
                else:
                    # The model "knew" our fake facts — either hallucinated or
                    # something truly weird. Mark as WARN, not FAIL.
                    self._add_result(
                        test_name=test_name, status=Status.WARN,
                        elapsed_ms=elapsed, category="negative_control",
                        facts_found=found,
                        detail=f"Model produced planted facts without context: {found}",
                        raw_output=response,
                    )

            except TimeoutError:
                self._add_result(
                    test_name=test_name, status=Status.FAIL,
                    category="negative_control", detail="Timeout",
                )
            except Exception as exc:
                self._add_result(
                    test_name=test_name, status=Status.FAIL,
                    category="negative_control", detail=str(exc)[:200],
                )

    # -------------------------------------------------------------------
    # Test 4: Retrieval precision — right document for right question
    # -------------------------------------------------------------------

    async def test_retrieval_precision(self) -> None:
        """Verify that search returns chunks from the correct document."""
        print("\n  --- Retrieval Precision ---")

        precision_cases = [
            ("FrostLattice-7 handshake overhead", "project_zephyr_report.md"),
            ("Luminara crystallophora bioluminescence frequency", "velaris_species_catalog.md"),
            ("IrPd-12 catalyst nanoparticle blend", "helix_protocol_memo.md"),
            ("velarin-9 protein molecular weight", "velaris_species_catalog.md"),
            ("Orion Gate NIST PQC validation", "project_zephyr_report.md"),
            ("Duskhollow Cartographers Accord Drift Surveyor", "ashenveil_chapter_7.md"),
            ("shadow currents Abyssal Observatory", "ashenveil_chapter_7.md"),
            ("VortexConsumer Tide Control storm threshold", "vortex_stream_api.md"),
            ("WindowAggregator late data dead letter topic", "vortex_stream_api.md"),
            ("Aurelia Dynamics Eigenstate 400 quantum magnetometer", "aurelia_q3_earnings.md"),
            ("NovaSpin acquisition CFIUS regulatory", "aurelia_q3_earnings.md"),
            ("Project Sable photonic tensor PFLOPS", "aurelia_q3_earnings.md"),
        ]

        for query, expected_filename in precision_cases:
            test_name = f"precision_{expected_filename.split('.')[0]}_{query[:30].replace(' ', '_')}"

            try:
                results = await self.harness.search(query, limit=3)

                if not results:
                    self._add_result(
                        test_name=test_name, status=Status.FAIL,
                        category="retrieval_precision",
                        detail="No results returned",
                    )
                    continue

                top_filename = results[0].get("filename", "")
                top_score = results[0].get("score", 0)

                self._log(f"Query: {query}")
                for r in results[:3]:
                    self._log(f"  {r['filename']} | score={r.get('score', 0):.4f} | {r['content'][:80]}")

                if top_filename == expected_filename:
                    self._add_result(
                        test_name=test_name, status=Status.PASS,
                        category="retrieval_precision",
                        detail=f"Top result: {top_filename} (score={top_score:.4f})",
                    )
                else:
                    # Check if the expected doc is in top 3
                    found_in_top3 = any(
                        r.get("filename") == expected_filename for r in results[:3]
                    )
                    self._add_result(
                        test_name=test_name,
                        status=Status.WARN if found_in_top3 else Status.FAIL,
                        category="retrieval_precision",
                        detail=f"Top: {top_filename}, expected: {expected_filename}"
                              + (" (in top 3)" if found_in_top3 else ""),
                    )

            except Exception as exc:
                self._add_result(
                    test_name=test_name, status=Status.FAIL,
                    category="retrieval_precision", detail=str(exc)[:200],
                )

    # -------------------------------------------------------------------
    # Test 5: Cross-document isolation — asking doc A question with doc B
    # -------------------------------------------------------------------

    async def test_cross_document_isolation(self) -> None:
        """With only doc B bound, doc A's facts should not appear."""
        if self.skip_cross:
            print("\n  --- Cross-Document Isolation (skipped) ---")
            return

        print("\n  --- Cross-Document Isolation ---")

        # Ask about Project Zephyr but only inject the species catalog
        species_id = self._doc_ids["velaris_species_catalog.md"]
        zephyr_query = "What is the budget for Project Zephyr?"
        zephyr_facts = ["4.73 million", "$4.73"]

        test_name = "cross_doc_isolation_wrong_doc"

        try:
            results = await self.harness.search_for_recall(
                zephyr_query, limit=3, document_ids=[species_id],
            )

            if results:
                context = self.harness.build_context_prompt(results)
            else:
                context = "[document_context]\n(No relevant documents found.)"

            system_prompt = (
                f"{context}\n\n"
                "You are a helpful assistant. Answer using ONLY the provided context. "
                "If the answer is not in the context, say 'I don't have that information.'"
            )

            response, elapsed, usage = await self._ask_llm(system_prompt, zephyr_query)

            found, _ = self._check_facts(response, zephyr_facts)

            if not found:
                self._add_result(
                    test_name=test_name, status=Status.PASS,
                    elapsed_ms=elapsed, category="cross_doc_isolation",
                    detail="Correctly refused — wrong document bound",
                    raw_output=response,
                )
            else:
                self._add_result(
                    test_name=test_name, status=Status.FAIL,
                    elapsed_ms=elapsed, category="cross_doc_isolation",
                    facts_found=found,
                    detail="LLM produced Project Zephyr facts from species catalog context",
                    raw_output=response,
                )

        except Exception as exc:
            self._add_result(
                test_name=test_name, status=Status.FAIL,
                category="cross_doc_isolation", detail=str(exc)[:200],
            )

    # -------------------------------------------------------------------
    # Test 6: Multi-turn coherence — follow-up questions
    # -------------------------------------------------------------------

    async def test_multi_turn_coherence(self) -> None:
        """Can the LLM answer follow-up questions using the same context?"""
        print("\n  --- Multi-Turn Coherence ---")

        doc_id = self._doc_ids["project_zephyr_report.md"]
        test_name = "multi_turn_zephyr"

        try:
            results = await self.harness.search_for_recall(
                "Project Zephyr team and timeline", limit=5, document_ids=[doc_id],
            )

            context = self.harness.build_context_prompt(results)

            system_prompt = (
                f"{context}\n\n"
                "You are a helpful assistant. Answer using the [document_context] above."
            )

            # Turn 1: broad question
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content="Tell me about Project Zephyr."),
            ]
            request = InternalChatRequest(
                model=self.model, messages=messages, stream=False, max_tokens=300,
            )
            start = time.monotonic()
            resp1 = await asyncio.wait_for(
                self.backend.chat(request), timeout=self.timeout,
            )
            turn1_content = resp1.message.content

            # Turn 2: follow-up referencing turn 1
            messages.append(Message(role="assistant", content=turn1_content))
            messages.append(Message(
                role="user",
                content="Who are the external auditors mentioned for the Orion Gate milestone?",
            ))
            request = InternalChatRequest(
                model=self.model, messages=messages, stream=False, max_tokens=200,
            )
            resp2 = await asyncio.wait_for(
                self.backend.chat(request), timeout=self.timeout,
            )
            elapsed = (time.monotonic() - start) * 1000
            turn2_content = resp2.message.content

            self._log(f"Turn 1: {turn1_content[:150]}")
            self._log(f"Turn 2: {turn2_content[:150]}")

            # Check turn 2 for the three auditor names
            auditor_facts = ["Kuiper Security", "Northvane Labs", "Stormgate Consulting"]
            found, missing = self._check_facts(turn2_content, auditor_facts)

            if len(found) >= 2:
                self._add_result(
                    test_name=test_name, status=Status.PASS,
                    elapsed_ms=elapsed, category="multi_turn",
                    facts_found=found, facts_missing=missing,
                    raw_output=f"T1: {turn1_content[:100]}... | T2: {turn2_content[:100]}...",
                )
            else:
                self._add_result(
                    test_name=test_name, status=Status.FAIL,
                    elapsed_ms=elapsed, category="multi_turn",
                    facts_found=found, facts_missing=missing,
                    detail=f"Follow-up missing auditors: {missing}",
                    raw_output=turn2_content,
                )

        except TimeoutError:
            self._add_result(
                test_name=test_name, status=Status.FAIL,
                category="multi_turn", detail="Timeout",
            )
        except Exception as exc:
            self._add_result(
                test_name=test_name, status=Status.FAIL,
                category="multi_turn", detail=str(exc)[:200],
            )

    # -------------------------------------------------------------------
    # Run all tests
    # -------------------------------------------------------------------

    async def run_all(self, doc_ids: dict[str, str]) -> ModelReport:
        """Run the complete test suite."""
        self._doc_ids = doc_ids

        await self.test_retrieval_precision()
        await self.test_positive_coherence_search_mode()
        await self.test_positive_coherence_full_mode()
        await self.test_negative_control()
        await self.test_cross_document_isolation()
        await self.test_multi_turn_coherence()

        report = ModelReport(model=self.model, results=self.results)
        return report


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _print_summary(reports: list[ModelReport]) -> None:
    """Print a summary table across all tested models."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for report in reports:
        total = len(report.results)
        print(f"\n  Model: {report.model}")
        print(f"    Passed: {report.passed}/{total}")
        print(f"    Failed: {report.failed}/{total}")
        print(f"    Warnings: {report.warnings}/{total}")
        print(f"    Coherence Rate: {report.coherence_rate:.1f}%")

        # Category breakdown
        categories: dict[str, list[TestResult]] = {}
        for r in report.results:
            categories.setdefault(r.category or "other", []).append(r)

        for cat, results in sorted(categories.items()):
            passed = sum(1 for r in results if r.status == Status.PASS)
            print(f"      {cat}: {passed}/{len(results)}")

    print()


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Live Document RAG Coherence Tests")
    parser.add_argument("--url", default="http://localhost:1234/v1",
                        help="OpenAI-compatible base URL")
    parser.add_argument("--model", default=None,
                        help="Test a single model (default: discover all)")
    parser.add_argument("--ollama", default=None,
                        help="Also test Ollama models at this URL")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show full outputs and injected context")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    parser.add_argument("--timeout", type=float, default=90.0,
                        help="Per-call timeout in seconds")
    parser.add_argument("--skip-negative", action="store_true",
                        help="Skip negative control tests")
    parser.add_argument("--skip-cross", action="store_true",
                        help="Skip cross-session/cross-model tests")
    args = parser.parse_args()

    print("=" * 70)
    print("AUGMENTUM — Live Document RAG Coherence Tests")
    print("=" * 70)

    # Phase 1: Set up document store
    print("\n[1/3] Setting up document store...")
    harness = DocumentRAGHarness()
    await harness.setup()

    print("\n[2/3] Ingesting planted documents...")
    doc_ids = await harness.ingest_all()

    # Phase 2: Discover models
    print("\n[3/3] Running tests...")
    reports: list[ModelReport] = []

    # Collect all (backend, model) pairs
    test_targets: list[tuple[LiveTestBackend, str]] = []

    backend = LiveTestBackend(args.url, timeout=args.timeout)
    if args.model:
        test_targets.append((backend, args.model))
    else:
        try:
            models = await backend.list_models()
            if models:
                print(f"\n  Discovered {len(models)} models at {args.url}:")
                for m in models:
                    print(f"    - {m}")
                test_targets.extend((backend, m) for m in models)
            else:
                print(f"  No models found at {args.url}")
        except Exception as exc:
            print(f"  Could not discover models at {args.url}: {exc}")

    # Also test Ollama if specified
    if args.ollama:
        ollama_backend = LiveTestBackend(
            f"{args.ollama.rstrip('/')}/v1", timeout=args.timeout,
        )
        try:
            models = await ollama_backend.list_models()
            if models:
                print(f"\n  Discovered {len(models)} Ollama models:")
                for m in models:
                    print(f"    - {m}")
                test_targets.extend((ollama_backend, m) for m in models)
        except Exception as exc:
            print(f"  Could not discover Ollama models: {exc}")

    if not test_targets:
        print("\n  No models available to test. Is LM Studio / Ollama running?")
        await harness.teardown()
        return

    # Run tests against each model
    for test_backend, model_name in test_targets:
        print(f"\n{'─' * 60}")
        print(f"  Model: {model_name}")
        print(f"{'─' * 60}")

        # Warmup: force LM Studio to load the model before testing
        if not await test_backend.warmup(model_name):
            print(f"  Skipping {model_name} — could not load.")
            continue

        tester = LiveDocumentRAGTester(
            backend=test_backend,
            harness=harness,
            model=model_name,
            verbose=args.verbose,
            timeout=args.timeout,
            skip_negative=args.skip_negative,
            skip_cross=args.skip_cross,
        )

        report = await tester.run_all(doc_ids)
        reports.append(report)

    # Summary
    _print_summary(reports)

    # JSON output
    if args.json:
        json_data = []
        for report in reports:
            json_data.append({
                "model": report.model,
                "passed": report.passed,
                "failed": report.failed,
                "warnings": report.warnings,
                "coherence_rate": report.coherence_rate,
                "results": [
                    {
                        "test_name": r.test_name,
                        "status": r.status.value,
                        "elapsed_ms": r.elapsed_ms,
                        "category": r.category,
                        "detail": r.detail,
                        "facts_found": r.facts_found,
                        "facts_missing": r.facts_missing,
                        "raw_output": r.raw_output if args.verbose else "",
                    }
                    for r in report.results
                ],
            })
        print("\n--- JSON ---")
        print(json.dumps(json_data, indent=2))

    # Cleanup
    await harness.teardown()
    await backend.close()

    # Exit code: fail if any model has <50% coherence
    if any(r.coherence_rate < 50 for r in reports):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(_main())
