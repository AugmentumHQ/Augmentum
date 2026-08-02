"""Structured document chunking strategy comparison test.

Tests how different chunking parameters affect RAG coherence for NON-narrative
content: technical reports, scientific catalogs, manufacturing memos, API docs,
and business/financial reports.  Companion to live_narrative_chunking_test.py.

Run manually:

    .venv/Scripts/python tests/live_structured_chunking_test.py [OPTIONS]

    --url URL           OpenAI-compatible base URL (default: http://localhost:1234/v1)
    --model NAME        Model to test (required — use whatever is loaded)
    --verbose / -v      Show injected context and model responses
    --timeout SECS      Per-call timeout (default: 120)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from augmentum.documents.chunker import chunk_text, chunk_with_parents, extract_text
from augmentum.memory.embeddings import EmbeddingService

# ---------------------------------------------------------------------------
# Chunking configurations — same 6 as narrative test for comparability
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
        "parent_size": None,
        "overlap": 200,
    },
}

# ---------------------------------------------------------------------------
# Planted documents — 5 structured types from the RAG test
# Each has planted facts no LLM could know from training.
# ---------------------------------------------------------------------------

STRUCTURED_DOCUMENTS = {
    # --- TECHNICAL REPORT ---
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
        "doc_type": "technical",
        "questions": [
            {
                "id": "tech_budget",
                "query": "What is the total approved budget for Project Zephyr?",
                "expected_facts": ["4.73 million", "$4.73"],
            },
            {
                "id": "tech_frost",
                "query": "What is FrostLattice-7 and what security level does it achieve?",
                "expected_facts": ["FrostLattice-7", "2^192", "lattice", "Meridian"],
            },
            {
                "id": "tech_orion",
                "query": "Who leads Project Zephyr and what is the Orion Gate milestone?",
                "expected_facts": ["Elara Voss", "Orion Gate", "2026-05-01", "NIST"],
            },
            {
                "id": "tech_overhead",
                "query": "How does the handshake overhead compare to competitors?",
                "expected_facts": ["847 bytes", "23%", "NTRUPrime"],
            },
        ],
    },

    # --- SCIENTIFIC CATALOG ---
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
        "doc_type": "scientific",
        "questions": [
            {
                "id": "sci_luminara",
                "query": "What is Luminara crystallophora and at what frequency does it glow?",
                "expected_facts": ["Luminara crystallophora", "0.73 Hz", "bioluminescent", "cnidarian"],
            },
            {
                "id": "sci_velarin",
                "query": "What protein was found in the Velaris Trench organisms and what does it do?",
                "expected_facts": ["velarin-9", "47.3 kDa", "piezoelectric"],
            },
            {
                "id": "sci_ferro",
                "query": "Describe the Ferrovermis magnetotacticus navigation mechanism",
                "expected_facts": ["magnetite", "nanocrystals", "50-80nm", "magnetic field"],
            },
            {
                "id": "sci_sub",
                "query": "What are the specs of the Abyssal Pioneer IV submersible?",
                "expected_facts": ["12,500m", "72 hours", "8K", "titanium"],
            },
        ],
    },

    # --- MANUFACTURING MEMO ---
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
        "doc_type": "manufacturing",
        "questions": [
            {
                "id": "mfg_temp",
                "query": "What is the new annealing temperature for HX-grade titanium billets?",
                "expected_facts": ["1,742", "1742", "62-degree", "62 degree"],
            },
            {
                "id": "mfg_catalyst",
                "query": "What catalyst replaced Pt-7 and what improvement does it bring?",
                "expected_facts": ["IrPd-12", "iridium", "palladium", "98.7%"],
            },
            {
                "id": "mfg_safety",
                "query": "What safety changes are required for the new annealing process?",
                "expected_facts": ["argon", "18 L/min", "12 L/min", "2025-12-01"],
            },
            {
                "id": "mfg_supplier",
                "query": "Who supplies the new catalyst and what are the order details?",
                "expected_facts": ["Aethon Nanomaterials", "AET-2025-4419", "6 weeks", "500g"],
            },
        ],
    },

    # --- API / CODING DOCS ---
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
        "doc_type": "coding",
        "questions": [
            {
                "id": "api_tide",
                "query": "What is the Tide Control storm threshold and how does back-pressure recovery work?",
                "expected_facts": ["847", "Tide Control", "212", "5 consecutive"],
            },
            {
                "id": "api_late",
                "query": "How does Vortex handle late data in the WindowAggregator?",
                "expected_facts": ["watermark_delay_ms", "allowed_lateness_ms", "dead letter", "X-Vortex-Late-By-Ms"],
            },
            {
                "id": "api_idemp",
                "query": "What is the idempotency mechanism in VortexProducer?",
                "expected_facts": ["idempotency_key_fn", "256", "LRU", "MurmurHash3"],
            },
            {
                "id": "api_errors",
                "query": "What errors can Vortex operations raise?",
                "expected_facts": ["VortexTimeoutError", "VortexBackPressureError", "retry_after_ms"],
            },
        ],
    },

    # --- BUSINESS / FINANCIAL ---
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
        "doc_type": "business",
        "questions": [
            {
                "id": "biz_revenue",
                "query": "What was Aurelia Dynamics Q3 revenue and EPS?",
                "expected_facts": ["287.4", "1.47"],
            },
            {
                "id": "biz_sable",
                "query": "What is Project Sable and what are its performance targets?",
                "expected_facts": ["Project Sable", "photonic", "$45M", "2.4 PFLOPS", "18 watts"],
            },
            {
                "id": "biz_novaspin",
                "query": "Describe the NovaSpin acquisition details",
                "expected_facts": ["NovaSpin", "73M", "Dresden", "spin-qubit", "CFIUS"],
            },
            {
                "id": "biz_eigen",
                "query": "What is the Eigenstate 400 and how many units shipped?",
                "expected_facts": ["Eigenstate 400", "1,247", "quantum magnetometer", "$112,400"],
            },
            {
                "id": "biz_guidance",
                "query": "What is the updated revenue guidance for FY2026?",
                "expected_facts": ["1.08B", "1.12B", "$75M", "buyback"],
            },
        ],
    },
}

# ---------------------------------------------------------------------------
# Backend + harness (shared with narrative test)
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
            tempfile.gettempdir(), f"aug_struct_chunk_{uuid.uuid4().hex[:8]}.db",
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
    doc_type: str
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

    for cfg_idx, (cfg_name, cfg) in enumerate(CHUNKING_CONFIGS.items(), 1):
        print(f"\n{'=' * 60}")
        print(f"  Config {cfg_idx}/{total_configs}: {cfg['label']}")
        print(f"{'=' * 60}")

        for doc_name, doc_data in STRUCTURED_DOCUMENTS.items():
            await harness.clear()

            doc_id, n_child, n_parent = await harness.ingest(
                doc_name, doc_data["content"],
                child_size=cfg["child_size"],
                parent_size=cfg["parent_size"],
                overlap=cfg["overlap"],
            )
            short_name = doc_name.split(".")[0][:20]
            print(f"\n  {short_name} [{doc_data['doc_type']}] -> {n_child} children, {n_parent} parents")

            for q in doc_data["questions"]:
                qid = q["id"]
                try:
                    print(f"    [{qid}] searching...", end="", flush=True)
                    context = await harness.search_for_context(q["query"], doc_id, limit=3)
                    print(f" asking LLM...", end="", flush=True)

                    if verbose:
                        print(f"\n    Context ({len(context)} chars): {context[:200]}...")

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
                        print(f"\n    Response: {response[:200]}")

                    found, missing = _check_facts(response, q["expected_facts"])
                    threshold = max(1, len(q["expected_facts"]) // 2)
                    passed = len(found) >= threshold

                    icon = "\u2713" if passed else "\u2717"
                    print(f"\n    {icon} {qid}: {len(found)}/{len(q['expected_facts'])} facts ({elapsed:.0f}ms)"
                          + (f" missing: {missing}" if missing else ""))

                    all_results.append(QuestionResult(
                        question_id=qid, config_name=cfg_name, doc_name=doc_name,
                        doc_type=doc_data["doc_type"],
                        facts_found=len(found), facts_total=len(q["expected_facts"]),
                        elapsed_ms=elapsed, child_chunks=n_child, parent_chunks=n_parent,
                        passed=passed, detail=", ".join(missing) if missing else "",
                    ))

                except asyncio.TimeoutError:
                    print(f"\n    \u2717 {qid}: TIMEOUT")
                    all_results.append(QuestionResult(
                        question_id=qid, config_name=cfg_name, doc_name=doc_name,
                        doc_type=doc_data["doc_type"],
                        facts_found=0, facts_total=len(q["expected_facts"]),
                        elapsed_ms=0, child_chunks=n_child, parent_chunks=n_parent,
                        passed=False, detail="timeout",
                    ))
                except Exception as exc:
                    print(f"\n    \u2717 {qid}: {str(exc)[:100]}")
                    all_results.append(QuestionResult(
                        question_id=qid, config_name=cfg_name, doc_name=doc_name,
                        doc_type=doc_data["doc_type"],
                        facts_found=0, facts_total=len(q["expected_facts"]),
                        elapsed_ms=0, child_chunks=n_child, parent_chunks=n_parent,
                        passed=False, detail=str(exc)[:100],
                    ))

    await harness.teardown()
    return all_results


def print_summary(results: list[QuestionResult], model: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"STRUCTURED DOCUMENT CHUNKING COMPARISON — {model}")
    print(f"{'=' * 70}")

    configs = sorted(set(r.config_name for r in results))
    doc_types = sorted(set(r.doc_type for r in results))

    # --- By config x doc_type matrix ---
    print(f"\n  {'Config':<35s}", end="")
    for dt in doc_types:
        print(f" {dt:>13s}", end="")
    print(f" {'TOTAL':>8s}")
    print(f"  {'-' * 35}", end="")
    for _ in doc_types:
        print(f" {'-' * 13}", end="")
    print(f" {'-' * 8}")

    config_totals = {}
    for cfg in configs:
        cfg_results = [r for r in results if r.config_name == cfg]
        label = CHUNKING_CONFIGS[cfg]["label"]
        print(f"  {label:<35s}", end="")

        cfg_passed = 0
        cfg_total = 0
        for dt in doc_types:
            dt_results = [r for r in cfg_results if r.doc_type == dt]
            passed = sum(1 for r in dt_results if r.passed)
            total = len(dt_results)
            pct = (passed / total * 100) if total else 0
            print(f" {passed}/{total} ({pct:4.0f}%)", end="")
            cfg_passed += passed
            cfg_total += total

        total_pct = (cfg_passed / cfg_total * 100) if cfg_total else 0
        print(f" {cfg_passed}/{cfg_total} ({total_pct:.0f}%)")
        config_totals[cfg] = (cfg_passed, cfg_total, total_pct)

    # Best config
    best = max(config_totals, key=lambda k: config_totals[k][2])
    best_label = CHUNKING_CONFIGS[best]["label"]
    best_pct = config_totals[best][2]
    print(f"\n  Best config: {best_label} ({best_pct:.0f}%)")

    # --- By doc type breakdown ---
    print(f"\n  --- By Document Type ---")
    for dt in doc_types:
        dt_results = [r for r in results if r.doc_type == dt]
        for cfg in configs:
            cfg_dt = [r for r in dt_results if r.config_name == cfg]
            passed = sum(1 for r in cfg_dt if r.passed)
            total = len(cfg_dt)
            chunks = cfg_dt[0].child_chunks if cfg_dt else 0
            parents = cfg_dt[0].parent_chunks if cfg_dt else 0
            label = CHUNKING_CONFIGS[cfg]["label"]
            print(f"    {dt:>13s} | {label:<35s} | {passed}/{total} | {chunks}c/{parents}p")

    # --- Parent vs No-Parent comparison ---
    print(f"\n  --- Parent vs No-Parent Delta ---")
    if "default" in config_totals and "no_parent" in config_totals:
        for dt in doc_types:
            def_results = [r for r in results if r.config_name == "default" and r.doc_type == dt]
            np_results = [r for r in results if r.config_name == "no_parent" and r.doc_type == dt]
            def_passed = sum(1 for r in def_results if r.passed)
            np_passed = sum(1 for r in np_results if r.passed)
            total = len(def_results)
            delta = np_passed - def_passed
            sign = "+" if delta > 0 else ""
            print(f"    {dt:>13s}: Default {def_passed}/{total} vs No-Parent {np_passed}/{total} ({sign}{delta})")

    # --- Chunk count table ---
    print(f"\n  --- Chunk Counts ---")
    docs = sorted(set(r.doc_name for r in results))
    print(f"  {'Config':<35s}", end="")
    for doc in docs:
        short = doc.split(".")[0][:12]
        print(f" {short:>12s}", end="")
    print()
    for cfg in configs:
        label = CHUNKING_CONFIGS[cfg]["label"]
        print(f"  {label:<35s}", end="")
        for doc in docs:
            dr = [r for r in results if r.config_name == cfg and r.doc_name == doc]
            if dr:
                print(f" {dr[0].child_chunks:>12d}", end="")
            else:
                print(f" {'?':>12s}", end="")
        print()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Structured document chunking comparison")
    parser.add_argument("--url", default="http://localhost:1234/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    total_questions = sum(len(d["questions"]) for d in STRUCTURED_DOCUMENTS.values())
    total_calls = total_questions * len(CHUNKING_CONFIGS)

    print("=" * 60)
    print("STRUCTURED DOCUMENT CHUNKING STRATEGY COMPARISON")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Documents: {len(STRUCTURED_DOCUMENTS)} (technical, scientific, manufacturing, coding, business)")
    print(f"  Configs: {len(CHUNKING_CONFIGS)}")
    print(f"  Questions per config: {total_questions}")
    print(f"  Total LLM calls: {total_calls}")

    backend = LiveTestBackend(args.url, timeout=args.timeout)
    if not await backend.warmup(args.model):
        print("FATAL: Could not warm up model.")
        sys.exit(1)

    results = await run_test(backend, args.model, args.verbose, args.timeout)
    await backend.close()

    print_summary(results, args.model)

    # Exit code
    best_cfg = max(
        set(r.config_name for r in results),
        key=lambda c: sum(1 for r in results if r.config_name == c and r.passed),
    )
    best_score = sum(1 for r in results if r.config_name == best_cfg and r.passed)
    total = sum(1 for r in results if r.config_name == best_cfg)
    if best_score / total < 0.5:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
