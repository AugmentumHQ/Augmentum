"""Search / knowledge / document-RAG / observation settings — the
retrieval substrate.
"""

from __future__ import annotations

from augmentum.registry.registry import SettingsRegistry
from augmentum.registry.settings import Setting

_SEARCH = ("search",)
_SEARCH_ADV = ("search", "advanced")
_KNOWLEDGE = ("knowledge",)
_KNOWLEDGE_ADV = ("knowledge", "advanced")
_DOC = ("document",)
_DOC_ADV = ("document", "advanced")
_OBS = ("observation", "advanced")


def register(r: SettingsRegistry) -> None:
    # ============== Search pipeline ==============
    r.register(
        Setting(
            key="search_expansion_enabled",
            kind="bool",
            default=True,
            label="Query expansion",
            description=(
                "Zero-cost query expansion (synonyms, type reformulation, "
                "site scoping) before sending to the search backend."
            ),
            section="search.pipeline",
            tags=_SEARCH,
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="search_expansion_max_variants",
            kind="int",
            default=3,
            label="Expansion variants per query",
            description=(
                "Maximum expansion variants per original query."
            ),
            section="search.pipeline",
            min_value=1,
            max_value=10,
            tags=_SEARCH_ADV,
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="search_expansion_max_total",
            kind="int",
            default=15,
            label="Expansion total cap",
            description=(
                "Maximum total queries after expansion across all originals."
            ),
            section="search.pipeline",
            min_value=5,
            max_value=50,
            tags=_SEARCH_ADV,
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="search_credibility_enabled",
            kind="bool",
            default=True,
            label="Source credibility scoring",
            description=(
                "Annotate results with source credibility scores (zero LLM "
                "cost — pulled from a static reputation table)."
            ),
            section="search.pipeline",
            tags=_SEARCH,
        )
    )
    r.register(
        Setting(
            key="search_direct_fetch_enabled",
            kind="bool",
            default=True,
            label="Direct URL fetch",
            description=(
                "Auto-fetch URLs found in user queries (bypass the search "
                "step entirely when the user pastes a link)."
            ),
            section="search.pipeline",
            tags=_SEARCH,
        )
    )
    r.register(
        Setting(
            key="search_direct_fetch_max_chars",
            kind="int",
            default=16000,
            label="Direct-fetch char cap",
            description=(
                "Maximum characters extracted per directly-fetched page."
            ),
            section="search.pipeline",
            min_value=1000,
            max_value=128000,
            tags=_SEARCH_ADV,
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="search_relevance_filter_enabled",
            kind="bool",
            default=True,
            label="Relevance filter",
            description=(
                "Drop search results unrelated to the query (zero LLM cost — "
                "heuristic match against query terms)."
            ),
            section="search.pipeline",
            tags=_SEARCH,
        )
    )
    r.register(
        Setting(
            key="search_relevance_min_score",
            kind="float",
            default=0.15,
            label="Relevance threshold",
            description=(
                "Minimum relevance score (0.0–1.0) for a result to survive "
                "the relevance filter."
            ),
            section="search.pipeline",
            min_value=0.0,
            max_value=1.0,
            tags=_SEARCH_ADV,
            advanced=True,
        )
    )

    # ---- Search proxies ----
    r.register(
        Setting(
            key="search_proxies",
            kind="str",
            default="",
            label="SearXNG outbound proxies",
            description=(
                "Newline-separated proxy URLs (http/https/socks5) for "
                "SearXNG outbound. Empty = direct connections only."
            ),
            section="search.proxy",
            max_length=4096,
            tags=("search", "advanced", "admin"),
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="search_proxy_rotation_enabled",
            kind="bool",
            default=False,
            label="Proxy rotation",
            description=(
                "Rotate among configured proxies per request. Provides some "
                "rate-limit relief at the cost of slightly higher latency."
            ),
            section="search.proxy",
            tags=("search", "advanced", "admin"),
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="search_proxy_healthcheck_interval_minutes",
            kind="int",
            default=5,
            label="Proxy health-check interval (min)",
            description=(
                "How often the proxy pool is health-checked. Lower = faster "
                "fail-over but more probe traffic."
            ),
            section="search.proxy",
            min_value=1,
            max_value=1440,
            tags=("search", "advanced", "admin"),
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="search_proxy_fallback_direct_enabled",
            kind="bool",
            default=True,
            label="Fallback to direct",
            description=(
                "If every proxy fails health-check, fall back to a direct "
                "(no-proxy) connection so search keeps working."
            ),
            section="search.proxy",
            tags=("search", "advanced", "admin"),
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ============== Knowledge packs / library ==============
    r.register(
        Setting(
            key="knowledge_library_enabled",
            kind="bool",
            default=True,
            label="Knowledge library",
            description=(
                "Enable the personal knowledge library (saved articles + "
                "PDFs + notes). Off = library surface hidden."
            ),
            section="knowledge.library",
            tags=_KNOWLEDGE,
            voice_aliases=("library", "knowledge library"),
        )
    )
    r.register(
        Setting(
            key="knowledge_library_in_chat",
            kind="bool",
            default=True,
            label="Library in chat",
            description=(
                "Let chat handlers inject relevant library items into "
                "context when a query matches."
            ),
            section="knowledge.library",
            tags=_KNOWLEDGE,
        )
    )
    r.register(
        Setting(
            key="knowledge_library_retention_days",
            kind="int",
            default=90,
            label="Library retention (days)",
            description=(
                "How long library items are kept before automatic cleanup. "
                "Library items you explicitly favorite are exempt."
            ),
            section="knowledge.library",
            min_value=1,
            max_value=3650,
            tags=_KNOWLEDGE,
        )
    )
    r.register(
        Setting(
            key="knowledge_packs_enabled",
            kind="bool",
            default=True,
            label="Knowledge packs",
            description=(
                "Enable knowledge-pack retrieval (Wikipedia, MDWiki, "
                "StackExchange, etc.). Off = no pack injection."
            ),
            section="knowledge.packs",
            tags=_KNOWLEDGE,
            voice_aliases=("packs", "wikipedia"),
        )
    )
    r.register(
        Setting(
            key="knowledge_max_results",
            kind="int",
            default=5,
            label="Max pack results",
            description=(
                "Fallback result cap when a per-mode override is unset."
            ),
            section="knowledge.packs",
            min_value=1,
            max_value=50,
            tags=_KNOWLEDGE_ADV,
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="knowledge_min_score",
            kind="float",
            default=0.3,
            label="Minimum pack score",
            description=(
                "Legacy minimum similarity score for pack hits. Hybrid path "
                "(RRF merge) supersedes this for most retrievals."
            ),
            section="knowledge.packs",
            min_value=0.0,
            max_value=1.0,
            tags=_KNOWLEDGE_ADV,
            advanced=True,
            deprecated="hybrid retrieval uses RRF — this is back-compat only",
        )
    )
    r.register(
        Setting(
            key="knowledge_embedding_use_gpu",
            kind="bool",
            default=True,
            label="GPU embeddings",
            description=(
                "Auto-detect GPU for knowledge-pack embedding. Falls back "
                "to CPU when no compatible GPU is present."
            ),
            section="knowledge.embedding",
            tags=_KNOWLEDGE_ADV,
            advanced=True,
            restart_required=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="knowledge_embedding_batch_size",
            kind="int",
            default=512,
            label="Embedding batch size",
            description=(
                "Chunks per embedding batch. Higher = better GPU utilization, "
                "more peak VRAM."
            ),
            section="knowledge.embedding",
            min_value=8,
            max_value=4096,
            tags=_KNOWLEDGE_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="knowledge_catalog_cache_ttl",
            kind="int",
            default=86400,
            label="Catalog cache TTL (s)",
            description=(
                "How long the available-packs catalog is cached. 86400 = 24h."
            ),
            section="knowledge.packs",
            min_value=60,
            max_value=604800,
            tags=_KNOWLEDGE_ADV,
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="knowledge_packs_custom_dir",
            kind="str",
            default="",
            label="Custom packs directory",
            description=(
                "Override the storage location for downloaded knowledge "
                "packs. Empty = use the default '{data_dir}/knowledge'."
            ),
            section="knowledge.packs",
            max_length=512,
            tags=_KNOWLEDGE_ADV,
            advanced=True,
            restart_required=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="knowledge_featured_packs",
            kind="str",
            default="",
            label="Featured packs override",
            description=(
                "Comma-separated pack IDs to feature on the packs page. "
                "Empty = use upstream-curated featured list."
            ),
            section="knowledge.packs",
            max_length=1024,
            tags=_KNOWLEDGE_ADV,
            advanced=True,
        )
    )

    # ============== Document RAG ==============
    r.register(
        Setting(
            key="document_rag_cliff_ratio",
            kind="float",
            default=0.3,
            label="Adaptive-K cliff ratio",
            description=(
                "Score drop-off threshold for adaptive K selection. Below this "
                "ratio of the top score, results are cut from the context."
            ),
            section="document.rag",
            min_value=0.0,
            max_value=1.0,
            tags=_DOC_ADV,
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="document_rag_max_context_tokens",
            kind="int",
            default=1500,
            label="RAG context budget",
            description=(
                "Injection token budget cap (~6000 chars). Higher = more "
                "document context, more tokens spent."
            ),
            section="document.rag",
            min_value=100,
            max_value=32000,
            tags=_DOC_ADV,
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="document_rag_query_analysis",
            kind="bool",
            default=True,
            label="LLM query analysis",
            description=(
                "Run LLM query classification before document search. "
                "Improves relevance at the cost of an extra LLM call per turn."
            ),
            section="document.rag",
            tags=_DOC,
        )
    )
    r.register(
        Setting(
            key="document_rag_query_analysis_model",
            kind="str",
            default="",
            label="Query-analysis model",
            description=(
                "Model used for query analysis. Empty = fall back to the "
                "memory extraction model."
            ),
            section="document.rag",
            max_length=256,
            tags=_DOC_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="document_rag_query_analysis_timeout",
            kind="float",
            default=2.0,
            label="Query-analysis timeout (s)",
            description=(
                "Max time the query-analysis pass can take before falling "
                "back to a raw similarity search."
            ),
            section="document.rag",
            min_value=0.1,
            max_value=30.0,
            tags=_DOC_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ============== Observation substrate ==============
    r.register(
        Setting(
            key="observation_substrate_enabled",
            kind="bool",
            default=False,
            label="Observation substrate",
            description=(
                "Enable the cross-modal sequential pattern memory (BOM). "
                "Off by default — the L0/L1/L2 store accumulates idle until "
                "you opt in."
            ),
            section="observation",
            tags=("observation", "advanced"),
            advanced=True,
            restart_required=False,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="observation_seed_chat_history",
            kind="bool",
            default=False,
            label="Seed chat history",
            description=(
                "When enabling the substrate, backfill observations from "
                "existing chat history. Off = start tracking from now only."
            ),
            section="observation",
            tags=_OBS,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="observation_lookup_cache_enabled",
            kind="bool",
            default=False,
            label="Lookup-cache hint",
            description=(
                "Hint llama-server to load a per-user observation-derived "
                "lookup cache at model start (--lookup-cache-static)."
            ),
            section="observation.lookup",
            tags=_OBS,
            advanced=True,
            restart_required=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="observation_lookup_cache_max_entries",
            kind="int",
            default=50_000,
            label="Lookup-cache entries cap",
            description=(
                "Hard cap on the top-K observations the exporter pulls into "
                "the lookup cache."
            ),
            section="observation.lookup",
            min_value=1000,
            max_value=1_000_000,
            tags=_OBS,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="observation_primary_user_id",
            kind="str",
            default="",
            label="Primary user ID",
            description=(
                "Single-tenant primary user whose observations seed the "
                "lookup cache. Empty = first user wins."
            ),
            section="observation",
            max_length=64,
            tags=_OBS,
            advanced=True,
            trust_tier="admin_only",
            companion_surfaceable=False,
        )
    )
