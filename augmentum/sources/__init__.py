"""Direct data sources — keyless structured APIs, bypassing web search.

Weather, rates, alerts and friends come from purpose-built keyless
APIs instead of the LLM → SearXNG → scrape path: typed JSON in, cached,
re-usable, and personalizable per user. Provider research + the full
breadth map live in project memory (``direct-sources-research``).

Substrate contract (``base.fetch_json``): descriptive User-Agent,
TTL response cache, per-provider throttle, never raises.
"""
