"""Preferred sources registry — curated knowledge base of web domains for AI tools.

Provides domain quality scoring, content metadata, and topic-aware routing so
the web tool can prioritize reliable, scrapeable sources and avoid sites that
block automated access.

Design principles:
  - Quality tiers drive auto-fetch URL ordering (best sources tried first)
  - Rich metadata per domain enables smarter decisions (content type, freshness,
    rate limits, extraction hints)
  - Topic mapping steers searches toward authoritative sources
  - The AVOID list prevents wasting time on sites that will 403/paywall/CAPTCHA
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Quality tiers
# ---------------------------------------------------------------------------

EXCELLENT = 2   # always prefer — clean HTML, reliable, no blocking
GOOD = 1        # generally works — occasional rate limits or light JS
UNKNOWN = 0     # not in registry — try if nothing better available
AVOID = -1      # known to block scrapers — try last or skip entirely

# ---------------------------------------------------------------------------
# Source metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """Metadata about a web domain's suitability for AI tool access."""

    quality: int                          # EXCELLENT / GOOD / AVOID
    categories: tuple[str, ...] = ()      # topic categories this source covers
    content_type: str = "article"         # article, reference, data, api, forum, news
    freshness: str = "static"             # realtime, hourly, daily, static
    notes: str = ""                       # human-readable notes for debugging
    rate_limit: str = ""                  # known rate limit info (e.g. "60/min")
    requires_js: bool = False             # True if content needs JS rendering
    has_paywall: bool = False             # True if content is paywalled
    extraction_hint: str = ""             # hint for content extraction (e.g. "use trafilatura")
    lang: str = "en"                      # primary language
    structured_data: bool = False         # True if returns JSON/structured data
    redirect_domain: str = ""             # if this domain redirects to another


# ---------------------------------------------------------------------------
# Source registry — the core knowledge base
# ---------------------------------------------------------------------------

_SOURCES: dict[str, SourceInfo] = {

    # ===== WEATHER & ENVIRONMENT =====
    "weather.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("weather", "forecast", "climate", "alerts"),
        content_type="data",
        freshness="realtime",
        notes="US National Weather Service. Clean HTML, no blocking, authoritative.",
        structured_data=True,
    ),
    "nws.noaa.gov": SourceInfo(
        quality=AVOID,
        categories=("weather", "forecast", "climate"),
        content_type="data",
        freshness="realtime",
        notes="NOAA NWS API. JSON endpoints available. Times out on automated access.",
        structured_data=True,
        requires_js=True,
    ),
    "openweathermap.org": SourceInfo(
        quality=GOOD,
        categories=("weather", "forecast"),
        content_type="data",
        freshness="realtime",
        notes="Requires API key for JSON. Website scrapeable.",
        rate_limit="60/min free tier",
    ),
    "accuweather.com": SourceInfo(
        quality=AVOID,
        categories=("weather",),
        content_type="article",
        freshness="realtime",
        notes="Heavy JS, aggressive anti-bot, frequent 403s.",
        requires_js=True,
    ),
    "weather.com": SourceInfo(
        quality=AVOID,
        categories=("weather",),
        content_type="article",
        freshness="realtime",
        notes="The Weather Channel. JS-heavy, anti-scraping.",
        requires_js=True,
    ),
    "wunderground.com": SourceInfo(
        quality=AVOID,
        categories=("weather",),
        content_type="article",
        freshness="realtime",
        notes="Weather Underground. Heavy JS rendering.",
        requires_js=True,
    ),
    "airnow.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("weather", "environment", "air quality"),
        content_type="data",
        freshness="hourly",
        notes="EPA air quality data. Clean, accessible.",
    ),
    "earthquake.usgs.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("science", "geology", "earthquakes"),
        content_type="data",
        freshness="realtime",
        notes="USGS earthquake data. JSON API available.",
        structured_data=True,
    ),
    "epa.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("environment", "science", "government"),
        content_type="reference",
        freshness="daily",
    ),

    # ===== NEWS & JOURNALISM =====
    "reuters.com": SourceInfo(
        quality=AVOID,
        categories=("news", "world", "business", "politics"),
        content_type="news",
        freshness="realtime",
        notes="Wire service. Now behind auth/bot detection — returns 401/403.",
        has_paywall=True,
    ),
    "apnews.com": SourceInfo(
        quality=EXCELLENT,
        categories=("news", "world", "politics"),
        content_type="news",
        freshness="realtime",
        notes="Associated Press. Clean, factual, scrapeable.",
    ),
    "bbc.com": SourceInfo(
        quality=EXCELLENT,
        categories=("news", "world", "science", "technology"),
        content_type="news",
        freshness="realtime",
        notes="BBC News. Reliable content extraction via trafilatura.",
        extraction_hint="trafilatura",
    ),
    "bbc.co.uk": SourceInfo(
        quality=EXCELLENT,
        categories=("news", "world"),
        content_type="news",
        freshness="realtime",
        redirect_domain="bbc.com",
    ),
    "npr.org": SourceInfo(
        quality=EXCELLENT,
        categories=("news", "politics", "culture", "science"),
        content_type="news",
        freshness="daily",
        notes="NPR. Clean article HTML.",
    ),
    "pbs.org": SourceInfo(
        quality=EXCELLENT,
        categories=("news", "education", "science"),
        content_type="news",
        freshness="daily",
    ),
    "aljazeera.com": SourceInfo(
        quality=GOOD,
        categories=("news", "world", "politics"),
        content_type="news",
        freshness="realtime",
    ),
    "theguardian.com": SourceInfo(
        quality=GOOD,
        categories=("news", "world", "opinion", "culture"),
        content_type="news",
        freshness="realtime",
        notes="Some sections have light paywalls.",
    ),
    "nytimes.com": SourceInfo(
        quality=AVOID,
        categories=("news", "world", "politics", "culture"),
        content_type="news",
        freshness="realtime",
        notes="Hard paywall + aggressive anti-bot.",
        has_paywall=True,
    ),
    "washingtonpost.com": SourceInfo(
        quality=AVOID,
        categories=("news", "politics"),
        content_type="news",
        freshness="realtime",
        notes="Hard paywall + anti-bot.",
        has_paywall=True,
    ),
    "arstechnica.com": SourceInfo(
        quality=GOOD,
        categories=("technology", "science", "news"),
        content_type="news",
        freshness="daily",
    ),
    "techcrunch.com": SourceInfo(
        quality=GOOD,
        categories=("technology", "startups", "news"),
        content_type="news",
        freshness="daily",
    ),
    "wired.com": SourceInfo(
        quality=GOOD,
        categories=("technology", "science", "culture"),
        content_type="news",
        freshness="daily",
    ),
    "theverge.com": SourceInfo(
        quality=GOOD,
        categories=("technology", "news"),
        content_type="news",
        freshness="daily",
    ),
    "bloomberg.com": SourceInfo(
        quality=AVOID,
        categories=("news", "finance", "business"),
        content_type="news",
        freshness="realtime",
        has_paywall=True,
        notes="Hard paywall + anti-bot.",
    ),
    "wsj.com": SourceInfo(
        quality=AVOID,
        categories=("news", "finance", "business"),
        content_type="news",
        freshness="realtime",
        has_paywall=True,
        notes="Wall Street Journal. Hard paywall.",
    ),
    "ft.com": SourceInfo(
        quality=AVOID,
        categories=("news", "finance"),
        content_type="news",
        freshness="realtime",
        has_paywall=True,
        notes="Financial Times. Hard paywall.",
    ),

    "thehill.com": SourceInfo(
        quality=GOOD,
        categories=("news", "politics"),
        content_type="news",
        freshness="realtime",
        notes="US political news. Mostly open access.",
    ),
    "politico.com": SourceInfo(
        quality=AVOID,
        categories=("news", "politics"),
        content_type="news",
        freshness="realtime",
        notes="Political news. Some premium content paywalled. Blocks automated access.",
        has_paywall=True,
        requires_js=True,
    ),
    "axios.com": SourceInfo(
        quality=AVOID,
        categories=("news", "politics", "technology", "business"),
        content_type="news",
        freshness="realtime",
        notes="Short-form news. Clean, concise articles. Blocks automated access.",
        requires_js=True,
    ),
    "cnn.com": SourceInfo(
        quality=GOOD,
        categories=("news", "world", "politics"),
        content_type="news",
        freshness="realtime",
        notes="Some JS but article content extractable.",
    ),
    "cbsnews.com": SourceInfo(
        quality=GOOD,
        categories=("news", "world", "politics"),
        content_type="news",
        freshness="realtime",
    ),
    "nbcnews.com": SourceInfo(
        quality=GOOD,
        categories=("news", "world", "politics"),
        content_type="news",
        freshness="realtime",
    ),
    "abcnews.go.com": SourceInfo(
        quality=GOOD,
        categories=("news", "world", "politics"),
        content_type="news",
        freshness="realtime",
    ),
    "c-span.org": SourceInfo(
        quality=EXCELLENT,
        categories=("news", "politics", "government"),
        content_type="reference",
        freshness="realtime",
        notes="C-SPAN. Government coverage, transcripts, clean HTML.",
    ),
    "propublica.org": SourceInfo(
        quality=EXCELLENT,
        categories=("news", "investigative", "government", "data"),
        content_type="news",
        freshness="daily",
        notes="Investigative journalism. Open access, clean HTML, data tools.",
    ),

    # ===== THINK TANKS & INTERNATIONAL AFFAIRS =====
    "brookings.edu": SourceInfo(
        quality=EXCELLENT,
        categories=("politics", "economics", "policy", "research"),
        content_type="reference",
        freshness="daily",
        notes="Brookings Institution. Open access research and analysis.",
    ),
    "atlanticcouncil.org": SourceInfo(
        quality=GOOD,
        categories=("politics", "foreign affairs", "security"),
        content_type="reference",
        freshness="daily",
        notes="Atlantic Council. Foreign policy analysis.",
    ),
    "stimson.org": SourceInfo(
        quality=GOOD,
        categories=("politics", "foreign affairs", "security"),
        content_type="reference",
        freshness="daily",
    ),
    "cfr.org": SourceInfo(
        quality=GOOD,
        categories=("politics", "foreign affairs", "policy"),
        content_type="reference",
        freshness="daily",
        notes="Council on Foreign Relations. Mostly open articles.",
    ),
    "rand.org": SourceInfo(
        quality=GOOD,
        categories=("research", "policy", "security", "technology"),
        content_type="reference",
        freshness="daily",
        notes="RAND Corporation. Some research paywalled.",
    ),
    "un.org": SourceInfo(
        quality=EXCELLENT,
        categories=("international", "government", "policy", "data"),
        content_type="reference",
        freshness="daily",
        notes="United Nations. Authoritative international data and resolutions.",
    ),
    "foreignaffairs.com": SourceInfo(
        quality=AVOID,
        categories=("politics", "foreign affairs"),
        content_type="news",
        freshness="daily",
        notes="Hard paywall.",
        has_paywall=True,
    ),
    "foreignpolicy.com": SourceInfo(
        quality=AVOID,
        categories=("politics", "foreign affairs"),
        content_type="news",
        freshness="daily",
        notes="Paywall + anti-bot.",
        has_paywall=True,
    ),

    # ===== REFERENCE & ENCYCLOPEDIAS =====
    "wikipedia.org": SourceInfo(
        quality=EXCELLENT,
        categories=("reference", "encyclopedia", "general"),
        content_type="reference",
        freshness="daily",
        notes="Use the wikipedia tool for direct MediaWiki API access instead.",
        structured_data=True,
    ),
    "en.wikipedia.org": SourceInfo(
        quality=EXCELLENT,
        categories=("reference", "encyclopedia", "general"),
        content_type="reference",
        freshness="daily",
        structured_data=True,
    ),
    "wiktionary.org": SourceInfo(
        quality=EXCELLENT,
        categories=("reference", "language", "dictionary"),
        content_type="reference",
        freshness="daily",
    ),
    "britannica.com": SourceInfo(
        quality=EXCELLENT,
        categories=("reference", "encyclopedia"),
        content_type="reference",
        freshness="static",
        notes="Clean article pages. Good for factual overviews.",
    ),
    "wikimedia.org": SourceInfo(
        quality=EXCELLENT,
        categories=("reference", "media"),
        content_type="reference",
        freshness="static",
    ),
    "gutenberg.org": SourceInfo(
        quality=EXCELLENT,
        categories=("reference", "literature", "books"),
        content_type="reference",
        freshness="static",
        notes="Public domain texts. Clean HTML.",
    ),
    "archive.org": SourceInfo(
        quality=EXCELLENT,
        categories=("reference", "archive", "history"),
        content_type="reference",
        freshness="static",
        notes="Internet Archive. Wayback Machine for historical pages.",
    ),
    "merriam-webster.com": SourceInfo(
        quality=GOOD,
        categories=("reference", "dictionary", "language"),
        content_type="reference",
        freshness="static",
    ),
    "dictionary.com": SourceInfo(
        quality=GOOD,
        categories=("reference", "dictionary", "language"),
        content_type="reference",
        freshness="static",
    ),
    "etymonline.com": SourceInfo(
        quality=GOOD,
        categories=("reference", "language", "etymology"),
        content_type="reference",
        freshness="static",
        notes="Etymology dictionary. Clean content.",
    ),

    # ===== PROGRAMMING & TECHNOLOGY =====
    "github.com": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "code", "open source"),
        content_type="reference",
        freshness="realtime",
        notes="README, issues, and code viewable. Raw files accessible.",
    ),
    "docs.python.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "python", "documentation"),
        content_type="reference",
        freshness="static",
        notes="Official Python docs. Clean HTML, well-structured.",
    ),
    "pypi.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "python", "packages"),
        content_type="reference",
        freshness="daily",
    ),
    "developer.mozilla.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "javascript", "html", "css", "web"),
        content_type="reference",
        freshness="static",
        notes="MDN Web Docs. Definitive web technology reference.",
    ),
    "stackoverflow.com": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "qa", "debugging"),
        content_type="forum",
        freshness="daily",
        notes="Q&A. Answers include code samples. Very scrapeable.",
    ),
    "stackexchange.com": SourceInfo(
        quality=EXCELLENT,
        categories=("qa", "reference"),
        content_type="forum",
        freshness="daily",
    ),
    "w3schools.com": SourceInfo(
        quality=GOOD,
        categories=("programming", "web", "tutorial"),
        content_type="reference",
        freshness="static",
        notes="Useful tutorials but historically inaccurate. Prefer MDN.",
    ),
    "docs.rs": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "rust", "documentation"),
        content_type="reference",
        freshness="daily",
    ),
    "rust-lang.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "rust"),
        content_type="reference",
        freshness="static",
    ),
    "doc.rust-lang.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "rust", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "go.dev": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "golang", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "pkg.go.dev": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "golang", "packages"),
        content_type="reference",
        freshness="daily",
    ),
    "learn.microsoft.com": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "microsoft", "documentation", "azure", "dotnet"),
        content_type="reference",
        freshness="daily",
        notes="Microsoft Learn docs. Clean HTML, comprehensive.",
    ),
    "devdocs.io": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "documentation"),
        content_type="reference",
        freshness="daily",
        notes="Aggregated API docs. Fast, clean.",
    ),
    "cppreference.com": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "cpp", "c", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "docs.oracle.com": SourceInfo(
        quality=AVOID,
        categories=("programming", "java", "documentation"),
        content_type="reference",
        freshness="static",
        notes="Returns empty content without JS.",
        requires_js=True,
    ),
    "kotlinlang.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "kotlin", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "typescriptlang.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "typescript", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "nodejs.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "javascript", "nodejs", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "docs.npmjs.com": SourceInfo(
        quality=GOOD,
        categories=("programming", "javascript", "packages"),
        content_type="reference",
        freshness="daily",
    ),
    "crates.io": SourceInfo(
        quality=AVOID,
        categories=("programming", "rust", "packages"),
        content_type="reference",
        freshness="daily",
        notes="Returns empty content without JS.",
        requires_js=True,
    ),
    "news.ycombinator.com": SourceInfo(
        quality=GOOD,
        categories=("technology", "news", "programming"),
        content_type="forum",
        freshness="realtime",
        notes="Hacker News. Minimal HTML, very scrapeable.",
    ),

    # ===== HARDWARE, BENCHMARKS & SPECS =====
    "tomshardware.com": SourceInfo(
        quality=EXCELLENT,
        categories=("technology", "benchmarks", "hardware", "specs"),
        content_type="data",
        freshness="daily",
        notes="Tom's Hardware. Detailed benchmarks, reviews, specs. Clean HTML.",
    ),
    "anandtech.com": SourceInfo(
        quality=GOOD,
        categories=("technology", "benchmarks", "hardware", "specs"),
        content_type="data",
        freshness="static",
        notes="AnandTech. Redirects to Tom's Hardware. Legacy archive.",
        redirect_domain="tomshardware.com",
    ),
    "techpowerup.com": SourceInfo(
        quality=EXCELLENT,
        categories=("technology", "benchmarks", "hardware", "gpu", "specs"),
        content_type="data",
        freshness="daily",
        notes="TechPowerUp. GPU specs database, reviews, benchmarks. Clean HTML.",
        structured_data=True,
    ),
    "notebookcheck.net": SourceInfo(
        quality=AVOID,
        categories=("technology", "benchmarks", "hardware", "specs"),
        content_type="data",
        freshness="daily",
        notes="Notebookcheck. Cloudflare JS challenge blocks scrapers (verify_sources).",
        structured_data=True,
    ),
    "cpu-monkey.com": SourceInfo(
        quality=AVOID,
        categories=("technology", "benchmarks", "hardware", "cpu", "specs"),
        content_type="data",
        freshness="daily",
        notes="CPU Monkey. Cloudflare JS challenge blocks automated access.",
        requires_js=True,
    ),
    "gpu.userbenchmark.com": SourceInfo(
        quality=GOOD,
        categories=("technology", "benchmarks", "gpu", "specs"),
        content_type="data",
        freshness="daily",
        notes="UserBenchmark GPU. Crowd-sourced GPU benchmarks.",
        structured_data=True,
    ),
    "cpu.userbenchmark.com": SourceInfo(
        quality=GOOD,
        categories=("technology", "benchmarks", "cpu", "specs"),
        content_type="data",
        freshness="daily",
        notes="UserBenchmark CPU. Crowd-sourced CPU benchmarks.",
        structured_data=True,
    ),
    "nanoreview.net": SourceInfo(
        quality=AVOID,
        categories=("technology", "benchmarks", "cpu", "gpu", "specs"),
        content_type="data",
        freshness="daily",
        notes="Nanoreview. Cloudflare JS challenge blocks automated access.",
        requires_js=True,
    ),
    "versus.com": SourceInfo(
        quality=GOOD,
        categories=("technology", "specs", "comparison"),
        content_type="data",
        freshness="daily",
        notes="Versus. Side-by-side device comparison. Structured specs.",
        structured_data=True,
    ),
    "pcgamingwiki.com": SourceInfo(
        quality=AVOID,
        categories=("technology", "games", "specs", "compatibility"),
        content_type="reference",
        freshness="daily",
        notes="PCGamingWiki. Cloudflare JS challenge blocks automated access.",
        requires_js=True,
    ),
    "ark.intel.com": SourceInfo(
        quality=EXCELLENT,
        categories=("technology", "cpu", "specs", "hardware"),
        content_type="data",
        freshness="static",
        notes="Intel ARK. Official Intel product specifications.",
        structured_data=True,
    ),
    "amd.com": SourceInfo(
        quality=AVOID,
        categories=("technology", "cpu", "gpu", "specs", "hardware"),
        content_type="data",
        freshness="static",
        notes="AMD official. Product specifications and documentation. Blocks automated access.",
        requires_js=True,
    ),
    "nvidia.com": SourceInfo(
        quality=GOOD,
        categories=("technology", "gpu", "specs", "hardware", "ai"),
        content_type="data",
        freshness="static",
        notes="NVIDIA official. Product specs. Heavy JS but specs pages work.",
        requires_js=True,
    ),
    "passmark.com": SourceInfo(
        quality=GOOD,
        categories=("technology", "benchmarks", "cpu", "gpu", "specs"),
        content_type="data",
        freshness="daily",
        notes="PassMark. CPU/GPU benchmark database. Some pages JS-heavy.",
        structured_data=True,
    ),
    "geekbench.com": SourceInfo(
        quality=GOOD,
        categories=("technology", "benchmarks", "cpu", "gpu"),
        content_type="data",
        freshness="daily",
        notes="Geekbench. Cross-platform benchmark results browser.",
        structured_data=True,
    ),
    "rtings.com": SourceInfo(
        quality=EXCELLENT,
        categories=("technology", "reviews", "monitors", "tv", "audio", "specs"),
        content_type="data",
        freshness="daily",
        notes="RTINGS. Objective reviews with measurement data. Clean HTML.",
        structured_data=True,
    ),
    "gsmarena.com": SourceInfo(
        quality=EXCELLENT,
        categories=("technology", "mobile", "specs", "phones"),
        content_type="data",
        freshness="daily",
        notes="GSMArena. Phone specifications database. Very scrapeable.",
        structured_data=True,
    ),
    "pcpartpicker.com": SourceInfo(
        quality=GOOD,
        categories=("technology", "hardware", "specs", "prices"),
        content_type="data",
        freshness="realtime",
        notes="PCPartPicker. PC component prices and compatibility. Some JS.",
        structured_data=True,
    ),
    "videocardz.com": SourceInfo(
        quality=AVOID,
        categories=("technology", "gpu", "hardware", "news", "specs"),
        content_type="news",
        freshness="daily",
        notes="VideoCardz. Cloudflare JS challenge blocks automated access.",
        requires_js=True,
    ),

    # ===== SPORTS SCORES & STATISTICS =====
    "cbssports.com": SourceInfo(
        quality=GOOD,
        categories=("sports", "scores", "nfl", "nba"),
        content_type="news",
        freshness="realtime",
        notes="CBS Sports. Scores, schedules, standings.",
    ),
    "sofascore.com": SourceInfo(
        quality=GOOD,
        categories=("sports", "scores", "soccer", "football"),
        content_type="data",
        freshness="realtime",
        notes="Sofascore. Live scores, stats. International focus.",
    ),
    "flashscore.com": SourceInfo(
        quality=GOOD,
        categories=("sports", "scores"),
        content_type="data",
        freshness="realtime",
        notes="FlashScore. Multi-sport live scores.",
    ),
    "nhl.com": SourceInfo(
        quality=GOOD,
        categories=("sports", "scores", "hockey"),
        content_type="data",
        freshness="realtime",
        notes="Official NHL. Scores, standings.",
    ),
    "soccerway.com": SourceInfo(
        quality=GOOD,
        categories=("sports", "scores", "soccer"),
        content_type="data",
        freshness="realtime",
        notes="Soccerway. International soccer scores, tables, fixtures.",
    ),

    # ===== FINANCE & MARKETS =====
    "cnbc.com": SourceInfo(
        quality=GOOD,
        categories=("finance", "stocks", "markets", "news", "business"),
        content_type="news",
        freshness="realtime",
        notes="CNBC. Business and market news.",
    ),
    "fool.com": SourceInfo(
        quality=GOOD,
        categories=("finance", "stocks", "investing"),
        content_type="article",
        freshness="daily",
        notes="Motley Fool. Investment analysis and stock picks.",
    ),
    "seekingalpha.com": SourceInfo(
        quality=GOOD,
        categories=("finance", "stocks", "investing"),
        content_type="article",
        freshness="daily",
        notes="Seeking Alpha. Stock analysis, earnings, dividends.",
    ),
    "stockanalysis.com": SourceInfo(
        quality=GOOD,
        categories=("finance", "stocks", "data", "markets"),
        content_type="data",
        freshness="realtime",
        notes="Stock Analysis. Free stock data, financials, screener.",
        structured_data=True,
    ),
    "finviz.com": SourceInfo(
        quality=GOOD,
        categories=("finance", "stocks", "data", "markets"),
        content_type="data",
        freshness="realtime",
        notes="Finviz. Stock screener, charts, maps.",
        structured_data=True,
    ),

    # ===== RECIPES & COOKING =====
    "cookieandkate.com": SourceInfo(
        quality=GOOD,
        categories=("recipe", "cooking", "food"),
        content_type="article",
        freshness="static",
        notes="Cookie and Kate. Vegetarian and whole food recipes.",
    ),

    # ===== HEALTH & MEDICINE =====
    "clevelandclinic.org": SourceInfo(
        quality=GOOD,
        categories=("health", "medical", "treatment"),
        content_type="reference",
        freshness="static",
        notes="Cleveland Clinic. Medical conditions and treatments.",
    ),

    # ===== ENTERTAINMENT =====
    "themoviedb.org": SourceInfo(
        quality=GOOD,
        categories=("movies", "entertainment", "data"),
        content_type="data",
        freshness="daily",
        notes="TMDB. Community-built movie and TV database.",
        structured_data=True,
    ),
    "thetvdb.com": SourceInfo(
        quality=GOOD,
        categories=("tv", "entertainment", "data"),
        content_type="data",
        freshness="daily",
        notes="TheTVDB. Community TV series database.",
        structured_data=True,
    ),
    "last.fm": SourceInfo(
        quality=GOOD,
        categories=("music", "entertainment"),
        content_type="data",
        freshness="realtime",
        notes="Last.fm. Music discovery, scrobbling, artist pages.",
    ),

    # ===== TRAVEL =====
    "lonelyplanet.com": SourceInfo(
        quality=GOOD,
        categories=("travel", "tourism"),
        content_type="article",
        freshness="static",
        notes="Lonely Planet. Travel guides and destination info.",
    ),
    "seat61.com": SourceInfo(
        quality=EXCELLENT,
        categories=("travel", "transportation"),
        content_type="article",
        freshness="static",
        notes="Seat 61. Train travel guides worldwide. Very clean HTML.",
    ),

    # ===== WEATHER =====
    "weatherspark.com": SourceInfo(
        quality=GOOD,
        categories=("weather", "climate", "data", "statistics"),
        content_type="data",
        freshness="static",
        notes="Weather Spark. Climate averages, year-round weather profiles.",
        structured_data=True,
    ),

    # ===== SHOPPING & DEALS =====
    "slickdeals.net": SourceInfo(
        quality=GOOD,
        categories=("shopping", "deals", "prices"),
        content_type="forum",
        freshness="realtime",
        notes="Slickdeals. Community-driven deals and coupons.",
    ),

    # ===== EDUCATION =====
    "howstuffworks.com": SourceInfo(
        quality=GOOD,
        categories=("reference", "education", "science"),
        content_type="article",
        freshness="static",
        notes="HowStuffWorks. Explanatory articles on diverse topics.",
    ),

    # ===== REAL ESTATE =====
    "redfin.com": SourceInfo(
        quality=GOOD,
        categories=("real estate", "housing", "prices"),
        content_type="data",
        freshness="realtime",
        notes="Redfin. Real estate listings and market data.",
    ),

    # ===== JOBS & SALARY =====
    "levels.fyi": SourceInfo(
        quality=GOOD,
        categories=("jobs", "salary", "technology"),
        content_type="data",
        freshness="daily",
        notes="Levels.fyi. Tech salary data and compensation comparisons.",
        structured_data=True,
    ),
    "trulia.com": SourceInfo(
        quality=GOOD,
        categories=("real estate", "housing", "prices"),
        content_type="data",
        freshness="realtime",
        notes="Trulia. Real estate listings and neighborhood data.",
    ),
    "salary.com": SourceInfo(
        quality=GOOD,
        categories=("jobs", "salary", "employment"),
        content_type="data",
        freshness="daily",
        notes="Salary.com. Compensation data and salary surveys.",
        structured_data=True,
    ),
    "payscale.com": SourceInfo(
        quality=GOOD,
        categories=("jobs", "salary", "employment"),
        content_type="data",
        freshness="daily",
        notes="PayScale. Salary comparison and compensation data.",
        structured_data=True,
    ),
    "ziprecruiter.com": SourceInfo(
        quality=GOOD,
        categories=("jobs", "employment"),
        content_type="data",
        freshness="daily",
        notes="ZipRecruiter. Job listings and salary estimates.",
    ),
    "builtin.com": SourceInfo(
        quality=GOOD,
        categories=("jobs", "technology", "startups"),
        content_type="article",
        freshness="daily",
        notes="Built In. Tech jobs, company profiles, salary data.",
    ),
    "wellfound.com": SourceInfo(
        quality=GOOD,
        categories=("jobs", "technology", "startups"),
        content_type="data",
        freshness="daily",
        notes="Wellfound (formerly AngelList). Startup jobs.",
    ),

    # ===== SHOPPING & REVIEWS =====
    "pricerunner.com": SourceInfo(
        quality=GOOD,
        categories=("shopping", "prices", "comparison"),
        content_type="data",
        freshness="realtime",
        notes="PriceRunner. Price comparison across retailers.",
        structured_data=True,
    ),
    "wirecutter.com": SourceInfo(
        quality=GOOD,
        categories=("shopping", "reviews", "recommendations"),
        content_type="article",
        freshness="daily",
        notes="Wirecutter (NYT). Expert product reviews and recommendations.",
    ),
    "consumersearch.com": SourceInfo(
        quality=GOOD,
        categories=("shopping", "reviews", "comparison"),
        content_type="article",
        freshness="static",
        notes="ConsumerSearch. Product review aggregation.",
    ),
    "bestproducts.com": SourceInfo(
        quality=GOOD,
        categories=("shopping", "reviews"),
        content_type="article",
        freshness="daily",
        notes="Best Products. Curated product recommendations.",
    ),
    "buymeonce.com": SourceInfo(
        quality=GOOD,
        categories=("shopping", "reviews", "sustainability"),
        content_type="article",
        freshness="static",
        notes="Buy Me Once. Durable product recommendations.",
    ),

    # ===== AI & MACHINE LEARNING =====
    "deepmind.google": SourceInfo(
        quality=GOOD,
        categories=("ai", "ml", "research"),
        content_type="article",
        freshness="daily",
        notes="Google DeepMind. AI research publications and blog.",
    ),
    "research.google": SourceInfo(
        quality=GOOD,
        categories=("ai", "ml", "research", "technology"),
        content_type="article",
        freshness="daily",
        notes="Google Research. Research publications across domains.",
    ),
    "ollama.com": SourceInfo(
        quality=GOOD,
        categories=("ai", "ml", "models", "technology"),
        content_type="data",
        freshness="daily",
        notes="Ollama. Local LLM model library and documentation.",
    ),
    "lmsys.org": SourceInfo(
        quality=GOOD,
        categories=("ai", "ml", "benchmarks", "models"),
        content_type="data",
        freshness="daily",
        notes="LMSYS. Chatbot Arena, model benchmarks, ELO rankings.",
        structured_data=True,
    ),
    "artificialintelligence-news.com": SourceInfo(
        quality=GOOD,
        categories=("ai", "ml", "news", "technology"),
        content_type="news",
        freshness="daily",
        notes="AI News. Industry news and developments.",
    ),
    "therundown.ai": SourceInfo(
        quality=GOOD,
        categories=("ai", "news", "technology"),
        content_type="news",
        freshness="daily",
        notes="The Rundown AI. AI news newsletter and articles.",
    ),

    # ===== MEDICAL RESEARCH =====
    "rxlist.com": SourceInfo(
        quality=GOOD,
        categories=("medical", "drugs", "pharmacy"),
        content_type="reference",
        freshness="static",
        notes="RxList. Drug information, interactions, side effects.",
    ),
    "merckmanuals.com": SourceInfo(
        quality=EXCELLENT,
        categories=("medical", "reference", "treatment"),
        content_type="reference",
        freshness="static",
        notes="Merck Manuals. Comprehensive medical reference. Clean HTML.",
    ),
    "psychologytoday.com": SourceInfo(
        quality=GOOD,
        categories=("health", "mental health", "psychology"),
        content_type="article",
        freshness="daily",
        notes="Psychology Today. Mental health articles, therapist directory.",
    ),

    # ===== LEGAL =====
    "casetext.com": SourceInfo(
        quality=GOOD,
        categories=("law", "legal", "research"),
        content_type="reference",
        freshness="daily",
        notes="Casetext. Legal research with AI assistant.",
    ),
    "lawinsider.com": SourceInfo(
        quality=GOOD,
        categories=("law", "legal", "contracts"),
        content_type="reference",
        freshness="static",
        notes="Law Insider. Contract clause database and definitions.",
    ),
    "nolo.com": SourceInfo(
        quality=GOOD,
        categories=("law", "legal", "reference"),
        content_type="article",
        freshness="static",
        notes="Nolo. Consumer-friendly legal guides and information.",
    ),
    "scotusblog.com": SourceInfo(
        quality=GOOD,
        categories=("law", "legal", "courts", "politics"),
        content_type="news",
        freshness="daily",
        notes="SCOTUSblog. Supreme Court news and analysis.",
    ),

    # ===== ACADEMIC =====
    "jstor.org": SourceInfo(
        quality=GOOD,
        categories=("research", "papers", "academic"),
        content_type="article",
        freshness="static",
        notes="JSTOR. Academic journal archive. Some content paywalled.",
        has_paywall=True,
    ),
    "peerj.com": SourceInfo(
        quality=GOOD,
        categories=("research", "papers", "science", "open access"),
        content_type="article",
        freshness="daily",
        notes="PeerJ. Open access peer-reviewed journal.",
    ),
    "biorxiv.org": SourceInfo(
        quality=GOOD,
        categories=("research", "papers", "biology", "science"),
        content_type="article",
        freshness="daily",
        notes="bioRxiv. Biology preprint server. Open access.",
    ),
    "medrxiv.org": SourceInfo(
        quality=AVOID,
        categories=("research", "papers", "medical", "science"),
        content_type="article",
        freshness="daily",
        notes="medRxiv preprints. Cloudflare JS challenge blocks scrapers (verify_sources).",
    ),

    # ===== AUTOMOTIVE =====
    "caranddriver.com": SourceInfo(
        quality=GOOD,
        categories=("automotive", "cars", "reviews"),
        content_type="article",
        freshness="daily",
        notes="Car and Driver. Car reviews, comparisons, news.",
    ),
    "motortrend.com": SourceInfo(
        quality=GOOD,
        categories=("automotive", "cars", "reviews"),
        content_type="article",
        freshness="daily",
        notes="MotorTrend. Car reviews, news, first drives.",
    ),
    "kbb.com": SourceInfo(
        quality=GOOD,
        categories=("automotive", "cars", "prices"),
        content_type="data",
        freshness="daily",
        notes="Kelley Blue Book. Car valuations and pricing.",
        structured_data=True,
    ),
    "fueleconomy.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("automotive", "cars", "data", "government"),
        content_type="data",
        freshness="static",
        notes="FuelEconomy.gov. Official US fuel economy data. Clean HTML.",
        structured_data=True,
    ),
    "carcomplaints.com": SourceInfo(
        quality=GOOD,
        categories=("automotive", "cars", "reviews", "safety"),
        content_type="data",
        freshness="daily",
        notes="CarComplaints. Car problems, recalls, TSBs database.",
        structured_data=True,
    ),

    # ===== HOME & DIY =====
    "homedepot.com": SourceInfo(
        quality=GOOD,
        categories=("home", "diy", "shopping", "prices"),
        content_type="data",
        freshness="realtime",
        notes="Home Depot. Home improvement products and how-to guides.",
    ),
    "familyhandyman.com": SourceInfo(
        quality=GOOD,
        categories=("home", "diy", "repair", "reference"),
        content_type="article",
        freshness="static",
        notes="Family Handyman. DIY repair and improvement guides.",
    ),
    "bobvila.com": SourceInfo(
        quality=GOOD,
        categories=("home", "diy", "repair", "reference"),
        content_type="article",
        freshness="static",
        notes="Bob Vila. Home improvement advice and how-tos.",
    ),
    "thisoldhouse.com": SourceInfo(
        quality=GOOD,
        categories=("home", "diy", "repair"),
        content_type="article",
        freshness="static",
        notes="This Old House. Home renovation and repair guides.",
    ),
    "instructables.com": SourceInfo(
        quality=GOOD,
        categories=("diy", "crafts", "projects", "reference"),
        content_type="article",
        freshness="static",
        notes="Instructables. Step-by-step DIY project guides.",
    ),

    # ===== GARDENING & PETS =====
    "almanac.com": SourceInfo(
        quality=GOOD,
        categories=("gardening", "weather", "reference", "agriculture"),
        content_type="reference",
        freshness="daily",
        notes="Old Farmer's Almanac. Gardening, weather, astronomy guides.",
    ),
    "gardeningknowhow.com": SourceInfo(
        quality=GOOD,
        categories=("gardening", "plants", "reference"),
        content_type="article",
        freshness="static",
        notes="Gardening Know How. Plant care and gardening guides.",
    ),
    "petmd.com": SourceInfo(
        quality=GOOD,
        categories=("pets", "veterinary", "health"),
        content_type="article",
        freshness="static",
        notes="PetMD. Pet health conditions, treatments, nutrition.",
    ),
    "akc.org": SourceInfo(
        quality=GOOD,
        categories=("pets", "dogs", "reference"),
        content_type="reference",
        freshness="static",
        notes="AKC. Dog breeds, training, health information.",
    ),
    "aspca.org": SourceInfo(
        quality=GOOD,
        categories=("pets", "veterinary", "safety"),
        content_type="reference",
        freshness="static",
        notes="ASPCA. Pet safety, toxic plants/foods, animal welfare.",
    ),

    # ===== SCIENCE NEWS =====
    "livescience.com": SourceInfo(
        quality=GOOD,
        categories=("science", "news", "reference"),
        content_type="article",
        freshness="daily",
        notes="Live Science. Science news and explainers.",
    ),

    # ===== PERSONAL FINANCE =====
    "creditkarma.com": SourceInfo(
        quality=GOOD,
        categories=("personal finance", "credit"),
        content_type="article",
        freshness="daily",
        notes="Credit Karma. Credit scores, monitoring, financial advice.",
    ),

    # ===== AVOID: BLOCKED/TIMEOUT =====
    "sports.yahoo.com": SourceInfo(
        quality=AVOID,
        categories=("sports", "scores", "news"),
        content_type="news",
        freshness="realtime",
        notes="Yahoo Sports. Times out on automated access.",
        requires_js=True,
    ),
    "camelcamelcamel.com": SourceInfo(
        quality=AVOID,
        categories=("shopping", "prices", "deals"),
        content_type="data",
        freshness="realtime",
        notes="CamelCamelCamel. Cloudflare blocks automated access.",
        requires_js=True,
    ),
    "upi.com": SourceInfo(
        quality=AVOID,
        categories=("news", "world"),
        content_type="news",
        freshness="realtime",
        notes="UPI. Cloudflare blocks automated access.",
        requires_js=True,
    ),
    "apartments.com": SourceInfo(quality=AVOID, categories=("real estate", "housing", "rental"), content_type="data", freshness="realtime", notes="Apartments.com. Access denied on automated access.", requires_js=True),
    "weworkremotely.com": SourceInfo(quality=AVOID, categories=("jobs", "remote", "employment"), content_type="data", freshness="daily", notes="We Work Remotely. Cloudflare blocks automated access.", requires_js=True),
    "remote.co": SourceInfo(quality=AVOID, categories=("jobs", "remote", "employment"), content_type="article", freshness="daily", notes="Remote.co. Times out on automated access.", requires_js=True),
    "openreview.net": SourceInfo(quality=AVOID, categories=("ai", "ml", "research", "papers"), content_type="article", freshness="daily", notes="OpenReview. Returns 403 on automated access.", requires_js=True),
    "ai.meta.com": SourceInfo(quality=AVOID, categories=("ai", "ml", "research"), content_type="article", freshness="daily", notes="Meta AI. Returns empty content without JS.", requires_js=True),
    "uptodate.com": SourceInfo(quality=AVOID, categories=("medical", "treatment", "research"), content_type="reference", freshness="daily", notes="UpToDate. Returns empty content without JS. Paywalled.", requires_js=True, has_paywall=True),
    "avvo.com": SourceInfo(quality=AVOID, categories=("law", "legal", "directory"), content_type="article", freshness="static", notes="Avvo. Cloudflare blocks automated access.", requires_js=True),
    "legalmatch.com": SourceInfo(quality=AVOID, categories=("law", "legal", "directory"), content_type="article", freshness="static", notes="LegalMatch. Cloudflare blocks automated access.", requires_js=True),
    "science.org": SourceInfo(quality=AVOID, categories=("research", "papers", "science"), content_type="article", freshness="daily", notes="Science (AAAS). Cloudflare blocks automated access.", requires_js=True),
    "edmunds.com": SourceInfo(quality=AVOID, categories=("automotive", "cars", "reviews", "prices"), content_type="data", freshness="daily", notes="Edmunds. Access denied on automated access.", requires_js=True),
    "lowes.com": SourceInfo(quality=AVOID, categories=("home", "diy", "shopping", "prices"), content_type="data", freshness="realtime", notes="Lowe's. Access denied on automated access.", requires_js=True),
    "shopzilla.com": SourceInfo(quality=AVOID, categories=("shopping", "prices", "comparison"), content_type="data", freshness="realtime", notes="Shopzilla. Returns empty content.", requires_js=True),
    "realtor.com": SourceInfo(quality=AVOID, categories=("real estate", "housing", "prices"), content_type="data", freshness="realtime", notes="Realtor.com. Request processing errors on automated access.", requires_js=True),
    "autoblog.com": SourceInfo(quality=AVOID, categories=("automotive", "cars", "news", "reviews"), content_type="news", freshness="daily", notes="Autoblog. Requires JS. Returns minimal content.", requires_js=True),
    "researchgate.net": SourceInfo(quality=AVOID, categories=("research", "papers", "academic"), content_type="article", freshness="daily", notes="ResearchGate. Temporarily unavailable responses.", requires_js=True),

    # ===== SCIENCE & RESEARCH =====
    "arxiv.org": SourceInfo(
        quality=EXCELLENT,
        categories=("science", "research", "papers", "physics", "math", "cs"),
        content_type="reference",
        freshness="daily",
        notes="Preprints. Abstract pages clean; full PDF also accessible.",
    ),
    "pubmed.ncbi.nlm.nih.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("science", "medical", "research", "papers"),
        content_type="reference",
        freshness="daily",
        notes="Biomedical literature. Abstracts always free.",
    ),
    "scholar.google.com": SourceInfo(
        quality=AVOID,
        categories=("science", "research", "papers"),
        content_type="reference",
        freshness="daily",
        notes="Google Scholar. Aggressive rate limiting and CAPTCHAs.",
        rate_limit="strict",
    ),
    "nature.com": SourceInfo(
        quality=GOOD,
        categories=("science", "research", "papers"),
        content_type="reference",
        freshness="daily",
        notes="Many articles open access. Some paywalled.",
        has_paywall=True,
    ),
    "sciencedirect.com": SourceInfo(
        quality=GOOD,
        categories=("science", "research", "papers"),
        content_type="reference",
        freshness="daily",
        has_paywall=True,
    ),
    "pnas.org": SourceInfo(
        quality=AVOID,
        categories=("science", "research"),
        content_type="reference",
        freshness="daily",
        notes="Blocks automated access.",
        requires_js=True,
    ),
    "ncbi.nlm.nih.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("science", "medical", "biology", "genomics"),
        content_type="data",
        freshness="daily",
        structured_data=True,
    ),
    "semanticscholar.org": SourceInfo(
        quality=GOOD,
        categories=("science", "research", "papers"),
        content_type="reference",
        freshness="daily",
        notes="AI-powered paper search. Clean HTML.",
    ),

    # ===== HEALTH & MEDICINE =====
    "cdc.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("health", "medical", "disease", "government"),
        content_type="reference",
        freshness="daily",
        notes="CDC. Authoritative health information.",
    ),
    "nih.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("health", "medical", "research"),
        content_type="reference",
        freshness="daily",
    ),
    "who.int": SourceInfo(
        quality=EXCELLENT,
        categories=("health", "medical", "disease", "global"),
        content_type="reference",
        freshness="daily",
        notes="World Health Organization. Authoritative global health data.",
    ),
    "mayoclinic.org": SourceInfo(
        quality=GOOD,
        categories=("health", "medical", "symptoms", "treatment"),
        content_type="reference",
        freshness="static",
        notes="Patient-facing medical info. Clean, well-structured.",
    ),
    "medlineplus.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("health", "medical", "drugs", "conditions"),
        content_type="reference",
        freshness="daily",
        notes="NIH consumer health info. Very clean HTML.",
    ),
    "webmd.com": SourceInfo(
        quality=GOOD,
        categories=("health", "medical", "symptoms"),
        content_type="article",
        freshness="static",
        notes="Ad-heavy but content extractable.",
    ),
    "healthline.com": SourceInfo(
        quality=GOOD,
        categories=("health", "medical", "nutrition", "fitness"),
        content_type="article",
        freshness="static",
    ),
    "drugs.com": SourceInfo(
        quality=AVOID,
        categories=("health", "medical", "drugs", "pharmacy"),
        content_type="reference",
        freshness="daily",
        notes="Drug info, interactions, dosages. Blocks automated access.",
        requires_js=True,
    ),
    "fda.gov": SourceInfo(
        quality=AVOID,
        categories=("health", "medical", "drugs", "food", "government"),
        content_type="reference",
        freshness="daily",
        notes="Returns empty content without JS.",
        requires_js=True,
    ),

    # ===== GOVERNMENT & LAW =====
    "usa.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("government", "services", "reference"),
        content_type="reference",
        freshness="daily",
    ),
    "irs.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("government", "tax", "finance"),
        content_type="reference",
        freshness="daily",
        notes="IRS. Tax forms, guidelines, calculators.",
    ),
    "congress.gov": SourceInfo(
        quality=AVOID,
        categories=("government", "law", "legislation", "politics"),
        content_type="reference",
        freshness="daily",
        notes="Bills, resolutions, congressional records. Blocks automated access.",
        requires_js=True,
    ),
    "law.cornell.edu": SourceInfo(
        quality=EXCELLENT,
        categories=("law", "legal", "reference"),
        content_type="reference",
        freshness="static",
        notes="Cornell LII. US legal code, Supreme Court opinions.",
    ),
    "supremecourt.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("law", "legal", "government"),
        content_type="reference",
        freshness="daily",
    ),
    "whitehouse.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("government", "politics", "policy"),
        content_type="reference",
        freshness="daily",
    ),
    "sec.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("government", "finance", "business", "filings"),
        content_type="data",
        freshness="daily",
        notes="SEC EDGAR filings. Structured financial data.",
        structured_data=True,
    ),
    "data.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("government", "data", "statistics"),
        content_type="data",
        freshness="daily",
        structured_data=True,
    ),
    "census.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("government", "statistics", "demographics"),
        content_type="data",
        freshness="static",
        structured_data=True,
    ),
    "bls.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("government", "economics", "statistics", "employment"),
        content_type="data",
        freshness="daily",
        notes="Bureau of Labor Statistics. CPI, unemployment, wages.",
        structured_data=True,
    ),
    "federalregister.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("government", "law", "regulations"),
        content_type="reference",
        freshness="daily",
    ),
    "nist.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("government", "science", "standards", "technology"),
        content_type="reference",
        freshness="static",
    ),
    "nasa.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("science", "space", "government"),
        content_type="reference",
        freshness="daily",
        notes="NASA. Rich media, clean articles.",
    ),

    # ===== FINANCE & ECONOMICS =====
    "investopedia.com": SourceInfo(
        quality=GOOD,
        categories=("finance", "investing", "economics", "reference"),
        content_type="reference",
        freshness="static",
        notes="Financial education. Clean article content.",
    ),
    "finance.yahoo.com": SourceInfo(
        quality=GOOD,
        categories=("finance", "stocks", "markets"),
        content_type="data",
        freshness="realtime",
        notes="Stock quotes, financials. Some JS but data extractable.",
    ),
    "macrotrends.net": SourceInfo(
        quality=GOOD,
        categories=("finance", "economics", "data", "statistics"),
        content_type="data",
        freshness="daily",
        notes="Historical financial data, charts.",
    ),
    "fred.stlouisfed.org": SourceInfo(
        quality=EXCELLENT,
        categories=("economics", "data", "statistics", "finance"),
        content_type="data",
        freshness="daily",
        notes="Federal Reserve Economic Data. Excellent API.",
        structured_data=True,
    ),
    "worldbank.org": SourceInfo(
        quality=EXCELLENT,
        categories=("economics", "data", "global", "development"),
        content_type="data",
        freshness="daily",
        structured_data=True,
    ),

    # ===== EDUCATION =====
    "khanacademy.org": SourceInfo(
        quality=AVOID,
        categories=("education", "math", "science", "tutorial"),
        content_type="reference",
        freshness="static",
        notes="Educational content. Some content requires JS. Returns empty content without JS.",
        requires_js=True,
    ),
    "coursera.org": SourceInfo(
        quality=GOOD,
        categories=("education", "courses"),
        content_type="reference",
        freshness="static",
        notes="Course descriptions accessible. Content behind login.",
    ),
    "mathworld.wolfram.com": SourceInfo(
        quality=EXCELLENT,
        categories=("math", "reference", "education"),
        content_type="reference",
        freshness="static",
        notes="Wolfram MathWorld. Comprehensive math reference.",
    ),
    "oeis.org": SourceInfo(
        quality=EXCELLENT,
        categories=("math", "reference", "sequences"),
        content_type="data",
        freshness="static",
        notes="Online Encyclopedia of Integer Sequences.",
    ),
    "plato.stanford.edu": SourceInfo(
        quality=EXCELLENT,
        categories=("philosophy", "reference", "education"),
        content_type="reference",
        freshness="static",
        notes="Stanford Encyclopedia of Philosophy. Authoritative, clean.",
    ),

    # ===== FOOD & COOKING =====
    "allrecipes.com": SourceInfo(
        quality=GOOD,
        categories=("food", "recipe", "cooking"),
        content_type="article",
        freshness="static",
    ),
    "seriouseats.com": SourceInfo(
        quality=GOOD,
        categories=("food", "recipe", "cooking"),
        content_type="article",
        freshness="static",
        notes="In-depth recipes with technique. Clean content.",
    ),
    "bonappetit.com": SourceInfo(
        quality=GOOD,
        categories=("food", "recipe", "cooking"),
        content_type="article",
        freshness="static",
    ),
    "foodnetwork.com": SourceInfo(
        quality=AVOID,
        categories=("food", "recipe", "cooking"),
        content_type="article",
        freshness="static",
        notes="Blocks automated access.",
        requires_js=True,
    ),
    "budgetbytes.com": SourceInfo(
        quality=GOOD,
        categories=("food", "recipe", "cooking", "budget"),
        content_type="article",
        freshness="static",
        notes="Budget-friendly recipes. Clean layout.",
    ),

    # ===== ENTERTAINMENT & MEDIA =====
    "imdb.com": SourceInfo(
        quality=AVOID,
        categories=("entertainment", "movies", "tv", "actors"),
        content_type="reference",
        freshness="daily",
        notes="Returns empty content without JS.",
        requires_js=True,
    ),
    "rottentomatoes.com": SourceInfo(
        quality=GOOD,
        categories=("entertainment", "movies", "tv", "reviews"),
        content_type="reference",
        freshness="daily",
    ),
    "metacritic.com": SourceInfo(
        quality=GOOD,
        categories=("entertainment", "games", "movies", "reviews"),
        content_type="reference",
        freshness="daily",
    ),
    "tvtropes.org": SourceInfo(
        quality=AVOID,
        categories=("entertainment", "reference", "writing", "tropes"),
        content_type="reference",
        freshness="static",
        notes="Wiki-style. Very scrapeable, community-edited. Blocks automated access.",
        requires_js=True,
    ),
    "goodreads.com": SourceInfo(
        quality=GOOD,
        categories=("books", "reviews", "literature"),
        content_type="reference",
        freshness="daily",
    ),

    # ===== SPORTS =====
    "espn.com": SourceInfo(
        quality=GOOD,
        categories=("sports", "news", "scores"),
        content_type="news",
        freshness="realtime",
    ),
    "sports-reference.com": SourceInfo(
        quality=EXCELLENT,
        categories=("sports", "statistics", "data", "history"),
        content_type="data",
        freshness="daily",
        notes="Sports stats. Pro-Football-Reference, Basketball-Reference, etc.",
    ),
    "baseball-reference.com": SourceInfo(
        quality=EXCELLENT,
        categories=("sports", "baseball", "statistics"),
        content_type="data",
        freshness="daily",
    ),
    "basketball-reference.com": SourceInfo(
        quality=EXCELLENT,
        categories=("sports", "basketball", "statistics"),
        content_type="data",
        freshness="daily",
    ),
    "pro-football-reference.com": SourceInfo(
        quality=AVOID,
        categories=("sports", "football", "statistics"),
        content_type="data",
        freshness="daily",
        notes="Blocks automated access.",
        requires_js=True,
    ),
    "hockey-reference.com": SourceInfo(
        quality=EXCELLENT,
        categories=("sports", "hockey", "statistics"),
        content_type="data",
        freshness="daily",
    ),
    "transfermarkt.com": SourceInfo(
        quality=GOOD,
        categories=("sports", "soccer", "football", "statistics"),
        content_type="data",
        freshness="daily",
    ),

    # ===== COMMUNITY & FORUMS =====
    "reddit.com": SourceInfo(
        quality=GOOD,
        categories=("forum", "community", "discussion"),
        content_type="forum",
        freshness="realtime",
        notes="Use old.reddit.com for cleaner scraping.",
    ),
    "old.reddit.com": SourceInfo(
        quality=GOOD,
        categories=("forum", "community", "discussion"),
        content_type="forum",
        freshness="realtime",
        notes="Old Reddit. Much cleaner HTML than new Reddit.",
    ),
    "medium.com": SourceInfo(
        quality=AVOID,
        categories=("blog", "technology", "opinion"),
        content_type="article",
        freshness="daily",
        notes="Login wall after ~3 articles. Increasingly aggressive.",
        has_paywall=True,
    ),
    "dev.to": SourceInfo(
        quality=GOOD,
        categories=("programming", "blog", "technology"),
        content_type="article",
        freshness="daily",
        notes="Developer blog platform. Clean HTML.",
    ),
    "hashnode.dev": SourceInfo(
        quality=AVOID,
        categories=("programming", "blog", "technology"),
        content_type="article",
        freshness="daily",
    ),
    "lobste.rs": SourceInfo(
        quality=GOOD,
        categories=("programming", "technology", "forum"),
        content_type="forum",
        freshness="daily",
        notes="Tech-focused link aggregator. Invite-only, clean HTML.",
    ),
    "lemmy.world": SourceInfo(
        quality=GOOD,
        categories=("forum", "community", "discussion"),
        content_type="forum",
        freshness="realtime",
        notes="Federated Reddit alternative. Clean HTML.",
    ),

    # ===== AI & MACHINE LEARNING =====
    "huggingface.co": SourceInfo(
        quality=EXCELLENT,
        categories=("ai", "ml", "programming", "models", "datasets"),
        content_type="reference",
        freshness="daily",
        notes="Model cards, datasets, docs all accessible. Hub API available.",
        structured_data=True,
    ),
    "pytorch.org": SourceInfo(
        quality=EXCELLENT,
        categories=("ai", "ml", "programming", "documentation"),
        content_type="reference",
        freshness="static",
        notes="PyTorch docs and tutorials. Clean HTML.",
    ),
    "tensorflow.org": SourceInfo(
        quality=EXCELLENT,
        categories=("ai", "ml", "programming", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "scikit-learn.org": SourceInfo(
        quality=EXCELLENT,
        categories=("ai", "ml", "python", "documentation"),
        content_type="reference",
        freshness="static",
        notes="Sklearn docs. Well-structured HTML with examples.",
    ),
    "keras.io": SourceInfo(
        quality=EXCELLENT,
        categories=("ai", "ml", "python", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "docs.anthropic.com": SourceInfo(
        quality=EXCELLENT,
        categories=("ai", "api", "documentation"),
        content_type="reference",
        freshness="daily",
        notes="Claude API docs.",
    ),
    "platform.openai.com": SourceInfo(
        quality=AVOID,
        categories=("ai", "api", "documentation"),
        content_type="reference",
        freshness="daily",
        notes="OpenAI API docs. Blocks automated access.",
        requires_js=True,
    ),
    "openai.com": SourceInfo(
        quality=AVOID,
        categories=("ai", "research"),
        content_type="reference",
        freshness="daily",
        notes="Blocks automated access.",
        requires_js=True,
    ),
    "paperswithcode.com": SourceInfo(
        quality=EXCELLENT,
        categories=("ai", "ml", "research", "papers"),
        content_type="reference",
        freshness="daily",
        notes="ML papers with code. Clean, structured, leaderboards.",
        structured_data=True,
    ),
    "kaggle.com": SourceInfo(
        quality=AVOID,
        categories=("ai", "ml", "data", "competition"),
        content_type="reference",
        freshness="daily",
        notes="Datasets, notebooks, competitions. Some content behind login. Returns empty content without JS.",
        requires_js=True,
    ),
    "wandb.ai": SourceInfo(
        quality=GOOD,
        categories=("ai", "ml", "tools"),
        content_type="reference",
        freshness="daily",
    ),
    "mlflow.org": SourceInfo(
        quality=GOOD,
        categories=("ai", "ml", "tools", "documentation"),
        content_type="reference",
        freshness="static",
    ),

    # ===== CLOUD & DEVOPS =====
    "docs.aws.amazon.com": SourceInfo(
        quality=AVOID,
        categories=("cloud", "aws", "documentation", "devops"),
        content_type="reference",
        freshness="daily",
        notes="AWS documentation. Comprehensive, clean HTML. Returns empty content without JS.",
        requires_js=True,
    ),
    "cloud.google.com": SourceInfo(
        quality=EXCELLENT,
        categories=("cloud", "gcp", "documentation", "devops"),
        content_type="reference",
        freshness="daily",
    ),
    "docs.docker.com": SourceInfo(
        quality=EXCELLENT,
        categories=("devops", "docker", "containers", "documentation"),
        content_type="reference",
        freshness="daily",
        notes="Docker official docs. Clean and well-structured.",
    ),
    "kubernetes.io": SourceInfo(
        quality=EXCELLENT,
        categories=("devops", "kubernetes", "containers", "documentation"),
        content_type="reference",
        freshness="daily",
    ),
    "terraform.io": SourceInfo(
        quality=EXCELLENT,
        categories=("devops", "iac", "documentation"),
        content_type="reference",
        freshness="daily",
        notes="Terraform/OpenTofu docs. Provider registry included.",
    ),
    "registry.terraform.io": SourceInfo(
        quality=EXCELLENT,
        categories=("devops", "iac", "documentation"),
        content_type="reference",
        freshness="daily",
    ),
    "docs.ansible.com": SourceInfo(
        quality=EXCELLENT,
        categories=("devops", "automation", "documentation"),
        content_type="reference",
        freshness="daily",
    ),
    "prometheus.io": SourceInfo(
        quality=EXCELLENT,
        categories=("devops", "monitoring", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "grafana.com": SourceInfo(
        quality=GOOD,
        categories=("devops", "monitoring", "documentation"),
        content_type="reference",
        freshness="daily",
    ),
    "nginx.org": SourceInfo(
        quality=EXCELLENT,
        categories=("devops", "web server", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "docs.deno.com": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "javascript", "typescript", "documentation"),
        content_type="reference",
        freshness="daily",
        notes="Deno runtime docs. Clean, modern.",
    ),
    "bun.sh": SourceInfo(
        quality=GOOD,
        categories=("programming", "javascript", "typescript", "documentation"),
        content_type="reference",
        freshness="daily",
        notes="Bun runtime docs.",
    ),
    "v8.dev": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "javascript", "documentation"),
        content_type="reference",
        freshness="static",
        notes="V8 JavaScript engine blog and docs.",
    ),
    "docs.github.com": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "devops", "documentation", "git"),
        content_type="reference",
        freshness="daily",
        notes="GitHub platform docs. Actions, API, etc.",
    ),
    "docs.gitlab.com": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "devops", "documentation", "git"),
        content_type="reference",
        freshness="daily",
    ),

    # ===== DATABASES =====
    "postgresql.org": SourceInfo(
        quality=EXCELLENT,
        categories=("database", "postgresql", "documentation"),
        content_type="reference",
        freshness="static",
        notes="PostgreSQL official docs. Extremely detailed.",
    ),
    "dev.mysql.com": SourceInfo(
        quality=EXCELLENT,
        categories=("database", "mysql", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "sqlite.org": SourceInfo(
        quality=EXCELLENT,
        categories=("database", "sqlite", "documentation"),
        content_type="reference",
        freshness="static",
        notes="SQLite docs. Clean, complete, authoritative.",
    ),
    "redis.io": SourceInfo(
        quality=EXCELLENT,
        categories=("database", "redis", "documentation", "caching"),
        content_type="reference",
        freshness="static",
    ),
    "mongodb.com": SourceInfo(
        quality=GOOD,
        categories=("database", "mongodb", "documentation"),
        content_type="reference",
        freshness="daily",
        notes="MongoDB docs. Some marketing pages mixed in.",
    ),
    "cassandra.apache.org": SourceInfo(
        quality=GOOD,
        categories=("database", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "elastic.co": SourceInfo(
        quality=GOOD,
        categories=("database", "search", "documentation"),
        content_type="reference",
        freshness="daily",
    ),
    "clickhouse.com": SourceInfo(
        quality=GOOD,
        categories=("database", "analytics", "documentation"),
        content_type="reference",
        freshness="daily",
    ),

    # ===== TOPIC_SITES BACKFILL (referenced but were missing from _SOURCES) =====
    "wiki.archlinux.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "linux", "documentation"),
        content_type="reference",
        freshness="daily",
        notes="ArchWiki. Best Linux documentation on the web.",
    ),
    "askubuntu.com": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "linux", "ubuntu", "qa"),
        content_type="forum",
        freshness="daily",
        notes="Ubuntu Q&A. Stack Exchange network.",
    ),
    "man7.org": SourceInfo(
        quality=GOOD,
        categories=("programming", "linux", "documentation"),
        content_type="reference",
        freshness="static",
        notes="Linux man pages online.",
    ),
    "pubchem.ncbi.nlm.nih.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("science", "chemistry", "data"),
        content_type="data",
        freshness="daily",
        notes="PubChem chemical database. Structured data.",
        structured_data=True,
    ),
    "climate.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("science", "climate", "environment", "government"),
        content_type="data",
        freshness="daily",
        notes="NOAA Climate.gov. Climate data and education.",
    ),
    "nhc.noaa.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("weather", "hurricane", "government"),
        content_type="data",
        freshness="realtime",
        notes="National Hurricane Center. Authoritative tropical weather.",
        structured_data=True,
    ),
    "swagger.io": SourceInfo(
        quality=GOOD,
        categories=("programming", "api", "documentation"),
        content_type="reference",
        freshness="static",
        notes="OpenAPI/Swagger specification and tools.",
    ),
    "graphql.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "api", "documentation"),
        content_type="reference",
        freshness="static",
        notes="Official GraphQL specification and docs.",
    ),
    "bitcoin.org": SourceInfo(
        quality=GOOD,
        categories=("finance", "cryptocurrency"),
        content_type="reference",
        freshness="static",
    ),
    "findlaw.com": SourceInfo(
        quality=AVOID,
        categories=("law", "legal", "reference"),
        content_type="reference",
        freshness="static",
        notes="Legal information for consumers. Some ads. Blocks automated access.",
        requires_js=True,
    ),
    "thesaurus.com": SourceInfo(
        quality=GOOD,
        categories=("reference", "language"),
        content_type="reference",
        freshness="static",
    ),
    "deepl.com": SourceInfo(
        quality=GOOD,
        categories=("reference", "language", "translation"),
        content_type="data",
        freshness="realtime",
        notes="Translation service. Web interface scrapeable for short texts.",
    ),

    # ===== ADDITIONAL PROGRAMMING LANGUAGES =====
    "swift.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "swift", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "developer.apple.com": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "swift", "ios", "macos", "documentation"),
        content_type="reference",
        freshness="daily",
        notes="Apple developer docs. Clean, comprehensive.",
    ),
    "php.net": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "php", "documentation"),
        content_type="reference",
        freshness="static",
        notes="PHP manual. User-contributed notes helpful.",
    ),
    "ruby-lang.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "ruby", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "ruby-doc.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "ruby", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "elixir-lang.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "elixir", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "hexdocs.pm": SourceInfo(
        quality=AVOID,
        categories=("programming", "elixir", "erlang", "documentation", "packages"),
        content_type="reference",
        freshness="daily",
        notes="Returns empty content without JS.",
        requires_js=True,
    ),
    "haskell.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "haskell", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "scala-lang.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "scala", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "dart.dev": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "dart", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "flutter.dev": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "dart", "flutter", "mobile", "documentation"),
        content_type="reference",
        freshness="daily",
    ),
    "dotnet.microsoft.com": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "csharp", "dotnet", "documentation"),
        content_type="reference",
        freshness="daily",
    ),
    "lua.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "lua", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "ziglang.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "zig", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "nim-lang.org": SourceInfo(
        quality=GOOD,
        categories=("programming", "nim", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "clojure.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "clojure", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "ocaml.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "ocaml", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "julialang.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "julia", "documentation", "science"),
        content_type="reference",
        freshness="static",
    ),
    "r-project.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "r", "statistics", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "cran.r-project.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "r", "packages"),
        content_type="reference",
        freshness="daily",
    ),

    # ===== WEB FRAMEWORKS =====
    "react.dev": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "javascript", "react", "documentation"),
        content_type="reference",
        freshness="static",
        notes="Official React docs. Clean, interactive examples.",
    ),
    "vuejs.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "javascript", "vue", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "angular.dev": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "typescript", "angular", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "svelte.dev": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "javascript", "svelte", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "nextjs.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "javascript", "react", "nextjs", "documentation"),
        content_type="reference",
        freshness="daily",
    ),
    "nuxt.com": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "javascript", "vue", "documentation"),
        content_type="reference",
        freshness="daily",
    ),
    "astro.build": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "javascript", "documentation"),
        content_type="reference",
        freshness="daily",
    ),
    "docs.djangoproject.com": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "python", "django", "documentation"),
        content_type="reference",
        freshness="static",
        notes="Django docs. Exceptionally well-written.",
    ),
    "flask.palletsprojects.com": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "python", "flask", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "fastapi.tiangolo.com": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "python", "fastapi", "documentation"),
        content_type="reference",
        freshness="static",
        notes="FastAPI docs. Interactive examples, clean.",
    ),
    "spring.io": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "java", "spring", "documentation"),
        content_type="reference",
        freshness="daily",
    ),
    "guides.rubyonrails.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "ruby", "rails", "documentation"),
        content_type="reference",
        freshness="static",
        notes="Rails Guides. Well-structured, comprehensive.",
    ),
    "expressjs.com": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "javascript", "nodejs", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "laravel.com": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "php", "laravel", "documentation"),
        content_type="reference",
        freshness="daily",
    ),
    "gin-gonic.com": SourceInfo(
        quality=GOOD,
        categories=("programming", "golang", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "actix.rs": SourceInfo(
        quality=GOOD,
        categories=("programming", "rust", "documentation"),
        content_type="reference",
        freshness="static",
    ),
    "tailwindcss.com": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "css", "documentation"),
        content_type="reference",
        freshness="daily",
        notes="Tailwind CSS docs. Clean, searchable.",
    ),
    "getbootstrap.com": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "css", "documentation"),
        content_type="reference",
        freshness="static",
    ),

    # ===== PACKAGE REGISTRIES =====
    "npmjs.com": SourceInfo(
        quality=AVOID,
        categories=("programming", "javascript", "packages"),
        content_type="reference",
        freshness="daily",
        notes="npm registry. Package pages with README and metadata. Blocks automated access.",
        requires_js=True,
    ),
    "rubygems.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "ruby", "packages"),
        content_type="reference",
        freshness="daily",
    ),
    "nuget.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "dotnet", "csharp", "packages"),
        content_type="reference",
        freshness="daily",
    ),
    "mvnrepository.com": SourceInfo(
        quality=AVOID,
        categories=("programming", "java", "packages"),
        content_type="reference",
        freshness="daily",
        notes="Maven Central search. Good metadata. Blocks automated access.",
        requires_js=True,
    ),
    "packagist.org": SourceInfo(
        quality=GOOD,
        categories=("programming", "php", "packages"),
        content_type="reference",
        freshness="daily",
    ),

    # ===== SECURITY =====
    "owasp.org": SourceInfo(
        quality=EXCELLENT,
        categories=("security", "programming", "reference"),
        content_type="reference",
        freshness="static",
        notes="OWASP. Security guidelines, top 10, cheat sheets.",
    ),
    "cve.mitre.org": SourceInfo(
        quality=EXCELLENT,
        categories=("security", "vulnerabilities", "reference"),
        content_type="data",
        freshness="daily",
        notes="CVE database. Authoritative vulnerability IDs.",
        structured_data=True,
    ),
    "nvd.nist.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("security", "vulnerabilities", "government"),
        content_type="data",
        freshness="daily",
        notes="National Vulnerability Database. CVSS scores, references.",
        structured_data=True,
    ),
    "cisa.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("security", "government", "advisories"),
        content_type="reference",
        freshness="daily",
    ),
    "exploit-db.com": SourceInfo(
        quality=GOOD,
        categories=("security", "vulnerabilities", "exploits"),
        content_type="data",
        freshness="daily",
    ),
    "portswigger.net": SourceInfo(
        quality=EXCELLENT,
        categories=("security", "web security", "education"),
        content_type="reference",
        freshness="static",
        notes="Web Security Academy. Clean educational content.",
    ),

    # ===== STANDARDS & SPECS =====
    "w3.org": SourceInfo(
        quality=EXCELLENT,
        categories=("standards", "web", "reference"),
        content_type="reference",
        freshness="static",
        notes="W3C standards. HTML/CSS/XML/accessibility specs.",
    ),
    "rfc-editor.org": SourceInfo(
        quality=EXCELLENT,
        categories=("standards", "networking", "reference"),
        content_type="reference",
        freshness="static",
        notes="IETF RFCs. Plain text, very scrapeable.",
    ),
    "ietf.org": SourceInfo(
        quality=EXCELLENT,
        categories=("standards", "networking", "reference"),
        content_type="reference",
        freshness="static",
    ),
    "datatracker.ietf.org": SourceInfo(
        quality=EXCELLENT,
        categories=("standards", "networking", "reference"),
        content_type="reference",
        freshness="static",
    ),
    "ecma-international.org": SourceInfo(
        quality=GOOD,
        categories=("standards", "programming", "reference"),
        content_type="reference",
        freshness="static",
        notes="ECMA standards (includes ECMAScript/JavaScript spec).",
    ),
    "tc39.es": SourceInfo(
        quality=EXCELLENT,
        categories=("standards", "javascript", "programming"),
        content_type="reference",
        freshness="daily",
        notes="TC39 proposals for JavaScript/ECMAScript.",
    ),
    "spec.whatwg.org": SourceInfo(
        quality=EXCELLENT,
        categories=("standards", "web", "reference"),
        content_type="reference",
        freshness="daily",
        notes="WHATWG living standards (HTML, DOM, Fetch, etc.).",
    ),
    "unicode.org": SourceInfo(
        quality=EXCELLENT,
        categories=("standards", "reference", "text"),
        content_type="reference",
        freshness="static",
    ),

    # ===== MAPS & GEOGRAPHY =====
    "openstreetmap.org": SourceInfo(
        quality=EXCELLENT,
        categories=("maps", "geography", "data"),
        content_type="data",
        freshness="daily",
        notes="Open map data. Wiki and API accessible.",
        structured_data=True,
    ),
    "geonames.org": SourceInfo(
        quality=EXCELLENT,
        categories=("geography", "data", "reference"),
        content_type="data",
        freshness="static",
        structured_data=True,
    ),
    "countrystudies.us": SourceInfo(
        quality=GOOD,
        categories=("geography", "reference", "history"),
        content_type="reference",
        freshness="static",
    ),
    "cia.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("government", "geography", "data", "reference"),
        content_type="data",
        freshness="static",
        notes="World Factbook. Comprehensive country data.",
        structured_data=True,
    ),

    # ===== MUSIC & AUDIO =====
    "musicbrainz.org": SourceInfo(
        quality=EXCELLENT,
        categories=("music", "reference", "data"),
        content_type="data",
        freshness="daily",
        notes="Open music encyclopedia. Structured metadata.",
        structured_data=True,
    ),
    "discogs.com": SourceInfo(
        quality=AVOID,
        categories=("music", "reference", "data"),
        content_type="reference",
        freshness="daily",
        notes="Music database. Discographies, release data. Blocks automated access.",
        requires_js=True,
    ),
    "genius.com": SourceInfo(
        quality=GOOD,
        categories=("music", "lyrics", "reference"),
        content_type="reference",
        freshness="daily",
        notes="Lyrics and annotations. Some content accessible.",
    ),
    "setlist.fm": SourceInfo(
        quality=GOOD,
        categories=("music", "concerts", "data"),
        content_type="data",
        freshness="daily",
    ),
    "allmusic.com": SourceInfo(
        quality=GOOD,
        categories=("music", "reviews", "reference"),
        content_type="reference",
        freshness="static",
    ),

    # ===== TRANSPORTATION =====
    "flightradar24.com": SourceInfo(
        quality=AVOID,
        categories=("transportation", "aviation"),
        notes="Heavy JS, WebSocket-based tracking.",
        requires_js=True,
    ),
    "flightaware.com": SourceInfo(
        quality=GOOD,
        categories=("transportation", "aviation", "data"),
        content_type="data",
        freshness="realtime",
        notes="Flight tracking. Some pages accessible.",
    ),
    "dot.gov": SourceInfo(
        quality=AVOID,
        categories=("government", "transportation"),
        content_type="reference",
        freshness="daily",
        notes="Blocks automated access.",
        requires_js=True,
    ),
    "nhtsa.gov": SourceInfo(
        quality=AVOID,
        categories=("government", "transportation", "safety", "data"),
        content_type="data",
        freshness="daily",
        notes="Vehicle safety ratings, recalls. Blocks automated access.",
        structured_data=True,
        requires_js=True,
    ),

    # ===== ADDITIONAL EDUCATION =====
    "ocw.mit.edu": SourceInfo(
        quality=EXCELLENT,
        categories=("education", "courses", "reference"),
        content_type="reference",
        freshness="static",
        notes="MIT OpenCourseWare. Free university-level content.",
    ),
    "edx.org": SourceInfo(
        quality=GOOD,
        categories=("education", "courses"),
        content_type="reference",
        freshness="daily",
        notes="Course listings accessible. Content behind enrollment.",
    ),
    "openstax.org": SourceInfo(
        quality=EXCELLENT,
        categories=("education", "textbooks", "reference"),
        content_type="reference",
        freshness="static",
        notes="Free peer-reviewed textbooks. Clean HTML.",
    ),
    "brilliant.org": SourceInfo(
        quality=GOOD,
        categories=("education", "math", "science"),
        content_type="reference",
        freshness="static",
        notes="Interactive lessons. Some free, most paywalled.",
        has_paywall=True,
    ),

    # ===== ADDITIONAL REFERENCE =====
    "snopes.com": SourceInfo(
        quality=GOOD,
        categories=("reference", "fact checking"),
        content_type="article",
        freshness="daily",
        notes="Fact-checking. Clean article content.",
    ),
    "factcheck.org": SourceInfo(
        quality=EXCELLENT,
        categories=("reference", "fact checking", "politics"),
        content_type="article",
        freshness="daily",
    ),
    "politifact.com": SourceInfo(
        quality=GOOD,
        categories=("reference", "fact checking", "politics"),
        content_type="article",
        freshness="daily",
    ),
    "worldometers.info": SourceInfo(
        quality=GOOD,
        categories=("statistics", "data", "reference"),
        content_type="data",
        freshness="realtime",
        notes="Real-time world statistics. Clean, scrapeable.",
        structured_data=True,
    ),
    "ourworldindata.org": SourceInfo(
        quality=EXCELLENT,
        categories=("statistics", "data", "research", "global"),
        content_type="data",
        freshness="daily",
        notes="Research-backed data visualizations and analysis.",
        structured_data=True,
    ),
    "data.worldbank.org": SourceInfo(
        quality=EXCELLENT,
        categories=("statistics", "data", "economics", "global"),
        content_type="data",
        freshness="daily",
        structured_data=True,
    ),
    "commons.wikimedia.org": SourceInfo(
        quality=EXCELLENT,
        categories=("reference", "media", "images"),
        content_type="reference",
        freshness="static",
    ),
    "simple.wikipedia.org": SourceInfo(
        quality=EXCELLENT,
        categories=("reference", "encyclopedia"),
        content_type="reference",
        freshness="daily",
        notes="Simple English Wikipedia. Easier to parse and summarize.",
    ),

    # ===== TOOLS & UTILITIES =====
    "regex101.com": SourceInfo(
        quality=GOOD,
        categories=("programming", "regex", "tools"),
        content_type="reference",
        freshness="static",
        requires_js=True,
        notes="Regex tester. Library of patterns accessible.",
    ),
    "jsonlint.com": SourceInfo(
        quality=GOOD,
        categories=("programming", "tools"),
        content_type="reference",
        freshness="static",
        requires_js=True,
    ),
    "caniuse.com": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "web", "reference"),
        content_type="data",
        freshness="daily",
        notes="Browser compatibility tables. Data accessible.",
        structured_data=True,
    ),
    "cdnjs.com": SourceInfo(
        quality=GOOD,
        categories=("programming", "web", "packages"),
        content_type="reference",
        freshness="daily",
    ),
    "bundlephobia.com": SourceInfo(
        quality=GOOD,
        categories=("programming", "javascript", "packages"),
        content_type="data",
        freshness="daily",
    ),
    "git-scm.com": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "git", "documentation"),
        content_type="reference",
        freshness="static",
        notes="Official Git documentation. Pro Git book online.",
    ),

    # ===== ADDITIONAL SCIENCE =====
    "space.com": SourceInfo(
        quality=GOOD,
        categories=("science", "space", "astronomy"),
        content_type="news",
        freshness="daily",
    ),
    "phys.org": SourceInfo(
        quality=GOOD,
        categories=("science", "physics", "news"),
        content_type="news",
        freshness="daily",
        notes="Science news aggregator. Clean articles.",
    ),
    "sciencenews.org": SourceInfo(
        quality=GOOD,
        categories=("science", "news"),
        content_type="news",
        freshness="daily",
    ),
    "pmc.ncbi.nlm.nih.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("science", "medical", "research", "papers"),
        content_type="reference",
        freshness="daily",
        notes="PubMed Central. Full-text open-access biomedical papers.",
    ),
    "iucnredlist.org": SourceInfo(
        quality=AVOID,
        categories=("science", "biology", "conservation", "data"),
        content_type="data",
        freshness="static",
        notes="IUCN Red List. Species conservation status. Blocks automated access.",
        structured_data=True,
        requires_js=True,
    ),
    "noaa.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("science", "weather", "ocean", "government"),
        content_type="data",
        freshness="daily",
        structured_data=True,
    ),
    "usda.gov": SourceInfo(
        quality=AVOID,
        categories=("government", "food", "agriculture", "data"),
        content_type="reference",
        freshness="daily",
        notes="Times out on automated access.",
        requires_js=True,
    ),
    "fnal.gov": SourceInfo(
        quality=AVOID,
        categories=("science", "physics", "research"),
        content_type="reference",
        freshness="static",
        notes="Fermilab. Particle physics. Blocks automated access.",
        requires_js=True,
    ),
    "energy.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("government", "science", "energy"),
        content_type="reference",
        freshness="daily",
    ),

    # ===== ADDITIONAL FINANCE =====
    "coinmarketcap.com": SourceInfo(
        quality=GOOD,
        categories=("finance", "cryptocurrency", "data"),
        content_type="data",
        freshness="realtime",
        notes="Crypto prices and market cap. Some JS but data extractable.",
    ),
    "coingecko.com": SourceInfo(
        quality=GOOD,
        categories=("finance", "cryptocurrency", "data"),
        content_type="data",
        freshness="realtime",
    ),
    "marketwatch.com": SourceInfo(
        quality=GOOD,
        categories=("finance", "stocks", "news"),
        content_type="news",
        freshness="realtime",
    ),
    "morningstar.com": SourceInfo(
        quality=AVOID,
        categories=("finance", "investing", "data"),
        content_type="data",
        freshness="daily",
        notes="Fund ratings and analysis. Some content paywalled. Returns empty content without JS.",
        has_paywall=True,
        requires_js=True,
    ),
    "bankrate.com": SourceInfo(
        quality=GOOD,
        categories=("finance", "personal finance", "rates"),
        content_type="reference",
        freshness="daily",
        notes="Interest rates, mortgage calculators.",
    ),
    "nerdwallet.com": SourceInfo(
        quality=GOOD,
        categories=("finance", "personal finance", "credit"),
        content_type="article",
        freshness="daily",
    ),

    # ===== ADDITIONAL HEALTH =====
    "clinicaltrials.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("health", "medical", "research", "data"),
        content_type="data",
        freshness="daily",
        notes="Clinical trial database. Structured, searchable.",
        structured_data=True,
    ),
    "nimh.nih.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("health", "mental health", "government"),
        content_type="reference",
        freshness="daily",
        notes="National Institute of Mental Health.",
    ),
    "samhsa.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("health", "mental health", "substance abuse", "government"),
        content_type="reference",
        freshness="daily",
    ),
    "ada.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("government", "disability", "law"),
        content_type="reference",
        freshness="static",
        notes="ADA information and technical assistance.",
    ),

    # ===== ADDITIONAL GOVERNMENT =====
    "state.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("government", "foreign affairs", "travel"),
        content_type="reference",
        freshness="daily",
        notes="US State Department. Travel advisories, country info.",
    ),
    "travel.state.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("government", "travel", "passports", "visas"),
        content_type="reference",
        freshness="daily",
    ),
    "ssa.gov": SourceInfo(
        quality=AVOID,
        categories=("government", "social security", "benefits"),
        content_type="reference",
        freshness="daily",
        notes="Blocks automated access.",
        requires_js=True,
    ),
    "va.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("government", "veterans", "health", "benefits"),
        content_type="reference",
        freshness="daily",
    ),
    "sba.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("government", "business", "small business"),
        content_type="reference",
        freshness="daily",
    ),
    "gao.gov": SourceInfo(
        quality=AVOID,
        categories=("government", "audit", "reports"),
        content_type="reference",
        freshness="daily",
        notes="Government Accountability Office reports. Blocks automated access.",
        requires_js=True,
    ),
    "cbo.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("government", "economics", "budget", "reports"),
        content_type="reference",
        freshness="daily",
        notes="Congressional Budget Office. Budget projections, cost estimates.",
    ),
    "fcc.gov": SourceInfo(
        quality=AVOID,
        categories=("government", "telecommunications", "regulation"),
        content_type="reference",
        freshness="daily",
        notes="Times out on automated access.",
        requires_js=True,
    ),
    "ftc.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("government", "consumer protection", "regulation"),
        content_type="reference",
        freshness="daily",
    ),
    "uscourts.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("government", "law", "courts"),
        content_type="reference",
        freshness="daily",
    ),
    "govinfo.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("government", "law", "documents"),
        content_type="reference",
        freshness="daily",
        notes="Official US government publications.",
    ),
    "patents.google.com": SourceInfo(
        quality=AVOID,
        categories=("patents", "intellectual property", "reference"),
        content_type="data",
        freshness="daily",
        notes="Google Patents. Full text patent search. Returns empty content without JS.",
        structured_data=True,
        requires_js=True,
    ),
    "uspto.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("government", "patents", "intellectual property"),
        content_type="data",
        freshness="daily",
    ),

    # ===== ADDITIONAL ENTERTAINMENT =====
    "igdb.com": SourceInfo(
        quality=AVOID,
        categories=("entertainment", "games", "reference"),
        content_type="reference",
        freshness="daily",
        notes="Game database backed by Twitch/Amazon. Blocks automated access.",
        requires_js=True,
    ),
    "howlongtobeat.com": SourceInfo(
        quality=GOOD,
        categories=("entertainment", "games", "reference"),
        content_type="data",
        freshness="daily",
    ),
    "letterboxd.com": SourceInfo(
        quality=GOOD,
        categories=("entertainment", "movies", "reviews"),
        content_type="reference",
        freshness="daily",
    ),
    "myanimelist.net": SourceInfo(
        quality=GOOD,
        categories=("entertainment", "anime", "manga", "reference"),
        content_type="reference",
        freshness="daily",
    ),
    "anilist.co": SourceInfo(
        quality=GOOD,
        categories=("entertainment", "anime", "manga", "reference"),
        content_type="reference",
        freshness="daily",
    ),
    "boardgamegeek.com": SourceInfo(
        quality=AVOID,
        categories=("entertainment", "games", "board games", "reference"),
        content_type="reference",
        freshness="daily",
        notes="Board game database. Wiki-style, scrapeable. Returns empty content without JS.",
        requires_js=True,
    ),

    # ===== ADDITIONAL SPORTS =====
    "fbref.com": SourceInfo(
        quality=AVOID,
        categories=("sports", "soccer", "football", "statistics"),
        content_type="data",
        freshness="daily",
        notes="Football/soccer stats from Sports Reference. Blocks automated access.",
        requires_js=True,
    ),
    "racing-reference.info": SourceInfo(
        quality=GOOD,
        categories=("sports", "motorsport", "statistics"),
        content_type="data",
        freshness="daily",
    ),
    "cricinfo.com": SourceInfo(
        quality=AVOID,
        categories=("sports", "cricket", "statistics"),
        content_type="data",
        freshness="daily",
        notes="Blocks automated access.",
        requires_js=True,
    ),
    "nfl.com": SourceInfo(
        quality=GOOD,
        categories=("sports", "football", "nfl"),
        content_type="news",
        freshness="realtime",
    ),
    "nba.com": SourceInfo(
        quality=AVOID,
        categories=("sports", "basketball", "nba"),
        content_type="news",
        freshness="realtime",
        notes="Some content JS-heavy. Times out on automated access.",
        requires_js=True,
    ),
    "mlb.com": SourceInfo(
        quality=GOOD,
        categories=("sports", "baseball", "mlb"),
        content_type="news",
        freshness="realtime",
    ),

    # ===== ADDITIONAL FOOD =====
    "kingarthurbaking.com": SourceInfo(
        quality=GOOD,
        categories=("food", "recipe", "baking"),
        content_type="article",
        freshness="static",
        notes="Baking recipes and technique guides.",
    ),
    "simplyrecipes.com": SourceInfo(
        quality=GOOD,
        categories=("food", "recipe", "cooking"),
        content_type="article",
        freshness="static",
    ),
    "epicurious.com": SourceInfo(
        quality=GOOD,
        categories=("food", "recipe", "cooking"),
        content_type="article",
        freshness="static",
    ),
    "fdc.nal.usda.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("food", "nutrition", "data", "government"),
        content_type="data",
        freshness="static",
        notes="FoodData Central. Nutritional values for foods.",
        structured_data=True,
    ),

    # ===== TRAVEL =====
    "wikitravel.org": SourceInfo(
        quality=GOOD,
        categories=("travel", "reference"),
        content_type="reference",
        freshness="static",
    ),
    "wikivoyage.org": SourceInfo(
        quality=EXCELLENT,
        categories=("travel", "reference"),
        content_type="reference",
        freshness="static",
        notes="Wikimedia travel guide. Clean, editable.",
    ),
    "tripadvisor.com": SourceInfo(
        quality=GOOD,
        categories=("travel", "reviews"),
        content_type="reference",
        freshness="daily",
        notes="Reviews accessible but rate-limited.",
        rate_limit="moderate",
    ),
    "rome2rio.com": SourceInfo(
        quality=AVOID,
        categories=("travel", "transportation"),
        content_type="data",
        freshness="daily",
        notes="Blocks automated access.",
        requires_js=True,
    ),

    # ===== OPEN DATA PORTALS =====
    "data.europa.eu": SourceInfo(
        quality=EXCELLENT,
        categories=("data", "government", "statistics"),
        content_type="data",
        freshness="daily",
        notes="EU Open Data Portal. 1.6M+ datasets. CKAN-based, REST/SPARQL API.",
        structured_data=True,
    ),
    "data.gov.uk": SourceInfo(
        quality=EXCELLENT,
        categories=("data", "government", "statistics"),
        content_type="data",
        freshness="daily",
        notes="UK government open data. CKAN-based, API available.",
        structured_data=True,
    ),
    "datacommons.org": SourceInfo(
        quality=AVOID,
        categories=("data", "statistics", "reference"),
        content_type="data",
        freshness="daily",
        notes="Google Data Commons. Aggregates data from UN, WHO, CDC, Census, World Bank. Returns empty content without JS.",
        structured_data=True,
        requires_js=True,
    ),
    "registry.opendata.aws": SourceInfo(
        quality=EXCELLENT,
        categories=("data", "cloud", "reference"),
        content_type="reference",
        freshness="static",
        notes="AWS Open Data Registry. Satellite imagery, genomic, climate datasets.",
    ),

    # ===== INTERNATIONAL SOURCES =====
    "gov.uk": SourceInfo(
        quality=EXCELLENT,
        categories=("government", "reference"),
        content_type="reference",
        freshness="daily",
        notes="UK Government. Famously clean, accessible HTML (GDS design system).",
    ),
    "legislation.gov.uk": SourceInfo(
        quality=EXCELLENT,
        categories=("law", "legal", "government"),
        content_type="reference",
        freshness="daily",
        notes="UK legislation. Full text of acts and statutory instruments. API with XML/HTML.",
        structured_data=True,
    ),
    "imf.org": SourceInfo(
        quality=AVOID,
        categories=("economics", "data", "global"),
        content_type="reference",
        freshness="daily",
        notes="International Monetary Fund. WEO database, country reports. Times out on automated access.",
        requires_js=True,
    ),
    "oecd.org": SourceInfo(
        quality=AVOID,
        categories=("economics", "data", "global", "policy"),
        content_type="reference",
        freshness="daily",
        notes="OECD. Cloudflare JS challenge blocks scrapers (verify_sources).",
    ),
    "wto.org": SourceInfo(
        quality=GOOD,
        categories=("economics", "trade", "global"),
        content_type="reference",
        freshness="daily",
        notes="World Trade Organization. Trade statistics, disputes, agreements.",
    ),

    # ===== LEGAL RESOURCES =====
    "courtlistener.com": SourceInfo(
        quality=EXCELLENT,
        categories=("law", "legal", "data"),
        content_type="data",
        freshness="daily",
        notes="Free Law Project. 10M+ court opinions, REST API, bulk data. Open source.",
        structured_data=True,
    ),
    "justia.com": SourceInfo(
        quality=AVOID,
        categories=("law", "legal", "reference"),
        content_type="reference",
        freshness="daily",
        notes="Free case law, codes, regulations. US Supreme Court, federal, state. Blocks automated access.",
        requires_js=True,
    ),
    "regulations.gov": SourceInfo(
        quality=GOOD,
        categories=("government", "law", "regulations"),
        content_type="reference",
        freshness="daily",
        notes="Federal rulemaking portal. Public comments, proposed rules. REST API.",
        structured_data=True,
    ),
    "uscode.house.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("government", "law", "reference"),
        content_type="reference",
        freshness="static",
        notes="Official US Code. Full text of federal statutes.",
    ),
    "ecfr.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("government", "law", "regulations"),
        content_type="reference",
        freshness="daily",
        notes="Electronic Code of Federal Regulations. Official, searchable, API.",
        structured_data=True,
    ),
    "eur-lex.europa.eu": SourceInfo(
        quality=AVOID,
        categories=("law", "legal", "government"),
        content_type="reference",
        freshness="daily",
        notes="EU law. Treaties, directives, regulations. SPARQL/API. Multi-language. Returns empty content without JS.",
        structured_data=True,
        requires_js=True,
    ),

    # ===== HOUSING & REAL ESTATE DATA =====
    "hud.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("government", "housing", "data"),
        content_type="reference",
        freshness="daily",
        notes="US Dept of Housing. Fair housing, FHA data, statistics.",
    ),
    "huduser.gov": SourceInfo(
        quality=AVOID,
        categories=("government", "housing", "data"),
        content_type="data",
        freshness="static",
        notes="HUD datasets. Fair market rents, income limits, housing affordability. Returns empty content without JS.",
        structured_data=True,
        requires_js=True,
    ),
    "freddiemac.com": SourceInfo(
        quality=GOOD,
        categories=("finance", "housing", "data"),
        content_type="data",
        freshness="daily",
        notes="Historical mortgage rates (PMMS), housing market outlook.",
    ),

    # ===== ACADEMIC & OPEN ACCESS =====
    "doaj.org": SourceInfo(
        quality=EXCELLENT,
        categories=("science", "research", "data"),
        content_type="data",
        freshness="daily",
        notes="Directory of Open Access Journals. 20K+ journals, REST API.",
        structured_data=True,
    ),
    "core.ac.uk": SourceInfo(
        quality=GOOD,
        categories=("science", "research", "papers"),
        content_type="reference",
        freshness="daily",
        notes="200M+ open access research papers. REST API.",
    ),
    "libretexts.org": SourceInfo(
        quality=EXCELLENT,
        categories=("education", "textbooks", "reference"),
        content_type="reference",
        freshness="static",
        notes="Open educational resources. Chemistry, biology, math textbooks. Wiki-style.",
    ),
    "ssrn.com": SourceInfo(
        quality=AVOID,
        categories=("research", "economics", "law", "papers"),
        content_type="reference",
        freshness="daily",
        notes="Social Science Research Network. Preprints, abstracts free. Blocks automated access.",
        requires_js=True,
    ),
    "theconversation.com": SourceInfo(
        quality=EXCELLENT,
        categories=("news", "science", "education"),
        content_type="article",
        freshness="daily",
        notes="Academic experts writing for public. Creative Commons, no paywall.",
    ),

    # ===== DEVELOPER TOOLS =====
    "cheat.sh": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "reference"),
        content_type="reference",
        freshness="static",
        notes="Programming cheat sheets. Curl-friendly, returns plain text. No JS.",
    ),
    "roadmap.sh": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "education", "reference"),
        content_type="reference",
        freshness="static",
        notes="Developer roadmaps (frontend, backend, DevOps). Clean static HTML.",
    ),
    "12factor.net": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "devops", "reference"),
        content_type="reference",
        freshness="static",
        notes="Twelve-Factor App methodology. Pure static HTML.",
    ),
    "peps.python.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "python", "reference"),
        content_type="reference",
        freshness="static",
        notes="Python Enhancement Proposals. Authoritative for Python decisions.",
    ),

    # ===== NICHE REFERENCE =====
    "loc.gov": SourceInfo(
        quality=EXCELLENT,
        categories=("reference", "archive", "history", "government"),
        content_type="reference",
        freshness="static",
        notes="Library of Congress. Digital collections, catalog. JSON/XML APIs.",
        structured_data=True,
    ),
    "wikidata.org": SourceInfo(
        quality=EXCELLENT,
        categories=("reference", "data", "knowledge graph"),
        content_type="data",
        freshness="daily",
        notes="Wikimedia structured data. 100M+ items. SPARQL endpoint, REST API.",
        structured_data=True,
    ),
    "openlibrary.org": SourceInfo(
        quality=EXCELLENT,
        categories=("reference", "books", "data"),
        content_type="data",
        freshness="daily",
        notes="Open Library (Internet Archive). 20M+ book records. REST API.",
        structured_data=True,
    ),
    "gbif.org": SourceInfo(
        quality=AVOID,
        categories=("science", "biology", "data"),
        content_type="data",
        freshness="daily",
        notes="Global Biodiversity Information Facility. 2B+ species records. REST API. Blocks automated access.",
        structured_data=True,
        requires_js=True,
    ),
    "simbad.u-strasbg.fr": SourceInfo(
        quality=EXCELLENT,
        categories=("science", "astronomy", "data"),
        content_type="data",
        freshness="static",
        notes="SIMBAD astronomical database. Star/object catalog. No blocking.",
        structured_data=True,
    ),
    "chroniclingamerica.loc.gov": SourceInfo(
        quality=AVOID,
        categories=("reference", "history", "news", "archive"),
        content_type="data",
        freshness="static",
        notes="Historic American newspapers (Library of Congress). REST API. Blocks automated access.",
        structured_data=True,
        requires_js=True,
    ),

    # ===== BUSINESS & COMPANY INFO =====
    "opencorporates.com": SourceInfo(
        quality=GOOD,
        categories=("business", "data"),
        content_type="data",
        freshness="daily",
        notes="200M+ company records worldwide. REST API with free tier.",
        structured_data=True,
    ),
    "sec.report": SourceInfo(
        quality=EXCELLENT,
        categories=("finance", "business", "data"),
        content_type="reference",
        freshness="daily",
        notes="SEC filings viewer. Cleaner interface than EDGAR. No auth.",
    ),
    "companieshouse.gov.uk": SourceInfo(
        quality=EXCELLENT,
        categories=("business", "data", "government"),
        content_type="data",
        freshness="daily",
        notes="UK company registry. Official, free, REST API.",
        structured_data=True,
    ),

    # ===== NEWS AGGREGATORS =====
    "techmeme.com": SourceInfo(
        quality=EXCELLENT,
        categories=("technology", "news"),
        content_type="news",
        freshness="realtime",
        notes="Tech news aggregator. Curated headlines, minimal HTML. No JS needed.",
    ),
    "allsides.com": SourceInfo(
        quality=AVOID,
        categories=("news", "reference"),
        content_type="reference",
        freshness="daily",
        notes="News with bias ratings (left/center/right). Media bias chart. Blocks automated access.",
        requires_js=True,
    ),
    "news.google.com": SourceInfo(
        quality=AVOID,
        categories=("news",),
        notes="Google News. Requires JS, aggressive bot detection.",
        requires_js=True,
    ),

    # ===== ACCESSIBILITY =====
    "webaim.org": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "accessibility", "web", "reference"),
        content_type="reference",
        freshness="static",
        notes="WebAIM accessibility reference. Contrast checker, WAVE tool docs, WCAG checklist.",
    ),
    "a11yproject.com": SourceInfo(
        quality=EXCELLENT,
        categories=("programming", "accessibility", "web", "reference"),
        content_type="reference",
        freshness="static",
        notes="Community accessibility resource. Checklist, patterns. Static site.",
    ),

    # ===== ADDITIONAL AVOID =====
    "snapchat.com": SourceInfo(
        quality=AVOID,
        categories=("social",),
        notes="App-only content.",
    ),
    "discord.com": SourceInfo(
        quality=AVOID,
        categories=("social", "chat"),
        notes="Login required. App-based.",
    ),
    "twitch.tv": SourceInfo(
        quality=AVOID,
        categories=("entertainment", "streaming"),
        notes="Video/live content. Heavy JS.",
        requires_js=True,
    ),
    "spotify.com": SourceInfo(
        quality=AVOID,
        categories=("music", "streaming"),
        notes="Login required. Audio content only.",
    ),
    "netflix.com": SourceInfo(
        quality=AVOID,
        categories=("entertainment", "streaming"),
        notes="Login required. Video content only.",
    ),
    "hulu.com": SourceInfo(
        quality=AVOID,
        categories=("entertainment", "streaming"),
        notes="Login required.",
    ),
    "disneyplus.com": SourceInfo(
        quality=AVOID,
        categories=("entertainment", "streaming"),
        notes="Login required.",
    ),
    "youtube.com": SourceInfo(
        quality=AVOID,
        categories=("media", "video"),
        notes="Video content. Use the youtube tool instead.",
        requires_js=True,
    ),
    "maps.google.com": SourceInfo(
        quality=AVOID,
        categories=("maps",),
        notes="Heavy JS. Use openstreetmap.org or geocoding APIs.",
        requires_js=True,
    ),
    "airbnb.com": SourceInfo(
        quality=AVOID,
        categories=("travel", "lodging"),
        notes="Heavy anti-bot, JS required.",
        requires_js=True,
    ),
    "booking.com": SourceInfo(
        quality=AVOID,
        categories=("travel", "lodging"),
        notes="Anti-bot protection.",
    ),
    "doordash.com": SourceInfo(
        quality=AVOID,
        categories=("food", "delivery"),
        notes="Heavy JS, login wall.",
        requires_js=True,
    ),
    "ubereats.com": SourceInfo(
        quality=AVOID,
        categories=("food", "delivery"),
        notes="Login required.",
    ),
    "grubhub.com": SourceInfo(
        quality=AVOID,
        categories=("food", "delivery"),
        notes="Heavy JS, login wall.",
        requires_js=True,
    ),
    "craigslist.org": SourceInfo(
        quality=GOOD,
        categories=("classifieds", "local"),
        content_type="forum",
        freshness="realtime",
        notes="Minimal HTML, very scrapeable. Rate-limited.",
        rate_limit="moderate",
    ),

    # ===== TRAVEL & GEOGRAPHY =====
    "timeanddate.com": SourceInfo(
        quality=EXCELLENT,
        categories=("time", "date", "timezone", "calendar", "astronomy"),
        content_type="data",
        freshness="realtime",
        notes="Time zones, countdowns, sun/moon data. Clean scraping.",
    ),
    "worldtimeserver.com": SourceInfo(
        quality=GOOD,
        categories=("time", "timezone"),
        content_type="data",
        freshness="realtime",
    ),
    "xe.com": SourceInfo(
        quality=GOOD,
        categories=("finance", "currency", "conversion"),
        content_type="data",
        freshness="realtime",
        notes="Currency conversion. Rates page scrapeable.",
    ),
    "numbeo.com": SourceInfo(
        quality=GOOD,
        categories=("travel", "cost of living", "statistics"),
        content_type="data",
        freshness="daily",
    ),

    # ===== MATH & DATA TOOLS =====
    "wolframalpha.com": SourceInfo(
        quality=GOOD,
        categories=("math", "calculation", "data", "reference"),
        content_type="data",
        freshness="realtime",
        notes="Computational knowledge engine. API available.",
        requires_js=True,
    ),
    "desmos.com": SourceInfo(
        quality=AVOID,
        categories=("math", "graphing", "calculator"),
        content_type="data",
        freshness="static",
        notes="Returns empty content without JS.",
        requires_js=True,
    ),
    "statista.com": SourceInfo(
        quality=GOOD,
        categories=("statistics", "data", "business"),
        content_type="data",
        freshness="daily",
        has_paywall=True,
        notes="Some stats visible, full reports paywalled.",
    ),

    # ===== AVOID: anti-bot, login walls, heavy JS =====
    "linkedin.com": SourceInfo(
        quality=AVOID,
        categories=("social", "professional"),
        notes="Requires login. Aggressive anti-scraping.",
    ),
    "facebook.com": SourceInfo(
        quality=AVOID,
        categories=("social",),
        notes="Requires login.",
    ),
    "instagram.com": SourceInfo(
        quality=AVOID,
        categories=("social", "media"),
        notes="Requires login. Heavy JS.",
        requires_js=True,
    ),
    "twitter.com": SourceInfo(
        quality=AVOID,
        categories=("social", "news"),
        notes="Requires login. Heavy JS.",
        requires_js=True,
        redirect_domain="x.com",
    ),
    "x.com": SourceInfo(
        quality=AVOID,
        categories=("social", "news"),
        notes="Formerly Twitter. Requires login.",
        requires_js=True,
    ),
    "tiktok.com": SourceInfo(
        quality=AVOID,
        categories=("social", "media"),
        notes="Requires JS. Video-only content.",
        requires_js=True,
    ),
    "pinterest.com": SourceInfo(
        quality=AVOID,
        categories=("social", "images"),
        notes="Login wall after a few views.",
    ),
    "quora.com": SourceInfo(
        quality=AVOID,
        categories=("qa", "forum"),
        notes="Login wall. Limited content visible.",
    ),
    "glassdoor.com": SourceInfo(
        quality=AVOID,
        categories=("employment", "reviews"),
        notes="Aggressive anti-bot, login wall.",
    ),
    "zillow.com": SourceInfo(
        quality=AVOID,
        categories=("real estate",),
        notes="Heavy anti-bot protection.",
    ),
    "indeed.com": SourceInfo(
        quality=AVOID,
        categories=("employment", "jobs"),
        notes="Anti-bot, CAPTCHAs.",
    ),
    "yelp.com": SourceInfo(
        quality=AVOID,
        categories=("reviews", "local"),
        notes="Anti-bot, limited content without interaction.",
    ),
    "amazon.com": SourceInfo(
        quality=AVOID,
        categories=("shopping", "reviews"),
        notes="Heavy anti-bot. CAPTCHAs.",
    ),
    "ebay.com": SourceInfo(
        quality=AVOID,
        categories=("shopping",),
        notes="Anti-bot protection.",
    ),
    "walmart.com": SourceInfo(
        quality=AVOID,
        categories=("shopping",),
        notes="Anti-bot protection.",
    ),
    "target.com": SourceInfo(
        quality=AVOID,
        categories=("shopping",),
        notes="Anti-bot protection.",
    ),
    "cloudflare.com": SourceInfo(
        quality=AVOID,
        categories=(),
        notes="Cloudflare challenge pages block scrapers.",
    ),

    # ===== TIER 1 EXPANSION — GOV / FINANCE REGULATORS =====
    "federalreserve.gov": SourceInfo(
        quality=EXCELLENT, categories=("finance", "economics", "monetary-policy", "fomc"),
        content_type="reference", freshness="daily",
        notes="US Federal Reserve. Clean HTML, authoritative.", structured_data=True,
    ),
    "treasury.gov": SourceInfo(
        quality=AVOID, categories=("finance", "government", "economics", "debt"),
        content_type="reference", freshness="daily",
        notes="US Treasury. Times out on automated access (verify_sources).",
    ),
    "bea.gov": SourceInfo(
        quality=EXCELLENT, categories=("economics", "gdp", "statistics", "data"),
        content_type="data", freshness="daily",
        notes="Bureau of Economic Analysis. GDP, trade, income stats.", structured_data=True,
    ),
    "eia.gov": SourceInfo(
        quality=EXCELLENT, categories=("energy", "statistics", "data", "oil", "gas"),
        content_type="data", freshness="daily",
        notes="Energy Information Administration. Clean HTML, rich data.", structured_data=True,
    ),
    "usgs.gov": SourceInfo(
        quality=EXCELLENT, categories=("geology", "earthquakes", "maps", "science"),
        content_type="data", freshness="realtime",
        notes="US Geological Survey. Clean HTML, authoritative.",
    ),
    "nps.gov": SourceInfo(
        quality=EXCELLENT, categories=("parks", "travel", "nature", "government"),
        content_type="reference", freshness="static",
        notes="National Park Service. Clean HTML.",
    ),
    "data.census.gov": SourceInfo(
        quality=GOOD, categories=("demographics", "statistics", "data", "population"),
        content_type="data", freshness="daily",
        notes="Census data explorer. JS-heavy but APIs available.", structured_data=True,
    ),
    "edgar.sec.gov": SourceInfo(
        quality=AVOID, categories=("finance", "sec", "filings", "companies"),
        content_type="data", freshness="realtime",
        notes="SEC EDGAR. Requires User-Agent header per SEC policy — fetch errors.",
        structured_data=True,
    ),
    "cftc.gov": SourceInfo(
        quality=GOOD, categories=("finance", "derivatives", "regulation"),
        content_type="reference", freshness="daily",
        notes="CFTC — commodity/derivatives regulator.",
    ),
    "finra.org": SourceInfo(
        quality=GOOD, categories=("finance", "broker", "regulation"),
        content_type="reference", freshness="daily",
        notes="FINRA — broker-dealer regulator.",
    ),
    "defense.gov": SourceInfo(
        quality=AVOID, categories=("defense", "military", "government"),
        content_type="news", freshness="daily",
        notes="US DoD. Bot-blocks automated access.",
    ),
    "justice.gov": SourceInfo(
        quality=AVOID, categories=("legal", "government", "law-enforcement"),
        content_type="reference", freshness="daily",
        notes="US DOJ. SSL/cert issues on automated fetch.",
    ),
    "gsa.gov": SourceInfo(
        quality=GOOD, categories=("government", "procurement", "federal"),
        content_type="reference", freshness="static",
        notes="General Services Administration.",
    ),

    # ===== TIER 1 — CENTRAL BANKS / INTL FINANCE =====
    "ecb.europa.eu": SourceInfo(
        quality=EXCELLENT, categories=("finance", "monetary-policy", "eurozone", "economics"),
        content_type="reference", freshness="daily",
        notes="European Central Bank. Clean HTML.", structured_data=True,
    ),
    "bankofengland.co.uk": SourceInfo(
        quality=AVOID, categories=("finance", "monetary-policy", "uk", "economics"),
        content_type="reference", freshness="daily",
        notes="Bank of England. Cert hostname mismatch on direct fetch.",
        structured_data=True,
    ),
    "bis.org": SourceInfo(
        quality=EXCELLENT, categories=("finance", "banking", "economics", "international"),
        content_type="reference", freshness="daily",
        notes="Bank for International Settlements.",
    ),

    # ===== TIER 1 — INTL SCIENCE & HEALTH =====
    "europa.eu": SourceInfo(
        quality=GOOD, categories=("government", "eu", "policy", "law"),
        content_type="reference", freshness="daily",
        notes="EU main portal.",
    ),
    "esa.int": SourceInfo(
        quality=EXCELLENT, categories=("space", "science", "astronomy"),
        content_type="article", freshness="daily",
        notes="European Space Agency. Clean HTML.",
    ),
    "cern.ch": SourceInfo(
        quality=EXCELLENT, categories=("physics", "science", "research"),
        content_type="article", freshness="daily",
        notes="CERN. Clean HTML, authoritative.",
    ),
    "iaea.org": SourceInfo(
        quality=AVOID, categories=("nuclear", "energy", "science", "international"),
        content_type="reference", freshness="daily",
        notes="International Atomic Energy Agency.",
    ),
    "fao.org": SourceInfo(
        quality=GOOD, categories=("agriculture", "food", "international", "data"),
        content_type="reference", freshness="daily",
        notes="UN Food and Agriculture Organization.",
    ),
    "unesco.org": SourceInfo(
        quality=GOOD, categories=("education", "culture", "international", "heritage"),
        content_type="reference", freshness="daily",
        notes="UNESCO.",
    ),
    "ecdc.europa.eu": SourceInfo(
        quality=EXCELLENT, categories=("health", "disease", "europe", "epidemiology"),
        content_type="reference", freshness="daily",
        notes="European CDC. Clean HTML.",
    ),
    "ema.europa.eu": SourceInfo(
        quality=EXCELLENT, categories=("health", "drugs", "pharmaceutical", "europe"),
        content_type="reference", freshness="daily",
        notes="European Medicines Agency.",
    ),
    "efsa.europa.eu": SourceInfo(
        quality=GOOD, categories=("health", "food-safety", "europe"),
        content_type="reference", freshness="daily",
        notes="European Food Safety Authority.",
    ),
    "echa.europa.eu": SourceInfo(
        quality=GOOD, categories=("chemicals", "safety", "europe", "regulation"),
        content_type="data", freshness="daily",
        notes="European Chemicals Agency.",
    ),
    "cochrane.org": SourceInfo(
        quality=EXCELLENT, categories=("health", "medical-evidence", "systematic-review"),
        content_type="reference", freshness="daily",
        notes="Cochrane — gold-standard medical evidence reviews.",
    ),
    "cochranelibrary.com": SourceInfo(
        quality=EXCELLENT, categories=("health", "medical-evidence", "systematic-review"),
        content_type="reference", freshness="daily",
        notes="Cochrane Library.",
    ),

    # ===== TIER 1 — TOP MEDICAL CENTERS =====
    "hopkinsmedicine.org": SourceInfo(
        quality=EXCELLENT, categories=("health", "medicine", "patient-info"),
        content_type="article", freshness="static",
        notes="Johns Hopkins Medicine. Clean HTML, authoritative.",
    ),
    "stanfordhealthcare.org": SourceInfo(
        quality=GOOD, categories=("health", "medicine", "patient-info"),
        content_type="article", freshness="static",
        notes="Stanford Health Care.",
    ),
    "mskcc.org": SourceInfo(
        quality=GOOD, categories=("health", "cancer", "oncology"),
        content_type="article", freshness="static",
        notes="Memorial Sloan Kettering Cancer Center.",
    ),

    # ===== TIER 2 CLEAN-HTML — FRAMEWORK DOCS =====
    "deno.land": SourceInfo(
        quality=EXCELLENT, categories=("programming", "javascript", "typescript", "runtime"),
        content_type="reference", freshness="static",
        notes="Deno runtime docs. Clean HTML.",
    ),
    "solidjs.com": SourceInfo(
        quality=AVOID, categories=("programming", "javascript", "framework", "frontend"),
        content_type="reference", freshness="static",
        notes="SolidJS homepage is JS-rendered SPA — near-empty raw HTML.",
        requires_js=True,
    ),
    "remix.run": SourceInfo(
        quality=EXCELLENT, categories=("programming", "javascript", "framework", "frontend"),
        content_type="reference", freshness="static",
        notes="Remix docs.",
    ),
    "mui.com": SourceInfo(
        quality=GOOD, categories=("programming", "ui", "react", "components"),
        content_type="reference", freshness="static",
        notes="Material UI docs.",
    ),
    "ant.design": SourceInfo(
        quality=GOOD, categories=("programming", "ui", "react", "components"),
        content_type="reference", freshness="static",
        notes="Ant Design docs.",
    ),

    # ===== TIER 2 — OS / INFRA =====
    "docker.com": SourceInfo(
        quality=GOOD, categories=("programming", "containers", "devops"),
        content_type="reference", freshness="static",
        notes="Docker docs. Clean HTML.",
    ),
    "kernel.org": SourceInfo(
        quality=EXCELLENT, categories=("linux", "kernel", "systems"),
        content_type="reference", freshness="daily",
        notes="Linux kernel. Clean HTML, authoritative.",
    ),
    "debian.org": SourceInfo(
        quality=EXCELLENT, categories=("linux", "debian", "os"),
        content_type="reference", freshness="static",
        notes="Debian Project. Clean HTML.",
    ),
    "ubuntu.com": SourceInfo(
        quality=GOOD, categories=("linux", "ubuntu", "os"),
        content_type="reference", freshness="static",
        notes="Ubuntu docs.",
    ),
    "archlinux.org": SourceInfo(
        quality=EXCELLENT, categories=("linux", "arch", "os"),
        content_type="reference", freshness="static",
        notes="Arch Linux. Clean HTML (Arch Wiki elsewhere).",
    ),
    "helm.sh": SourceInfo(
        quality=EXCELLENT, categories=("kubernetes", "devops", "packaging"),
        content_type="reference", freshness="static",
        notes="Helm docs. Clean HTML.",
    ),
    "hashicorp.com": SourceInfo(
        quality=GOOD, categories=("devops", "infrastructure", "terraform"),
        content_type="reference", freshness="static",
        notes="HashiCorp docs.",
    ),

    # ===== TIER 2 — DATABASES =====
    "mysql.com": SourceInfo(
        quality=GOOD, categories=("database", "sql", "programming"),
        content_type="reference", freshness="static",
        notes="MySQL docs.",
    ),
    "mariadb.org": SourceInfo(
        quality=EXCELLENT, categories=("database", "sql", "programming"),
        content_type="reference", freshness="static",
        notes="MariaDB. Clean HTML.",
    ),
    "duckdb.org": SourceInfo(
        quality=EXCELLENT, categories=("database", "olap", "analytics", "sql"),
        content_type="reference", freshness="static",
        notes="DuckDB docs. Clean HTML.",
    ),

    # ===== TIER 2 — AI LABS / RESEARCH =====
    "anthropic.com": SourceInfo(
        quality=GOOD, categories=("ai", "research", "llm"),
        content_type="article", freshness="daily",
        notes="Anthropic. Clean HTML, occasional CF.",
    ),
    "deepmind.com": SourceInfo(
        quality=GOOD, categories=("ai", "research", "llm"),
        content_type="article", freshness="daily",
        notes="Google DeepMind.",
    ),
    "mistral.ai": SourceInfo(
        quality=GOOD, categories=("ai", "llm", "research"),
        content_type="article", freshness="daily",
        notes="Mistral AI.",
    ),
    "cohere.com": SourceInfo(
        quality=GOOD, categories=("ai", "llm", "enterprise"),
        content_type="article", freshness="daily",
        notes="Cohere.",
    ),
    "distill.pub": SourceInfo(
        quality=EXCELLENT, categories=("ai", "machine-learning", "research", "interactive"),
        content_type="article", freshness="static",
        notes="Distill — high-quality interactive ML research articles.",
    ),
    "connectedpapers.com": SourceInfo(
        quality=GOOD, categories=("academic", "research", "paper-discovery"),
        content_type="reference", freshness="daily",
        notes="Connected Papers — citation graph explorer. JS-heavy.",
        requires_js=True,
    ),

    # ========================================================================
    # PASS 1 — INTERNATIONAL GOV / STAT / CENTRAL BANKS / NEWS (verified live)
    # ========================================================================

    # ----- Central banks -----
    "bankofcanada.ca": SourceInfo(
        quality=EXCELLENT, categories=("finance", "monetary-policy", "canada", "economics"),
        content_type="reference", freshness="daily",
        notes="Bank of Canada. Clean HTML.", structured_data=True,
    ),
    "rba.gov.au": SourceInfo(
        quality=EXCELLENT, categories=("finance", "monetary-policy", "australia", "economics"),
        content_type="reference", freshness="daily",
        notes="Reserve Bank of Australia.", structured_data=True,
    ),
    "rbi.org.in": SourceInfo(
        quality=EXCELLENT, categories=("finance", "monetary-policy", "india", "economics"),
        content_type="reference", freshness="daily",
        notes="Reserve Bank of India.", structured_data=True,
    ),
    "snb.ch": SourceInfo(
        quality=EXCELLENT, categories=("finance", "monetary-policy", "switzerland", "economics"),
        content_type="reference", freshness="daily",
        notes="Swiss National Bank.", structured_data=True,
    ),
    "riksbank.se": SourceInfo(
        quality=EXCELLENT, categories=("finance", "monetary-policy", "sweden", "economics"),
        content_type="reference", freshness="daily",
        notes="Sveriges Riksbank.", structured_data=True,
    ),
    "norges-bank.no": SourceInfo(
        quality=EXCELLENT, categories=("finance", "monetary-policy", "norway", "economics"),
        content_type="reference", freshness="daily",
        notes="Norges Bank.", structured_data=True,
    ),
    "bcb.gov.br": SourceInfo(
        quality=GOOD, categories=("finance", "monetary-policy", "brazil", "economics"),
        content_type="reference", freshness="daily",
        notes="Banco Central do Brasil. Portuguese primary, English subset.",
    ),

    # ----- US Fed regional banks -----
    "newyorkfed.org": SourceInfo(
        quality=EXCELLENT, categories=("finance", "economics", "research", "monetary-policy"),
        content_type="reference", freshness="daily",
        notes="NY Fed. Strong research publications.", structured_data=True,
    ),
    "frbsf.org": SourceInfo(
        quality=EXCELLENT, categories=("finance", "economics", "research"),
        content_type="reference", freshness="daily",
        notes="San Francisco Fed. Econ research & letters.",
    ),
    "chicagofed.org": SourceInfo(
        quality=EXCELLENT, categories=("finance", "economics", "research"),
        content_type="reference", freshness="daily",
        notes="Chicago Fed.",
    ),
    "richmondfed.org": SourceInfo(
        quality=EXCELLENT, categories=("finance", "economics", "research"),
        content_type="reference", freshness="daily",
        notes="Richmond Fed.",
    ),
    "dallasfed.org": SourceInfo(
        quality=EXCELLENT, categories=("finance", "economics", "research"),
        content_type="reference", freshness="daily",
        notes="Dallas Fed.",
    ),
    "atlantafed.org": SourceInfo(
        quality=EXCELLENT, categories=("finance", "economics", "research"),
        content_type="reference", freshness="daily",
        notes="Atlanta Fed. GDPNow forecast.",
    ),

    # ----- International statistics agencies -----
    "ons.gov.uk": SourceInfo(
        quality=EXCELLENT, categories=("statistics", "uk", "data", "demographics"),
        content_type="data", freshness="daily",
        notes="UK Office for National Statistics.", structured_data=True,
    ),
    "statcan.gc.ca": SourceInfo(
        quality=EXCELLENT, categories=("statistics", "canada", "data", "demographics"),
        content_type="data", freshness="daily",
        notes="Statistics Canada.", structured_data=True,
    ),
    "abs.gov.au": SourceInfo(
        quality=EXCELLENT, categories=("statistics", "australia", "data", "demographics"),
        content_type="data", freshness="daily",
        notes="Australian Bureau of Statistics.", structured_data=True,
    ),
    "destatis.de": SourceInfo(
        quality=EXCELLENT, categories=("statistics", "germany", "data", "europe"),
        content_type="data", freshness="daily",
        notes="Germany Federal Statistical Office.", structured_data=True,
    ),
    "insee.fr": SourceInfo(
        quality=EXCELLENT, categories=("statistics", "france", "data", "europe"),
        content_type="data", freshness="daily",
        notes="INSEE — France National Institute of Statistics.", lang="fr",
    ),
    "ine.es": SourceInfo(
        quality=GOOD, categories=("statistics", "spain", "data", "europe"),
        content_type="data", freshness="daily",
        notes="Spain National Statistics Institute.", lang="es",
    ),
    "cbs.nl": SourceInfo(
        quality=EXCELLENT, categories=("statistics", "netherlands", "data", "europe"),
        content_type="data", freshness="daily",
        notes="CBS Netherlands.",
    ),
    "scb.se": SourceInfo(
        quality=EXCELLENT, categories=("statistics", "sweden", "data", "europe"),
        content_type="data", freshness="daily",
        notes="Statistics Sweden.",
    ),
    "ssb.no": SourceInfo(
        quality=EXCELLENT, categories=("statistics", "norway", "data", "europe"),
        content_type="data", freshness="daily",
        notes="Statistics Norway.",
    ),
    "stat.fi": SourceInfo(
        quality=EXCELLENT, categories=("statistics", "finland", "data", "europe"),
        content_type="data", freshness="daily",
        notes="Statistics Finland.",
    ),
    "dst.dk": SourceInfo(
        quality=GOOD, categories=("statistics", "denmark", "data", "europe"),
        content_type="data", freshness="daily",
        notes="Statistics Denmark.",
    ),
    "ibge.gov.br": SourceInfo(
        quality=GOOD, categories=("statistics", "brazil", "data"),
        content_type="data", freshness="daily",
        notes="IBGE Brazil statistics.", lang="pt",
    ),
    "inegi.org.mx": SourceInfo(
        quality=GOOD, categories=("statistics", "mexico", "data"),
        content_type="data", freshness="daily",
        notes="INEGI Mexico statistics.", lang="es",
    ),
    "statssa.gov.za": SourceInfo(
        quality=AVOID, categories=("statistics", "south-africa", "data"),
        content_type="data", freshness="daily",
        notes="Statistics South Africa.",
    ),

    # ----- Open data portals -----
    "data.gouv.fr": SourceInfo(
        quality=EXCELLENT, categories=("open-data", "france", "data", "europe"),
        content_type="data", freshness="daily",
        notes="France open data portal.", structured_data=True,
    ),
    "govdata.de": SourceInfo(
        quality=EXCELLENT, categories=("open-data", "germany", "data", "europe"),
        content_type="data", freshness="daily",
        notes="Germany open data portal.", structured_data=True,
    ),
    "opendata.swiss": SourceInfo(
        quality=EXCELLENT, categories=("open-data", "switzerland", "data"),
        content_type="data", freshness="daily",
        notes="Switzerland open data.",
    ),
    "data.gov.au": SourceInfo(
        quality=EXCELLENT, categories=("open-data", "australia", "data"),
        content_type="data", freshness="daily",
        notes="Australia open data portal.",
    ),
    "open.canada.ca": SourceInfo(
        quality=EXCELLENT, categories=("open-data", "canada", "data"),
        content_type="data", freshness="daily",
        notes="Canada open government.",
    ),
    "data.gov.sg": SourceInfo(
        quality=EXCELLENT, categories=("open-data", "singapore", "data"),
        content_type="data", freshness="daily",
        notes="Singapore open data.",
    ),
    "data.gov.in": SourceInfo(
        quality=AVOID, categories=("open-data", "india", "data"),
        content_type="data", freshness="daily",
        notes="India open data.",
    ),

    # ----- National gov portals (non-US) -----
    "canada.ca": SourceInfo(
        quality=GOOD, categories=("government", "canada", "reference"),
        content_type="reference", freshness="daily",
        notes="Government of Canada main portal.",
    ),
    "india.gov.in": SourceInfo(
        quality=AVOID, categories=("government", "india", "reference"),
        content_type="reference", freshness="daily",
        notes="India national portal.",
    ),
    "gov.ie": SourceInfo(
        quality=EXCELLENT, categories=("government", "ireland", "reference"),
        content_type="reference", freshness="daily",
        notes="Ireland gov portal.",
    ),
    "gov.scot": SourceInfo(
        quality=EXCELLENT, categories=("government", "scotland", "reference"),
        content_type="reference", freshness="daily",
        notes="Scottish Government.",
    ),
    "gov.wales": SourceInfo(
        quality=EXCELLENT, categories=("government", "wales", "reference"),
        content_type="reference", freshness="daily",
        notes="Welsh Government.",
    ),

    # ----- International English-language news -----
    "france24.com": SourceInfo(
        quality=EXCELLENT, categories=("news", "france", "world", "journalism"),
        content_type="news", freshness="realtime",
        notes="France 24 English news.",
    ),
    "channelnewsasia.com": SourceInfo(
        quality=EXCELLENT, categories=("news", "singapore", "asia", "world"),
        content_type="news", freshness="realtime",
        notes="Channel News Asia.",
    ),
    "straitstimes.com": SourceInfo(
        quality=GOOD, categories=("news", "singapore", "asia"),
        content_type="news", freshness="realtime",
        notes="The Straits Times (Singapore).",
    ),
    "japantimes.co.jp": SourceInfo(
        quality=GOOD, categories=("news", "japan", "asia"),
        content_type="news", freshness="realtime",
        notes="The Japan Times — English Japan news.",
    ),
    "timesofindia.indiatimes.com": SourceInfo(
        quality=GOOD, categories=("news", "india", "asia"),
        content_type="news", freshness="realtime",
        notes="Times of India. Ad-heavy but scrapeable.",
    ),
    "thehindu.com": SourceInfo(
        quality=GOOD, categories=("news", "india", "asia", "journalism"),
        content_type="news", freshness="realtime",
        notes="The Hindu — long-form India news.",
    ),
    "irishtimes.com": SourceInfo(
        quality=GOOD, categories=("news", "ireland", "europe"),
        content_type="news", freshness="realtime",
        notes="The Irish Times.",
    ),
    "rte.ie": SourceInfo(
        quality=EXCELLENT, categories=("news", "ireland", "europe", "public-broadcaster"),
        content_type="news", freshness="realtime",
        notes="RTÉ — Ireland public broadcaster.",
    ),
    "abc.net.au": SourceInfo(
        quality=EXCELLENT, categories=("news", "australia", "public-broadcaster"),
        content_type="news", freshness="realtime",
        notes="ABC News Australia.",
    ),
    "rnz.co.nz": SourceInfo(
        quality=EXCELLENT, categories=("news", "new-zealand", "public-broadcaster"),
        content_type="news", freshness="realtime",
        notes="Radio New Zealand.",
    ),
    "ctvnews.ca": SourceInfo(
        quality=GOOD, categories=("news", "canada"),
        content_type="news", freshness="realtime",
        notes="CTV News Canada.",
    ),
    "swissinfo.ch": SourceInfo(
        quality=EXCELLENT, categories=("news", "switzerland", "europe"),
        content_type="news", freshness="realtime",
        notes="Swissinfo — English Switzerland news.",
    ),
    "euronews.com": SourceInfo(
        quality=GOOD, categories=("news", "europe", "world"),
        content_type="news", freshness="realtime",
        notes="Euronews. EU-focused.",
    ),
    "politico.eu": SourceInfo(
        quality=EXCELLENT, categories=("news", "europe", "politics", "policy"),
        content_type="news", freshness="realtime",
        notes="Politico EU — EU policy coverage.",
    ),

    # ========================================================================
    # PASS 2 — SCIENTIFIC DATABASES & OPEN-ACCESS PUBLISHERS (verified live)
    # ========================================================================

    # ----- Bioinformatics & molecular databases -----
    "uniprot.org": SourceInfo(
        quality=EXCELLENT, categories=("biology", "proteins", "research", "database"),
        content_type="data", freshness="daily",
        notes="UniProt — universal protein sequence DB.", structured_data=True,
    ),
    "rcsb.org": SourceInfo(
        quality=EXCELLENT, categories=("biology", "proteins", "structure", "pdb", "database"),
        content_type="data", freshness="daily",
        notes="RCSB PDB — 3D protein structures.", structured_data=True,
    ),
    "reactome.org": SourceInfo(
        quality=EXCELLENT, categories=("biology", "pathways", "research", "database"),
        content_type="data", freshness="daily",
        notes="Reactome — curated biological pathways.", structured_data=True,
    ),
    "ebi.ac.uk": SourceInfo(
        quality=EXCELLENT, categories=("biology", "bioinformatics", "research", "database"),
        content_type="data", freshness="daily",
        notes="EMBL-EBI. Hub for dozens of bio DBs.",
    ),
    "genome.ucsc.edu": SourceInfo(
        quality=EXCELLENT, categories=("biology", "genomics", "research", "database"),
        content_type="data", freshness="daily",
        notes="UCSC Genome Browser.",
    ),
    "omim.org": SourceInfo(
        quality=EXCELLENT, categories=("biology", "genetics", "medical", "database"),
        content_type="reference", freshness="daily",
        notes="OMIM — Mendelian inheritance in man.",
    ),
    "string-db.org": SourceInfo(
        quality=EXCELLENT, categories=("biology", "proteins", "interactions", "database"),
        content_type="data", freshness="daily",
        notes="STRING — protein-protein interaction network.",
    ),
    "flybase.org": SourceInfo(
        quality=EXCELLENT, categories=("biology", "genetics", "drosophila", "database"),
        content_type="data", freshness="daily",
        notes="FlyBase — Drosophila genetics.",
    ),
    "zfin.org": SourceInfo(
        quality=EXCELLENT, categories=("biology", "genetics", "zebrafish", "database"),
        content_type="data", freshness="daily",
        notes="ZFIN — zebrafish model organism DB.",
    ),
    "pangaea.de": SourceInfo(
        quality=EXCELLENT, categories=("earth-science", "data", "climate", "oceanography"),
        content_type="data", freshness="daily",
        notes="PANGAEA — earth & environmental science data.",
    ),
    "itis.gov": SourceInfo(
        quality=EXCELLENT, categories=("biology", "taxonomy", "species", "database"),
        content_type="reference", freshness="static",
        notes="ITIS — Integrated Taxonomic Information System.",
    ),
    "catalogueoflife.org": SourceInfo(
        quality=EXCELLENT, categories=("biology", "taxonomy", "species", "database"),
        content_type="reference", freshness="daily",
        notes="Catalogue of Life.",
    ),
    "eol.org": SourceInfo(
        quality=EXCELLENT, categories=("biology", "species", "reference"),
        content_type="reference", freshness="daily",
        notes="Encyclopedia of Life.",
    ),
    "chemspider.com": SourceInfo(
        quality=EXCELLENT, categories=("chemistry", "compounds", "database"),
        content_type="data", freshness="daily",
        notes="ChemSpider — free chemical structure DB (RSC).",
    ),
    "nomad-lab.eu": SourceInfo(
        quality=EXCELLENT, categories=("materials", "science", "data", "research"),
        content_type="data", freshness="daily",
        notes="NOMAD — materials science repository.",
    ),

    # ----- Open-access publishers & journals -----
    "plos.org": SourceInfo(
        quality=EXCELLENT, categories=("academic", "open-access", "publisher"),
        content_type="reference", freshness="daily",
        notes="PLOS — open-access publisher.",
    ),
    "journals.plos.org": SourceInfo(
        quality=EXCELLENT, categories=("academic", "open-access", "journals", "research"),
        content_type="article", freshness="daily",
        notes="PLOS journals — full-text open access.",
    ),
    "frontiersin.org": SourceInfo(
        quality=EXCELLENT, categories=("academic", "open-access", "journals", "research"),
        content_type="article", freshness="daily",
        notes="Frontiers — open-access publisher.",
    ),
    "elifesciences.org": SourceInfo(
        quality=EXCELLENT, categories=("academic", "open-access", "biology", "research"),
        content_type="article", freshness="daily",
        notes="eLife — open-access biology journal.",
    ),
    "biomedcentral.com": SourceInfo(
        quality=EXCELLENT, categories=("academic", "open-access", "medical", "biology"),
        content_type="article", freshness="daily",
        notes="BioMed Central — open-access publisher.",
    ),
    "mdpi.com": SourceInfo(
        quality=AVOID, categories=("academic", "open-access", "publisher"),
        content_type="article", freshness="daily",
        notes="MDPI — open-access publisher. Variable peer-review quality.",
    ),

    # ----- Major commercial publishers (non-Cloudflared landing) -----
    "acm.org": SourceInfo(
        quality=GOOD, categories=("academic", "computer-science", "publisher"),
        content_type="reference", freshness="daily",
        notes="ACM — main site (dl.acm.org article pages Cloudflare-blocked).",
    ),
    "ieeexplore.ieee.org": SourceInfo(
        quality=GOOD, categories=("academic", "engineering", "publisher", "computer-science"),
        content_type="reference", freshness="daily",
        notes="IEEE Xplore — abstracts accessible.",
    ),
    "springer.com": SourceInfo(
        quality=EXCELLENT, categories=("academic", "publisher"),
        content_type="reference", freshness="daily",
        notes="Springer main site.",
    ),
    "link.springer.com": SourceInfo(
        quality=EXCELLENT, categories=("academic", "publisher", "journals"),
        content_type="article", freshness="daily",
        notes="SpringerLink — article metadata & OA content.",
    ),
    "wiley.com": SourceInfo(
        quality=GOOD, categories=("academic", "publisher"),
        content_type="reference", freshness="daily",
        notes="Wiley main site (onlinelibrary.wiley.com article pages CF-blocked).",
    ),
    "sage.com": SourceInfo(
        quality=GOOD, categories=("academic", "publisher"),
        content_type="reference", freshness="daily",
        notes="SAGE main site.",
    ),
    "cambridge.org": SourceInfo(
        quality=AVOID, categories=("academic", "publisher"),
        content_type="reference", freshness="daily",
        notes="Cambridge University Press.",
    ),
    "muse.jhu.edu": SourceInfo(
        quality=EXCELLENT, categories=("academic", "humanities", "journals", "publisher"),
        content_type="article", freshness="daily",
        notes="Project MUSE — humanities & social sciences.",
    ),

    # ----- Repositories, libraries & abstracting services -----
    "zenodo.org": SourceInfo(
        quality=EXCELLENT, categories=("research", "data", "repository", "open-data"),
        content_type="data", freshness="daily",
        notes="Zenodo — CERN general research repository.",
    ),
    "dataverse.org": SourceInfo(
        quality=AVOID, categories=("research", "data", "repository", "open-data"),
        content_type="data", freshness="daily",
        notes="Harvard Dataverse project.",
    ),
    "worldcat.org": SourceInfo(
        quality=EXCELLENT, categories=("reference", "books", "library", "catalog"),
        content_type="reference", freshness="daily",
        notes="WorldCat — global library catalog.",
    ),
    "adsabs.harvard.edu": SourceInfo(
        quality=EXCELLENT, categories=("academic", "astronomy", "physics", "abstracts"),
        content_type="reference", freshness="daily",
        notes="NASA ADS — astrophysics abstracts.",
    ),

    # ========================================================================
    # PASS 3 — HEALTH / MEDICAL SOCIETIES & PATIENT INFO (verified live)
    # ========================================================================

    # ----- Disease-specific foundations & specialty societies -----
    "heart.org": SourceInfo(
        quality=EXCELLENT, categories=("health", "cardiology", "patient-info"),
        content_type="reference", freshness="daily",
        notes="American Heart Association.",
    ),
    "diabetes.org": SourceInfo(
        quality=EXCELLENT, categories=("health", "diabetes", "patient-info"),
        content_type="reference", freshness="daily",
        notes="American Diabetes Association.",
    ),
    "cancer.org": SourceInfo(
        quality=EXCELLENT, categories=("health", "cancer", "oncology", "patient-info"),
        content_type="reference", freshness="daily",
        notes="American Cancer Society.",
    ),
    "aap.org": SourceInfo(
        quality=EXCELLENT, categories=("health", "pediatrics", "children", "medical"),
        content_type="reference", freshness="daily",
        notes="American Academy of Pediatrics.",
    ),
    "psychiatry.org": SourceInfo(
        quality=EXCELLENT, categories=("health", "mental-health", "psychiatry"),
        content_type="reference", freshness="daily",
        notes="American Psychiatric Association.",
    ),
    "endocrine.org": SourceInfo(
        quality=EXCELLENT, categories=("health", "endocrinology", "hormones"),
        content_type="reference", freshness="daily",
        notes="Endocrine Society.",
    ),
    "lung.org": SourceInfo(
        quality=EXCELLENT, categories=("health", "pulmonology", "respiratory"),
        content_type="reference", freshness="daily",
        notes="American Lung Association.",
    ),
    "kidney.org": SourceInfo(
        quality=EXCELLENT, categories=("health", "nephrology", "kidney"),
        content_type="reference", freshness="daily",
        notes="National Kidney Foundation.",
    ),
    "liverfoundation.org": SourceInfo(
        quality=EXCELLENT, categories=("health", "hepatology", "liver"),
        content_type="reference", freshness="daily",
        notes="American Liver Foundation.",
    ),
    "alz.org": SourceInfo(
        quality=EXCELLENT, categories=("health", "alzheimers", "dementia", "aging"),
        content_type="reference", freshness="daily",
        notes="Alzheimer's Association.",
    ),
    "parkinson.org": SourceInfo(
        quality=EXCELLENT, categories=("health", "parkinsons", "neurology"),
        content_type="reference", freshness="daily",
        notes="Parkinson's Foundation.",
    ),
    "arthritis.org": SourceInfo(
        quality=EXCELLENT, categories=("health", "arthritis", "rheumatology"),
        content_type="reference", freshness="daily",
        notes="Arthritis Foundation.",
    ),
    "epilepsy.com": SourceInfo(
        quality=EXCELLENT, categories=("health", "epilepsy", "neurology"),
        content_type="reference", freshness="daily",
        notes="Epilepsy Foundation.",
    ),
    "lupus.org": SourceInfo(
        quality=EXCELLENT, categories=("health", "lupus", "autoimmune"),
        content_type="reference", freshness="daily",
        notes="Lupus Foundation of America.",
    ),
    "aafp.org": SourceInfo(
        quality=EXCELLENT, categories=("health", "primary-care", "family-medicine"),
        content_type="reference", freshness="daily",
        notes="American Academy of Family Physicians.",
    ),
    "acog.org": SourceInfo(
        quality=EXCELLENT, categories=("health", "obstetrics", "gynecology", "womens-health"),
        content_type="reference", freshness="daily",
        notes="American College of Obstetricians and Gynecologists.",
    ),
    "aad.org": SourceInfo(
        quality=EXCELLENT, categories=("health", "dermatology", "skin"),
        content_type="reference", freshness="daily",
        notes="American Academy of Dermatology.",
    ),
    "rheumatology.org": SourceInfo(
        quality=EXCELLENT, categories=("health", "rheumatology", "autoimmune"),
        content_type="reference", freshness="daily",
        notes="American College of Rheumatology.",
    ),
    "gastro.org": SourceInfo(
        quality=EXCELLENT, categories=("health", "gastroenterology"),
        content_type="reference", freshness="daily",
        notes="American Gastroenterological Association.",
    ),
    "acc.org": SourceInfo(
        quality=EXCELLENT, categories=("health", "cardiology"),
        content_type="reference", freshness="daily",
        notes="American College of Cardiology.",
    ),
    "chestnet.org": SourceInfo(
        quality=EXCELLENT, categories=("health", "pulmonology", "critical-care"),
        content_type="reference", freshness="daily",
        notes="American College of Chest Physicians.",
    ),
    "aaaai.org": SourceInfo(
        quality=EXCELLENT, categories=("health", "allergy", "immunology"),
        content_type="reference", freshness="daily",
        notes="American Academy of Allergy, Asthma & Immunology.",
    ),
    "asco.org": SourceInfo(
        quality=EXCELLENT, categories=("health", "oncology", "cancer", "medical"),
        content_type="reference", freshness="daily",
        notes="American Society of Clinical Oncology.",
    ),

    # ----- National health info portals (non-US) -----
    "nice.org.uk": SourceInfo(
        quality=EXCELLENT, categories=("health", "uk", "clinical-guidelines", "evidence"),
        content_type="reference", freshness="daily",
        notes="NICE — UK clinical guidelines, gold-standard.",
    ),
    "nhs.uk": SourceInfo(
        quality=EXCELLENT, categories=("health", "uk", "patient-info"),
        content_type="reference", freshness="daily",
        notes="UK NHS patient info.",
    ),
    "healthdirect.gov.au": SourceInfo(
        quality=EXCELLENT, categories=("health", "australia", "patient-info"),
        content_type="reference", freshness="daily",
        notes="Australian national health info service.",
    ),
    "betterhealth.vic.gov.au": SourceInfo(
        quality=EXCELLENT, categories=("health", "australia", "patient-info"),
        content_type="reference", freshness="daily",
        notes="Victoria state health info — comprehensive patient content.",
    ),
    "healthlinkbc.ca": SourceInfo(
        quality=EXCELLENT, categories=("health", "canada", "patient-info"),
        content_type="reference", freshness="daily",
        notes="HealthLink BC (Canada).",
    ),

    # ----- Consumer health & drug info -----
    "goodrx.com": SourceInfo(
        quality=GOOD, categories=("health", "drugs", "pricing", "consumer"),
        content_type="data", freshness="daily",
        notes="GoodRx — drug pricing and info.",
    ),
    "medscape.com": SourceInfo(
        quality=GOOD, categories=("health", "medical", "professional"),
        content_type="reference", freshness="daily",
        notes="Medscape — clinician-focused medical reference.",
    ),
    "verywellhealth.com": SourceInfo(
        quality=GOOD, categories=("health", "consumer", "patient-info"),
        content_type="article", freshness="daily",
        notes="Verywell Health — editor-reviewed consumer health.",
    ),
    "verywellmind.com": SourceInfo(
        quality=GOOD, categories=("health", "mental-health", "consumer"),
        content_type="article", freshness="daily",
        notes="Verywell Mind — mental health consumer info.",
    ),
    "everydayhealth.com": SourceInfo(
        quality=GOOD, categories=("health", "consumer", "patient-info"),
        content_type="article", freshness="daily",
        notes="Everyday Health.",
    ),
    "health.harvard.edu": SourceInfo(
        quality=EXCELLENT, categories=("health", "consumer", "medical"),
        content_type="article", freshness="daily",
        notes="Harvard Health Publishing.",
    ),
    "patient.info": SourceInfo(
        quality=EXCELLENT, categories=("health", "consumer", "patient-info"),
        content_type="reference", freshness="daily",
        notes="Patient.info — UK clinician-written patient articles.",
    ),

    # ----- Pediatrics & developmental -----
    "healthychildren.org": SourceInfo(
        quality=EXCELLENT, categories=("health", "pediatrics", "parenting", "children"),
        content_type="reference", freshness="daily",
        notes="AAP consumer site — HealthyChildren.org.",
    ),
    "zerotothree.org": SourceInfo(
        quality=EXCELLENT, categories=("health", "parenting", "child-development", "infants"),
        content_type="reference", freshness="daily",
        notes="Zero to Three — early childhood development.",
    ),

    # ----- Mental health -----
    "mentalhealth.org.uk": SourceInfo(
        quality=EXCELLENT, categories=("mental-health", "uk", "consumer"),
        content_type="reference", freshness="daily",
        notes="Mental Health Foundation UK.",
    ),
    "nami.org": SourceInfo(
        quality=EXCELLENT, categories=("mental-health", "support", "patient-info"),
        content_type="reference", freshness="daily",
        notes="NAMI — National Alliance on Mental Illness.",
    ),
    "mhanational.org": SourceInfo(
        quality=EXCELLENT, categories=("mental-health", "support", "consumer"),
        content_type="reference", freshness="daily",
        notes="Mental Health America.",
    ),
    "apa.org.au": SourceInfo(
        quality=GOOD, categories=("mental-health", "australia", "psychology"),
        content_type="reference", freshness="daily",
        notes="Australian Psychological Society.",
    ),
    "ranzcp.org": SourceInfo(
        quality=GOOD, categories=("mental-health", "psychiatry", "australia", "new-zealand"),
        content_type="reference", freshness="daily",
        notes="Royal ANZ College of Psychiatrists.",
    ),

    # ----- NIH institutes -----
    "drugabuse.gov": SourceInfo(
        quality=EXCELLENT, categories=("health", "substance-abuse", "nih"),
        content_type="reference", freshness="daily",
        notes="NIDA — National Institute on Drug Abuse.",
    ),
    "niehs.nih.gov": SourceInfo(
        quality=EXCELLENT, categories=("health", "environment", "nih"),
        content_type="reference", freshness="daily",
        notes="NIEHS — environmental health sciences.",
    ),
    "nhlbi.nih.gov": SourceInfo(
        quality=EXCELLENT, categories=("health", "cardiology", "pulmonology", "nih"),
        content_type="reference", freshness="daily",
        notes="NHLBI — heart, lung, blood.",
    ),
    "niddk.nih.gov": SourceInfo(
        quality=EXCELLENT, categories=("health", "diabetes", "digestive", "kidney", "nih"),
        content_type="reference", freshness="daily",
        notes="NIDDK — diabetes, digestive, kidney diseases.",
    ),

    # ----- Medical education -----
    "aamc.org": SourceInfo(
        quality=EXCELLENT, categories=("health", "medical-education", "research"),
        content_type="reference", freshness="daily",
        notes="Association of American Medical Colleges.",
    ),
    "ama-assn.org": SourceInfo(
        quality=EXCELLENT, categories=("health", "medical", "professional"),
        content_type="reference", freshness="daily",
        notes="American Medical Association.",
    ),

    # ========================================================================
    # PASS 4 — ENTERTAINMENT / CULTURE / INSTITUTIONS (verified live)
    # ========================================================================

    # ----- Film & TV -----
    "tvmaze.com": SourceInfo(
        quality=EXCELLENT, categories=("entertainment", "tv", "database"),
        content_type="data", freshness="daily",
        notes="TVmaze — TV show DB with open API.", structured_data=True,
    ),
    "tasteofcinema.com": SourceInfo(
        quality=GOOD, categories=("entertainment", "film", "lists", "criticism"),
        content_type="article", freshness="daily",
        notes="Taste of Cinema — film lists/criticism.",
    ),
    "criticker.com": SourceInfo(
        quality=GOOD, categories=("entertainment", "film", "reviews"),
        content_type="reference", freshness="daily",
        notes="Criticker — film ratings community.",
    ),
    "flickchart.com": SourceInfo(
        quality=GOOD, categories=("entertainment", "film", "rankings"),
        content_type="reference", freshness="daily",
        notes="Flickchart — movie ranking tool.",
    ),
    "mubi.com": SourceInfo(
        quality=GOOD, categories=("entertainment", "film", "streaming", "criticism"),
        content_type="article", freshness="daily",
        notes="MUBI — curated cinema + Notebook publication.",
    ),
    "boxofficemojo.com": SourceInfo(
        quality=EXCELLENT, categories=("entertainment", "film", "box-office", "data"),
        content_type="data", freshness="daily",
        notes="Box Office Mojo — film revenue data.", structured_data=True,
    ),
    "the-numbers.com": SourceInfo(
        quality=EXCELLENT, categories=("entertainment", "film", "box-office", "data"),
        content_type="data", freshness="daily",
        notes="The Numbers — film industry stats.",
    ),

    # ----- Music -----
    "nme.com": SourceInfo(
        quality=GOOD, categories=("music", "news", "reviews", "entertainment"),
        content_type="news", freshness="realtime",
        notes="NME music news & reviews.",
    ),
    "pitchfork.com": SourceInfo(
        quality=EXCELLENT, categories=("music", "reviews", "criticism"),
        content_type="article", freshness="daily",
        notes="Pitchfork music reviews.",
    ),
    "rollingstone.com": SourceInfo(
        quality=EXCELLENT, categories=("music", "news", "reviews", "entertainment"),
        content_type="news", freshness="realtime",
        notes="Rolling Stone.",
    ),
    "billboard.com": SourceInfo(
        quality=EXCELLENT, categories=("music", "charts", "news"),
        content_type="data", freshness="daily",
        notes="Billboard — music charts and news.",
    ),
    "officialcharts.com": SourceInfo(
        quality=EXCELLENT, categories=("music", "charts", "uk"),
        content_type="data", freshness="daily",
        notes="UK Official Charts.",
    ),
    "ableton.com": SourceInfo(
        quality=EXCELLENT, categories=("music", "production", "software", "reference"),
        content_type="reference", freshness="static",
        notes="Ableton — music production docs/blog.",
    ),
    "bandcamp.com": SourceInfo(
        quality=GOOD, categories=("music", "streaming", "indie"),
        content_type="reference", freshness="daily",
        notes="Bandcamp artist pages.",
    ),
    "soundcloud.com": SourceInfo(
        quality=GOOD, categories=("music", "streaming", "audio"),
        content_type="reference", freshness="daily",
        notes="SoundCloud — artist pages.",
    ),
    "beatport.com": SourceInfo(
        quality=GOOD, categories=("music", "electronic", "dj", "store"),
        content_type="reference", freshness="daily",
        notes="Beatport — electronic music DJ store.",
    ),
    "whosampled.com": SourceInfo(
        quality=EXCELLENT, categories=("music", "samples", "database"),
        content_type="reference", freshness="daily",
        notes="WhoSampled — sample/cover/remix DB.",
    ),
    "jaxsta.com": SourceInfo(
        quality=GOOD, categories=("music", "credits", "database"),
        content_type="reference", freshness="daily",
        notes="Jaxsta — music credits DB.",
    ),
    "songexploder.net": SourceInfo(
        quality=EXCELLENT, categories=("music", "analysis", "podcast"),
        content_type="article", freshness="weekly",
        notes="Song Exploder podcast breakdowns.",
    ),

    # ----- Museums & arts institutions -----
    "metmuseum.org": SourceInfo(
        quality=EXCELLENT, categories=("art", "museum", "culture", "history"),
        content_type="reference", freshness="daily",
        notes="Metropolitan Museum of Art.",
    ),
    "tate.org.uk": SourceInfo(
        quality=EXCELLENT, categories=("art", "museum", "uk", "culture"),
        content_type="reference", freshness="daily",
        notes="Tate galleries (UK).",
    ),
    "louvre.fr": SourceInfo(
        quality=EXCELLENT, categories=("art", "museum", "france", "culture"),
        content_type="reference", freshness="daily",
        notes="Musée du Louvre.",
    ),
    "smithsonianmag.com": SourceInfo(
        quality=EXCELLENT, categories=("culture", "history", "science", "magazine"),
        content_type="article", freshness="daily",
        notes="Smithsonian Magazine.",
    ),
    "si.edu": SourceInfo(
        quality=EXCELLENT, categories=("culture", "museum", "research", "history"),
        content_type="reference", freshness="daily",
        notes="Smithsonian Institution.",
    ),
    "nationalgallery.org.uk": SourceInfo(
        quality=EXCELLENT, categories=("art", "museum", "uk"),
        content_type="reference", freshness="daily",
        notes="National Gallery London.",
    ),
    "britishmuseum.org": SourceInfo(
        quality=EXCELLENT, categories=("art", "museum", "history", "uk"),
        content_type="reference", freshness="daily",
        notes="British Museum.",
    ),
    "guggenheim.org": SourceInfo(
        quality=EXCELLENT, categories=("art", "museum", "modern"),
        content_type="reference", freshness="daily",
        notes="Guggenheim Museum.",
    ),
    "getty.edu": SourceInfo(
        quality=EXCELLENT, categories=("art", "museum", "research", "history"),
        content_type="reference", freshness="daily",
        notes="Getty — museum + research.",
    ),
    "artic.edu": SourceInfo(
        quality=EXCELLENT, categories=("art", "museum", "chicago"),
        content_type="reference", freshness="daily",
        notes="Art Institute of Chicago.",
    ),
    "lacma.org": SourceInfo(
        quality=EXCELLENT, categories=("art", "museum", "los-angeles"),
        content_type="reference", freshness="daily",
        notes="LACMA.",
    ),
    "whitney.org": SourceInfo(
        quality=EXCELLENT, categories=("art", "museum", "american-art"),
        content_type="reference", freshness="daily",
        notes="Whitney Museum.",
    ),
    "philamuseum.org": SourceInfo(
        quality=EXCELLENT, categories=("art", "museum", "philadelphia"),
        content_type="reference", freshness="daily",
        notes="Philadelphia Museum of Art.",
    ),
    "rijksmuseum.nl": SourceInfo(
        quality=EXCELLENT, categories=("art", "museum", "netherlands"),
        content_type="reference", freshness="daily",
        notes="Rijksmuseum.",
    ),
    "vam.ac.uk": SourceInfo(
        quality=EXCELLENT, categories=("art", "museum", "design", "uk"),
        content_type="reference", freshness="daily",
        notes="V&A — Victoria and Albert Museum.",
    ),
    "atlasobscura.com": SourceInfo(
        quality=EXCELLENT, categories=("travel", "history", "culture", "curiosities"),
        content_type="article", freshness="daily",
        notes="Atlas Obscura — unusual places.",
    ),

    # ----- Industry press & theater -----
    "variety.com": SourceInfo(
        quality=EXCELLENT, categories=("entertainment", "industry", "news", "film", "tv"),
        content_type="news", freshness="realtime",
        notes="Variety.",
    ),
    "deadline.com": SourceInfo(
        quality=EXCELLENT, categories=("entertainment", "industry", "news", "film", "tv"),
        content_type="news", freshness="realtime",
        notes="Deadline Hollywood.",
    ),
    "indiewire.com": SourceInfo(
        quality=EXCELLENT, categories=("entertainment", "film", "tv", "news"),
        content_type="news", freshness="realtime",
        notes="IndieWire.",
    ),
    "playbill.com": SourceInfo(
        quality=EXCELLENT, categories=("entertainment", "theater", "broadway"),
        content_type="news", freshness="daily",
        notes="Playbill — theater news.",
    ),
    "broadwayworld.com": SourceInfo(
        quality=GOOD, categories=("entertainment", "theater", "broadway"),
        content_type="news", freshness="daily",
        notes="BroadwayWorld.",
    ),
    "ibdb.com": SourceInfo(
        quality=EXCELLENT, categories=("entertainment", "theater", "broadway", "database"),
        content_type="data", freshness="daily",
        notes="Internet Broadway Database.",
    ),

    # ========================================================================
    # PASS 5 — REGIONAL US NEWS / MAGAZINES / THINK TANKS (verified live)
    # ========================================================================

    # ----- Major US regional papers -----
    "sfchronicle.com": SourceInfo(
        quality=EXCELLENT, categories=("news", "regional", "california", "san-francisco"),
        content_type="news", freshness="realtime",
        notes="San Francisco Chronicle.",
    ),
    "seattletimes.com": SourceInfo(
        quality=EXCELLENT, categories=("news", "regional", "washington", "seattle"),
        content_type="news", freshness="realtime",
        notes="The Seattle Times.",
    ),
    "startribune.com": SourceInfo(
        quality=GOOD, categories=("news", "regional", "minnesota", "minneapolis"),
        content_type="news", freshness="realtime",
        notes="Star Tribune (Minneapolis).",
    ),
    "bostonglobe.com": SourceInfo(
        quality=EXCELLENT, categories=("news", "regional", "massachusetts", "boston"),
        content_type="news", freshness="realtime",
        notes="The Boston Globe.",
    ),
    "inquirer.com": SourceInfo(
        quality=EXCELLENT, categories=("news", "regional", "pennsylvania", "philadelphia"),
        content_type="news", freshness="realtime",
        notes="The Philadelphia Inquirer.",
    ),
    "denverpost.com": SourceInfo(
        quality=EXCELLENT, categories=("news", "regional", "colorado", "denver"),
        content_type="news", freshness="realtime",
        notes="The Denver Post.",
    ),
    "latimes.com": SourceInfo(
        quality=EXCELLENT, categories=("news", "regional", "california", "los-angeles"),
        content_type="news", freshness="realtime",
        notes="Los Angeles Times.",
    ),
    "chicagotribune.com": SourceInfo(
        quality=EXCELLENT, categories=("news", "regional", "illinois", "chicago"),
        content_type="news", freshness="realtime",
        notes="Chicago Tribune.",
    ),
    "nydailynews.com": SourceInfo(
        quality=GOOD, categories=("news", "regional", "new-york"),
        content_type="news", freshness="realtime",
        notes="NY Daily News.",
    ),
    "nypost.com": SourceInfo(
        quality=GOOD, categories=("news", "regional", "new-york", "tabloid"),
        content_type="news", freshness="realtime",
        notes="New York Post.",
    ),
    "dallasnews.com": SourceInfo(
        quality=EXCELLENT, categories=("news", "regional", "texas", "dallas"),
        content_type="news", freshness="realtime",
        notes="The Dallas Morning News.",
    ),
    "houstonchronicle.com": SourceInfo(
        quality=EXCELLENT, categories=("news", "regional", "texas", "houston"),
        content_type="news", freshness="realtime",
        notes="Houston Chronicle.",
    ),
    "tampabay.com": SourceInfo(
        quality=EXCELLENT, categories=("news", "regional", "florida", "tampa"),
        content_type="news", freshness="realtime",
        notes="Tampa Bay Times.",
    ),
    "ajc.com": SourceInfo(
        quality=EXCELLENT, categories=("news", "regional", "georgia", "atlanta"),
        content_type="news", freshness="realtime",
        notes="The Atlanta Journal-Constitution.",
    ),
    "azcentral.com": SourceInfo(
        quality=GOOD, categories=("news", "regional", "arizona", "phoenix"),
        content_type="news", freshness="realtime",
        notes="The Arizona Republic.",
    ),
    "postandcourier.com": SourceInfo(
        quality=EXCELLENT, categories=("news", "regional", "south-carolina", "charleston"),
        content_type="news", freshness="realtime",
        notes="The Post and Courier.",
    ),
    "mercurynews.com": SourceInfo(
        quality=GOOD, categories=("news", "regional", "california", "silicon-valley"),
        content_type="news", freshness="realtime",
        notes="The Mercury News.",
    ),
    "sandiegouniontribune.com": SourceInfo(
        quality=GOOD, categories=("news", "regional", "california", "san-diego"),
        content_type="news", freshness="realtime",
        notes="San Diego Union-Tribune.",
    ),
    "baltimoresun.com": SourceInfo(
        quality=GOOD, categories=("news", "regional", "maryland", "baltimore"),
        content_type="news", freshness="realtime",
        notes="The Baltimore Sun.",
    ),
    "detroitnews.com": SourceInfo(
        quality=GOOD, categories=("news", "regional", "michigan", "detroit"),
        content_type="news", freshness="realtime",
        notes="The Detroit News.",
    ),
    "freep.com": SourceInfo(
        quality=GOOD, categories=("news", "regional", "michigan", "detroit"),
        content_type="news", freshness="realtime",
        notes="Detroit Free Press.",
    ),
    "stltoday.com": SourceInfo(
        quality=GOOD, categories=("news", "regional", "missouri", "st-louis"),
        content_type="news", freshness="realtime",
        notes="St. Louis Post-Dispatch.",
    ),
    "tennessean.com": SourceInfo(
        quality=GOOD, categories=("news", "regional", "tennessee", "nashville"),
        content_type="news", freshness="realtime",
        notes="The Tennessean.",
    ),
    "courier-journal.com": SourceInfo(
        quality=GOOD, categories=("news", "regional", "kentucky", "louisville"),
        content_type="news", freshness="realtime",
        notes="The Courier Journal.",
    ),
    "jsonline.com": SourceInfo(
        quality=GOOD, categories=("news", "regional", "wisconsin", "milwaukee"),
        content_type="news", freshness="realtime",
        notes="Milwaukee Journal Sentinel.",
    ),

    # ----- Magazines -----
    "newyorker.com": SourceInfo(
        quality=EXCELLENT, categories=("magazine", "journalism", "culture", "politics"),
        content_type="article", freshness="daily",
        notes="The New Yorker.",
    ),
    "theatlantic.com": SourceInfo(
        quality=EXCELLENT, categories=("magazine", "journalism", "culture", "politics"),
        content_type="article", freshness="daily",
        notes="The Atlantic.",
    ),
    "harpers.org": SourceInfo(
        quality=EXCELLENT, categories=("magazine", "journalism", "culture", "politics"),
        content_type="article", freshness="monthly",
        notes="Harper's Magazine.",
    ),
    "vox.com": SourceInfo(
        quality=EXCELLENT, categories=("news", "magazine", "explainer", "politics"),
        content_type="article", freshness="realtime",
        notes="Vox.",
    ),
    "slate.com": SourceInfo(
        quality=EXCELLENT, categories=("magazine", "politics", "culture"),
        content_type="article", freshness="realtime",
        notes="Slate.",
    ),
    "thecut.com": SourceInfo(
        quality=GOOD, categories=("magazine", "culture", "style"),
        content_type="article", freshness="daily",
        notes="The Cut (New York Magazine).",
    ),
    "newrepublic.com": SourceInfo(
        quality=EXCELLENT, categories=("magazine", "politics", "journalism"),
        content_type="article", freshness="daily",
        notes="The New Republic.",
    ),
    "thenation.com": SourceInfo(
        quality=EXCELLENT, categories=("magazine", "politics", "progressive"),
        content_type="article", freshness="daily",
        notes="The Nation.",
    ),
    "motherjones.com": SourceInfo(
        quality=EXCELLENT, categories=("magazine", "politics", "investigative"),
        content_type="article", freshness="daily",
        notes="Mother Jones.",
    ),
    "jacobin.com": SourceInfo(
        quality=GOOD, categories=("magazine", "politics", "left"),
        content_type="article", freshness="daily",
        notes="Jacobin.",
    ),
    "dissentmagazine.org": SourceInfo(
        quality=GOOD, categories=("magazine", "politics", "left"),
        content_type="article", freshness="monthly",
        notes="Dissent Magazine.",
    ),
    "commentary.org": SourceInfo(
        quality=GOOD, categories=("magazine", "politics", "conservative"),
        content_type="article", freshness="monthly",
        notes="Commentary Magazine.",
    ),
    "thebulwark.com": SourceInfo(
        quality=GOOD, categories=("magazine", "politics", "center-right"),
        content_type="article", freshness="daily",
        notes="The Bulwark.",
    ),
    "lawfareblog.com": SourceInfo(
        quality=EXCELLENT, categories=("policy", "national-security", "law", "analysis"),
        content_type="article", freshness="daily",
        notes="Lawfare — national security law analysis.",
    ),

    # ----- Think tanks -----
    "aei.org": SourceInfo(
        quality=EXCELLENT, categories=("think-tank", "policy", "conservative"),
        content_type="reference", freshness="daily",
        notes="American Enterprise Institute.",
    ),
    "heritage.org": SourceInfo(
        quality=GOOD, categories=("think-tank", "policy", "conservative"),
        content_type="reference", freshness="daily",
        notes="Heritage Foundation.",
    ),
    "urban.org": SourceInfo(
        quality=EXCELLENT, categories=("think-tank", "policy", "research", "data"),
        content_type="reference", freshness="daily",
        notes="Urban Institute.",
    ),
    "chathamhouse.org": SourceInfo(
        quality=EXCELLENT, categories=("think-tank", "policy", "international", "uk"),
        content_type="reference", freshness="daily",
        notes="Chatham House (Royal Institute of International Affairs).",
    ),
    "pewresearch.org": SourceInfo(
        quality=EXCELLENT, categories=("research", "polling", "social-science", "data"),
        content_type="data", freshness="daily",
        notes="Pew Research Center. Authoritative surveys.",
    ),
    "fivethirtyeight.com": SourceInfo(
        quality=EXCELLENT, categories=("data-journalism", "politics", "sports", "polling"),
        content_type="article", freshness="daily",
        notes="FiveThirtyEight — data journalism (now mostly archive).",
    ),
    "theintercept.com": SourceInfo(
        quality=EXCELLENT, categories=("journalism", "investigative", "politics"),
        content_type="article", freshness="daily",
        notes="The Intercept.",
    ),

    # ----- Journalism meta / media criticism -----
    "niemanlab.org": SourceInfo(
        quality=EXCELLENT, categories=("journalism", "media", "research"),
        content_type="article", freshness="daily",
        notes="Nieman Journalism Lab (Harvard).",
    ),
    "poynter.org": SourceInfo(
        quality=EXCELLENT, categories=("journalism", "media", "fact-checking", "ethics"),
        content_type="article", freshness="daily",
        notes="Poynter Institute.",
    ),

    # ========================================================================
    # PASS 6 — TRANSPORTATION / TRAVEL / REAL-ESTATE DEPTH (verified live)
    # ========================================================================

    # ----- Aviation & regulators -----
    "faa.gov": SourceInfo(
        quality=EXCELLENT, categories=("transportation", "aviation", "government", "regulation"),
        content_type="reference", freshness="daily",
        notes="FAA — Federal Aviation Administration.",
    ),
    "ntsb.gov": SourceInfo(
        quality=EXCELLENT, categories=("transportation", "safety", "government", "investigation"),
        content_type="reference", freshness="daily",
        notes="NTSB — National Transportation Safety Board.",
    ),
    "iata.org": SourceInfo(
        quality=EXCELLENT, categories=("transportation", "aviation", "international"),
        content_type="reference", freshness="daily",
        notes="IATA — International Air Transport Association.",
    ),
    "icao.int": SourceInfo(
        quality=EXCELLENT, categories=("transportation", "aviation", "international", "regulation"),
        content_type="reference", freshness="daily",
        notes="ICAO — International Civil Aviation Organization.",
    ),
    "tsa.gov": SourceInfo(
        quality=EXCELLENT, categories=("transportation", "security", "government"),
        content_type="reference", freshness="daily",
        notes="Transportation Security Administration.",
    ),
    "skyvector.com": SourceInfo(
        quality=EXCELLENT, categories=("aviation", "charts", "navigation", "pilots"),
        content_type="data", freshness="daily",
        notes="SkyVector — aeronautical charts.",
    ),
    "airnav.com": SourceInfo(
        quality=EXCELLENT, categories=("aviation", "airports", "database", "pilots"),
        content_type="data", freshness="daily",
        notes="AirNav — airport info & FBO.",
    ),
    "openflights.org": SourceInfo(
        quality=EXCELLENT, categories=("aviation", "airports", "database", "open-data"),
        content_type="data", freshness="daily",
        notes="OpenFlights — open airport/route DB.",
    ),

    # ----- Maps & geography -----
    "worldatlas.com": SourceInfo(
        quality=EXCELLENT, categories=("geography", "reference", "countries"),
        content_type="reference", freshness="daily",
        notes="WorldAtlas.",
    ),
    "wiki.openstreetmap.org": SourceInfo(
        quality=EXCELLENT, categories=("maps", "geography", "reference", "documentation"),
        content_type="reference", freshness="daily",
        notes="OpenStreetMap wiki — tagging & documentation.",
    ),
    "transitapp.com": SourceInfo(
        quality=GOOD, categories=("transportation", "transit", "mobile"),
        content_type="reference", freshness="realtime",
        notes="Transit app — public transit schedules.",
    ),
    "moovitapp.com": SourceInfo(
        quality=GOOD, categories=("transportation", "transit", "mobile"),
        content_type="reference", freshness="realtime",
        notes="Moovit — urban mobility.",
    ),

    # ----- Rail (international) -----
    "nationalrail.co.uk": SourceInfo(
        quality=EXCELLENT, categories=("transportation", "rail", "uk", "schedules"),
        content_type="data", freshness="realtime",
        notes="National Rail UK.",
    ),
    "ns.nl": SourceInfo(
        quality=EXCELLENT, categories=("transportation", "rail", "netherlands"),
        content_type="reference", freshness="realtime",
        notes="NS — Dutch Railways.",
    ),
    "trenitalia.com": SourceInfo(
        quality=GOOD, categories=("transportation", "rail", "italy"),
        content_type="reference", freshness="realtime",
        notes="Trenitalia.",
    ),
    "deutschebahn.com": SourceInfo(
        quality=EXCELLENT, categories=("transportation", "rail", "germany"),
        content_type="reference", freshness="realtime",
        notes="Deutsche Bahn.",
    ),

    # ----- US transit agencies -----
    "bart.gov": SourceInfo(
        quality=EXCELLENT, categories=("transportation", "transit", "california", "san-francisco"),
        content_type="reference", freshness="realtime",
        notes="Bay Area Rapid Transit.",
    ),
    "mta.info": SourceInfo(
        quality=EXCELLENT, categories=("transportation", "transit", "new-york"),
        content_type="reference", freshness="realtime",
        notes="NYC Metropolitan Transportation Authority.",
    ),
    "wmata.com": SourceInfo(
        quality=EXCELLENT, categories=("transportation", "transit", "washington-dc"),
        content_type="reference", freshness="realtime",
        notes="WMATA — DC Metro.",
    ),
    "cta.com": SourceInfo(
        quality=EXCELLENT, categories=("transportation", "transit", "chicago"),
        content_type="reference", freshness="realtime",
        notes="Chicago Transit Authority.",
    ),

    # ----- Housing & real-estate research -----
    "fhfa.gov": SourceInfo(
        quality=EXCELLENT, categories=("real-estate", "finance", "government", "data"),
        content_type="data", freshness="daily",
        notes="FHFA — housing finance data.",
    ),
    "jchs.harvard.edu": SourceInfo(
        quality=EXCELLENT, categories=("real-estate", "research", "housing", "harvard"),
        content_type="reference", freshness="daily",
        notes="Harvard Joint Center for Housing Studies.",
    ),
    "fanniemae.com": SourceInfo(
        quality=EXCELLENT, categories=("real-estate", "finance", "mortgages"),
        content_type="reference", freshness="daily",
        notes="Fannie Mae.",
    ),
    "nar.realtor": SourceInfo(
        quality=EXCELLENT, categories=("real-estate", "research", "industry"),
        content_type="reference", freshness="daily",
        notes="National Association of Realtors.",
    ),
    "apartmentlist.com": SourceInfo(
        quality=GOOD, categories=("real-estate", "rentals", "research"),
        content_type="reference", freshness="daily",
        notes="Apartment List — research/rent index.",
    ),
    "rentometer.com": SourceInfo(
        quality=GOOD, categories=("real-estate", "rentals", "data"),
        content_type="data", freshness="daily",
        notes="Rentometer — rent comp data.",
    ),
    "homelight.com": SourceInfo(
        quality=GOOD, categories=("real-estate", "consumer", "guides"),
        content_type="reference", freshness="daily",
        notes="HomeLight.",
    ),
    "loopnet.com": SourceInfo(
        quality=AVOID, categories=("real-estate", "commercial"),
        content_type="reference", freshness="daily",
        notes="LoopNet — commercial real estate.",
    ),
    "costar.com": SourceInfo(
        quality=AVOID, categories=("real-estate", "commercial", "data"),
        content_type="data", freshness="daily",
        notes="CoStar — commercial real estate data.",
    ),
    "rentcafe.com": SourceInfo(
        quality=GOOD, categories=("real-estate", "rentals", "research"),
        content_type="reference", freshness="daily",
        notes="RentCafe — rental search + research blog.",
    ),
}


# ---------------------------------------------------------------------------
# Topic → recommended search site operators
# ---------------------------------------------------------------------------

# Maps query keywords/phrases to preferred domains. Used to add site: operators
# to SearXNG queries and to boost matching domains in search result ranking.
# Multiple keywords can match — results are deduplicated.
_TOPIC_SITES: dict[str, list[str]] = {
    # Weather & environment
    "weather": ["weather.gov", "openweathermap.org"],
    "forecast": ["weather.gov"],
    "hurricane": ["weather.gov", "nhc.noaa.gov"],
    "tornado": ["weather.gov"],
    "earthquake": ["earthquake.usgs.gov"],
    "air quality": ["airnow.gov"],
    "climate": ["climate.gov", "nasa.gov"],

    # Programming & tech
    "python": ["docs.python.org", "stackoverflow.com", "pypi.org"],
    "javascript": ["developer.mozilla.org", "stackoverflow.com", "nodejs.org"],
    "typescript": ["typescriptlang.org", "developer.mozilla.org"],
    "rust language": ["docs.rs", "rust-lang.org", "doc.rust-lang.org"],
    "rust programming": ["docs.rs", "rust-lang.org", "doc.rust-lang.org"],
    "rust lang": ["docs.rs", "rust-lang.org"],
    "golang": ["go.dev", "pkg.go.dev", "stackoverflow.com"],
    "go lang": ["go.dev", "pkg.go.dev"],
    "java language": ["docs.oracle.com", "stackoverflow.com", "spring.io"],
    "java programming": ["docs.oracle.com", "stackoverflow.com", "spring.io"],
    "kotlin": ["kotlinlang.org", "stackoverflow.com"],
    "c++": ["cppreference.com", "stackoverflow.com"],
    "cpp": ["cppreference.com", "stackoverflow.com"],
    "c#": ["learn.microsoft.com", "dotnet.microsoft.com", "stackoverflow.com"],
    "csharp": ["learn.microsoft.com", "dotnet.microsoft.com"],
    ".net": ["learn.microsoft.com", "dotnet.microsoft.com", "nuget.org"],
    "dotnet": ["learn.microsoft.com", "dotnet.microsoft.com"],
    "swift language": ["swift.org", "developer.apple.com"],
    "swift programming": ["swift.org", "developer.apple.com"],
    "swiftui": ["developer.apple.com", "swift.org"],
    "php": ["php.net", "stackoverflow.com", "laravel.com"],
    "ruby language": ["ruby-lang.org", "ruby-doc.org", "guides.rubyonrails.org"],
    "ruby programming": ["ruby-lang.org", "ruby-doc.org"],
    "ruby gems": ["rubygems.org", "ruby-lang.org"],
    "elixir": ["elixir-lang.org", "hexdocs.pm"],
    "haskell": ["haskell.org", "stackoverflow.com"],
    "scala": ["scala-lang.org", "stackoverflow.com"],
    "clojure": ["clojure.org", "stackoverflow.com"],
    "lua": ["lua.org"],
    "zig": ["ziglang.org"],
    "julia": ["julialang.org"],
    "r language": ["r-project.org", "cran.r-project.org"],
    "r programming": ["r-project.org", "cran.r-project.org"],
    "ocaml": ["ocaml.org"],
    "dart": ["dart.dev", "flutter.dev"],
    "flutter": ["flutter.dev", "dart.dev"],
    "html": ["developer.mozilla.org", "w3schools.com"],
    "css": ["developer.mozilla.org", "w3schools.com", "tailwindcss.com"],
    "tailwind": ["tailwindcss.com"],
    "bootstrap": ["getbootstrap.com"],
    "react": ["react.dev", "stackoverflow.com", "nextjs.org"],
    "nextjs": ["nextjs.org", "react.dev"],
    "next.js": ["nextjs.org", "react.dev"],
    "vue": ["vuejs.org", "stackoverflow.com", "nuxt.com"],
    "nuxt": ["nuxt.com", "vuejs.org"],
    "angular": ["angular.dev", "stackoverflow.com"],
    "svelte": ["svelte.dev"],
    "astro": ["astro.build"],
    "django": ["docs.djangoproject.com", "stackoverflow.com"],
    "flask": ["flask.palletsprojects.com", "stackoverflow.com"],
    "fastapi": ["fastapi.tiangolo.com", "stackoverflow.com"],
    "express": ["expressjs.com", "nodejs.org"],
    "spring framework": ["spring.io", "docs.oracle.com"],
    "spring boot": ["spring.io"],
    "ruby on rails": ["guides.rubyonrails.org", "ruby-lang.org"],
    "rails framework": ["guides.rubyonrails.org"],
    "laravel": ["laravel.com", "php.net"],
    "docker": ["docs.docker.com", "stackoverflow.com"],
    "kubernetes": ["kubernetes.io", "stackoverflow.com"],
    "k8s": ["kubernetes.io"],
    "terraform": ["terraform.io", "registry.terraform.io"],
    "ansible": ["docs.ansible.com"],
    "nginx": ["nginx.org"],
    "linux": ["man7.org", "wiki.archlinux.org", "askubuntu.com"],
    "ubuntu": ["askubuntu.com", "wiki.archlinux.org"],
    "arch linux": ["wiki.archlinux.org"],
    "git": ["git-scm.com", "stackoverflow.com", "docs.github.com"],
    "github actions": ["docs.github.com"],
    "ci/cd": ["docs.github.com", "docs.gitlab.com"],
    "sql": ["stackoverflow.com", "w3schools.com", "postgresql.org"],
    "postgresql": ["postgresql.org", "stackoverflow.com"],
    "postgres": ["postgresql.org"],
    "mysql": ["dev.mysql.com", "stackoverflow.com"],
    "sqlite": ["sqlite.org"],
    "mongodb": ["mongodb.com", "stackoverflow.com"],
    "redis": ["redis.io"],
    "elasticsearch": ["elastic.co"],
    "regex": ["regex101.com", "developer.mozilla.org"],
    "api": ["developer.mozilla.org", "stackoverflow.com"],
    "rest api": ["developer.mozilla.org", "swagger.io"],
    "graphql": ["graphql.org", "stackoverflow.com"],
    "deno": ["docs.deno.com", "stackoverflow.com"],
    "bun": ["bun.sh", "stackoverflow.com"],
    "npm": ["npmjs.com", "stackoverflow.com"],
    "pip": ["pypi.org", "docs.python.org"],
    "cargo": ["crates.io", "docs.rs"],
    "nuget": ["nuget.org"],
    "gem": ["rubygems.org"],
    "maven": ["mvnrepository.com"],
    "browser compatibility": ["caniuse.com", "developer.mozilla.org"],
    "can i use": ["caniuse.com"],

    # AI & machine learning
    "machine learning": ["scikit-learn.org", "pytorch.org", "paperswithcode.com"],
    "deep learning": ["pytorch.org", "tensorflow.org", "arxiv.org"],
    "neural network": ["pytorch.org", "tensorflow.org", "arxiv.org"],
    "transformer": ["huggingface.co", "arxiv.org"],
    "llm": ["huggingface.co", "arxiv.org", "docs.anthropic.com"],
    "large language model": ["huggingface.co", "arxiv.org"],
    "ai model": ["huggingface.co", "paperswithcode.com"],
    "pytorch": ["pytorch.org"],
    "tensorflow": ["tensorflow.org"],
    "hugging face": ["huggingface.co"],
    "huggingface": ["huggingface.co"],
    "keras": ["keras.io"],
    "sklearn": ["scikit-learn.org"],
    "scikit": ["scikit-learn.org"],
    "claude api": ["docs.anthropic.com"],
    "anthropic": ["docs.anthropic.com"],
    "openai api": ["platform.openai.com"],
    "gpt": ["platform.openai.com", "huggingface.co"],
    "kaggle": ["kaggle.com"],
    "fine tuning": ["huggingface.co", "pytorch.org"],
    "fine-tuning": ["huggingface.co", "pytorch.org"],

    # Cloud & DevOps
    "aws": ["docs.aws.amazon.com"],
    "amazon web services": ["docs.aws.amazon.com"],
    "gcp": ["cloud.google.com"],
    "google cloud": ["cloud.google.com"],
    "azure": ["learn.microsoft.com"],
    "cloud computing": ["docs.aws.amazon.com", "cloud.google.com", "learn.microsoft.com"],
    "devops": ["docs.docker.com", "kubernetes.io", "terraform.io"],
    "infrastructure": ["terraform.io", "docs.ansible.com"],
    "monitoring": ["prometheus.io", "grafana.com"],
    "container": ["docs.docker.com", "kubernetes.io"],

    # Health & medicine
    "medical": ["cdc.gov", "nih.gov", "mayoclinic.org", "medlineplus.gov"],
    "health": ["cdc.gov", "nih.gov", "mayoclinic.org", "healthline.com"],
    "disease": ["cdc.gov", "who.int", "nih.gov"],
    "symptoms": ["mayoclinic.org", "medlineplus.gov", "webmd.com"],
    "drug": ["drugs.com", "medlineplus.gov", "fda.gov"],
    "medication": ["drugs.com", "medlineplus.gov", "fda.gov"],
    "vaccine": ["cdc.gov", "who.int"],
    "nutrition": ["nih.gov", "healthline.com", "fda.gov", "fdc.nal.usda.gov"],
    "calories": ["fdc.nal.usda.gov", "healthline.com"],
    "mental health": ["nimh.nih.gov", "cdc.gov", "mayoclinic.org"],
    "anxiety": ["nimh.nih.gov", "mayoclinic.org"],
    "depression": ["nimh.nih.gov", "mayoclinic.org"],
    "clinical trial": ["clinicaltrials.gov", "pubmed.ncbi.nlm.nih.gov"],
    "side effects": ["drugs.com", "medlineplus.gov"],
    "first aid": ["mayoclinic.org", "cdc.gov"],
    "pregnancy": ["medlineplus.gov", "cdc.gov", "mayoclinic.org"],

    # Science & research
    "science": ["arxiv.org", "nature.com", "pubmed.ncbi.nlm.nih.gov"],
    "research": ["arxiv.org", "scholar.google.com", "semanticscholar.org"],
    "paper": ["arxiv.org", "scholar.google.com", "semanticscholar.org"],
    "physics": ["arxiv.org", "wikipedia.org"],
    "chemistry": ["pubchem.ncbi.nlm.nih.gov", "wikipedia.org"],
    "biology": ["ncbi.nlm.nih.gov", "wikipedia.org"],
    "astronomy": ["nasa.gov", "arxiv.org", "space.com"],
    "outer space": ["nasa.gov", "space.com"],
    "space exploration": ["nasa.gov", "space.com"],
    "climate change": ["nasa.gov", "noaa.gov", "epa.gov"],
    "ocean": ["noaa.gov", "nasa.gov"],
    "endangered species": ["iucnredlist.org", "wikipedia.org"],
    "evolution": ["wikipedia.org", "ncbi.nlm.nih.gov"],
    "geology": ["earthquake.usgs.gov", "wikipedia.org"],
    "energy": ["energy.gov", "epa.gov"],

    # General news
    "us news": ["apnews.com", "npr.org", "propublica.org"],
    "world news": ["apnews.com", "bbc.com", "aljazeera.com"],
    "investigative": ["propublica.org"],
    "congress": ["congress.gov", "c-span.org"],
    "united nations": ["un.org"],
    "foreign policy": ["cfr.org", "brookings.edu"],

    # Government & law
    "tax": ["irs.gov"],
    "taxes": ["irs.gov"],
    "tax return": ["irs.gov"],
    "law": ["law.cornell.edu", "congress.gov", "supremecourt.gov"],
    "legal": ["law.cornell.edu", "findlaw.com"],
    "regulation": ["federalregister.gov", "law.cornell.edu"],
    "bill": ["congress.gov"],
    "patent": ["patents.google.com", "uspto.gov"],
    "supreme court": ["supremecourt.gov", "law.cornell.edu"],
    "passport": ["travel.state.gov"],
    "visa": ["travel.state.gov"],
    "social security": ["ssa.gov"],
    "veterans": ["va.gov"],
    "small business": ["sba.gov"],
    "fda": ["fda.gov"],
    "consumer protection": ["ftc.gov"],
    "government data": ["data.gov", "census.gov"],

    # Finance & economics
    "stock": ["finance.yahoo.com", "sec.gov"],
    "stocks": ["finance.yahoo.com", "sec.gov"],
    "stock market": ["finance.yahoo.com", "marketwatch.com"],
    "investing": ["investopedia.com", "sec.gov"],
    "inflation": ["bls.gov", "fred.stlouisfed.org"],
    "unemployment": ["bls.gov", "fred.stlouisfed.org"],
    "gdp": ["bls.gov", "fred.stlouisfed.org", "worldbank.org"],
    "economic": ["fred.stlouisfed.org", "bls.gov", "worldbank.org"],
    "economy": ["fred.stlouisfed.org", "bls.gov"],
    "currency": ["xe.com"],
    "exchange rate": ["xe.com"],
    "cryptocurrency": ["coinmarketcap.com", "coingecko.com"],
    "bitcoin": ["coinmarketcap.com", "bitcoin.org"],
    "ethereum": ["coinmarketcap.com", "coingecko.com"],
    "crypto": ["coinmarketcap.com", "coingecko.com"],
    "mortgage": ["bankrate.com", "nerdwallet.com"],
    "credit card": ["nerdwallet.com", "bankrate.com"],
    "credit score": ["nerdwallet.com"],
    "interest rate": ["fred.stlouisfed.org", "bankrate.com"],
    "budget": ["cbo.gov", "investopedia.com"],
    "retirement": ["ssa.gov", "investopedia.com"],
    "401k": ["investopedia.com", "irs.gov"],

    # Food & cooking
    "recipe": ["allrecipes.com", "seriouseats.com", "budgetbytes.com"],
    "cooking": ["allrecipes.com", "seriouseats.com", "bonappetit.com"],
    "baking": ["kingarthurbaking.com", "allrecipes.com"],
    "food nutrition": ["fdc.nal.usda.gov"],
    "meal prep": ["budgetbytes.com", "allrecipes.com"],
    "vegan recipe": ["allrecipes.com", "budgetbytes.com"],
    "vegetarian recipe": ["allrecipes.com", "epicurious.com"],

    # Math & reference
    "math": ["mathworld.wolfram.com", "wikipedia.org"],
    "equation": ["mathworld.wolfram.com", "wikipedia.org"],
    "statistics": ["wikipedia.org", "ourworldindata.org"],
    "data visualization": ["ourworldindata.org"],
    "definition": ["merriam-webster.com", "wikipedia.org"],
    "etymology": ["etymonline.com", "merriam-webster.com"],
    "synonym": ["merriam-webster.com", "thesaurus.com"],
    "thesaurus": ["merriam-webster.com"],
    "translate": ["translate.google.com", "deepl.com"],
    "translation": ["translate.google.com", "deepl.com"],
    "philosophy": ["plato.stanford.edu", "wikipedia.org"],
    "history": ["wikipedia.org", "britannica.com", "archive.org"],
    "world war": ["wikipedia.org", "britannica.com"],
    "biography": ["wikipedia.org", "britannica.com"],
    "fact check": ["factcheck.org", "snopes.com", "politifact.com"],
    "is it true": ["factcheck.org", "snopes.com"],
    "hoax": ["snopes.com"],
    "world population": ["worldometers.info", "census.gov"],

    # Sports
    "baseball": ["baseball-reference.com", "espn.com", "mlb.com"],
    "basketball": ["basketball-reference.com", "espn.com", "nba.com"],
    "football": ["pro-football-reference.com", "espn.com", "nfl.com"],
    "soccer": ["transfermarkt.com", "fbref.com", "espn.com"],
    "hockey": ["hockey-reference.com", "espn.com"],
    "cricket": ["cricinfo.com"],
    "nfl": ["pro-football-reference.com", "espn.com", "nfl.com"],
    "nba": ["basketball-reference.com", "espn.com", "nba.com"],
    "mlb": ["baseball-reference.com", "espn.com", "mlb.com"],
    "nhl": ["hockey-reference.com", "espn.com"],
    "world cup": ["fbref.com", "transfermarkt.com"],
    "premier league": ["fbref.com", "transfermarkt.com"],
    "formula 1": ["racing-reference.info"],
    "f1": ["racing-reference.info"],
    "olympics": ["wikipedia.org", "espn.com"],
    "sports stats": ["sports-reference.com"],

    # Time & geography
    "time zone": ["timeanddate.com"],
    "timezone": ["timeanddate.com"],
    "what time": ["timeanddate.com"],
    "sunrise": ["timeanddate.com"],
    "sunset": ["timeanddate.com"],
    "calendar": ["timeanddate.com"],
    "population": ["census.gov", "worldbank.org", "worldometers.info"],
    "cost of living": ["numbeo.com"],
    "country info": ["cia.gov", "worldbank.org"],
    "map": ["openstreetmap.org"],
    "coordinates": ["openstreetmap.org", "geonames.org"],
    "travel advisory": ["travel.state.gov"],
    "flight": ["flightaware.com"],

    # Entertainment
    "movie": ["imdb.com", "rottentomatoes.com", "letterboxd.com"],
    "film": ["imdb.com", "rottentomatoes.com"],
    "tv show": ["imdb.com", "rottentomatoes.com"],
    "television": ["imdb.com", "rottentomatoes.com"],
    "book": ["goodreads.com", "gutenberg.org"],
    "book review": ["goodreads.com"],
    "game": ["metacritic.com", "igdb.com"],
    "video game": ["metacritic.com", "igdb.com", "pcgamingwiki.com"],
    "board game": ["boardgamegeek.com"],
    "anime": ["myanimelist.net", "anilist.co"],
    "manga": ["myanimelist.net", "anilist.co"],
    "lyrics": ["genius.com"],
    "song": ["genius.com", "musicbrainz.org"],
    "album": ["musicbrainz.org", "allmusic.com"],
    "artist": ["musicbrainz.org", "allmusic.com", "discogs.com"],
    "band": ["musicbrainz.org", "allmusic.com"],
    "music": ["musicbrainz.org", "allmusic.com"],
    "how long to beat": ["howlongtobeat.com"],

    # Security
    "security": ["owasp.org", "cisa.gov", "nvd.nist.gov"],
    "vulnerability": ["nvd.nist.gov", "cve.mitre.org"],
    "cve": ["cve.mitre.org", "nvd.nist.gov"],
    "owasp": ["owasp.org"],
    "web security": ["owasp.org", "portswigger.net"],
    "exploit": ["exploit-db.com", "nvd.nist.gov"],
    "cybersecurity": ["cisa.gov", "owasp.org"],

    # Education
    "tutorial": ["w3schools.com", "developer.mozilla.org"],
    "learn programming": ["w3schools.com", "khanacademy.org"],
    "online course": ["coursera.org", "edx.org"],
    "textbook": ["openstax.org", "ocw.mit.edu"],
    "mit course": ["ocw.mit.edu"],

    # Legal
    "case law": ["courtlistener.com", "justia.com", "law.cornell.edu"],
    "federal regulation": ["ecfr.gov", "regulations.gov", "federalregister.gov"],
    "eu law": ["eur-lex.europa.eu"],
    "uk law": ["legislation.gov.uk"],
    "court opinion": ["courtlistener.com", "law.cornell.edu"],

    # Housing & real estate
    "housing": ["hud.gov", "huduser.gov"],
    "mortgage rates": ["freddiemac.com", "bankrate.com"],
    "fair market rent": ["huduser.gov"],

    # Open data
    "open data": ["data.gov", "data.europa.eu", "data.gov.uk", "datacommons.org"],
    "dataset": ["datacommons.org", "registry.opendata.aws", "data.gov"],
    "public data": ["data.gov", "datacommons.org"],

    # Accessibility
    "accessibility": ["webaim.org", "a11yproject.com", "w3.org"],
    "wcag": ["webaim.org", "w3.org"],
    "a11y": ["webaim.org", "a11yproject.com"],
    "screen reader": ["webaim.org", "a11yproject.com"],

    # Biology & nature
    "species": ["gbif.org", "iucnredlist.org"],
    "biodiversity": ["gbif.org", "iucnredlist.org"],
    "taxonomy": ["gbif.org", "wikipedia.org"],

    # Library & archives
    "library": ["loc.gov", "openlibrary.org"],
    "newspaper archive": ["chroniclingamerica.loc.gov", "archive.org"],
    "historic newspaper": ["chroniclingamerica.loc.gov"],

    # Business & company
    "company info": ["opencorporates.com", "sec.gov"],
    "sec filing": ["sec.report", "sec.gov"],
    "annual report": ["sec.report", "sec.gov"],

    # Academic
    "open access": ["doaj.org", "core.ac.uk", "pmc.ncbi.nlm.nih.gov"],
    "open textbook": ["libretexts.org", "openstax.org"],
    "preprint": ["arxiv.org", "ssrn.com"],
    "pep": ["peps.python.org"],

    # Developer reference
    "cheat sheet": ["cheat.sh", "devdocs.io"],
    "developer roadmap": ["roadmap.sh"],
    "twelve factor": ["12factor.net"],

    # Astronomy
    "star catalog": ["simbad.u-strasbg.fr"],
    "star": ["simbad.u-strasbg.fr", "nasa.gov"],

    # Standards & specs
    "rfc": ["rfc-editor.org", "datatracker.ietf.org"],
    "w3c": ["w3.org"],
    "specification": ["w3.org", "rfc-editor.org"],
    "unicode": ["unicode.org"],
    "ecmascript": ["tc39.es", "ecma-international.org"],
}


# ---------------------------------------------------------------------------
# Backward-compatible quality lookup (flat dict for sort_urls_by_quality)
# ---------------------------------------------------------------------------

_DOMAIN_QUALITY: dict[str, int] = {
    domain: info.quality for domain, info in _SOURCES.items()
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_source_info(url: str) -> SourceInfo | None:
    """Return full metadata for a URL's domain, or None if not in registry."""
    host = _extract_host(url)
    if not host:
        return None
    parts = host.split(".")
    for i in range(len(parts) - 1):
        candidate = ".".join(parts[i:])
        if candidate in _SOURCES:
            return _SOURCES[candidate]
    return None


def domain_quality(url: str) -> int:
    """Return the quality tier for a URL's domain.

    Checks the full domain first, then walks up subdomains.
    e.g. "api.weather.gov" → checks "api.weather.gov", then "weather.gov".
    """
    host = _extract_host(url)
    if not host:
        return UNKNOWN
    parts = host.split(".")
    for i in range(len(parts) - 1):
        candidate = ".".join(parts[i:])
        if candidate in _DOMAIN_QUALITY:
            return _DOMAIN_QUALITY[candidate]
    return UNKNOWN


def sort_urls_by_quality(urls: list[str]) -> list[str]:
    """Sort URLs by domain quality tier (best first), preserving order within tiers."""
    return sorted(urls, key=lambda u: -domain_quality(u))


def sort_urls_by_quality_with_diversity(
    urls: list[str], per_domain_cap: int = 2,
) -> list[str]:
    """Quality-ranked URLs with per-domain diversity.

    Prevents a single authoritative domain from dominating the top slots.
    After quality sort, keeps at most `per_domain_cap` URLs per domain in
    the primary block; the rest are appended in quality order so nothing
    is lost — just deprioritized.

    Why: curated registries concentrate results on a few EXCELLENT domains
    (three wikipedia pages in a row). Users want authoritative *and* varied.
    """
    if per_domain_cap < 1:
        per_domain_cap = 1
    ranked = sorted(urls, key=lambda u: -domain_quality(u))
    seen: dict[str, int] = {}
    diversified: list[str] = []
    overflow: list[str] = []
    for url in ranked:
        host = _extract_host(url)
        count = seen.get(host, 0)
        if count < per_domain_cap:
            diversified.append(url)
            seen[host] = count + 1
        else:
            overflow.append(url)
    return diversified + overflow


def merge_learned_reputation(reputation: dict[str, int]) -> int:
    """Merge learned domain reputation scores into the quality lookup.

    Maps reputation scores (from domain_reputation SQLite table) to
    quality tiers and updates _DOMAIN_QUALITY. Learned scores override
    static curated tiers for domains that have enough data.

    Only overrides if the learned score is decisive (>=3 fetches or
    a user action). This prevents a single failed fetch from demoting
    a curated EXCELLENT source.

    Returns the number of domains updated.
    """
    updated = 0
    for domain, score in reputation.items():
        # Map reputation score → quality tier
        if score >= 5:
            tier = EXCELLENT
        elif score >= 2:
            tier = GOOD
        elif score >= 0:
            continue  # not enough signal — keep the curated tier
        else:
            tier = AVOID

        current = _DOMAIN_QUALITY.get(domain)
        if current != tier:
            _DOMAIN_QUALITY[domain] = tier
            updated += 1

    return updated


def _query_matches_topic_keyword(query_lower: str, query_tokens: set[str], keyword: str) -> bool:
    """Return True when a topic keyword matches as a real token or phrase.

    Short keywords like ``api`` and ``tax`` should not fire inside unrelated
    words (e.g. ``rapid`` or ``syntax``). For simple alphanumeric keywords we
    require exact token matches. Keywords with punctuation keep a regex-based
    phrase match so entries like ``c++`` and ``next.js`` still work.
    """
    normalized = keyword.lower().strip()
    if not normalized:
        return False

    if re.fullmatch(r"[a-z0-9 ]+", normalized):
        parts = normalized.split()
        if len(parts) == 1:
            return parts[0] in query_tokens
        if any(part not in query_tokens for part in parts):
            return False
        pattern = rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])"
        return re.search(pattern, query_lower) is not None

    escaped = re.escape(normalized).replace(r"\ ", r"\s+")
    pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
    return re.search(pattern, query_lower) is not None


def get_topic_sites(query: str) -> list[str]:
    """Return recommended site domains for a query based on topic keywords.

    Matches multi-word phrases first, then single words.
    Results are deduplicated while preserving order.
    """
    query_lower = query.lower()
    query_tokens = set(re.findall(r"[a-z0-9]+", query_lower))
    sites: list[str] = []
    seen: set[str] = set()

    # Sort by key length descending so multi-word phrases match first
    for keyword in sorted(_TOPIC_SITES, key=len, reverse=True):
        if _query_matches_topic_keyword(query_lower, query_tokens, keyword):
            for d in _TOPIC_SITES[keyword]:
                if d not in seen:
                    sites.append(d)
                    seen.add(d)
    return sites


def get_sources_by_category(category: str) -> list[tuple[str, SourceInfo]]:
    """Return all sources matching a category, sorted by quality."""
    matches = [
        (domain, info)
        for domain, info in _SOURCES.items()
        if category in info.categories
    ]
    matches.sort(key=lambda x: -x[1].quality)
    return matches


def describe_source(url: str) -> str:
    """Return a human-readable description of a source for AI context injection.

    Returns empty string if the domain is unknown.
    """
    info = get_source_info(url)
    if not info:
        return ""
    host = _extract_host(url) or url
    parts = []
    if info.categories:
        parts.append(f"categories: {', '.join(info.categories)}")
    parts.append(f"content: {info.content_type}")
    parts.append(f"freshness: {info.freshness}")
    if info.has_paywall:
        parts.append("paywalled")
    if info.requires_js:
        parts.append("requires-js")
    if info.notes:
        parts.append(info.notes)
    quality_label = {EXCELLENT: "excellent", GOOD: "good", AVOID: "avoid"}.get(
        info.quality, "unknown"
    )
    return f"[{host}] quality={quality_label}, {', '.join(parts)}"


def get_registry_stats() -> dict:
    """Return summary stats about the source registry."""
    total = len(_SOURCES)
    by_quality = {
        "excellent": sum(1 for s in _SOURCES.values() if s.quality == EXCELLENT),
        "good": sum(1 for s in _SOURCES.values() if s.quality == GOOD),
        "avoid": sum(1 for s in _SOURCES.values() if s.quality == AVOID),
    }
    all_categories: set[str] = set()
    for info in _SOURCES.values():
        all_categories.update(info.categories)
    return {
        "total_sources": total,
        "by_quality": by_quality,
        "total_categories": len(all_categories),
        "total_topic_mappings": len(_TOPIC_SITES),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_host(url: str) -> str:
    """Extract lowercase hostname from a URL, stripping port."""
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ""
    if ":" in host:
        host = host.rsplit(":", 1)[0]
    return host
