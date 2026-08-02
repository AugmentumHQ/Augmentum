"""Test candidate sources for the preferred_sources registry.

Tests a batch of URLs across categories we're thin on:
sports, finance, recipes, health, entertainment, local news, shopping, travel.
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from augmentum.tools.web_fetch import WebFetchTool

_BLOCK_SIGS = [
    "just a moment", "enable javascript and cookies", "checking your browser",
    "cloudflare", "ray id", "access denied", "403 forbidden", "captcha",
    "are you a robot", "please verify", "bot detection",
]


def _status(content: str, chars: int) -> str:
    lower = (content or "")[:500].lower()
    if chars < 50:
        return "EMPTY"
    if any(s in lower for s in _BLOCK_SIGS) and chars < 500:
        return "BLOCKED"
    return "OK"


# (domain, test_url, proposed_categories, proposed_content_type, proposed_freshness)
CANDIDATES = [
    # ===== SPORTS SCORES & STATS =====
    ("espn.com", "https://www.espn.com/nfl/scoreboard", "sports,scores,nfl,nba,mlb", "news", "realtime"),
    ("cbssports.com", "https://www.cbssports.com/nfl/scores/", "sports,scores,nfl,nba", "news", "realtime"),
    ("sports.yahoo.com", "https://sports.yahoo.com/", "sports,scores,news", "news", "realtime"),
    ("sofascore.com", "https://www.sofascore.com/", "sports,scores,soccer,football", "data", "realtime"),
    ("flashscore.com", "https://www.flashscore.com/", "sports,scores", "data", "realtime"),
    ("nfl.com", "https://www.nfl.com/scores/", "sports,scores,nfl,football", "data", "realtime"),
    ("mlb.com", "https://www.mlb.com/scores", "sports,scores,mlb,baseball", "data", "realtime"),
    ("nhl.com", "https://www.nhl.com/scores", "sports,scores,hockey", "data", "realtime"),
    ("basketball-reference.com", "https://www.basketball-reference.com/", "sports,scores,nba,basketball,statistics", "data", "daily"),
    ("baseball-reference.com", "https://www.baseball-reference.com/", "sports,scores,mlb,baseball,statistics", "data", "daily"),
    ("hockey-reference.com", "https://www.hockey-reference.com/", "sports,scores,hockey,statistics", "data", "daily"),
    ("soccerway.com", "https://int.soccerway.com/", "sports,scores,soccer", "data", "realtime"),

    # ===== FINANCE & STOCKS =====
    ("finance.yahoo.com", "https://finance.yahoo.com/", "finance,stocks,markets,investing", "data", "realtime"),
    ("marketwatch.com", "https://www.marketwatch.com/", "finance,stocks,markets,news", "news", "realtime"),
    ("google.com/finance", "https://www.google.com/finance/", "finance,stocks,markets", "data", "realtime"),
    ("cnbc.com", "https://www.cnbc.com/", "finance,stocks,markets,news,business", "news", "realtime"),
    ("fool.com", "https://www.fool.com/", "finance,stocks,investing", "article", "daily"),
    ("seekingalpha.com", "https://seekingalpha.com/", "finance,stocks,investing", "article", "daily"),
    ("stockanalysis.com", "https://stockanalysis.com/", "finance,stocks,data,markets", "data", "realtime"),
    ("finviz.com", "https://finviz.com/", "finance,stocks,data,markets", "data", "realtime"),
    ("coinmarketcap.com", "https://coinmarketcap.com/", "finance,cryptocurrency,markets", "data", "realtime"),
    ("coingecko.com", "https://www.coingecko.com/", "finance,cryptocurrency,markets", "data", "realtime"),

    # ===== RECIPES & COOKING =====
    ("allrecipes.com", "https://www.allrecipes.com/", "recipe,cooking,food", "article", "static"),
    ("simplyrecipes.com", "https://www.simplyrecipes.com/", "recipe,cooking,food", "article", "static"),
    ("budgetbytes.com", "https://www.budgetbytes.com/", "recipe,cooking,food,budget", "article", "static"),
    ("seriouseats.com", "https://www.seriouseats.com/", "recipe,cooking,food", "article", "static"),
    ("epicurious.com", "https://www.epicurious.com/", "recipe,cooking,food", "article", "static"),
    ("bonappetit.com", "https://www.bonappetit.com/", "recipe,cooking,food", "article", "static"),
    ("cookieandkate.com", "https://cookieandkate.com/", "recipe,cooking,food", "article", "static"),
    ("kingarthurbaking.com", "https://www.kingarthurbaking.com/recipes", "recipe,baking,food", "article", "static"),

    # ===== HEALTH & MEDICINE =====
    ("mayoclinic.org", "https://www.mayoclinic.org/", "health,medical,symptoms,treatment", "reference", "static"),
    ("webmd.com", "https://www.webmd.com/", "health,medical,symptoms", "article", "static"),
    ("healthline.com", "https://www.healthline.com/", "health,medical,nutrition,fitness", "article", "daily"),
    ("medlineplus.gov", "https://medlineplus.gov/", "health,medical,drugs,treatment", "reference", "static"),
    ("clevelandclinic.org", "https://my.clevelandclinic.org/health", "health,medical,treatment", "reference", "static"),
    ("nutritiondata.self.com", "https://nutritiondata.self.com/", "nutrition,food,health,data", "data", "static"),

    # ===== ENTERTAINMENT =====
    ("rottentomatoes.com", "https://www.rottentomatoes.com/", "movies,entertainment,reviews", "data", "daily"),
    ("metacritic.com", "https://www.metacritic.com/", "movies,games,entertainment,reviews", "data", "daily"),
    ("letterboxd.com", "https://letterboxd.com/", "movies,entertainment,reviews", "article", "daily"),
    ("themoviedb.org", "https://www.themoviedb.org/", "movies,entertainment,data", "data", "daily"),
    ("thetvdb.com", "https://thetvdb.com/", "tv,entertainment,data", "data", "daily"),
    ("last.fm", "https://www.last.fm/", "music,entertainment", "data", "realtime"),
    ("setlist.fm", "https://www.setlist.fm/", "music,concerts,entertainment", "data", "daily"),

    # ===== TRAVEL =====
    ("tripadvisor.com", "https://www.tripadvisor.com/", "travel,reviews,lodging,food", "article", "daily"),
    ("lonelyplanet.com", "https://www.lonelyplanet.com/", "travel,tourism", "article", "static"),
    ("seat61.com", "https://www.seat61.com/", "travel,transportation", "article", "static"),
    ("timeanddate.com", "https://www.timeanddate.com/", "time,timezone,date,calendar", "data", "realtime"),

    # ===== SHOPPING & PRICES =====
    ("camelcamelcamel.com", "https://camelcamelcamel.com/", "shopping,prices,deals", "data", "realtime"),
    ("slickdeals.net", "https://slickdeals.net/", "shopping,deals,prices", "forum", "realtime"),

    # ===== EDUCATION & REFERENCE =====
    ("coursera.org", "https://www.coursera.org/", "education,courses,learning", "article", "static"),
    ("edx.org", "https://www.edx.org/", "education,courses,learning", "article", "static"),
    ("quora.com", "https://www.quora.com/", "qa,discussion,reference", "forum", "daily"),
    ("howstuffworks.com", "https://www.howstuffworks.com/", "reference,education,science", "article", "static"),

    # ===== WEATHER (fill gaps) =====
    ("wunderground.com", "https://www.wunderground.com/", "weather,forecast,climate", "data", "realtime"),
    ("accuweather.com", "https://www.accuweather.com/", "weather,forecast", "data", "realtime"),
    ("weatherspark.com", "https://weatherspark.com/", "weather,climate,data,statistics", "data", "static"),

    # ===== LOCAL NEWS / WIRE SERVICES =====
    ("upi.com", "https://www.upi.com/", "news,world", "news", "realtime"),
    ("thehill.com", "https://thehill.com/", "news,politics", "news", "realtime"),
    ("propublica.org", "https://www.propublica.org/", "news,investigative", "news", "daily"),
    ("pbs.org", "https://www.pbs.org/newshour/", "news,world,education", "news", "daily"),
    ("aljazeera.com", "https://www.aljazeera.com/", "news,world,international", "news", "realtime"),
    ("abcnews.go.com", "https://abcnews.go.com/", "news,world,politics", "news", "realtime"),

    # ===== REAL ESTATE =====
    ("zillow.com", "https://www.zillow.com/", "real estate,housing,prices", "data", "realtime"),
    ("redfin.com", "https://www.redfin.com/", "real estate,housing,prices", "data", "realtime"),

    # ===== JOBS =====
    ("glassdoor.com", "https://www.glassdoor.com/", "jobs,salary,reviews,employment", "data", "daily"),
    ("indeed.com", "https://www.indeed.com/", "jobs,employment", "data", "daily"),
    ("levels.fyi", "https://www.levels.fyi/", "jobs,salary,technology", "data", "daily"),
]


async def main():
    tool = WebFetchTool()

    print(f"Testing {len(CANDIDATES)} candidate sources...")
    print(f"{'Status':>8}  {'Domain':<35} {'Chars':>6}  {'Time':>5}  {'Preview'}")
    print("-" * 100)

    batch_size = 10
    ok_sites = []
    blocked_sites = []

    for i in range(0, len(CANDIDATES), batch_size):
        batch = CANDIDATES[i:i + batch_size]

        async def test_one(entry):
            domain, url, cats, ctype, fresh = entry
            try:
                result = await asyncio.wait_for(
                    tool.execute(url=url, max_chars=800),
                    timeout=10.0,
                )
                content = result.output or ""
                chars = len(content)
                st = _status(content, chars)
                preview = content[:60].replace("\n", " ")
                return (domain, st, chars, cats, ctype, fresh, preview)
            except asyncio.TimeoutError:
                return (domain, "TIMEOUT", 0, cats, ctype, fresh, "Timed out")
            except Exception as e:
                return (domain, "ERROR", 0, cats, ctype, fresh, str(e)[:60])

        results = await asyncio.gather(*[test_one(e) for e in batch])

        for domain, st, chars, cats, ctype, fresh, preview in results:
            icon = st
            print(f"  [{icon:>7}]  {domain:<35} {chars:>5}ch  {'':>5}  {preview[:50]}")
            if st == "OK":
                ok_sites.append((domain, cats, ctype, fresh))
            else:
                blocked_sites.append((domain, st))

        if i + batch_size < len(CANDIDATES):
            await asyncio.sleep(0.5)

    print(f"\n{'=' * 80}")
    print(f"ACCESSIBLE ({len(ok_sites)}):\n")
    for domain, cats, ctype, fresh in ok_sites:
        print(f'    "{domain}": SourceInfo(')
        print(f'        quality=GOOD,')
        print(f'        categories=({", ".join(repr(c) for c in cats.split(","))}),')
        print(f'        content_type="{ctype}",')
        print(f'        freshness="{fresh}",')
        print(f'    ),')

    print(f"\nBLOCKED/FAILED ({len(blocked_sites)}):")
    for domain, st in blocked_sites:
        print(f"    {domain:<35} ({st})")


if __name__ == "__main__":
    asyncio.run(main())
