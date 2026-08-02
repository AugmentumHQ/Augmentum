"""Test candidate sources round 2 — filling gaps in real estate, jobs, shopping,
research/academic, AI/ML, medical research, legal, automotive, pets, gardening,
DIY/home improvement, and more.
"""

from __future__ import annotations

import asyncio
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


CANDIDATES = [
    # ===== REAL ESTATE (currently 1) =====
    ("realtor.com", "https://www.realtor.com/", "real estate,housing,prices", "data", "realtime"),
    ("apartments.com", "https://www.apartments.com/", "real estate,housing,rental", "data", "realtime"),
    ("trulia.com", "https://www.trulia.com/", "real estate,housing,prices", "data", "realtime"),
    ("rentcafe.com", "https://www.rentcafe.com/", "real estate,housing,rental", "data", "realtime"),

    # ===== JOBS & SALARY (currently 1) =====
    ("salary.com", "https://www.salary.com/", "jobs,salary,employment", "data", "daily"),
    ("payscale.com", "https://www.payscale.com/", "jobs,salary,employment", "data", "daily"),
    ("ziprecruiter.com", "https://www.ziprecruiter.com/", "jobs,employment", "data", "daily"),
    ("builtin.com", "https://builtin.com/", "jobs,technology,startups", "article", "daily"),
    ("wellfound.com", "https://wellfound.com/", "jobs,technology,startups", "data", "daily"),
    ("remote.co", "https://remote.co/", "jobs,remote,employment", "article", "daily"),
    ("weworkremotely.com", "https://weworkremotely.com/", "jobs,remote,employment", "data", "daily"),

    # ===== SHOPPING & PRICES (currently 1) =====
    ("pricerunner.com", "https://www.pricerunner.com/", "shopping,prices,comparison", "data", "realtime"),
    ("shopzilla.com", "https://www.shopzilla.com/", "shopping,prices,comparison", "data", "realtime"),
    ("wirecutter.com", "https://www.nytimes.com/wirecutter/", "shopping,reviews,recommendations", "article", "daily"),
    ("consumersearch.com", "https://www.consumersearch.com/", "shopping,reviews,comparison", "article", "static"),
    ("bestproducts.com", "https://www.bestproducts.com/", "shopping,reviews", "article", "daily"),
    ("buymeonce.com", "https://buymeonce.com/", "shopping,reviews,sustainability", "article", "static"),

    # ===== AI & MACHINE LEARNING RESEARCH =====
    ("huggingface.co", "https://huggingface.co/", "ai,ml,models,research", "data", "daily"),
    ("paperswithcode.com", "https://paperswithcode.com/", "ai,ml,research,papers,benchmarks", "data", "daily"),
    ("openreview.net", "https://openreview.net/", "ai,ml,research,papers", "article", "daily"),
    ("deepmind.google", "https://deepmind.google/", "ai,ml,research", "article", "daily"),
    ("research.google", "https://research.google/", "ai,ml,research,technology", "article", "daily"),
    ("ai.meta.com", "https://ai.meta.com/", "ai,ml,research", "article", "daily"),
    ("ollama.com", "https://ollama.com/", "ai,ml,models,technology", "data", "daily"),
    ("lmsys.org", "https://lmsys.org/", "ai,ml,benchmarks,models", "data", "daily"),
    ("artificialintelligence-news.com", "https://www.artificialintelligence-news.com/", "ai,ml,news,technology", "news", "daily"),
    ("therundown.ai", "https://www.therundown.ai/", "ai,news,technology", "news", "daily"),

    # ===== MEDICAL & HEALTH RESEARCH =====
    ("pubmed.ncbi.nlm.nih.gov", "https://pubmed.ncbi.nlm.nih.gov/", "medical,research,papers,health", "data", "daily"),
    ("ncbi.nlm.nih.gov", "https://www.ncbi.nlm.nih.gov/", "medical,research,health,genomics", "data", "daily"),
    ("who.int", "https://www.who.int/", "health,medical,global,disease", "reference", "daily"),
    ("drugs.com", "https://www.drugs.com/", "medical,drugs,pharmacy,health", "reference", "static"),
    ("rxlist.com", "https://www.rxlist.com/", "medical,drugs,pharmacy", "reference", "static"),
    ("uptodate.com", "https://www.uptodate.com/", "medical,treatment,research", "reference", "daily"),
    ("merckmanuals.com", "https://www.merckmanuals.com/", "medical,reference,treatment", "reference", "static"),
    ("psychologytoday.com", "https://www.psychologytoday.com/", "health,mental health,psychology", "article", "daily"),
    ("nih.gov", "https://www.nih.gov/", "medical,research,health,government", "reference", "daily"),
    ("cdc.gov", "https://www.cdc.gov/", "health,medical,disease,government", "reference", "daily"),

    # ===== LEGAL RESEARCH =====
    ("law.cornell.edu", "https://www.law.cornell.edu/", "law,legal,legislation,reference", "reference", "static"),
    ("courtlistener.com", "https://www.courtlistener.com/", "law,legal,courts", "data", "daily"),
    ("casetext.com", "https://casetext.com/", "law,legal,research", "reference", "daily"),
    ("lawinsider.com", "https://www.lawinsider.com/", "law,legal,contracts", "reference", "static"),
    ("nolo.com", "https://www.nolo.com/", "law,legal,reference", "article", "static"),
    ("avvo.com", "https://www.avvo.com/", "law,legal,directory", "article", "static"),
    ("legalmatch.com", "https://www.legalmatch.com/", "law,legal,directory", "article", "static"),
    ("scotusblog.com", "https://www.scotusblog.com/", "law,legal,courts,politics", "news", "daily"),

    # ===== ACADEMIC / RESEARCH (general) =====
    ("scholar.google.com", "https://scholar.google.com/", "research,papers,academic", "data", "daily"),
    ("semanticscholar.org", "https://www.semanticscholar.org/", "research,papers,academic", "data", "daily"),
    ("researchgate.net", "https://www.researchgate.net/", "research,papers,academic", "article", "daily"),
    ("jstor.org", "https://www.jstor.org/", "research,papers,academic", "article", "static"),
    ("sciencedirect.com", "https://www.sciencedirect.com/", "research,papers,science,medical", "article", "daily"),
    ("nature.com", "https://www.nature.com/", "research,papers,science", "article", "daily"),
    ("science.org", "https://www.science.org/", "research,papers,science", "article", "daily"),
    ("peerj.com", "https://peerj.com/", "research,papers,science,open access", "article", "daily"),
    ("biorxiv.org", "https://www.biorxiv.org/", "research,papers,biology,science", "article", "daily"),
    ("medrxiv.org", "https://www.medrxiv.org/", "research,papers,medical,science", "article", "daily"),

    # ===== AUTOMOTIVE =====
    ("edmunds.com", "https://www.edmunds.com/", "automotive,cars,reviews,prices", "data", "daily"),
    ("caranddriver.com", "https://www.caranddriver.com/", "automotive,cars,reviews", "article", "daily"),
    ("motortrend.com", "https://www.motortrend.com/", "automotive,cars,reviews", "article", "daily"),
    ("kbb.com", "https://www.kbb.com/", "automotive,cars,prices", "data", "daily"),
    ("fueleconomy.gov", "https://www.fueleconomy.gov/", "automotive,cars,data,government", "data", "static"),
    ("autoblog.com", "https://www.autoblog.com/", "automotive,cars,news,reviews", "news", "daily"),
    ("carcomplaints.com", "https://www.carcomplaints.com/", "automotive,cars,reviews,safety", "data", "daily"),

    # ===== HOME & DIY =====
    ("homedepot.com", "https://www.homedepot.com/", "home,diy,shopping,prices", "data", "realtime"),
    ("lowes.com", "https://www.lowes.com/", "home,diy,shopping,prices", "data", "realtime"),
    ("familyhandyman.com", "https://www.familyhandyman.com/", "home,diy,repair,reference", "article", "static"),
    ("bobvila.com", "https://www.bobvila.com/", "home,diy,repair,reference", "article", "static"),
    ("thisoldhouse.com", "https://www.thisoldhouse.com/", "home,diy,repair", "article", "static"),
    ("instructables.com", "https://www.instructables.com/", "diy,crafts,projects,reference", "article", "static"),

    # ===== GARDENING & PETS =====
    ("almanac.com", "https://www.almanac.com/", "gardening,weather,reference,agriculture", "reference", "daily"),
    ("gardeningknowhow.com", "https://www.gardeningknowhow.com/", "gardening,plants,reference", "article", "static"),
    ("petmd.com", "https://www.petmd.com/", "pets,veterinary,health", "article", "static"),
    ("akc.org", "https://www.akc.org/", "pets,dogs,reference", "reference", "static"),
    ("aspca.org", "https://www.aspca.org/", "pets,veterinary,safety", "reference", "static"),

    # ===== SCIENCE & DATA (filling gaps) =====
    ("ourworldindata.org", "https://ourworldindata.org/", "data,statistics,research,global", "data", "daily"),
    ("statista.com", "https://www.statista.com/", "data,statistics,research,business", "data", "daily"),
    ("data.worldbank.org", "https://data.worldbank.org/", "data,statistics,economics,global", "data", "daily"),
    ("fred.stlouisfed.org", "https://fred.stlouisfed.org/", "data,statistics,economics,finance", "data", "realtime"),
    ("usgs.gov", "https://www.usgs.gov/", "science,geology,earthquakes,environment,government", "data", "daily"),
    ("space.com", "https://www.space.com/", "science,space,astronomy,news", "news", "daily"),
    ("livescience.com", "https://www.livescience.com/", "science,news,reference", "article", "daily"),
    ("phys.org", "https://phys.org/", "science,research,news", "news", "daily"),
    ("sciencenews.org", "https://www.sciencenews.org/", "science,research,news", "news", "daily"),

    # ===== PERSONAL FINANCE (filling gaps) =====
    ("nerdwallet.com", "https://www.nerdwallet.com/", "personal finance,credit,banking", "article", "daily"),
    ("bankrate.com", "https://www.bankrate.com/", "personal finance,rates,banking,credit", "data", "daily"),
    ("creditkarma.com", "https://www.creditkarma.com/", "personal finance,credit", "article", "daily"),
    ("investopedia.com", "https://www.investopedia.com/", "finance,investing,reference,education", "reference", "static"),
]


async def main():
    tool = WebFetchTool()
    print(f"Testing {len(CANDIDATES)} candidate sources...\n")

    batch_size = 10
    ok = []
    fail = []

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
                return (domain, "TIMEOUT", 0, cats, ctype, fresh, "")
            except Exception as e:
                return (domain, "ERROR", 0, cats, ctype, fresh, str(e)[:60])

        results = await asyncio.gather(*[test_one(e) for e in batch])
        for domain, st, chars, cats, ctype, fresh, preview in results:
            icon = "OK" if st == "OK" else st
            print(f"  [{icon:>7}]  {domain:<40} {chars:>5}ch  {preview[:45]}")
            if st == "OK":
                ok.append((domain, cats, ctype, fresh))
            else:
                fail.append((domain, st))

        if i + batch_size < len(CANDIDATES):
            await asyncio.sleep(0.5)

    print(f"\n{'='*80}")
    print(f"PASSED: {len(ok)}  |  FAILED: {len(fail)}")
    print(f"\nFAILED:")
    for d, st in fail:
        print(f"  {d:<40} ({st})")


if __name__ == "__main__":
    asyncio.run(main())
