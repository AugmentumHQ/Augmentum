"""RAG Pipeline v2 — LLM Generation Quality Tests

Tests whether the LLM correctly USES the injected document context.
Previous tests measured retrieval accuracy; this measures generation faithfulness.

Dimensions tested:
  A. Faithfulness — answer contains facts FROM the context
  B. Hallucination resistance — answer does NOT invent facts NOT in context
  C. Sufficiency response — LLM acknowledges when context is partial/absent
  D. Grounding instruction compliance — LLM cites the reference material
  E. Noise rejection — LLM ignores irrelevant chunks in the context

Requires LM Studio running on localhost:1234.
"""

from __future__ import annotations

import re

import httpx
import pytest

# ---------------------------------------------------------------------------
# LM Studio client
# ---------------------------------------------------------------------------

async def _lm_studio_available() -> list[str]:
    """Return list of available model IDs, or empty if unavailable."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get("http://localhost:1234/v1/models")
            return [m["id"] for m in resp.json().get("data", [])]
    except Exception:
        return []


async def _chat(model: str, system: str, user: str, temperature: float = 0.0) -> str:
    """Send a chat request to LM Studio, return the response content."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "http://localhost:1234/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": 300,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def _contains_any(text: str, keywords: list[str]) -> list[str]:
    """Return which keywords appear in text (case-insensitive word boundary)."""
    found = []
    for kw in keywords:
        pattern = re.escape(kw)
        if re.search(pattern, text, re.IGNORECASE):
            found.append(kw)
    return found


# ---------------------------------------------------------------------------
# Test context blocks (simulating what _build_document_context produces)
# ---------------------------------------------------------------------------

_SUFFICIENT_CONTEXT = """<reference_material>
[Document: contract.pdf p.3] (relevance: high)
The annual base compensation shall be One Hundred Forty-Five Thousand Dollars ($145,000),
payable in bi-weekly installments. The Employee shall be eligible for quarterly performance
bonuses of up to fifteen percent (15%) of the base compensation.

[Document: contract.pdf p.7] (relevance: high)
Either party may terminate this Agreement with ninety (90) days written notice. In the event
of termination without cause, the Employee shall receive severance equal to six (6) months
of base compensation plus accrued vacation time.
</reference_material>

Ground your response in the reference material above. Cite specific details when possible."""

_PARTIAL_CONTEXT = """<reference_material>
[Document: api_docs.md p.1] (relevance: moderate)
All API requests require a valid API key passed in the X-API-Key header.
OAuth 2.0 bearer tokens are also supported for user-delegated access.
</reference_material>

The reference material above may not fully address the query. Use it where relevant but indicate when you are drawing on general knowledge rather than the provided material."""

_NOISY_CONTEXT = """<reference_material>
[Document: recipes.md] (relevance: moderate)
Braise 2 pounds of beef chuck in red wine at 325 degrees Fahrenheit for 3 hours.
Add pearl onions, mushrooms, and carrots during the last hour. Deglaze the pan with
cognac before adding the wine. Season with thyme, bay leaf, and black pepper.

[Document: contract.pdf p.3] (relevance: high)
The annual base compensation shall be One Hundred Forty-Five Thousand Dollars ($145,000),
payable in bi-weekly installments.
</reference_material>

Ground your response in the reference material above. Cite specific details when possible."""

_EMPTY_CONTEXT_NOTE = """[context_note: The available documents do not appear to cover the topic of this query. The response should indicate that this information is not available in the provided documents.]"""


# ---------------------------------------------------------------------------
# Test runner per model
# ---------------------------------------------------------------------------

async def _run_generation_suite(model: str) -> dict:
    """Run all generation quality tests against a single model.

    Returns dict of {test_name: {passed: bool, details: str}}
    """
    results = {}

    # --- A. Faithfulness: answer contains facts from context ---

    # A1: Salary question with sufficient context
    answer = await _chat(
        model,
        system=_SUFFICIENT_CONTEXT,
        user="What is the employee's base compensation?",
    )
    facts_found = _contains_any(answer, ["$145,000", "145,000", "bi-weekly", "fifteen percent", "15%"])
    results["A1_salary_faithfulness"] = {
        "passed": len(facts_found) >= 1,
        "details": f"Found {facts_found} in answer ({len(answer)} chars)",
    }

    # A2: Termination question
    answer = await _chat(
        model,
        system=_SUFFICIENT_CONTEXT,
        user="What happens if the employee is terminated without cause?",
    )
    facts_found = _contains_any(answer, ["ninety", "90 days", "six months", "6 months", "severance", "vacation"])
    results["A2_termination_faithfulness"] = {
        "passed": len(facts_found) >= 2,
        "details": f"Found {facts_found}",
    }

    # --- B. Hallucination resistance: don't invent facts ---

    # B1: Ask about something NOT in the context
    answer = await _chat(
        model,
        system=_SUFFICIENT_CONTEXT,
        user="What is the employee's health insurance coverage?",
    )
    # The context says NOTHING about health insurance
    hallucination_markers = _contains_any(answer, ["PPO", "HMO", "deductible", "copay", "premium", "dental", "vision"])
    # Check if model acknowledges lack of info
    acknowledges = _contains_any(answer, ["not mentioned", "not specified", "not included", "doesn't mention",
                                          "does not mention", "no information", "not covered", "not addressed",
                                          "not available", "cannot find", "don't have"])
    results["B1_hallucination_resistance"] = {
        "passed": len(hallucination_markers) == 0 or len(acknowledges) > 0,
        "details": f"Hallucinations: {hallucination_markers}, Acknowledges gap: {acknowledges}",
    }

    # B2: Ask for a specific number not in context
    answer = await _chat(
        model,
        system=_SUFFICIENT_CONTEXT,
        user="How many vacation days does the employee get per year?",
    )
    # Context mentions "accrued vacation time" but NOT a specific number
    invented_numbers = re.findall(r'\b\d{1,2}\s*(?:days?|weeks?)\b', answer, re.IGNORECASE)
    acknowledges = _contains_any(answer, ["not specified", "not mentioned", "doesn't specify", "does not specify",
                                          "not stated", "accrued", "not provided"])
    results["B2_specific_number_hallucination"] = {
        "passed": len(invented_numbers) == 0 or len(acknowledges) > 0,
        "details": f"Invented numbers: {invented_numbers}, Acknowledges: {acknowledges}",
    }

    # --- C. Sufficiency response: acknowledge partial/missing context ---

    # C1: Partial context — should indicate uncertainty
    answer = await _chat(
        model,
        system=_PARTIAL_CONTEXT,
        user="What are all the API rate limits and pricing tiers?",
    )
    hedging = _contains_any(answer, ["not fully", "limited information", "based on the available",
                                     "document doesn't", "not mentioned", "additional",
                                     "not specified", "general knowledge", "beyond what",
                                     "only mentions", "doesn't cover", "not provided"])
    results["C1_partial_context_hedging"] = {
        "passed": len(hedging) > 0,
        "details": f"Hedging language: {hedging}",
    }

    # C2: No context — should clearly state info unavailable
    answer = await _chat(
        model,
        system=_EMPTY_CONTEXT_NOTE,
        user="What is the company's refund policy?",
    )
    unavailable = _contains_any(answer, ["not available", "no information", "not covered",
                                         "not found", "don't have", "cannot find",
                                         "not in the documents", "not provided",
                                         "not mentioned", "unable to find"])
    results["C2_no_context_acknowledgment"] = {
        "passed": len(unavailable) > 0,
        "details": f"Acknowledgment: {unavailable}",
    }

    # --- D. Grounding: references the document ---

    # D1: Answer should reference the source document
    answer = await _chat(
        model,
        system=_SUFFICIENT_CONTEXT,
        user="Summarize the key financial terms of this employment agreement.",
    )
    grounding = _contains_any(answer, ["contract", "agreement", "document", "reference material",
                                       "$145,000", "severance", "bonus"])
    results["D1_grounding_references"] = {
        "passed": len(grounding) >= 2,
        "details": f"Grounding terms: {grounding}",
    }

    # --- E. Noise rejection: ignore irrelevant chunks ---

    # E1: Ask about salary when context has recipe + salary chunks
    answer = await _chat(
        model,
        system=_NOISY_CONTEXT,
        user="What is the employee's annual compensation?",
    )
    salary_facts = _contains_any(answer, ["$145,000", "145,000", "bi-weekly"])
    recipe_leakage = _contains_any(answer, ["beef", "braise", "cognac", "thyme", "mushrooms", "wine"])
    results["E1_noise_rejection"] = {
        "passed": len(salary_facts) >= 1 and len(recipe_leakage) == 0,
        "details": f"Salary facts: {salary_facts}, Recipe leakage: {recipe_leakage}",
    }

    return results


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generation_quality():
    """Run generation quality suite against all available small models."""
    models = await _lm_studio_available()
    if not models:
        pytest.skip("LM Studio not available")

    target_models = ["nvidia/nemotron-3-nano-4b", "gemma-3-4b-it"]
    available_targets = [m for m in target_models if m in models]
    if not available_targets:
        pytest.skip(f"Neither nemotron nor gemma found. Available: {models[:5]}")

    all_results = {}
    for model in available_targets:
        print(f"\n{'='*70}")
        print(f"MODEL: {model}")
        print(f"{'='*70}")

        results = await _run_generation_suite(model)
        all_results[model] = results

        passed = sum(1 for r in results.values() if r["passed"])
        total = len(results)

        for test_name, result in results.items():
            status = "PASS" if result["passed"] else "FAIL"
            print(f"  {status}  {test_name}: {result['details']}")

        print(f"\n  SCORE: {passed}/{total} ({passed/total:.0%})")

    # Print comparison if multiple models
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"{'='*70}")
        test_names = list(next(iter(all_results.values())).keys())
        print(f"{'Test':<40} " + " ".join(f"{m.split('/')[-1]:>15}" for m in all_results))
        print("-" * (40 + 16 * len(all_results)))
        for tn in test_names:
            row = f"{tn:<40} "
            for model, results in all_results.items():
                status = "PASS" if results[tn]["passed"] else "FAIL"
                row += f"{status:>15} "
            print(row)

        for model, results in all_results.items():
            passed = sum(1 for r in results.values() if r["passed"])
            total = len(results)
            short = model.split("/")[-1]
            print(f"\n{short}: {passed}/{total} ({passed/total:.0%})")
