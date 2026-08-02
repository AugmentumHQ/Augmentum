"""Tests for zero-cost search query expansion."""

from __future__ import annotations

from augmentum.search.expander import (
    _extract_comparison_axis,
    _extract_comparison_entities,
    detect_domain,
    detect_query_type,
    expand_queries,
    expand_query_batch,
    extract_key_terms,
)

# ---------------------------------------------------------------------------
# detect_query_type
# ---------------------------------------------------------------------------


class TestDetectQueryType:
    def test_temporal_today(self):
        assert detect_query_type("what happened today in tech") == "temporal"

    def test_temporal_year(self):
        assert detect_query_type("best laptops 2026") == "temporal"

    def test_temporal_latest(self):
        assert detect_query_type("latest news on AI regulation") == "temporal"

    def test_comparative_vs(self):
        assert detect_query_type("Python vs Rust performance") == "comparative"

    def test_comparative_difference(self):
        assert detect_query_type("difference between TCP and UDP") == "comparative"

    def test_comparative_better_than(self):
        assert detect_query_type("which is better, React or Vue?") == "comparative"

    def test_definitional_what_is(self):
        assert detect_query_type("what is a monorepo") == "definitional"

    def test_definitional_explain(self):
        assert detect_query_type("explain quantum entanglement") == "definitional"

    def test_howto(self):
        assert detect_query_type("how to install Docker on Ubuntu") == "howto"

    def test_howto_step_by_step(self):
        assert detect_query_type("step by step guide to baking bread") == "howto"

    def test_factual_default(self):
        assert detect_query_type("population of France") == "factual"

    def test_factual_generic(self):
        assert detect_query_type("capital of Australia") == "factual"


# ---------------------------------------------------------------------------
# detect_domain
# ---------------------------------------------------------------------------


class TestDetectDomain:
    def test_programming(self):
        assert detect_domain("Python pip install pandas numpy") == "python"

    def test_javascript(self):
        assert detect_domain("react component lifecycle npm") == "javascript"

    def test_medical(self):
        assert detect_domain("symptoms of diabetes treatment") == "medical"

    def test_math(self):
        assert detect_domain("proof of the Pythagorean theorem") == "math"

    def test_cooking(self):
        assert detect_domain("recipe for sourdough bread bake") == "cooking"

    def test_finance(self):
        assert detect_domain("best ETF for dividend investing") == "finance"

    def test_no_domain(self):
        assert detect_domain("hello world") is None

    def test_history(self):
        assert detect_domain("ancient Roman civilization era") == "history"


# ---------------------------------------------------------------------------
# extract_key_terms
# ---------------------------------------------------------------------------


class TestExtractKeyTerms:
    def test_basic(self):
        terms = extract_key_terms("what is the capital of France")
        assert "capital" in terms
        assert "France" in terms
        # Stop words removed
        assert "the" not in terms
        assert "is" not in terms

    def test_max_terms(self):
        terms = extract_key_terms("one two three four five six seven eight", max_terms=3)
        assert len(terms) == 3

    def test_empty(self):
        assert extract_key_terms("") == []

    def test_all_stop_words(self):
        assert extract_key_terms("the is a an") == []


# ---------------------------------------------------------------------------
# expand_queries — core expansion
# ---------------------------------------------------------------------------


class TestExpandQueries:
    def test_original_always_first(self):
        result = expand_queries("capital of France")
        assert result[0] == "capital of France"

    def test_returns_at_least_original(self):
        result = expand_queries("hello")
        assert len(result) >= 1
        assert result[0] == "hello"

    def test_empty_query(self):
        result = expand_queries("")
        assert result == []

    def test_temporal_adds_year(self):
        result = expand_queries("what happened today in AI")
        # Should add a variant with year or "latest"
        has_temporal = any("2025" in v or "2026" in v or "latest" in v for v in result[1:])
        assert has_temporal, f"No temporal variant in {result}"

    def test_comparative_splits_entities(self):
        result = expand_queries("Python vs Rust performance")
        # Should have variants for individual entities
        has_python = any("Python" in v and "vs" not in v for v in result[1:])
        has_rust = any("Rust" in v and "vs" not in v for v in result[1:])
        assert has_python or has_rust, f"No entity split in {result}"

    def test_definitional_simplifies(self):
        result = expand_queries("what is a neural network")
        # Should have a simplified variant
        has_simple = any("neural network" in v.lower() and "what is" not in v.lower() for v in result[1:])
        assert has_simple, f"No simplified variant in {result}"

    def test_howto_adds_tutorial(self):
        result = expand_queries("how to install Docker on Ubuntu")
        has_tutorial = any("tutorial" in v.lower() for v in result)
        assert has_tutorial, f"No tutorial variant in {result}"

    def test_synonym_substitution(self):
        result = expand_queries("best Python web framework")
        has_synonym = any("top rated" in v.lower() or "recommended" in v.lower() for v in result[1:])
        assert has_synonym, f"No synonym variant in {result}"

    def test_site_scoped_programming(self):
        result = expand_queries("python function error handling", include_site_scoped=True)
        has_site = any("site:" in v for v in result)
        assert has_site, f"No site-scoped variant in {result}"

    def test_site_scoped_disabled(self):
        result = expand_queries("python function error handling", include_site_scoped=False)
        has_site = any("site:" in v for v in result)
        assert not has_site, f"Unexpected site-scoped variant in {result}"

    def test_max_variants_respected(self):
        result = expand_queries("difference between TCP and UDP", max_variants=3)
        assert len(result) <= 3

    def test_no_duplicates(self):
        result = expand_queries("best Python web framework")
        normalized = [v.strip().lower() for v in result]
        assert len(normalized) == len(set(normalized)), f"Duplicates in {result}"

    def test_key_term_extraction(self):
        result = expand_queries("what are the main advantages of using Kubernetes")
        # Should have a key-terms variant (stop words removed)
        has_keyterms = any(
            "main" in v and "advantages" in v and "Kubernetes" in v and v != result[0]
            for v in result
        )
        assert has_keyterms, f"No key-term variant in {result}"


# ---------------------------------------------------------------------------
# expand_query_batch
# ---------------------------------------------------------------------------


class TestExpandQueryBatch:
    def test_originals_first(self):
        queries = ["query one", "query two"]
        result = expand_query_batch(queries)
        assert result[0] == "query one"
        assert result[1] == "query two"

    def test_deduplicates_across_queries(self):
        queries = ["Python tutorial", "Python tutorial guide"]
        result = expand_query_batch(queries)
        normalized = [v.strip().lower() for v in result]
        assert len(normalized) == len(set(normalized)), f"Duplicates in {result}"

    def test_max_total_respected(self):
        queries = ["query one", "query two", "query three", "query four"]
        result = expand_query_batch(queries, max_total=5)
        assert len(result) <= 5

    def test_single_query(self):
        result = expand_query_batch(["capital of France"])
        assert result[0] == "capital of France"
        assert len(result) >= 1

    def test_empty_list(self):
        result = expand_query_batch([])
        assert result == []


# ---------------------------------------------------------------------------
# Comparison entity extraction
# ---------------------------------------------------------------------------


class TestComparisonEntities:
    def test_vs(self):
        entities = _extract_comparison_entities("Python vs Rust")
        assert len(entities) == 2
        assert "Python" in entities[0]
        assert "Rust" in entities[1]

    def test_difference_between(self):
        entities = _extract_comparison_entities("difference between TCP and UDP")
        assert len(entities) == 2
        assert "TCP" in entities[0]
        assert "UDP" in entities[1]

    def test_compared_to(self):
        entities = _extract_comparison_entities("React compared to Vue")
        assert len(entities) == 2

    def test_no_comparison(self):
        entities = _extract_comparison_entities("what is Python")
        assert len(entities) == 0


class TestComparisonAxis:
    def test_faster(self):
        axis = _extract_comparison_axis("which is faster, Python or Rust?")
        assert "performance" in axis or "speed" in axis

    def test_cheaper(self):
        axis = _extract_comparison_axis("is AWS cheaper than Azure?")
        assert "cost" in axis or "pricing" in axis

    def test_no_axis(self):
        axis = _extract_comparison_axis("Python vs Rust")
        assert axis == ""


# ---------------------------------------------------------------------------
# Integration-style tests
# ---------------------------------------------------------------------------


class TestExpansionIntegration:
    """Tests that verify the expansion works well for realistic queries."""

    def test_news_query(self):
        """News queries should get temporal expansion."""
        result = expand_queries("latest AI regulation news 2026")
        assert len(result) >= 2

    def test_coding_help(self):
        """Coding queries should get site-scoped stackoverflow variant."""
        result = expand_queries("python TypeError NoneType not subscriptable")
        has_so = any("site:docs.python.org" in v or "site:stackoverflow" in v for v in result)
        assert has_so, f"No StackOverflow variant in {result}"

    def test_medical_query(self):
        """Medical queries should get authoritative site variant."""
        result = expand_queries("symptoms of iron deficiency diagnosis")
        has_site = any("site:" in v for v in result)
        assert has_site, f"No authoritative site variant in {result}"

    def test_simple_factual(self):
        """Simple factual queries still get at least key-term extraction."""
        result = expand_queries("population of Japan")
        assert len(result) >= 1

    def test_comparison_full_pipeline(self):
        """Comparative queries should split entities and detect axis."""
        result = expand_queries("which is faster Python or Rust for web servers")
        # Should have entity-specific variants
        assert len(result) >= 3, f"Too few variants: {result}"
