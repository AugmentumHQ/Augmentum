"""Tests for intent classifier — zero-cost heuristic query classification."""

from __future__ import annotations

import pytest

from augmentum.tools.intent import classify_intent, QueryIntent


class TestSearchIntent:
    """Search intent detection from keyword signals."""

    def test_news_query_detected_as_search(self):
        intent = classify_intent("What is the latest news about AI?")
        assert intent.action == "search"
        assert intent.confidence > 0.3

    def test_weather_query_detected_as_search(self):
        intent = classify_intent("What's the weather forecast today?")
        assert intent.action == "search"
        assert intent.temporal is True

    def test_reference_query_detected(self):
        intent = classify_intent("Who was Albert Einstein?")
        assert intent.action == "search"
        assert intent.source_type == "reference"

    def test_data_query_detected(self):
        intent = classify_intent("What are the specifications of RTX 4090?")
        assert intent.action == "search"
        assert intent.source_type == "data"

    def test_fresh_query_temporal(self):
        intent = classify_intent("What are the latest stock prices today?")
        assert intent.temporal is True
        assert intent.source_type == "fresh"


class TestFetchIntent:
    """URL fetch detection."""

    def test_url_detected_as_fetch(self):
        intent = classify_intent("Summarize https://example.com/article")
        assert intent.action == "fetch_url"
        assert intent.url == "https://example.com/article"
        assert intent.confidence >= 0.9

    def test_http_url_detected(self):
        intent = classify_intent("Read http://docs.python.org/3/library/")
        assert intent.action == "fetch_url"
        assert intent.url is not None


class TestCalcIntent:
    """Calculate intent detection."""

    def test_math_expression_detected(self):
        intent = classify_intent("What is 42 + 58?")
        assert intent.action == "calculate"
        assert intent.math_expr is not None

    def test_calculate_keyword_detected(self):
        intent = classify_intent("Calculate 15% of 200")
        assert intent.action == "calculate"
        assert intent.confidence > 0.3

    def test_complex_expression(self):
        intent = classify_intent("100 * 3.14 + 50")
        assert intent.action == "calculate"


class TestConvertIntent:
    """Unit conversion intent detection."""

    def test_convert_temperature(self):
        intent = classify_intent("Convert 100 celsius to fahrenheit")
        assert intent.action == "convert"

    def test_how_many_miles(self):
        intent = classify_intent("Convert 10 km to miles")
        assert intent.action == "convert"


class TestDatetimeIntent:
    """Datetime intent detection."""

    def test_what_time_is_it(self):
        intent = classify_intent("What time is it in Tokyo?")
        assert intent.action == "datetime"

    def test_what_day_is_date(self):
        intent = classify_intent("What day is March 15?")
        assert intent.action == "datetime"

    def test_how_many_days_until(self):
        intent = classify_intent("How many days until Christmas?")
        assert intent.action == "datetime"


class TestBuildAppIntent:
    """Application building intent detection."""

    def test_build_app(self):
        intent = classify_intent("Build me a calculator app")
        assert intent.action == "build_app"
        assert intent.confidence >= 0.8

    def test_create_website(self):
        intent = classify_intent("Create a website for my portfolio")
        assert intent.action == "build_app"

    def test_make_dashboard(self):
        intent = classify_intent("Make me a dashboard for sales data")
        assert intent.action == "build_app"


class TestNoneIntent:
    """No detectable intent (greetings, creative writing)."""

    def test_greeting_low_confidence(self):
        intent = classify_intent("Hello there, good morning")
        assert intent.action == "none"
        assert intent.confidence < 0.3

    def test_empty_string(self):
        intent = classify_intent("")
        assert intent.action == "none"
        assert intent.confidence == 0.0

    def test_creative_prompt(self):
        intent = classify_intent("Write me a poem about the sea")
        assert intent.action == "none" or intent.confidence < 0.5


class TestTopicHints:
    """Topic keyword matching and site hints."""

    def test_gpu_query_has_tech_topics(self):
        intent = classify_intent("What are the latest GPU benchmarks?")
        assert "gpu" in intent.topics or "benchmarks" in intent.topics

    def test_sports_query_has_sport_topics(self):
        intent = classify_intent("What are the NFL scores today?")
        assert any(t in intent.topics for t in ["nfl", "football", "sports", "scores"])
