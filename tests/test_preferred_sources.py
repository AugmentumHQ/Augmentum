"""Tests for the preferred sources registry."""

from augmentum.tools.preferred_sources import (
    AVOID,
    EXCELLENT,
    GOOD,
    UNKNOWN,
    describe_source,
    domain_quality,
    get_registry_stats,
    get_source_info,
    get_sources_by_category,
    get_topic_sites,
    sort_urls_by_quality,
    sort_urls_by_quality_with_diversity,
)


class TestSortWithDiversity:
    def test_caps_per_domain(self):
        urls = [
            "https://en.wikipedia.org/a",
            "https://en.wikipedia.org/b",
            "https://en.wikipedia.org/c",
            "https://weather.gov/forecast",
        ]
        result = sort_urls_by_quality_with_diversity(urls, per_domain_cap=2)
        assert result[:3] == [
            "https://en.wikipedia.org/a",
            "https://en.wikipedia.org/b",
            "https://weather.gov/forecast",
        ]
        assert result[3] == "https://en.wikipedia.org/c"

    def test_nothing_lost(self):
        urls = [f"https://en.wikipedia.org/p{i}" for i in range(5)]
        result = sort_urls_by_quality_with_diversity(urls, per_domain_cap=1)
        assert sorted(result) == sorted(urls)

    def test_empty(self):
        assert sort_urls_by_quality_with_diversity([]) == []


class TestDomainQuality:
    def test_excellent_domain(self):
        assert domain_quality("https://en.wikipedia.org/wiki/Python") == EXCELLENT

    def test_good_domain(self):
        assert domain_quality("https://www.nytimes.com/article") == GOOD

    def test_avoid_domain(self):
        assert domain_quality("https://www.accuweather.com/forecast") == AVOID

    def test_unknown_domain(self):
        assert domain_quality("https://randomsite.xyz/page") == UNKNOWN

    def test_subdomain_walks_up(self):
        assert domain_quality("https://api.weather.gov/forecast") == EXCELLENT

    def test_gov_weather(self):
        assert domain_quality("https://weather.gov") == EXCELLENT

    def test_strips_port(self):
        assert domain_quality("https://github.com:443/repo") == EXCELLENT

    def test_empty_url(self):
        assert domain_quality("") == UNKNOWN

    def test_malformed_url(self):
        assert domain_quality("not-a-url") == UNKNOWN

    def test_social_media_avoid(self):
        assert domain_quality("https://twitter.com/post") == AVOID
        assert domain_quality("https://x.com/post") == AVOID
        assert domain_quality("https://linkedin.com/in/user") == AVOID
        assert domain_quality("https://facebook.com/page") == AVOID

    def test_government_excellent(self):
        assert domain_quality("https://irs.gov/forms") == EXCELLENT
        assert domain_quality("https://cdc.gov/disease") == EXCELLENT
        assert domain_quality("https://nasa.gov/mission") == EXCELLENT

    def test_programming_docs(self):
        assert domain_quality("https://docs.python.org/3/library") == EXCELLENT
        assert domain_quality("https://developer.mozilla.org/en-US") == EXCELLENT
        assert domain_quality("https://docs.rs/tokio") == EXCELLENT

    def test_ai_ml_sources(self):
        assert domain_quality("https://huggingface.co/models") == EXCELLENT
        assert domain_quality("https://pytorch.org/docs") == EXCELLENT
        assert domain_quality("https://paperswithcode.com/sota") == EXCELLENT

    def test_cloud_devops(self):
        assert domain_quality("https://docs.aws.amazon.com/s3") == EXCELLENT
        assert domain_quality("https://kubernetes.io/docs") == EXCELLENT
        assert domain_quality("https://docs.docker.com/engine") == EXCELLENT

    def test_database_docs(self):
        assert domain_quality("https://postgresql.org/docs") == EXCELLENT
        assert domain_quality("https://sqlite.org/lang.html") == EXCELLENT
        assert domain_quality("https://redis.io/commands") == EXCELLENT

    def test_security_sources(self):
        assert domain_quality("https://owasp.org/Top10") == EXCELLENT
        assert domain_quality("https://nvd.nist.gov/vuln") == EXCELLENT
        assert domain_quality("https://cve.mitre.org/cgi-bin/cvename.cgi") == EXCELLENT

    def test_standards_sources(self):
        assert domain_quality("https://w3.org/TR/html5") == EXCELLENT
        assert domain_quality("https://rfc-editor.org/rfc/rfc7231") == EXCELLENT

    def test_streaming_avoid(self):
        assert domain_quality("https://netflix.com/title/123") == AVOID
        assert domain_quality("https://spotify.com/track/abc") == AVOID
        assert domain_quality("https://youtube.com/watch?v=abc") == AVOID
        assert domain_quality("https://twitch.tv/channel") == AVOID


class TestSortByQuality:
    def test_sorts_excellent_first(self):
        urls = [
            "https://accuweather.com/nyc",
            "https://randomsite.com/weather",
            "https://weather.gov/forecast/nyc",
            "https://nytimes.com/weather",
        ]
        result = sort_urls_by_quality(urls)
        assert result[0] == "https://weather.gov/forecast/nyc"
        assert result[-1] == "https://accuweather.com/nyc"

    def test_preserves_order_within_tier(self):
        urls = [
            "https://wikipedia.org/wiki/A",
            "https://github.com/repo",
        ]
        result = sort_urls_by_quality(urls)
        assert result == urls

    def test_empty_list(self):
        assert sort_urls_by_quality([]) == []

    def test_avoids_go_last(self):
        urls = [
            "https://linkedin.com/in/user",
            "https://stackoverflow.com/q/123",
            "https://randomsite.xyz/page",
        ]
        result = sort_urls_by_quality(urls)
        assert result[0] == "https://stackoverflow.com/q/123"
        assert result[-1] == "https://linkedin.com/in/user"


class TestTopicSites:
    def test_weather_topic(self):
        sites = get_topic_sites("what's the weather today")
        assert "weather.gov" in sites

    def test_python_topic(self):
        sites = get_topic_sites("python list comprehension")
        assert "docs.python.org" in sites
        assert "stackoverflow.com" in sites

    def test_medical_topic(self):
        sites = get_topic_sites("health effects of caffeine")
        assert "cdc.gov" in sites or "nih.gov" in sites

    def test_no_match(self):
        sites = get_topic_sites("random unrelated query")
        assert sites == []

    def test_no_duplicates(self):
        sites = get_topic_sites("medical health advice")
        assert len(sites) == len(set(sites))

    def test_multiword_phrase(self):
        sites = get_topic_sites("what is the cost of living in NYC")
        assert "numbeo.com" in sites

    def test_finance_topic(self):
        sites = get_topic_sites("current inflation rate")
        assert "bls.gov" in sites or "fred.stlouisfed.org" in sites

    def test_sports_topic(self):
        sites = get_topic_sites("NBA standings 2026")
        assert "basketball-reference.com" in sites

    def test_programming_language(self):
        # Disambiguated multi-word forms required (bare 'rust' was removed
        # because it collides with non-tech queries like "remove rust from car").
        sites = get_topic_sites("rust language ownership and borrowing")
        assert "docs.rs" in sites or "rust-lang.org" in sites

    def test_law_topic(self):
        sites = get_topic_sites("US tax deductions for 2026")
        assert "irs.gov" in sites

    def test_ai_topic(self):
        sites = get_topic_sites("how to fine-tune a transformer model")
        assert "huggingface.co" in sites or "pytorch.org" in sites

    def test_cloud_topic(self):
        sites = get_topic_sites("deploy to aws lambda")
        assert "docs.aws.amazon.com" in sites

    def test_docker_topic(self):
        sites = get_topic_sites("docker compose networking")
        assert "docs.docker.com" in sites

    def test_database_topic(self):
        sites = get_topic_sites("postgresql index optimization")
        assert "postgresql.org" in sites

    def test_security_topic(self):
        sites = get_topic_sites("owasp top 10 vulnerabilities")
        assert "owasp.org" in sites

    def test_fact_check_topic(self):
        sites = get_topic_sites("is it true that")
        assert "factcheck.org" in sites or "snopes.com" in sites

    def test_framework_topic(self):
        sites = get_topic_sites("django rest framework tutorial")
        assert "docs.djangoproject.com" in sites

    def test_nextjs_topic(self):
        sites = get_topic_sites("nextjs server components")
        assert "nextjs.org" in sites

    def test_recipe_topic(self):
        sites = get_topic_sites("easy recipe for banana bread")
        assert "allrecipes.com" in sites or "seriouseats.com" in sites


class TestSourceInfo:
    def test_get_source_info_known(self):
        info = get_source_info("https://weather.gov/forecast")
        assert info is not None
        assert info.quality == EXCELLENT
        assert "weather" in info.categories
        assert info.freshness == "realtime"

    def test_get_source_info_unknown(self):
        assert get_source_info("https://unknown-site-xyz.com") is None

    def test_get_source_info_subdomain(self):
        info = get_source_info("https://api.weather.gov/gridpoints")
        assert info is not None
        assert info.quality == EXCELLENT

    def test_source_has_paywall(self):
        info = get_source_info("https://wsj.com/article/123")
        assert info is not None
        assert info.has_paywall is True

    def test_source_requires_js(self):
        info = get_source_info("https://instagram.com/user")
        assert info is not None
        assert info.requires_js is True

    def test_source_categories(self):
        info = get_source_info("https://stackoverflow.com/questions/123")
        assert info is not None
        assert "programming" in info.categories
        assert "qa" in info.categories

    def test_source_structured_data(self):
        info = get_source_info("https://fred.stlouisfed.org/series/GDP")
        assert info is not None
        assert info.structured_data is True


class TestDescribeSource:
    def test_known_source(self):
        desc = describe_source("https://weather.gov/forecast")
        assert "excellent" in desc
        assert "weather" in desc
        assert "realtime" in desc

    def test_unknown_source(self):
        assert describe_source("https://unknown-xyz.com") == ""

    def test_paywall_noted(self):
        desc = describe_source("https://wsj.com/article")
        assert "paywalled" in desc

    def test_js_noted(self):
        desc = describe_source("https://instagram.com/user")
        assert "requires-js" in desc


class TestSourcesByCategory:
    def test_weather_sources(self):
        sources = get_sources_by_category("weather")
        domains = [d for d, _ in sources]
        assert "weather.gov" in domains
        # Should be sorted by quality — excellent first
        qualities = [info.quality for _, info in sources]
        assert qualities == sorted(qualities, reverse=True)

    def test_programming_sources(self):
        sources = get_sources_by_category("programming")
        domains = [d for d, _ in sources]
        assert "stackoverflow.com" in domains
        assert "github.com" in domains
        assert len(sources) >= 5

    def test_empty_category(self):
        sources = get_sources_by_category("nonexistent_category_xyz")
        assert sources == []


class TestRegistryStats:
    def test_stats_structure(self):
        stats = get_registry_stats()
        assert "total_sources" in stats
        assert "by_quality" in stats
        assert "total_categories" in stats
        assert "total_topic_mappings" in stats

    def test_reasonable_counts(self):
        stats = get_registry_stats()
        assert stats["total_sources"] >= 200
        assert stats["by_quality"]["excellent"] >= 80
        assert stats["by_quality"]["avoid"] >= 25
        assert stats["total_categories"] >= 40
        assert stats["total_topic_mappings"] >= 150


class TestTopicHintCollisions:
    """Regression guard: ablation harness found that single-word programming
    keys (rust, java, swift, ruby, rails, spring, space) were hijacking
    non-technical queries. They were replaced with disambiguated forms.
    """

    def test_interior_design_does_not_trigger_space_hints(self):
        # "small space living room" used to inject site:nasa.gov
        assert get_topic_sites("small space living room ideas") == []
        assert "nasa.gov" not in get_topic_sites("how to organize a small space")

    def test_travel_does_not_trigger_java_hints(self):
        # "java" the Indonesian island vs Java the language
        assert "docs.oracle.com" not in get_topic_sites("java the island travel guide")
        assert "docs.oracle.com" not in get_topic_sites("best beaches in java indonesia")

    def test_car_maintenance_does_not_trigger_rust_hints(self):
        assert "docs.rs" not in get_topic_sites("how to remove rust from car")
        assert "rust-lang.org" not in get_topic_sites("preventing rust on metal tools")

    def test_train_does_not_trigger_rails_hints(self):
        assert "guides.rubyonrails.org" not in get_topic_sites("train rails replacement project")

    def test_water_does_not_trigger_spring_hints(self):
        assert "spring.io" not in get_topic_sites("spring water vs tap water")
        assert "spring.io" not in get_topic_sites("when does spring start")

    def test_jewelry_does_not_trigger_ruby_hints(self):
        assert "ruby-lang.org" not in get_topic_sites("ruby earrings vintage")
        assert "ruby-lang.org" not in get_topic_sites("what is a ruby gemstone worth")

    def test_disambiguated_tech_queries_still_work(self):
        # Replacements must still surface the right docs for explicit tech queries
        assert "docs.rs" in get_topic_sites("how do I learn the rust language")
        assert "swift.org" in get_topic_sites("swift language vs objective-c")
        assert "spring.io" in get_topic_sites("what is the spring framework")
        assert "ruby-lang.org" in get_topic_sites("ruby on rails tutorial")
        assert "nasa.gov" in get_topic_sites("outer space exploration")
        assert "nasa.gov" in get_topic_sites("astronomy facts")
        assert "docs.oracle.com" in get_topic_sites("java programming tutorial")
