"""Tests for augmentum/utils/metrics.py — Prometheus-compatible metrics."""

from __future__ import annotations

from augmentum.utils.metrics import (
    ACTIVE_SESSIONS,
    REGISTRY,
    REQUEST_COUNT,
    REQUEST_DURATION,
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
)


class TestCounter:
    """Verify counter metric behavior."""

    def test_increment_default(self):
        c = Counter("test_counter", "Test")
        c.inc()
        collected = c.collect()
        assert len(collected) == 1
        assert collected[0][1] == 1.0

    def test_increment_by_amount(self):
        c = Counter("test_counter2", "Test")
        c.inc(amount=5.0)
        collected = c.collect()
        assert collected[0][1] == 5.0

    def test_increment_with_labels(self):
        c = Counter("test_counter3", "Test", labels=("mode",))
        c.inc(mode="analytical")
        c.inc(mode="analytical")
        c.inc(mode="narrative")
        collected = c.collect()
        label_map = {tuple(d.items()): v for d, v in collected}
        assert label_map[(("mode", "analytical"),)] == 2.0
        assert label_map[(("mode", "narrative"),)] == 1.0


class TestGauge:
    """Verify gauge metric behavior."""

    def test_set_value(self):
        g = Gauge("test_gauge", "Test")
        g.set(42.0)
        collected = g.collect()
        assert collected[0][1] == 42.0

    def test_inc_and_dec(self):
        g = Gauge("test_gauge2", "Test")
        g.inc(amount=10)
        g.dec(amount=3)
        collected = g.collect()
        assert collected[0][1] == 7.0

    def test_gauge_with_labels(self):
        g = Gauge("test_gauge3", "Test", labels=("service",))
        g.set(1.0, service="searxng")
        g.set(0.0, service="executor")
        collected = g.collect()
        assert len(collected) == 2


class TestHistogram:
    """Verify histogram metric behavior."""

    def test_observe(self):
        h = Histogram("test_hist", "Test")
        h.observe(0.5)
        h.observe(1.5)
        collected = h.collect()
        assert len(collected) == 1
        labels, counts, sum_val, total = collected[0]
        assert total == 2
        assert sum_val == 2.0

    def test_timer_context_manager(self):
        h = Histogram("test_hist2", "Test")
        with h.time():
            pass  # Near-zero duration
        collected = h.collect()
        assert collected[0][3] == 1  # total count

    def test_custom_buckets(self):
        h = Histogram("test_hist3", "Test", buckets=(1.0, 5.0, 10.0, float("inf")))
        h.observe(3.0)
        collected = h.collect()
        _, counts, _, _ = collected[0]
        assert len(counts) == 4


class TestMetricsRegistry:
    """Verify registry and rendering."""

    def test_create_counter(self):
        reg = MetricsRegistry()
        c = reg.counter("my_counter", "A counter")
        assert isinstance(c, Counter)

    def test_dedup_same_name(self):
        reg = MetricsRegistry()
        c1 = reg.counter("dedup_counter", "First")
        c2 = reg.counter("dedup_counter", "Second")
        assert c1 is c2

    def test_render_prometheus_format(self):
        reg = MetricsRegistry()
        c = reg.counter("render_test", "Render test", labels=("mode",))
        c.inc(mode="analytical")
        output = reg.render()
        assert "# HELP render_test Render test" in output
        assert "# TYPE render_test counter" in output
        assert "render_test" in output


class TestGlobalMetrics:
    """Verify pre-defined global metrics exist."""

    def test_request_duration_exists(self):
        assert isinstance(REQUEST_DURATION, Histogram)

    def test_request_count_exists(self):
        assert isinstance(REQUEST_COUNT, Counter)

    def test_active_sessions_exists(self):
        assert isinstance(ACTIVE_SESSIONS, Gauge)

    def test_registry_singleton(self):
        assert REGISTRY is not None
        assert isinstance(REGISTRY, MetricsRegistry)
