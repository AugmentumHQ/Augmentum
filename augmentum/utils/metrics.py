"""Lightweight Prometheus-compatible metrics (no external dependencies)."""
from __future__ import annotations

import time
import threading
from collections import defaultdict
from typing import Any

import structlog

log = structlog.get_logger(__name__)


class Counter:
    """Monotonically increasing counter."""
    def __init__(self, name: str, help_text: str, labels: tuple[str, ...] = ()):
        self.name = name
        self.help = help_text
        self.labels = labels
        self._values: dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, **label_values) -> None:
        key = tuple(label_values.get(l, "") for l in self.labels)
        with self._lock:
            self._values[key] += amount

    def collect(self) -> list[tuple[dict[str, str], float]]:
        with self._lock:
            return [
                (dict(zip(self.labels, k)), v)
                for k, v in self._values.items()
            ]


class Gauge:
    """Value that can go up and down."""
    def __init__(self, name: str, help_text: str, labels: tuple[str, ...] = ()):
        self.name = name
        self.help = help_text
        self.labels = labels
        self._values: dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    def set(self, value: float, **label_values) -> None:
        key = tuple(label_values.get(l, "") for l in self.labels)
        with self._lock:
            self._values[key] = value

    def inc(self, amount: float = 1.0, **label_values) -> None:
        key = tuple(label_values.get(l, "") for l in self.labels)
        with self._lock:
            self._values[key] += amount

    def dec(self, amount: float = 1.0, **label_values) -> None:
        key = tuple(label_values.get(l, "") for l in self.labels)
        with self._lock:
            self._values[key] -= amount

    def collect(self) -> list[tuple[dict[str, str], float]]:
        with self._lock:
            return [
                (dict(zip(self.labels, k)), v)
                for k, v in self._values.items()
            ]


class Histogram:
    """Tracks distributions with configurable buckets."""
    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, float("inf"))

    def __init__(self, name: str, help_text: str, labels: tuple[str, ...] = (), buckets=None):
        self.name = name
        self.help = help_text
        self.labels = labels
        self._buckets = buckets or self.DEFAULT_BUCKETS
        self._counts: dict[tuple, list[int]] = defaultdict(lambda: [0] * len(self._buckets))
        self._sums: dict[tuple, float] = defaultdict(float)
        self._totals: dict[tuple, int] = defaultdict(int)
        self._lock = threading.Lock()

    def observe(self, value: float, **label_values) -> None:
        key = tuple(label_values.get(l, "") for l in self.labels)
        with self._lock:
            self._sums[key] += value
            self._totals[key] += 1
            for i, b in enumerate(self._buckets):
                if value <= b:
                    self._counts[key][i] += 1

    def time(self, **label_values):
        """Context manager for timing."""
        return _HistogramTimer(self, label_values)

    def collect(self) -> list[tuple[dict[str, str], list[int], float, int]]:
        with self._lock:
            return [
                (dict(zip(self.labels, k)), list(self._counts[k]), self._sums[k], self._totals[k])
                for k in set(list(self._counts.keys()) + list(self._sums.keys()))
            ]


class _HistogramTimer:
    def __init__(self, histogram: Histogram, labels: dict):
        self._histogram = histogram
        self._labels = labels
        self._start = 0.0

    def __enter__(self):
        self._start = time.monotonic()
        return self

    def __exit__(self, *args):
        self._histogram.observe(time.monotonic() - self._start, **self._labels)


class MetricsRegistry:
    """Global metrics registry."""
    def __init__(self):
        self._metrics: dict[str, Counter | Gauge | Histogram] = {}

    def counter(self, name: str, help_text: str, labels: tuple[str, ...] = ()) -> Counter:
        if name not in self._metrics:
            self._metrics[name] = Counter(name, help_text, labels)
        return self._metrics[name]

    def gauge(self, name: str, help_text: str, labels: tuple[str, ...] = ()) -> Gauge:
        if name not in self._metrics:
            self._metrics[name] = Gauge(name, help_text, labels)
        return self._metrics[name]

    def histogram(self, name: str, help_text: str, labels: tuple[str, ...] = (), buckets=None) -> Histogram:
        if name not in self._metrics:
            self._metrics[name] = Histogram(name, help_text, labels, buckets)
        return self._metrics[name]

    def render(self) -> str:
        """Render all metrics in Prometheus text exposition format."""
        lines = []
        for name, metric in sorted(self._metrics.items()):
            lines.append(f"# HELP {name} {metric.help}")

            if isinstance(metric, Counter):
                lines.append(f"# TYPE {name} counter")
                for labels, value in metric.collect():
                    label_str = _format_labels(labels)
                    lines.append(f"{name}{label_str} {value}")

            elif isinstance(metric, Gauge):
                lines.append(f"# TYPE {name} gauge")
                for labels, value in metric.collect():
                    label_str = _format_labels(labels)
                    lines.append(f"{name}{label_str} {value}")

            elif isinstance(metric, Histogram):
                lines.append(f"# TYPE {name} histogram")
                for labels, counts, sum_val, total in metric.collect():
                    label_str = _format_labels(labels)
                    cumulative = 0
                    for i, bucket in enumerate(metric._buckets):
                        cumulative += counts[i]
                        le = "+Inf" if bucket == float("inf") else str(bucket)
                        bucket_labels = {**labels, "le": le}
                        lines.append(f"{name}_bucket{_format_labels(bucket_labels)} {cumulative}")
                    lines.append(f"{name}_sum{label_str} {sum_val}")
                    lines.append(f"{name}_count{label_str} {total}")

            lines.append("")

        return "\n".join(lines)


def _format_labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    pairs = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()) if v)
    return f"{{{pairs}}}" if pairs else ""


# Global singleton
REGISTRY = MetricsRegistry()

# Pre-define standard metrics
REQUEST_DURATION = REGISTRY.histogram(
    "augmentum_request_duration_seconds",
    "Request duration in seconds",
    labels=("mode", "endpoint"),
)

REQUEST_COUNT = REGISTRY.counter(
    "augmentum_requests_total",
    "Total requests processed",
    labels=("mode", "status"),
)

TOOL_CALLS = REGISTRY.counter(
    "augmentum_tool_calls_total",
    "Total tool calls",
    labels=("tool", "success"),
)

ACTIVE_SESSIONS = REGISTRY.gauge(
    "augmentum_active_sessions",
    "Number of active sessions",
)

IMAGE_QUEUE_DEPTH = REGISTRY.gauge(
    "augmentum_image_queue_depth",
    "Current image generation queue depth",
)

IMAGE_GENERATION_DURATION = REGISTRY.histogram(
    "augmentum_image_generation_seconds",
    "Image generation duration in seconds",
    labels=("model",),
    buckets=(1, 2, 5, 10, 20, 30, 60, 120, 300, float("inf")),
)

EMBEDDING_DURATION = REGISTRY.histogram(
    "augmentum_embedding_seconds",
    "Embedding computation duration",
    labels=("operation",),
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, float("inf")),
)

LLM_TOKENS = REGISTRY.counter(
    "augmentum_llm_tokens_total",
    "Total LLM tokens processed",
    labels=("direction",),  # "prompt" or "completion"
)

MEMORY_OPERATIONS = REGISTRY.counter(
    "augmentum_memory_operations_total",
    "Memory store operations",
    labels=("operation",),  # "store", "recall", "consolidate", "compact"
)
