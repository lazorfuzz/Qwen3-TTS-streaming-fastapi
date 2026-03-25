"""Prometheus metric definitions for TTS server.

All metrics are module-level singletons, auto-registered with the default
prometheus_client registry.  Import the objects you need and call
.observe() / .inc() / .dec() at the appropriate points.

Supports prometheus_client multiprocess mode: when PROMETHEUS_MULTIPROC_DIR
is set, each worker writes metrics to shared mmap files on disk, and the
/metrics endpoint aggregates from all workers.
"""

from prometheus_client import Counter, Gauge, Histogram


# ---------------------------------------------------------------------------
# Request-level metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "tts_requests_total",
    "Total TTS requests",
    ["status"],
)

REQUEST_DURATION = Histogram(
    "tts_request_duration_seconds",
    "End-to-end TTS request duration",
)

TTFB_HISTOGRAM = Histogram(
    "tts_time_to_first_byte_seconds",
    "Time from request start to first audio byte sent",
)

ACTIVE_REQUESTS = Gauge(
    "tts_active_requests",
    "Number of TTS requests currently in progress",
    multiprocess_mode="livesum",
)

# ---------------------------------------------------------------------------
# Batch metrics
# ---------------------------------------------------------------------------
BATCH_SIZE_HISTOGRAM = Histogram(
    "tts_batch_size",
    "Number of requests per dispatched batch",
    buckets=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16],
)
