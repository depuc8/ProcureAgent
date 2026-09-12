"""OpenTelemetry -> Google Cloud Trace.

Uses ADK's own exporter helpers (google.adk.telemetry.google_cloud) so ADK's spans
and ours land in one trace. Exporting is opt-in (OTEL_TO_CLOUD_TRACE=true) so tests and
offline runs never need credentials.

Other runtimes configure tracing themselves, and the TelemetryPlugin works with
all of them because it only decorates whatever spans are current:
  * `adk web --otel_to_cloud`            ADK sets up the exporters
  * Agent Engine                         AdkApp(enable_tracing=True)
  * OTEL_EXPORTER_OTLP_ENDPOINT          honoured by ADK's maybe_set_otel_providers
"""

from __future__ import annotations

import logging

from settings import get_settings

log = logging.getLogger(__name__)
_done = False


def setup_telemetry() -> bool:
    """Idempotent. Returns True if Cloud Trace export was configured."""
    global _done
    if _done:
        return False
    _done = True
    if not get_settings().otel_to_cloud_trace:
        return False
    try:
        from google.adk.telemetry.google_cloud import get_gcp_exporters
        from google.adk.telemetry.setup import maybe_set_otel_providers

        hooks = get_gcp_exporters(enable_cloud_tracing=True)
        maybe_set_otel_providers([hooks])
        log.info("OpenTelemetry spans will be exported to Cloud Trace")
        return True
    except Exception:  # missing creds / project must not stop the app from serving
        log.exception("Cloud Trace export could not be configured; continuing without it")
        return False
