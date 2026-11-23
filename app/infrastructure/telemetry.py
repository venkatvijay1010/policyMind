"""OpenTelemetry setup; a no-op unless OTEL_EXPORTER_OTLP_ENDPOINT is set."""

import os

import structlog
from opentelemetry import trace

logger = structlog.get_logger()
tracer = trace.get_tracer("policymind")


def add_trace_ids(_logger, _method, event_dict: dict) -> dict:
    """structlog processor that correlates log lines with the active trace."""
    context = trace.get_current_span().get_span_context()
    if context.is_valid:
        event_dict["trace_id"] = format(context.trace_id, "032x")
        event_dict["span_id"] = format(context.span_id, "016x")
    return event_dict


def setup_telemetry(app) -> None:
    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning(
            "OTEL endpoint set but OpenTelemetry extras are missing: pip install .[otel]"
        )
        return

    provider = TracerProvider(
        resource=Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "policymind")})
    )
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app, excluded_urls="metrics,health")
    HTTPXClientInstrumentor().instrument()
