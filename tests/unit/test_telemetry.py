import pytest
from opentelemetry.sdk.trace import TracerProvider

from app.infrastructure.telemetry import add_trace_ids

pytestmark = pytest.mark.unit


def test_no_trace_ids_outside_a_span():
    assert "trace_id" not in add_trace_ids(None, "info", {"event": "x"})


def test_trace_ids_added_inside_a_span():
    tracer = TracerProvider().get_tracer("test")
    with tracer.start_as_current_span("s") as span:
        event = add_trace_ids(None, "info", {"event": "x"})
        assert event["trace_id"] == format(span.get_span_context().trace_id, "032x")
        assert len(event["span_id"]) == 16
