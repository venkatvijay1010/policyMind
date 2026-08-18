"""Unit tests for safe source ingestion helpers."""

from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile
from pydantic import ValidationError

from app.api.routes.ingest import _source_request_from_upload, _validate_source_uri, chunk_text
from app.api.schemas.schemas import KnowledgeSourceRequest

pytestmark = pytest.mark.unit


def test_knowledge_source_rejects_non_progressing_overlap():
    with pytest.raises(ValidationError, match="segment_overlap must be smaller"):
        KnowledgeSourceRequest(
            source_text="A source document", segment_length=100, segment_overlap=100
        )


@pytest.mark.parametrize("chunk_size,chunk_overlap", [(100, 100), (100, 101), (0, 0)])
def test_chunk_text_rejects_non_progressing_configuration(chunk_size, chunk_overlap):
    with pytest.raises(ValueError):
        chunk_text("content", chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def test_chunk_text_preserves_correct_source_offsets():
    source = "  " + "policy terms " * 30 + "  "

    chunks = chunk_text(source, chunk_size=60, chunk_overlap=10)

    assert len(chunks) > 1
    assert [chunk["passage_order"] for chunk in chunks] == list(range(len(chunks)))
    assert all(
        source[chunk["source_offset_start"] : chunk["source_offset_end"]] == chunk["content"]
        for chunk in chunks
    )


def test_url_ingestion_requires_an_explicit_host_allowlist():
    with pytest.raises(HTTPException) as exc_info:
        _validate_source_uri("https://example.com/policy.txt")

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_uploaded_markdown_becomes_an_indexable_source_request():
    uploaded = UploadFile(
        filename="benefits-guide.md",
        file=BytesIO(b"# Benefits Guide\n\nCoverage is available after the waiting period."),
    )

    source_request = await _source_request_from_upload(
        uploaded,
        chunk_size=100,
        chunk_overlap=10,
    )

    assert source_request.source_text is not None
    assert "Benefits Guide" in source_request.source_text
    assert source_request.segment_length == 100
