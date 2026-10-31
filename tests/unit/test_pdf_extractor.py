"""PDF extraction tests use generated fixtures and no remote providers."""

from __future__ import annotations

import hashlib

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.infrastructure.documents import pdf_extractor
from app.infrastructure.documents.pdf_extractor import (
    ExtractedElement,
    PDFExtractionError,
    extract_pdf,
)

pytestmark = pytest.mark.unit


def _text_pdf(*pages: str) -> bytes:
    import pymupdf

    document = pymupdf.open()
    for text in pages:
        page = document.new_page()
        page.insert_text((72, 72), text)
    result = document.tobytes()
    document.close()
    return result


def _scanned_pdf(with_native_text: str | None = None) -> bytes:
    import pymupdf

    image_doc = pymupdf.open()
    image_page = image_doc.new_page()
    image_page.insert_text((72, 72), "Scanned clause")
    image_bytes = image_page.get_pixmap().tobytes("png")
    image_doc.close()

    document = pymupdf.open()
    page = document.new_page()
    if with_native_text:
        page.insert_text((72, 72), with_native_text)
    page.insert_image(pymupdf.Rect(50, 150, 550, 750), stream=image_bytes)
    result = document.tobytes()
    document.close()
    return result


def test_native_text_pdf_keeps_page_numbers_and_original_hash(monkeypatch):
    monkeypatch.setattr(pdf_extractor.settings, "pdf_min_native_chars_per_page", 0)
    original = _text_pdf("Coverage begins on day one.", "Annual limit is 5000.")

    result = extract_pdf(original)

    assert result.content_sha256 == hashlib.sha256(original).hexdigest()
    assert result.page_count == 2
    assert [page.page_number for page in result.pages] == [1, 2]
    assert "Coverage begins" in result.pages[0].text
    assert "Annual limit" in result.pages[1].text
    assert all(
        element.element_type == "native_text" for page in result.pages for element in page.elements
    )


def test_rotated_pdf_page_keeps_native_content_and_original_page_number(monkeypatch):
    import pymupdf

    monkeypatch.setattr(pdf_extractor.settings, "pdf_min_native_chars_per_page", 0)
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Rotated policy page has native text.")
    page.set_rotation(90)
    original = document.tobytes()
    document.close()

    result = extract_pdf(original)

    assert result.pages[0].page_number == 1
    assert "Rotated policy page" in result.pages[0].text


def test_mixed_page_preserves_native_and_separate_ocr(monkeypatch):
    monkeypatch.setattr(pdf_extractor.settings, "pdf_min_native_chars_per_page", 0)
    monkeypatch.setattr(
        pdf_extractor,
        "_ocr_page",
        lambda page, blocks, page_number, pymupdf: [
            ExtractedElement(
                page_number=page_number,
                element_type="ocr_text",
                text_content="Scanned clause",
                confidence=42.5,
                parser_name="tesseract",
                needs_review=True,
            )
        ],
    )
    original = _scanned_pdf("Native clause remains in the text layer. " * 3)

    result = extract_pdf(original)

    assert "Native clause remains in the text layer." in result.pages[0].text
    assert "Scanned clause" in result.pages[0].text
    assert {element.element_type for element in result.pages[0].elements} == {
        "native_text",
        "ocr_text",
    }
    assert result.pages[0].needs_review
    assert result.pages[0].elements[-1].confidence == 42.5


def test_scanned_page_uses_ocr_and_records_review_confidence(monkeypatch):
    monkeypatch.setattr(
        pdf_extractor,
        "_ocr_page",
        lambda page, blocks, page_number, pymupdf: [
            ExtractedElement(
                page_number=page_number,
                element_type="ocr_text",
                text_content="Scanned clause",
                bbox=[50.0, 150.0, 180.0, 175.0],
                confidence=91.0,
                parser_name="tesseract",
                parser_version="test-version",
            )
        ],
    )

    result = extract_pdf(_scanned_pdf())

    assert result.pages[0].text == "Scanned clause"
    assert result.pages[0].elements[0].element_type == "ocr_text"
    assert result.pages[0].elements[0].bbox == [50.0, 150.0, 180.0, 175.0]
    assert result.pages[0].elements[0].confidence == 91.0
    assert result.needs_review


def test_ocr_tsv_confidence_and_pixel_box_are_saved(monkeypatch):
    import pymupdf
    import pytesseract

    monkeypatch.setattr(pdf_extractor.settings, "pdf_ocr_review_confidence", 60.0)
    monkeypatch.setattr(
        pytesseract,
        "image_to_data",
        lambda *args, **kwargs: {
            "text": ["uncertain"],
            "conf": ["35"],
            "left": [200],
            "top": [100],
            "width": [80],
            "height": [20],
            "block_num": [1],
            "par_num": [1],
            "line_num": [1],
        },
    )
    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    elements = pdf_extractor._ocr_page(page, [], 1, pymupdf)
    document.close()

    assert len(elements) == 1
    assert elements[0].confidence == 35.0
    assert elements[0].bbox is not None
    assert elements[0].bbox[0] < elements[0].bbox[2]
    assert elements[0].needs_review


def test_table_cells_headers_and_bbox_are_retained(monkeypatch):
    monkeypatch.setattr(pdf_extractor.settings, "pdf_min_native_chars_per_page", 0)
    import pymupdf

    document = pymupdf.open()
    page = document.new_page()
    page.draw_line((50, 80), (250, 80))
    page.draw_line((50, 110), (250, 110))
    page.draw_line((50, 140), (250, 140))
    page.draw_line((50, 80), (50, 140))
    page.draw_line((150, 80), (150, 140))
    page.draw_line((250, 80), (250, 140))
    page.insert_text((60, 100), "Benefit")
    page.insert_text((160, 100), "Limit")
    page.insert_text((60, 130), "Dental")
    page.insert_text((160, 130), "5000")
    original = document.tobytes()
    document.close()

    result = extract_pdf(original)

    table = next(element for element in result.pages[0].elements if element.element_type == "table")
    assert table.structured_data is not None
    assert table.structured_data["headers"] == ["Benefit", "Limit"]
    assert table.structured_data["rows"] == [["Dental", "5000"]]
    assert table.bbox is not None
    assert "Dental | 5000" in result.pages[0].text


def test_malformed_encrypted_page_limit_and_output_limit_errors(monkeypatch):
    monkeypatch.setattr(pdf_extractor.settings, "pdf_min_native_chars_per_page", 0)
    with pytest.raises(PDFExtractionError, match="not a readable PDF"):
        extract_pdf(b"not a pdf")

    import pymupdf

    document = pymupdf.open()
    document.new_page()
    encrypted = document.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user"
    )
    document.close()
    with pytest.raises(PDFExtractionError, match="Encrypted"):
        extract_pdf(encrypted)

    monkeypatch.setattr(pdf_extractor.settings, "pdf_max_pages", 1)
    with pytest.raises(PDFExtractionError, match="page count"):
        extract_pdf(_text_pdf("Page one", "Page two"))

    monkeypatch.setattr(pdf_extractor.settings, "pdf_max_pages", 200)
    monkeypatch.setattr(pdf_extractor.settings, "pdf_max_extracted_chars", 5)
    with pytest.raises(PDFExtractionError, match="output limit"):
        extract_pdf(_text_pdf("A longer native text line"))


@pytest.mark.asyncio
async def test_upload_index_persists_unchanged_source_and_page_links(monkeypatch):
    from io import BytesIO

    import pymupdf
    from fastapi import UploadFile

    from app.api.routes import ingest
    from app.api.routes.ingest import _index_contract_source, _source_request_from_upload
    from app.infrastructure.database.models import (
        ContractPassage,
        DocumentElement,
        PassageSource,
        Policy,
        SourceDocument,
    )
    from app.infrastructure.database.postgres import Base

    monkeypatch.setattr(pdf_extractor.settings, "pdf_min_native_chars_per_page", 0)
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_textbox(
        pymupdf.Rect(60, 60, 540, 700),
        "Policy benefits and eligibility details. " * 10,
        fontsize=11,
    )
    original = pdf.tobytes()
    pdf.close()
    upload = UploadFile(filename="source-original.pdf", file=BytesIO(original))
    source_request = await _source_request_from_upload(upload, 100, 10)

    class FakeEmbeddings:
        async def embed_batch(self, texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(ingest, "get_embedding_service", lambda: FakeEmbeddings())
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        policy = Policy(contract_ref="PDF-TEST", contract_title="Test PDF")
        session.add(policy)
        await session.flush()
        response = await _index_contract_source(policy.id, source_request, session)

        stored_source = await session.get(SourceDocument, response.source_document_id)
        passages = (await session.scalars(select(ContractPassage))).all()
        elements = (await session.scalars(select(DocumentElement))).all()
        links = (await session.scalars(select(PassageSource))).all()

        assert stored_source is not None
        assert stored_source.original_bytes == original
        assert stored_source.content_sha256 == hashlib.sha256(original).hexdigest()
        assert stored_source.filename == "source-original.pdf"
        assert stored_source.page_count == 1
        assert passages
        assert elements and all(element.page_number == 1 for element in elements)
        assert links and all(link.source_document_id == stored_source.id for link in links)
        assert all(link.page_number == 1 for link in links)
        assert all(link.element_ids for link in links)
        assert response.content_sha256 == stored_source.content_sha256

    await engine.dispose()
