"""Bounded, provenance-preserving extraction for uploaded PDF files."""

from __future__ import annotations

import hashlib
import io
import multiprocessing
import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.config import settings

EXTRACTION_VERSION = "pdf-extraction-v1"


class PDFExtractionError(ValueError):
    """A safe, user-actionable PDF validation or extraction error."""


class PDFExtractionTimeoutError(PDFExtractionError):
    """The isolated PDF parser exceeded its wall-clock deadline."""


@dataclass
class ExtractedElement:
    page_number: int
    element_type: str
    text_content: str
    structured_data: dict[str, Any] | None = None
    bbox: list[float] | None = None
    confidence: float | None = None
    parser_name: str = "pymupdf"
    parser_version: str | None = None
    needs_review: bool = False


@dataclass
class ExtractedPage:
    page_number: int
    text: str
    elements: list[ExtractedElement] = field(default_factory=list)
    needs_review: bool = False


@dataclass
class PDFExtraction:
    content_sha256: str
    page_count: int
    pages: list[ExtractedPage]
    needs_review: bool

    @property
    def retrieval_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)


def _extract_pdf_worker(pdf_bytes: bytes, connection: Any) -> None:
    """Spawn target: parse in an isolated process and send the result back."""
    try:
        connection.send((True, extract_pdf(pdf_bytes)))
    except PDFExtractionError as exc:
        connection.send((False, str(exc)))
    except Exception:
        # Do not send parser exception details or source content to the API caller.
        connection.send((False, "PDF extraction worker failed"))
    finally:
        connection.close()


def extract_pdf_with_timeout(pdf_bytes: bytes, timeout_seconds: float) -> PDFExtraction:
    """Run extraction in a killable process and enforce an overall deadline."""
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_extract_pdf_worker, args=(pdf_bytes, sender), daemon=True)
    try:
        process.start()
        sender.close()
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if receiver.poll(min(0.1, remaining)):
                succeeded, payload = receiver.recv()
                process.join(timeout=1)
                if succeeded:
                    if not isinstance(payload, PDFExtraction):
                        raise PDFExtractionError("PDF extraction worker returned invalid data")
                    return payload
                raise PDFExtractionError(payload)
            if not process.is_alive():
                break
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
            raise PDFExtractionTimeoutError("PDF extraction exceeded its time limit")
        raise PDFExtractionError("PDF extraction worker exited unexpectedly")
    finally:
        receiver.close()
        if not sender.closed:
            sender.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)


def extract_pdf(pdf_bytes: bytes) -> PDFExtraction:
    """Extract native text, OCR spans, and detectable tables from PDF bytes.

    Call from a worker thread: this method performs synchronous parsing and OCR.
    Tesseract's subprocess timeout bounds each OCR page independently.
    """
    try:
        import pymupdf
    except ImportError as exc:
        raise PDFExtractionError("PDF parsing support is unavailable") from exc

    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise PDFExtractionError("The uploaded file is not a readable PDF") from exc

    with document:
        if document.is_encrypted:
            raise PDFExtractionError("Encrypted PDFs are not supported")
        page_count = document.page_count
        if page_count == 0:
            raise PDFExtractionError("The PDF has no pages")
        if page_count > settings.pdf_max_pages:
            raise PDFExtractionError(
                f"PDF page count exceeds the limit of {settings.pdf_max_pages} pages"
            )

        pages: list[ExtractedPage] = []
        extracted_chars = 0
        for page_index in range(page_count):
            page = document.load_page(page_index)
            page_number = page_index + 1
            page_dict = page.get_text("dict", sort=True)
            native_blocks = [
                block
                for block in page_dict.get("blocks", [])
                if block.get("type") == 0 and _block_text(block).strip()
            ]
            native_text = "\n".join(_block_text(block) for block in native_blocks).strip()
            # get_image_info includes inline image occurrences, unlike get_images.
            has_images = bool(page.get_image_info())
            elements: list[ExtractedElement] = []
            for block in native_blocks:
                text = _block_text(block).strip()
                if text:
                    elements.append(
                        ExtractedElement(
                            page_number=page_number,
                            element_type="native_text",
                            text_content=text,
                            bbox=_rect(block.get("bbox")),
                            parser_name="pymupdf",
                            parser_version=pymupdf.VersionBind,
                        )
                    )

            page_text_parts = [native_text] if native_text else []
            page_review = False
            if len(native_text) < settings.pdf_min_native_chars_per_page or has_images:
                ocr_elements = _ocr_page(page, native_blocks, page_number, pymupdf)
                elements.extend(ocr_elements)
                # OCR order and any scanned table structure require source review.
                page_review = True
                ocr_text = "\n".join(element.text_content for element in ocr_elements)
                if ocr_text.strip():
                    page_text_parts.append(ocr_text)
                    if any(
                        element.confidence is None
                        or element.confidence < settings.pdf_ocr_review_confidence
                        for element in ocr_elements
                    ):
                        page_review = True
                else:
                    page_review = True

            table_elements, table_text, table_review = _extract_tables(page, page_number, pymupdf)
            elements.extend(table_elements)
            if table_text:
                page_text_parts.append(table_text)
            page_review = page_review or table_review
            page_text = "\n\n".join(part for part in page_text_parts if part.strip())
            extracted_chars += len(page_text)
            if extracted_chars > settings.pdf_max_extracted_chars:
                raise PDFExtractionError("PDF extracted text exceeds the configured output limit")
            pages.append(ExtractedPage(page_number, page_text, elements, page_review))

    if not any(page.text.strip() for page in pages):
        raise PDFExtractionError("No readable text could be extracted from this PDF")
    return PDFExtraction(
        content_sha256=hashlib.sha256(pdf_bytes).hexdigest(),
        page_count=page_count,
        pages=pages,
        needs_review=any(page.needs_review for page in pages),
    )


def _block_text(block: dict[str, Any]) -> str:
    return "\n".join(
        "".join(span.get("text", "") for span in line.get("spans", []))
        for line in block.get("lines", [])
    )


def _rect(value: Any) -> list[float] | None:
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        return None
    try:
        return [float(part) for part in value]
    except (TypeError, ValueError):
        return None


def _ocr_page(page: Any, native_blocks: list[dict[str, Any]], page_number: int, pymupdf: Any):
    try:
        import pytesseract
        from PIL import Image
        from pytesseract import Output
    except ImportError as exc:
        raise PDFExtractionError(
            "OCR support requires Pillow, pytesseract, and the Tesseract executable"
        ) from exc

    estimated_width = page.rect.width * settings.pdf_render_dpi / 72
    estimated_height = page.rect.height * settings.pdf_render_dpi / 72
    if estimated_width * estimated_height > settings.pdf_max_render_pixels:
        raise PDFExtractionError("PDF page render exceeds the configured pixel limit")

    try:
        pixmap = page.get_pixmap(dpi=settings.pdf_render_dpi, alpha=False)
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        data = pytesseract.image_to_data(
            image,
            output_type=Output.DICT,
            timeout=settings.pdf_ocr_timeout_seconds,
        )
    except RuntimeError as exc:
        raise PDFExtractionError("OCR timed out while processing a PDF page") from exc
    except Exception as exc:
        # Do not return native-only output for a page where OCR was required.
        raise PDFExtractionError(
            "OCR could not process a page; verify that Tesseract and its language data are installed"
        ) from exc

    native_words: list[tuple[str, list[float]]] = []
    for block in native_blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                for word in re.findall(r"\w+", span.get("text", "").casefold()):
                    bbox = _rect(span.get("bbox"))
                    if bbox:
                        native_words.append((word, bbox))

    scale_x = page.rect.width / pixmap.width
    scale_y = page.rect.height / pixmap.height
    line_words: dict[tuple[int, int, int], list[tuple[str, float, list[float]]]] = {}
    for index, raw_text in enumerate(data["text"]):
        word = raw_text.strip()
        if not word:
            continue
        try:
            confidence = float(data["conf"][index])
            x, y = int(data["left"][index]), int(data["top"][index])
            width, height = int(data["width"][index]), int(data["height"][index])
            line_key = (
                int(data["block_num"][index]),
                int(data["par_num"][index]),
                int(data["line_num"][index]),
            )
        except (TypeError, ValueError):
            continue
        if confidence < 0:
            continue
        bbox = [x * scale_x, y * scale_y, (x + width) * scale_x, (y + height) * scale_y]
        word_norms = re.findall(r"\w+", word.casefold())
        if any(_duplicate_word(token, bbox, native_words) for token in word_norms):
            continue
        line_words.setdefault(line_key, []).append((word, confidence, bbox))

    # Keep parser provenance without launching an unbounded second Tesseract process.
    version = getattr(pytesseract, "__version__", None)
    result = []
    for words in line_words.values():
        text = " ".join(word for word, _, _ in words)
        confidence = sum(item[1] for item in words) / len(words)
        boxes = [item[2] for item in words]
        bbox = [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ]
        result.append(
            ExtractedElement(
                page_number=page_number,
                element_type="ocr_text",
                text_content=text,
                bbox=bbox,
                confidence=confidence,
                parser_name="tesseract",
                parser_version=f"pytesseract-{version}" if version else None,
                needs_review=confidence < settings.pdf_ocr_review_confidence,
            )
        )
    return result


def _duplicate_word(
    word: str, bbox: list[float], native_words: list[tuple[str, list[float]]]
) -> bool:
    for native_word, native_box in native_words:
        if word != native_word:
            continue
        intersection_width = max(0.0, min(bbox[2], native_box[2]) - max(bbox[0], native_box[0]))
        intersection_height = max(0.0, min(bbox[3], native_box[3]) - max(bbox[1], native_box[1]))
        intersection = intersection_width * intersection_height
        area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        if intersection / area >= 0.5:
            return True
    return False


def _extract_tables(page: Any, page_number: int, pymupdf: Any):
    try:
        finder = page.find_tables()
        tables = finder.tables
    except Exception:
        # Layout detection is heuristic. Preserve page text and mark it for review.
        return [], "", True

    elements: list[ExtractedElement] = []
    readable_tables: list[str] = []
    # Geometric reconstruction is useful evidence, but is never a verified transcript.
    needs_review = bool(tables)
    for table_index, table in enumerate(tables, 1):
        table_needs_review = True
        rows = [[cell or "" for cell in row] for row in table.extract()]
        header_names = [name or "" for name in table.header.names]
        has_header = any(name.strip() for name in header_names)
        if has_header and not table.header.external and rows:
            rows = rows[1:]
        elif not has_header and rows:
            # Guessing the first row as the header is useful for retrieval but uncertain.
            header_names = rows.pop(0)
            has_header = any(cell.strip() for cell in header_names)
            needs_review = True
            table_needs_review = True
        if not has_header:
            needs_review = True
            table_needs_review = True
        structured = {
            "table_index": table_index,
            "headers": header_names,
            "rows": rows,
            "row_count": len(rows),
            "column_count": max([len(header_names), *(len(row) for row in rows)], default=0),
            "header_detected": bool(any(name.strip() for name in header_names)),
        }
        lines = []
        if header_names:
            lines.append(" | ".join(header_names))
            lines.append(" | ".join("---" for _ in header_names))
        lines.extend(" | ".join(row) for row in rows)
        table_text = "\n".join(lines).strip()
        if table_text:
            readable_tables.append(table_text)
        elements.append(
            ExtractedElement(
                page_number=page_number,
                element_type="table",
                text_content=table_text,
                structured_data=structured,
                bbox=_rect(table.bbox),
                parser_name="pymupdf-table-finder",
                parser_version=pymupdf.VersionBind,
                needs_review=table_needs_review,
            )
        )
    return elements, "\n\n".join(readable_tables), needs_review
