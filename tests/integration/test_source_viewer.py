"""Integration tests for opening the exact evidence source behind a citation."""

import importlib.util
from collections import Counter
from io import BytesIO
from pathlib import Path

import pymupdf
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from PIL import Image, ImageChops
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.api.routes import ingest
from app.application import source_documents
from app.config import settings
from app.infrastructure.database.models import (
    ContractPassage,
    DocumentElement,
    PassageSource,
    Policy,
    SourceDocument,
)
from app.infrastructure.database.postgres import Base, get_db_session
from app.infrastructure.documents.pdf_extractor import extract_pdf
from app.infrastructure.storage.object_storage import StorageError, StoredObject

pytestmark = pytest.mark.integration

API = "/api/v2/knowledge"

PAGE_ONE = [
    "Eligibility overview. Employees become members after ninety days of continuous service "
    "and remain members while they stay on the payroll of the sponsoring employer."
]
PAGE_TWO = [
    "Dental benefits. Preventive dental cleanings and routine examinations are covered twice "
    "per calendar year with no deductible applied to the member.",
    "Vision benefits. One eye examination and one pair of corrective lenses are covered every "
    "twenty four months under this plan for each enrolled member.",
    "Maternity benefits. Prenatal visits, delivery and postnatal care are covered when they "
    "are provided by an in-network provider under this benefit plan.",
]
MARKDOWN = (
    "# Policy Guide \U0001f642\n\n"
    "## Eligibility\nMembers qualify after ninety days of continuous service under this plan.\n\n"
    "## Dental\nPreventive cleanings are covered twice per year with no deductible applied.\n\n"
    "## Vision\nOne eye examination and one pair of lenses are covered every two years.\n"
)


class FakeEmbeddings:
    async def embed_batch(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeStorage:
    """In-memory stand-in for the S3 bucket."""

    bucket = "policy-bucket"

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.deleted: list[tuple[str, str]] = []
        self.fail_put = False
        self.fail_get = False

    async def put(self, key, data, content_type):
        if self.fail_put:
            raise StorageError("denied")
        self.objects[(self.bucket, key)] = data
        return StoredObject(bucket=self.bucket, key=key, etag="etag-1")

    async def get(self, bucket, key):
        if self.fail_get:
            raise StorageError("unavailable")
        return self.objects[(bucket, key)]

    async def delete(self, bucket, key):
        self.deleted.append((bucket, key))
        self.objects.pop((bucket, key), None)


@pytest.fixture
def storage(monkeypatch):
    fake = FakeStorage()
    monkeypatch.setattr(ingest, "get_object_storage", lambda: fake)
    monkeypatch.setattr(source_documents, "get_object_storage", lambda: fake)
    monkeypatch.setattr(ingest, "get_embedding_service", lambda: FakeEmbeddings())
    # Extract in-process: the production wrapper spawns a worker process per upload.
    monkeypatch.setattr(
        ingest, "extract_pdf_with_timeout", lambda content, timeout: extract_pdf(content)
    )
    return fake


@pytest_asyncio.fixture
async def session_factory():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


@pytest_asyncio.fixture
async def client(session_factory, storage):
    from app.main import app

    async def override_session():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db_session] = override_session
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
            yield http
    finally:
        app.dependency_overrides.pop(get_db_session, None)


def _pdf(pages: list[list[str]], rotation: int = 0) -> bytes:
    """Build a PDF in which every string on a page is its own text block."""
    document = pymupdf.open()
    for blocks in pages:
        page = document.new_page()
        for index, block in enumerate(blocks):
            top = 60 + index * 150
            page.insert_textbox(pymupdf.Rect(60, top, 540, top + 120), block, fontsize=11)
        if rotation:
            page.set_rotation(rotation)
    data = document.tobytes()
    document.close()
    return data


async def _upload(client, filename, content, content_type, *, chunk_size=100, chunk_overlap=10):
    return await client.post(
        f"{API}/scopes/file",
        params={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        files={"file": (filename, content, content_type)},
        data={"title": "Test Policy"},
    )


async def _upload_pdf(client, data, **options):
    response = await _upload(client, "policy.pdf", data, "application/pdf", **options)
    assert response.status_code == 201, response.text
    return response.json()


async def _upload_markdown(client, content=MARKDOWN, filename="guide.md"):
    response = await _upload(client, filename, content.encode("utf-8"), "text/markdown")
    assert response.status_code == 201, response.text
    return response.json()


async def _passage_pages(session_factory, source_document_id) -> dict[int, list[int]]:
    async with session_factory() as session:
        rows = (
            await session.execute(
                select(PassageSource.passage_id, PassageSource.page_number).where(
                    PassageSource.source_document_id == source_document_id
                )
            )
        ).all()
    pages: dict[int, list[int]] = {}
    for passage_id, page in rows:
        pages.setdefault(passage_id, []).append(page)
    return {passage_id: sorted(values) for passage_id, values in pages.items()}


def _passage_on(pages_by_passage: dict[int, list[int]], pages: list[int]) -> int:
    matches = [passage_id for passage_id, found in pages_by_passage.items() if found == pages]
    assert matches, f"no passage covers exactly pages {pages}: {pages_by_passage}"
    return matches[0]


def _tint_and_ink_boxes(png: bytes):
    """Bounding boxes of the amber highlight and of the dark text in a rendered page."""
    image = Image.open(BytesIO(png)).convert("RGB")
    red, _, blue = image.split()
    tint = ImageChops.subtract(red, blue).point(lambda value: 255 if value > 40 else 0)
    ink = image.convert("L").point(lambda value: 255 if value < 100 else 0)
    return tint.getbbox(), ink.getbbox()


def _load_backfill():
    path = Path(__file__).resolve().parents[2] / "scripts" / "backfill_source_storage.py"
    spec = importlib.util.spec_from_file_location("backfill_source_storage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestUploadStorage:
    async def test_upload_records_the_s3_location_and_keeps_the_database_copy(
        self, client, session_factory, storage
    ):
        original = _pdf([PAGE_ONE, PAGE_TWO])

        body = await _upload_pdf(client, original)

        async with session_factory() as session:
            source = await session.get(SourceDocument, body["source_document_id"])
        assert source.storage_backend == "s3"
        assert source.storage_bucket == storage.bucket
        assert source.storage_key.startswith(f"{settings.s3_prefix}/{body['scope_key']}/")
        assert source.storage_key.endswith(".pdf")
        assert source.storage_etag == "etag-1"
        assert source.storage_uploaded_at is not None
        assert storage.objects[(source.storage_bucket, source.storage_key)] == original
        assert source.original_bytes == original

    async def test_failed_upload_aborts_indexing_and_leaves_no_rows(
        self, client, session_factory, storage
    ):
        storage.fail_put = True

        response = await _upload(client, "notes.md", MARKDOWN.encode(), "text/markdown")

        assert response.status_code == 502
        assert response.json()["detail"] == "Unable to store the source document"
        async with session_factory() as session:
            assert (await session.scalars(select(SourceDocument))).all() == []
            assert (await session.scalars(select(Policy))).all() == []

    async def test_misconfigured_storage_is_reported_as_unavailable(self, client, monkeypatch):
        def broken():
            raise StorageError("no bucket configured")

        monkeypatch.setattr(ingest, "get_object_storage", broken)

        response = await _upload(client, "notes.md", MARKDOWN.encode(), "text/markdown")

        assert response.status_code == 503
        assert "bucket" not in response.text

    async def test_text_only_ingestion_creates_an_openable_source(
        self, client, session_factory, storage
    ):
        async with session_factory() as session:
            policy = Policy(contract_ref="TXT-1", contract_title="Plain Policy")
            session.add(policy)
            await session.commit()
            scope_key = policy.id

        response = await client.put(
            f"{API}/scopes/{scope_key}/source",
            json={"source_text": MARKDOWN, "segment_length": 100, "segment_overlap": 10},
        )

        assert response.status_code == 200, response.text
        source_id = response.json()["source_document_id"]
        async with session_factory() as session:
            source = await session.get(SourceDocument, source_id)
            links = (
                await session.scalars(
                    select(PassageSource).where(PassageSource.source_document_id == source_id)
                )
            ).all()
            passages = (
                await session.scalars(select(ContractPassage).order_by(ContractPassage.id))
            ).all()
        assert source.filename == "Plain Policy.txt"
        assert source.content_type == "text/plain"
        assert source.upload_metadata == {"source": "source_text"}
        assert storage.objects[(source.storage_bucket, source.storage_key)] == MARKDOWN.encode()
        assert len(links) == len(passages) > 1
        assert all(link.page_number is None for link in links)

        opened = await client.get(
            f"{API}/sources/{source_id}/open", params={"passage_id": passages[1].id}
        )
        assert opened.json()["highlighted"] is True

    async def test_deleting_an_uploaded_document_removes_its_stored_object(
        self, client, session_factory, storage
    ):
        body = await _upload_markdown(client)
        assert len(storage.objects) == 1
        stored = list(storage.objects)

        response = await client.delete(f"{API}/scopes/{body['scope_key']}")

        assert response.status_code == 200
        assert storage.deleted == stored
        assert storage.objects == {}


class TestOpenSource:
    async def test_pdf_source_opens_on_the_cited_page_with_the_paragraph_marked(
        self, client, session_factory
    ):
        body = await _upload_pdf(client, _pdf([PAGE_ONE, PAGE_TWO]))
        source_id = body["source_document_id"]
        pages = await _passage_pages(session_factory, source_id)

        second_page = await client.get(
            f"{API}/sources/{source_id}/open", params={"passage_id": _passage_on(pages, [2])}
        )
        opened = second_page.json()
        assert second_page.status_code == 200
        assert opened["kind"] == "pdf"
        assert opened["filename"] == "policy.pdf"
        assert opened["page"] == 2
        assert opened["page_count"] == 2
        assert opened["highlighted"] is True
        assert opened["file_url"] == f"{API}/sources/{source_id}/file"

        # A passage that spans two pages opens on the page where it starts.
        spanning = await client.get(
            f"{API}/sources/{source_id}/open", params={"passage_id": _passage_on(pages, [1, 2])}
        )
        assert spanning.json()["page"] == 1

        plain = (await client.get(f"{API}/sources/{source_id}/open")).json()
        assert plain["page"] == 1
        assert plain["highlighted"] is False

    async def test_text_source_highlight_uses_utf16_offsets(self, client, session_factory):
        body = await _upload_markdown(client)
        source_id = body["source_document_id"]
        async with session_factory() as session:
            passages = (
                await session.scalars(select(ContractPassage).order_by(ContractPassage.id))
            ).all()
        cited = passages[-1]

        response = await client.get(
            f"{API}/sources/{source_id}/open", params={"passage_id": cited.id}
        )

        opened = response.json()
        assert opened["kind"] == "text"
        assert opened["page"] is None
        assert opened["highlighted"] is True
        assert opened["text"] == MARKDOWN
        # Browsers slice by UTF-16 units; the emoji in the title shifts every later offset.
        utf16 = opened["text"].encode("utf-16-le")
        selected = utf16[opened["highlight_start"] * 2 : opened["highlight_end"] * 2]
        assert selected.decode("utf-16-le") == cited.content
        assert opened["highlight_start"] == cited.source_offset_start + 1

    async def test_a_passage_from_another_document_is_rejected(self, client, session_factory):
        first = await _upload_markdown(client)
        second = await _upload_markdown(client, MARKDOWN + "\n## Extra\nAnother section.\n")
        first_passages = await _passage_pages(session_factory, first["source_document_id"])

        response = await client.get(
            f"{API}/sources/{second['source_document_id']}/open",
            params={"passage_id": next(iter(first_passages))},
        )

        assert response.status_code == 404

    async def test_unknown_and_unsupported_sources(self, client, session_factory):
        assert (await client.get(f"{API}/sources/999/open")).status_code == 404

        async with session_factory() as session:
            policy = Policy(contract_ref="DOCX-1", contract_title="Letter")
            session.add(policy)
            await session.flush()
            session.add(
                SourceDocument(
                    contract_id=policy.id,
                    filename="letter.docx",
                    content_type="application/octet-stream",
                    content_sha256="0" * 64,
                    original_bytes=b"PK\x03\x04",
                    upload_metadata={},
                    extraction_status="complete",
                    needs_review=False,
                )
            )
            await session.commit()
            source_id = (await session.scalars(select(SourceDocument.id))).one()

        assert (await client.get(f"{API}/sources/{source_id}/open")).status_code == 415
        # Unknown types are only ever offered as a download, never rendered inline.
        download = await client.get(f"{API}/sources/{source_id}/file")
        assert download.headers["content-type"] == "application/octet-stream"
        assert download.headers["content-disposition"].startswith("attachment;")


class TestSourceFile:
    async def test_pdf_and_text_files_are_served_with_safe_headers(self, client):
        original = _pdf([PAGE_ONE])
        pdf = await _upload_pdf(client, original)
        markdown = await _upload_markdown(client)

        pdf_response = await client.get(f"{API}/sources/{pdf['source_document_id']}/file")
        text_response = await client.get(f"{API}/sources/{markdown['source_document_id']}/file")

        assert pdf_response.content == original
        assert pdf_response.headers["content-type"] == "application/pdf"
        assert pdf_response.headers["content-disposition"].startswith(
            'inline; filename="policy.pdf"'
        )
        assert text_response.headers["content-type"] == "text/plain; charset=utf-8"
        for response in (pdf_response, text_response):
            assert response.headers["x-content-type-options"] == "nosniff"

    async def test_file_is_read_from_the_bucket_and_falls_back_to_the_database(
        self, client, session_factory, storage
    ):
        original = _pdf([PAGE_ONE])
        body = await _upload_pdf(client, original)
        source_id = body["source_document_id"]
        async with session_factory() as session:
            source = await session.get(SourceDocument, source_id)
        replacement = _pdf([PAGE_TWO])
        storage.objects[(source.storage_bucket, source.storage_key)] = replacement

        assert (await client.get(f"{API}/sources/{source_id}/file")).content == replacement

        storage.fail_get = True
        assert (await client.get(f"{API}/sources/{source_id}/file")).content == original

    async def test_filenames_cannot_inject_headers(self, client, session_factory):
        body = await _upload_markdown(client)
        async with session_factory() as session:
            source = await session.get(SourceDocument, body["source_document_id"])
            source.filename = 'evil"\r\nX-Injected: 1.md'
            await session.commit()

        response = await client.get(f"{API}/sources/{source.id}/file")

        assert "x-injected" not in response.headers
        disposition = response.headers["content-disposition"]
        assert "\r" not in disposition and "\n" not in disposition
        assert 'filename="evil_X-Injected_1.md"' in disposition


class TestPageImage:
    async def test_image_marks_only_the_cited_paragraphs(self, client, session_factory):
        body = await _upload_pdf(client, _pdf([PAGE_ONE, PAGE_TWO]))
        source_id = body["source_document_id"]
        passage_id = _passage_on(await _passage_pages(session_factory, source_id), [2])

        response = await client.get(
            f"{API}/sources/{source_id}/pages/2/image", params={"passage_id": passage_id}
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.headers["x-content-type-options"] == "nosniff"
        async with session_factory() as session:
            blocks = (
                await session.scalars(
                    select(DocumentElement).where(
                        DocumentElement.source_document_id == source_id,
                        DocumentElement.page_number == 2,
                    )
                )
            ).all()
        assert len(blocks) == len(PAGE_TWO)
        assert 1 <= int(response.headers["x-highlight-count"]) < len(blocks)
        tint, _ = _tint_and_ink_boxes(response.content)
        assert tint is not None
        height = Image.open(BytesIO(response.content)).height
        assert float(response.headers["x-highlight-top"]) == pytest.approx(
            tint[1] / height, abs=0.02
        )

        unmarked = await client.get(f"{API}/sources/{source_id}/pages/2/image")
        assert unmarked.headers["x-highlight-count"] == "0"
        assert "x-highlight-top" not in unmarked.headers
        assert _tint_and_ink_boxes(unmarked.content)[0] is None

    @pytest.mark.parametrize("rotation", [0, 90, 270])
    async def test_highlight_lines_up_with_the_text_on_rotated_pages(
        self, client, session_factory, rotation
    ):
        body = await _upload_pdf(
            client, _pdf([PAGE_TWO[:1]], rotation=rotation), chunk_size=2000, chunk_overlap=0
        )
        source_id = body["source_document_id"]
        passage_id = next(iter(await _passage_pages(session_factory, source_id)))

        response = await client.get(
            f"{API}/sources/{source_id}/pages/1/image", params={"passage_id": passage_id}
        )

        assert response.headers["x-highlight-count"] == "1"
        tint, ink = _tint_and_ink_boxes(response.content)
        assert tint is not None and ink is not None
        margin = 6
        assert tint[0] - margin <= ink[0] and tint[1] - margin <= ink[1]
        assert tint[2] + margin >= ink[2] and tint[3] + margin >= ink[3]
        tint_area = (tint[2] - tint[0]) * (tint[3] - tint[1])
        ink_area = (ink[2] - ink[0]) * (ink[3] - ink[1])
        assert tint_area < 1.5 * ink_area

    async def test_image_rejects_bad_requests(self, client, session_factory):
        pdf = await _upload_pdf(client, _pdf([PAGE_ONE]))
        markdown = await _upload_markdown(client)
        other_pages = await _passage_pages(session_factory, markdown["source_document_id"])
        pdf_id, markdown_id = pdf["source_document_id"], markdown["source_document_id"]

        assert (await client.get(f"{API}/sources/{pdf_id}/pages/0/image")).status_code == 422
        assert (await client.get(f"{API}/sources/{pdf_id}/pages/2/image")).status_code == 404
        assert (await client.get(f"{API}/sources/{markdown_id}/pages/1/image")).status_code == 404
        foreign = await client.get(
            f"{API}/sources/{pdf_id}/pages/1/image", params={"passage_id": next(iter(other_pages))}
        )
        assert foreign.status_code == 404


class TestBackfill:
    async def test_legacy_scope_gets_a_reconstructed_source_document(self, session_factory):
        backfill = _load_backfill()
        chunks = ingest.chunk_text(MARKDOWN, chunk_size=100, chunk_overlap=10)
        async with session_factory() as session:
            policy = Policy(contract_ref="LEGACY-1", contract_title="Legacy Policy")
            session.add(policy)
            await session.flush()
            session.add_all(
                ContractPassage(
                    contract_id=policy.id,
                    content=chunk["content"],
                    passage_order=chunk["passage_order"],
                    source_offset_start=chunk["source_offset_start"],
                    source_offset_end=chunk["source_offset_end"],
                    embedding=[0.1],
                )
                for chunk in chunks
            )
            await session.commit()

        async with session_factory() as session:
            preview = Counter()
            await backfill._attach_missing_sources(session, True, preview)
            assert preview["scopes_attached"] == 1
            assert (await session.scalars(select(SourceDocument))).all() == []

            stats = Counter()
            await backfill._attach_missing_sources(session, False, stats)
            assert stats["scopes_attached"] == 1

            source = (await session.scalars(select(SourceDocument))).one()
            passages = (
                await session.scalars(select(ContractPassage).order_by(ContractPassage.id))
            ).all()
            links = (await session.scalars(select(PassageSource))).all()
            rebuilt = source.original_bytes.decode("utf-8")
            assert all(
                rebuilt[passage.source_offset_start : passage.source_offset_end] == passage.content
                for passage in passages
            )
            assert source.filename == "Legacy Policy.txt"
            assert source.upload_metadata == {"source": "reconstructed_from_passages"}
            assert len(links) == len(passages)

            repeat = Counter()
            await backfill._attach_missing_sources(session, False, repeat)
            assert not repeat

    async def test_pdf_element_offsets_are_restored(self, client, session_factory):
        backfill = _load_backfill()
        await _upload_pdf(client, _pdf([PAGE_ONE, PAGE_TWO]))

        async def offsets():
            async with session_factory() as session:
                elements = (await session.scalars(select(DocumentElement))).all()
            return {item.id: (item.text_offset_start, item.text_offset_end) for item in elements}

        before = await offsets()
        assert before and all(start is not None for start, _ in before.values())
        async with session_factory() as session:
            await session.execute(
                text(
                    "UPDATE document_elements SET text_offset_start = NULL, text_offset_end = NULL"
                )
            )
            await session.commit()

        async with session_factory() as session:
            stats = Counter()
            await backfill._fill_element_offsets(session, False, stats)

        assert await offsets() == before
        assert stats["elements_located"] == len(before)

    async def test_documents_missing_from_the_bucket_are_uploaded(
        self, client, session_factory, storage, monkeypatch
    ):
        backfill = _load_backfill()
        monkeypatch.setattr(backfill, "get_object_storage", lambda: storage)
        body = await _upload_markdown(client)
        async with session_factory() as session:
            await session.execute(
                text(
                    "UPDATE source_documents SET storage_backend = NULL, storage_bucket = NULL, "
                    "storage_key = NULL, storage_etag = NULL, storage_uploaded_at = NULL"
                )
            )
            await session.commit()
        storage.objects.clear()

        async with session_factory() as session:
            preview = Counter()
            await backfill._upload_missing_objects(session, True, preview)
            assert preview["objects_to_upload"] == 1
            assert storage.objects == {}

            stats = Counter()
            await backfill._upload_missing_objects(session, False, stats)
        assert stats["objects_uploaded"] == 1

        async with session_factory() as session:
            source = await session.get(SourceDocument, body["source_document_id"])
        assert storage.objects[(source.storage_bucket, source.storage_key)] == (
            source.original_bytes
        )
