"""Unit tests for source-document provenance, object storage and the schema upgrade."""

import hashlib
import io

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, inspect, text

from app.application import source_documents
from app.application.source_documents import (
    delete_stored_objects,
    locate_texts,
    new_text_source_document,
    read_original,
    reconstruct_text,
    source_kind,
    store_original,
)
from app.config import Settings
from app.domain.entities.models import DocumentChunk
from app.domain.services.citation_builder import CitationBuilder
from app.infrastructure.database.models import SourceDocument
from app.infrastructure.database.postgres import add_missing_columns
from app.infrastructure.storage import object_storage
from app.infrastructure.storage.object_storage import (
    S3ObjectStorage,
    StorageError,
    StoredObject,
    build_object_key,
)

pytestmark = pytest.mark.unit


class FakeS3Client:
    """Records boto3-style calls and can be told to fail."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], tuple[bytes, str]] = {}
        self.fail = False

    def put_object(self, *, Bucket, Key, Body, ContentType):  # noqa: N803 - boto3 keyword names
        if self.fail:
            raise RuntimeError("AccessDenied: secret-detail")
        self.objects[(Bucket, Key)] = (Body, ContentType)
        return {"ETag": '"abc123"'}

    def get_object(self, *, Bucket, Key):  # noqa: N803 - boto3 keyword names
        if self.fail:
            raise RuntimeError("AccessDenied: secret-detail")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)][0])}

    def delete_object(self, *, Bucket, Key):  # noqa: N803 - boto3 keyword names
        if self.fail:
            raise RuntimeError("AccessDenied: secret-detail")
        self.objects.pop((Bucket, Key), None)
        return {}


class FakeStorage:
    """In-memory stand-in for S3ObjectStorage used by the provenance helpers."""

    bucket = "policy-bucket"

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.fail_get = False
        self.failing_keys: set[str] = set()
        self.deleted: list[tuple[str, str]] = []

    async def put(self, key, data, content_type):
        self.objects[(self.bucket, key)] = data
        return StoredObject(bucket=self.bucket, key=key, etag="etag-1")

    async def get(self, bucket, key):
        if self.fail_get:
            raise StorageError("unavailable")
        return self.objects[(bucket, key)]

    async def delete(self, bucket, key):
        if key in self.failing_keys:
            raise StorageError("denied")
        self.deleted.append((bucket, key))
        self.objects.pop((bucket, key), None)


def _source(**overrides) -> SourceDocument:
    values = {
        "id": 42,
        "contract_id": 7,
        "filename": "policy.pdf",
        "content_type": "application/pdf",
        "content_sha256": "abcdef0123456789" * 4,
        "original_bytes": b"%PDF-original",
    }
    values.update(overrides)
    return SourceDocument(**values)


@pytest.mark.parametrize(
    ("filename", "content_type", "expected"),
    [
        ("Policy.PDF", "application/octet-stream", "pdf"),
        ("notes.md", "", "text"),
        ("notes.txt", "application/octet-stream", "text"),
        ("blob.bin", "application/pdf", "pdf"),
        ("blob.bin", "text/plain; charset=utf-8", "text"),
        ("letter.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml", "other"),
    ],
)
def test_source_kind_uses_extension_then_media_type(filename, content_type, expected):
    assert source_kind(filename, content_type) == expected


def test_object_key_is_built_from_ids_and_hash_only(monkeypatch):
    monkeypatch.setattr(object_storage.settings, "s3_prefix", "/archive/docs/")

    key = build_object_key(7, 42, "abcdef0123456789ffff", "../../Evil Name.PDF")

    assert key == "archive/docs/7/42-abcdef0123456789.pdf"


def test_object_key_drops_unknown_extensions_and_empty_prefix(monkeypatch):
    monkeypatch.setattr(object_storage.settings, "s3_prefix", "")

    assert build_object_key(7, 42, "abcdef0123456789ffff", "payload.exe") == (
        "7/42-abcdef0123456789"
    )


def test_locate_texts_follows_reading_order_across_whitespace():
    haystack = "Alpha beta\n\nGamma delta"

    assert locate_texts(haystack, ["Alpha beta", "Gamma delta"]) == [(0, 10), (12, 23)]


def test_locate_texts_reports_missing_elements_without_losing_the_rest():
    haystack = "Alpha beta\n\nGamma delta"

    located = locate_texts(haystack, ["Alpha beta", "Zeta", "", "Gamma delta"])

    assert located == [(0, 10), None, None, (12, 23)]


def test_locate_texts_never_matches_a_repeat_past_unrelated_text():
    haystack = "Intro\n\nTotal: 5\n\nMore text\n\nTotal: 5"

    located = locate_texts(haystack, ["Intro", "Total: 5", "Total: 5"])

    assert located == [(0, 5), (7, 15), None]


def test_reconstruct_text_places_passages_at_their_exact_offsets():
    original = "Alpha beta gamma.\n\nDelta epsilon."

    rebuilt = reconstruct_text([("Alpha beta gamma.", 0), ("Delta epsilon.", 19)])

    assert rebuilt == (original, [0, 19])


def test_reconstruct_text_realigns_passages_recorded_too_early():
    # The second passage truly starts at 6 but an older chunker recorded 4.
    rebuilt = reconstruct_text([("abcdefghij", 0), ("ghijklmnop", 4)])

    assert rebuilt == ("abcdefghijklmnop", [0, 6])


def test_reconstruct_text_gives_up_when_overlaps_contradict():
    assert reconstruct_text([("abcdefghij" * 20, 0), ("Z" * 50, 10)]) is None


def test_new_text_source_document_sanitises_name_and_hashes_content():
    source = new_text_source_document(7, "Coverage text é", "Policy: A/B?.md", origin="seed")

    assert source.filename == "Policy_ A_B_.md"
    assert source.content_type == "text/markdown"
    assert source.original_bytes == "Coverage text é".encode("utf-8")
    assert source.content_sha256 == hashlib.sha256(source.original_bytes).hexdigest()
    assert source.upload_metadata == {"source": "seed"}
    assert source.extraction_status == "complete"
    assert source.page_count is None


def test_new_text_source_document_falls_back_for_unusable_names():
    assert new_text_source_document(7, "text", "...", origin="seed").filename == "source.txt"
    assert new_text_source_document(7, "text", "title.txt", origin="seed").content_type == (
        "text/plain"
    )


@pytest.mark.asyncio
async def test_s3_storage_round_trips_bytes_and_unquotes_the_etag():
    client = FakeS3Client()
    storage = S3ObjectStorage(client, "policy-bucket")

    stored = await storage.put("a/b.pdf", b"data", "application/pdf")

    assert stored == StoredObject(bucket="policy-bucket", key="a/b.pdf", etag="abc123")
    assert client.objects[("policy-bucket", "a/b.pdf")] == (b"data", "application/pdf")
    assert await storage.get("policy-bucket", "a/b.pdf") == b"data"
    await storage.delete("policy-bucket", "a/b.pdf")
    assert client.objects == {}


@pytest.mark.asyncio
async def test_s3_storage_hides_provider_errors_behind_storage_error():
    client = FakeS3Client()
    client.fail = True
    storage = S3ObjectStorage(client, "policy-bucket")

    for call in (
        storage.put("k", b"x", "text/plain"),
        storage.get("policy-bucket", "k"),
        storage.delete("policy-bucket", "k"),
    ):
        with pytest.raises(StorageError) as error:
            await call
        assert "secret-detail" not in str(error.value)


def test_database_backend_does_not_create_a_storage_client(monkeypatch):
    monkeypatch.setattr(object_storage.settings, "storage_backend", "database")
    object_storage.get_object_storage.cache_clear()
    try:
        assert object_storage.get_object_storage() is None
    finally:
        object_storage.get_object_storage.cache_clear()


def test_s3_backend_builds_a_client_for_the_configured_bucket(monkeypatch):
    monkeypatch.setattr(object_storage.settings, "storage_backend", "s3")
    monkeypatch.setattr(object_storage.settings, "s3_bucket", "policy-bucket")
    monkeypatch.setattr(object_storage.settings, "s3_region", "us-east-1")
    monkeypatch.setattr(object_storage.settings, "s3_endpoint_url", "http://127.0.0.1:9")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    object_storage.get_object_storage.cache_clear()
    try:
        storage = object_storage.get_object_storage()
        assert isinstance(storage, S3ObjectStorage)
        assert storage.bucket == "policy-bucket"
    finally:
        object_storage.get_object_storage.cache_clear()


def test_s3_backend_requires_a_bucket():
    with pytest.raises(ValidationError, match="S3_BUCKET"):
        Settings(_env_file=None, storage_backend="s3", s3_bucket="  ")


@pytest.mark.asyncio
async def test_store_original_records_the_object_location():
    storage = FakeStorage()
    source = _source()

    await store_original(source, storage)

    assert source.storage_backend == "s3"
    assert source.storage_bucket == "policy-bucket"
    assert source.storage_key == build_object_key(7, 42, source.content_sha256, "policy.pdf")
    assert source.storage_etag == "etag-1"
    assert source.storage_uploaded_at is not None
    assert storage.objects[("policy-bucket", source.storage_key)] == b"%PDF-original"
    # The database copy is kept, so reads keep working if the bucket is unreachable.
    assert source.original_bytes == b"%PDF-original"


@pytest.mark.asyncio
async def test_read_original_prefers_object_storage_and_falls_back_to_database(monkeypatch):
    storage = FakeStorage()
    monkeypatch.setattr(source_documents, "get_object_storage", lambda: storage)
    source = _source()
    await store_original(source, storage)
    storage.objects[(source.storage_bucket, source.storage_key)] = b"from-bucket"

    assert await read_original(source) == b"from-bucket"

    storage.fail_get = True
    assert await read_original(source) == b"%PDF-original"


@pytest.mark.asyncio
async def test_read_original_uses_the_database_copy_without_a_storage_location(monkeypatch):
    def unexpected():
        raise AssertionError("storage must not be consulted")

    monkeypatch.setattr(source_documents, "get_object_storage", unexpected)

    assert await read_original(_source()) == b"%PDF-original"


@pytest.mark.asyncio
async def test_delete_stored_objects_continues_after_a_failure(monkeypatch):
    storage = FakeStorage()
    storage.objects = {("policy-bucket", "a"): b"1", ("policy-bucket", "b"): b"2"}
    storage.failing_keys = {"a"}
    monkeypatch.setattr(source_documents, "get_object_storage", lambda: storage)

    await delete_stored_objects([("policy-bucket", "a"), ("policy-bucket", "b")])

    assert storage.deleted == [("policy-bucket", "b")]


@pytest.mark.asyncio
async def test_delete_stored_objects_ignores_unavailable_storage(monkeypatch):
    def broken():
        raise StorageError("misconfigured")

    monkeypatch.setattr(source_documents, "get_object_storage", broken)

    await delete_stored_objects([("policy-bucket", "a")])


def test_add_missing_columns_upgrades_an_old_schema_and_is_idempotent():
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        add_missing_columns(connection)  # no tables yet: nothing to upgrade
        connection.execute(
            text("CREATE TABLE source_documents (id INTEGER PRIMARY KEY, filename VARCHAR(255))")
        )
        connection.execute(
            text("CREATE TABLE document_elements (id INTEGER PRIMARY KEY, page_number INTEGER)")
        )
        connection.execute(text("INSERT INTO source_documents (id, filename) VALUES (1, 'a.pdf')"))

        add_missing_columns(connection)
        add_missing_columns(connection)

        inspector = inspect(connection)
        source_columns = {column["name"] for column in inspector.get_columns("source_documents")}
        element_columns = {column["name"] for column in inspector.get_columns("document_elements")}
        row = connection.execute(text("SELECT filename, storage_key FROM source_documents")).one()

    assert {
        "storage_backend",
        "storage_bucket",
        "storage_key",
        "storage_etag",
        "storage_uploaded_at",
    } <= source_columns
    assert {"text_offset_start", "text_offset_end"} <= element_columns
    assert tuple(row) == ("a.pdf", None)
    engine.dispose()


def test_citations_carry_the_passage_and_source_document_ids():
    chunk = DocumentChunk(
        7,
        1,
        "Dental care is covered.",
        source_page=2,
        source_document_id=3,
        source_filename="p.pdf",
    )

    citation = CitationBuilder.build_citations([chunk])[0]

    assert citation.passage_id == 7
    assert citation.source_document_id == 3
    assert citation.source_filename == "p.pdf"
    assert citation.page == 2
