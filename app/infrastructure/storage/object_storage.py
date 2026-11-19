"""Private S3 storage for original source documents."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import lru_cache
from pathlib import PurePath
from typing import Any

from app.config import settings

# The client-supplied filename never becomes part of an object key. Only these known
# extensions are kept so a stored object is recognisable in the S3 console.
_KEY_EXTENSIONS = {".pdf", ".md", ".txt"}


class StorageError(RuntimeError):
    """The configured object store could not complete a request."""


@dataclass(frozen=True)
class StoredObject:
    bucket: str
    key: str
    etag: str | None


def build_object_key(
    contract_id: int, source_document_id: int, content_sha256: str, filename: str
) -> str:
    """Build a unique key from database ids and the content hash."""
    suffix = PurePath(filename).suffix.lower()
    extension = suffix if suffix in _KEY_EXTENSIONS else ""
    name = f"{source_document_id}-{content_sha256[:16]}{extension}"
    prefix = settings.s3_prefix.strip("/")
    return "/".join(part for part in (prefix, str(contract_id), name) if part)


class S3ObjectStorage:
    """Async wrapper over a boto3 S3 client; blocking calls run in a worker thread."""

    def __init__(self, client: Any, bucket: str) -> None:
        self._client = client
        self.bucket = bucket

    async def put(self, key: str, data: bytes, content_type: str) -> StoredObject:
        try:
            response = await asyncio.to_thread(
                self._client.put_object,
                Bucket=self.bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
        except Exception as exc:  # botocore raises many unrelated exception types
            raise StorageError("Unable to store the source document") from exc
        etag = str(response.get("ETag") or "").strip('"') or None
        return StoredObject(bucket=self.bucket, key=key, etag=etag)

    async def get(self, bucket: str, key: str) -> bytes:
        try:
            response = await asyncio.to_thread(self._client.get_object, Bucket=bucket, Key=key)
            return await asyncio.to_thread(response["Body"].read)
        except Exception as exc:
            raise StorageError("Unable to read the source document") from exc

    async def delete(self, bucket: str, key: str) -> None:
        try:
            await asyncio.to_thread(self._client.delete_object, Bucket=bucket, Key=key)
        except Exception as exc:
            raise StorageError("Unable to delete the source document") from exc


@lru_cache
def get_object_storage() -> S3ObjectStorage | None:
    """Return the configured object store, or None when originals stay in the database."""
    if settings.storage_backend != "s3":
        return None
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise StorageError("STORAGE_BACKEND=s3 requires the boto3 package") from exc

    client = boto3.client(
        "s3",
        region_name=settings.s3_region or None,
        endpoint_url=settings.s3_endpoint_url or None,
        config=Config(
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=5,
            read_timeout=30,
        ),
    )
    return S3ObjectStorage(client, settings.s3_bucket)
