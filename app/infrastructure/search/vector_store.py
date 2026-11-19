"""Qdrant vector index for passage embeddings; the database stays the source of truth."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from qdrant_client import AsyncQdrantClient, models

from app.config import settings

_BATCH_SIZE = 256


class VectorStoreError(RuntimeError):
    """Qdrant could not complete the request."""


@dataclass(frozen=True)
class VectorHit:
    passage_id: int
    contract_id: Optional[int]
    score: float  # cosine similarity


def _scope_filter(contract_id: int) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="contract_id", match=models.MatchValue(value=contract_id))]
    )


class QdrantVectorStore:
    """Points use the passage row id as their id and carry the owning scope as payload."""

    def __init__(self, client: AsyncQdrantClient, collection: str):
        self._client = client
        self._collection = collection
        self._ready = False
        self._lock = asyncio.Lock()

    async def _collection_exists(self) -> bool:
        if not self._ready and await self._client.collection_exists(self._collection):
            self._ready = True
        return self._ready

    async def _ensure_collection(self, size: int) -> None:
        async with self._lock:
            if await self._collection_exists():
                return
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
            )
            await self._client.create_payload_index(
                collection_name=self._collection,
                field_name="contract_id",
                field_schema=models.PayloadSchemaType.INTEGER,
            )
            self._ready = True

    async def search(
        self, vector: Sequence[float], contract_id: Optional[int], limit: int
    ) -> list[VectorHit]:
        try:
            if not await self._collection_exists():
                return []
            response = await self._client.query_points(
                collection_name=self._collection,
                query=list(vector),
                query_filter=_scope_filter(contract_id) if contract_id is not None else None,
                limit=limit,
                with_payload=["contract_id"],
            )
        except Exception as exc:
            raise VectorStoreError("Vector search failed") from exc
        return [
            VectorHit(
                passage_id=int(point.id),
                contract_id=(point.payload or {}).get("contract_id"),
                score=float(point.score),
            )
            for point in response.points
        ]

    async def replace_scope(
        self, contract_id: int, items: Sequence[tuple[int, Sequence[float]]]
    ) -> None:
        """Upsert the scope's passage vectors, then drop that scope's older points."""
        if not items:
            await self.delete_scope(contract_id)
            return
        try:
            await self._ensure_collection(len(items[0][1]))
            for start in range(0, len(items), _BATCH_SIZE):
                await self._client.upsert(
                    collection_name=self._collection,
                    points=[
                        models.PointStruct(
                            id=passage_id,
                            vector=list(vector),
                            payload={"contract_id": contract_id},
                        )
                        for passage_id, vector in items[start : start + _BATCH_SIZE]
                    ],
                    wait=True,
                )
            stale = models.Filter(
                must=_scope_filter(contract_id).must,
                must_not=[models.HasIdCondition(has_id=[passage_id for passage_id, _ in items])],
            )
            await self._client.delete(
                collection_name=self._collection,
                points_selector=models.FilterSelector(filter=stale),
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreError("Vector index update failed") from exc

    async def delete_scope(self, contract_id: int) -> None:
        try:
            if not await self._collection_exists():
                return
            await self._client.delete(
                collection_name=self._collection,
                points_selector=models.FilterSelector(filter=_scope_filter(contract_id)),
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreError("Vector index cleanup failed") from exc

    async def close(self) -> None:
        await self._client.close()


_store: Optional[QdrantVectorStore] = None


def get_vector_store() -> Optional[QdrantVectorStore]:
    """Return the Qdrant store, or None when vectors live in the database."""
    global _store
    if settings.vector_backend != "qdrant":
        return None
    if _store is None:
        client = AsyncQdrantClient(
            location=settings.qdrant_url, api_key=settings.qdrant_api_key or None
        )
        _store = QdrantVectorStore(client, settings.qdrant_collection)
    return _store


async def close_vector_store() -> None:
    global _store
    if _store is not None:
        await _store.close()
        _store = None
