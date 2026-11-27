"""Elasticsearch BM25 index for passage text; the database stays the source of truth."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk

from app.config import settings


class KeywordStoreError(RuntimeError):
    """Elasticsearch could not complete the request."""


@dataclass(frozen=True)
class KeywordHit:
    passage_id: int
    contract_id: Optional[int]
    score: float  # raw BM25 score


def _scope_filter(contract_id: int) -> dict:
    return {"term": {"contract_id": contract_id}}


class ElasticsearchKeywordStore:
    """Documents use the passage row id as their id and carry the owning scope."""

    def __init__(self, client: AsyncElasticsearch, index: str):
        self._client = client
        self._index = index
        self._ready = False
        self._lock = asyncio.Lock()

    async def _ensure_index(self) -> None:
        async with self._lock:
            if self._ready:
                return
            if not await self._client.indices.exists(index=self._index):
                await self._client.indices.create(
                    index=self._index,
                    mappings={
                        "properties": {
                            "contract_id": {"type": "integer"},
                            "content": {"type": "text", "analyzer": "english"},
                        }
                    },
                )
            self._ready = True

    async def search(self, query: str, contract_id: Optional[int], limit: int) -> list[KeywordHit]:
        try:
            if not await self._client.indices.exists(index=self._index):
                return []
            response = await self._client.search(
                index=self._index,
                size=limit,
                query={
                    "bool": {
                        "must": [{"match": {"content": query}}],
                        "filter": [_scope_filter(contract_id)] if contract_id is not None else [],
                    }
                },
                source=["contract_id"],
            )
        except Exception as exc:
            raise KeywordStoreError("Keyword search failed") from exc
        return [
            KeywordHit(
                passage_id=int(hit["_id"]),
                contract_id=hit["_source"].get("contract_id"),
                score=float(hit["_score"]),
            )
            for hit in response["hits"]["hits"]
        ]

    async def replace_scope(self, contract_id: int, items: Sequence[tuple[int, str]]) -> None:
        """Index the scope's passages, then drop that scope's older documents."""
        if not items:
            await self.delete_scope(contract_id)
            return
        try:
            await self._ensure_index()
            await async_bulk(
                self._client,
                (
                    {
                        "_index": self._index,
                        "_id": passage_id,
                        "_source": {"contract_id": contract_id, "content": content},
                    }
                    for passage_id, content in items
                ),
                refresh=True,
            )
            await self._client.delete_by_query(
                index=self._index,
                query={
                    "bool": {
                        "filter": [_scope_filter(contract_id)],
                        "must_not": [{"ids": {"values": [str(i) for i, _ in items]}}],
                    }
                },
                refresh=True,
            )
        except Exception as exc:
            raise KeywordStoreError("Keyword index update failed") from exc

    async def delete_scope(self, contract_id: int) -> None:
        try:
            if not await self._client.indices.exists(index=self._index):
                return
            await self._client.delete_by_query(
                index=self._index, query=_scope_filter(contract_id), refresh=True
            )
        except Exception as exc:
            raise KeywordStoreError("Keyword index cleanup failed") from exc

    async def close(self) -> None:
        await self._client.close()


_store: Optional[ElasticsearchKeywordStore] = None


def get_keyword_store() -> Optional[ElasticsearchKeywordStore]:
    """Return the Elasticsearch store, or None when BM25 runs in-process."""
    global _store
    if settings.bm25_backend != "elasticsearch":
        return None
    if _store is None:
        client = AsyncElasticsearch(
            settings.elasticsearch_url,
            api_key=settings.elasticsearch_api_key or None,
            request_timeout=settings.node_timeout_seconds,
        )
        _store = ElasticsearchKeywordStore(client, settings.elasticsearch_index)
    return _store


async def close_keyword_store() -> None:
    global _store
    if _store is not None:
        await _store.close()
        _store = None
