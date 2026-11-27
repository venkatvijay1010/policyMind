"""Unit tests for the Elasticsearch keyword store and its use by hybrid search."""

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.infrastructure.database.models import Base, ContractPassage, Policy
from app.infrastructure.search import hybrid_search
from app.infrastructure.search.hybrid_search import HybridSearch
from app.infrastructure.search.keyword_store import (
    ElasticsearchKeywordStore,
    KeywordHit,
    KeywordStoreError,
)

pytestmark = pytest.mark.unit


def test_elasticsearch_backend_requires_a_url():
    with pytest.raises(ValidationError, match="ELASTICSEARCH_URL"):
        Settings(_env_file=None, bm25_backend="elasticsearch", elasticsearch_url="")


async def test_client_failure_raises_keyword_store_error():
    class FailingIndices:
        async def exists(self, index):
            return True

    class FailingClient:
        indices = FailingIndices()

        async def search(self, **kwargs):
            raise RuntimeError("connection refused")

    store = ElasticsearchKeywordStore(FailingClient(), "test")

    with pytest.raises(KeywordStoreError):
        await store.search("query", contract_id=None, limit=3)


async def test_missing_index_is_empty():
    class Indices:
        async def exists(self, index):
            return False

    class Client:
        indices = Indices()

    store = ElasticsearchKeywordStore(Client(), "test")

    assert await store.search("query", contract_id=1, limit=3) == []
    await store.delete_scope(1)


class FakeKeywordStore:
    def __init__(self, hits):
        self.hits = hits

    async def search(self, query, contract_id, limit):
        return self.hits


async def test_hybrid_bm25_uses_elasticsearch_and_skips_stale_documents(monkeypatch):
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with sessions() as session:
        first = Policy(contract_ref="P-1", contract_title="First")
        second = Policy(contract_ref="P-2", contract_title="Second")
        session.add_all([first, second])
        await session.flush()
        best = ContractPassage(contract_id=first.id, content="best", passage_order=0)
        weak = ContractPassage(contract_id=first.id, content="weak", passage_order=1)
        other = ContractPassage(contract_id=second.id, content="other", passage_order=0)
        session.add_all([best, weak, other])
        await session.commit()

        hits = [
            KeywordHit(best.id, first.id, 8.0),
            KeywordHit(9999, first.id, 6.0),
            KeywordHit(other.id, first.id, 5.0),
            KeywordHit(weak.id, first.id, 2.0),
        ]
        monkeypatch.setattr(hybrid_search, "get_keyword_store", lambda: FakeKeywordStore(hits))

        results = await HybridSearch(session).bm25_search("query", contract_id=None, top_k=10)

    await engine.dispose()
    assert [item.chunk.content for item in results] == ["best", "weak"]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.25)
    assert all(item.source == "bm25" for item in results)
