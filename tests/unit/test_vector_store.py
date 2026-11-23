"""Unit tests for the Qdrant vector store and its use by hybrid search."""

import pytest
from pydantic import ValidationError
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.infrastructure.database.models import Base, ContractPassage, Policy
from app.infrastructure.search import hybrid_search
from app.infrastructure.search.hybrid_search import HybridSearch
from app.infrastructure.search.vector_store import QdrantVectorStore, VectorStoreError

pytestmark = pytest.mark.unit


@pytest.fixture
async def store():
    client = AsyncQdrantClient(location=":memory:")
    yield QdrantVectorStore(client, "test_passages")
    await client.close()


async def test_search_ranks_by_similarity_and_filters_by_scope(store):
    await store.replace_scope(1, [(10, [1.0, 0.0]), (11, [0.7, 0.7])])
    await store.replace_scope(2, [(20, [1.0, 0.0])])

    scoped = await store.search([1.0, 0.0], contract_id=1, limit=5)
    unscoped = await store.search([1.0, 0.0], contract_id=None, limit=5)

    assert [hit.passage_id for hit in scoped] == [10, 11]
    assert scoped[0].score > scoped[1].score
    assert {hit.contract_id for hit in scoped} == {1}
    assert {hit.passage_id for hit in unscoped} == {10, 11, 20}


async def test_replace_scope_removes_stale_points_only_in_that_scope(store):
    await store.replace_scope(1, [(10, [1.0, 0.0]), (11, [0.0, 1.0])])
    await store.replace_scope(2, [(20, [1.0, 0.0])])

    await store.replace_scope(1, [(12, [1.0, 0.0])])

    hits = await store.search([1.0, 0.0], contract_id=None, limit=10)
    assert {hit.passage_id for hit in hits} == {12, 20}


async def test_empty_replace_deletes_the_scope(store):
    await store.replace_scope(1, [(10, [1.0, 0.0])])
    await store.replace_scope(2, [(20, [1.0, 0.0])])

    await store.replace_scope(1, [])

    hits = await store.search([1.0, 0.0], contract_id=None, limit=10)
    assert [hit.passage_id for hit in hits] == [20]


async def test_delete_scope(store):
    await store.replace_scope(1, [(10, [1.0, 0.0])])
    await store.replace_scope(2, [(20, [1.0, 0.0])])

    await store.delete_scope(1)

    hits = await store.search([1.0, 0.0], contract_id=None, limit=10)
    assert [hit.passage_id for hit in hits] == [20]


async def test_missing_collection_is_empty_and_deletable(store):
    assert await store.search([1.0, 0.0], contract_id=1, limit=5) == []
    await store.delete_scope(1)


async def test_client_failure_raises_vector_store_error():
    class FailingClient:
        async def collection_exists(self, collection_name):
            return True

        async def query_points(self, **kwargs):
            raise RuntimeError("connection refused")

    failing = QdrantVectorStore(FailingClient(), "test_passages")

    with pytest.raises(VectorStoreError):
        await failing.search([1.0, 0.0], contract_id=None, limit=3)


class FakeEmbeddings:
    async def embed_query(self, query):
        return [1.0, 0.0]


async def test_hybrid_vector_search_uses_qdrant_and_skips_stale_points(store, monkeypatch):
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
        near = ContractPassage(contract_id=first.id, content="near", passage_order=0)
        far = ContractPassage(contract_id=first.id, content="far", passage_order=1)
        other = ContractPassage(contract_id=second.id, content="other", passage_order=0)
        session.add_all([near, far, other])
        await session.commit()

        await store.replace_scope(first.id, [(near.id, [1.0, 0.0]), (far.id, [0.6, 0.8])])
        # Points whose row is missing, or whose row now belongs to another scope.
        await store.replace_scope(99, [(9999, [1.0, 0.0]), (other.id, [1.0, 0.0])])
        monkeypatch.setattr(hybrid_search, "get_vector_store", lambda: store)
        search = HybridSearch(session)
        search.embedding_service = FakeEmbeddings()

        results = await search.vector_search("query", contract_id=None, top_k=10)
        scoped = await search.vector_search("query", contract_id=first.id, top_k=10)

    await engine.dispose()
    assert [item.chunk.content for item in results if item.chunk.contract_id == first.id] == [
        "near",
        "far",
    ]
    assert all(item.chunk.id not in (9999, other.id) for item in results)
    assert [item.chunk.content for item in scoped] == ["near", "far"]
    assert scoped[0].score == pytest.approx(1.0)
    assert scoped[1].score == pytest.approx(0.6)
    assert all(item.source == "vector" for item in scoped)


def test_qdrant_backend_requires_a_url():
    with pytest.raises(ValidationError, match="QDRANT_URL"):
        Settings(_env_file=None, vector_backend="qdrant", qdrant_url="")


async def test_exact_search_matches_the_approximate_result_on_a_tiny_collection(store):
    await store.replace_scope(1, [(10, [1.0, 0.0]), (11, [0.7, 0.7]), (12, [0.0, 1.0])])
    approximate = await store.search([1.0, 0.0], contract_id=None, limit=3, hnsw_ef=16)
    exact = await store.search([1.0, 0.0], contract_id=None, limit=3, exact=True)
    assert [h.passage_id for h in approximate] == [h.passage_id for h in exact]


async def test_sample_vectors_returns_stored_vectors(store):
    await store.replace_scope(1, [(10, [1.0, 0.0]), (11, [0.0, 1.0])])
    assert len(await store.sample_vectors(5)) == 2
