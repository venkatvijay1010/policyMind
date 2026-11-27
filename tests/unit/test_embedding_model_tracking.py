"""Passages record their embedding model, and vectors from another model are not ranked."""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.infrastructure.database.models import Base, ContractPassage, Policy
from app.infrastructure.database.postgres import add_missing_columns
from app.infrastructure.search.hybrid_search import HybridSearch

pytestmark = pytest.mark.unit


class FakeEmbeddings:
    async def embed_query(self, query):
        return [1.0, 0.0]


async def test_vector_search_skips_passages_from_another_model(monkeypatch):
    monkeypatch.setattr(settings, "embedding_model", "current-model")
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with sessions() as session:
        policy = Policy(contract_ref="P-1", contract_title="First")
        session.add(policy)
        await session.flush()
        session.add_all(
            [
                ContractPassage(
                    contract_id=policy.id,
                    content="current",
                    passage_order=0,
                    embedding=[1.0, 0.0],
                    embedding_model="current-model",
                ),
                ContractPassage(
                    contract_id=policy.id,
                    content="other",
                    passage_order=1,
                    embedding=[1.0, 0.0],
                    embedding_model="old-model",
                ),
                ContractPassage(
                    contract_id=policy.id,
                    content="legacy",
                    passage_order=2,
                    embedding=[1.0, 0.0],
                ),
            ]
        )
        await session.commit()
        search = HybridSearch(session)
        search.embedding_service = FakeEmbeddings()

        results = await search.vector_search("query", contract_id=policy.id, top_k=10)

    await engine.dispose()
    assert sorted(item.chunk.content for item in results) == ["current", "legacy"]


async def test_older_databases_gain_the_embedding_model_column():
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    async with engine.begin() as connection:
        await connection.execute(
            text("CREATE TABLE contract_passages (id INTEGER PRIMARY KEY, content TEXT)")
        )
        await connection.run_sync(add_missing_columns)
        await connection.run_sync(add_missing_columns)
        columns = await connection.execute(text("PRAGMA table_info(contract_passages)"))
        names = {row[1] for row in columns}
    await engine.dispose()
    assert "embedding_model" in names
