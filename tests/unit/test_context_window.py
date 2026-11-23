import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.application.agents.rag_agent import RAGAgent
from app.config import settings
from app.domain.entities.models import DocumentChunk
from app.domain.services.context_window import merge_passage_window
from app.infrastructure.database.models import Base, ContractPassage, Policy

pytestmark = pytest.mark.unit

TEXT = "Alpha clause one. Beta clause two. Gamma clause three. Delta clause four."


def _rows(*spans: tuple[int, int]):
    return [(TEXT[start:end], start, end) for start, end in spans]


def test_overlap_is_trimmed_using_offsets():
    # Second passage starts inside the first (overlap of 7 chars).
    assert merge_passage_window(_rows((0, 24), (17, 42))) == TEXT[:42]


def test_gap_between_passages_is_joined_with_a_space():
    assert merge_passage_window(_rows((0, 17), (18, 35))) == f"{TEXT[:17]} {TEXT[18:35]}"


def test_missing_offsets_fall_back_to_plain_join():
    assert merge_passage_window([("a", None, None), ("b", None, None)]) == "a b"


@pytest.fixture
async def session():
    engine = create_async_engine(
        "sqlite+aiosqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as db:
        db.add(Policy(id=1, contract_ref="P-1"))
        spans = [(0, 17), (18, 35), (36, 54), (55, 72)]
        for order, (start, end) in enumerate(spans):
            db.add(
                ContractPassage(
                    contract_id=1,
                    content=TEXT[start:end],
                    passage_order=order,
                    source_offset_start=start,
                    source_offset_end=end,
                )
            )
        await db.commit()
        yield db
    await engine.dispose()


async def test_expansion_is_off_by_default(session):
    chunk = DocumentChunk(id=2, contract_id=1, content="Beta clause two.", passage_order=1)
    result = await RAGAgent(session)._expand_with_neighbors([chunk])
    assert result[0].context_text is None


async def test_expansion_adds_neighbours_but_keeps_matched_content(session, monkeypatch):
    monkeypatch.setattr(settings, "context_neighbor_passages", 1)
    chunk = DocumentChunk(id=2, contract_id=1, content="Beta clause two.", passage_order=1)
    result = await RAGAgent(session)._expand_with_neighbors([chunk])
    assert result[0].content == "Beta clause two."
    assert result[0].context_text == f"{TEXT[:17]} {TEXT[18:35]} {TEXT[36:54]}"
