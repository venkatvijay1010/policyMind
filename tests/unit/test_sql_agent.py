"""Unit tests for SQL read-only enforcement."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.agents.sql_agent import SQLAgent

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_successful_retry_clears_previous_error():
    agent = SQLAgent(AsyncMock())
    agent.generate_sql = AsyncMock(return_value={"sql": "SELECT count(*) FROM service_cases"})
    agent.execute_sql = AsyncMock(side_effect=[([], "temporary failure"), ([{"count": 2}], None)])
    agent.explain_results = AsyncMock(return_value="Two cases.")

    result = await agent.answer("How many cases?")

    assert result.answer == "Two cases."
    assert agent.execute_sql.await_count == 2


@pytest.fixture
def sql_agent() -> tuple[SQLAgent, AsyncMock]:
    session = AsyncMock()
    session.begin_nested = MagicMock(return_value=AsyncMock())
    session.bind = None
    result = MagicMock()
    result.fetchall.return_value = [(1,), (2,)]
    result.keys.return_value = ["id"]
    session.execute.return_value = result
    return SQLAgent(session), session


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id, updated_at FROM service_cases",
        "WITH totals AS (SELECT count(*) AS count FROM service_cases) SELECT * FROM totals",
        "SELECT * FROM service_cases;",
    ],
)
def test_safe_sql_allows_single_read_only_statements(sql_agent, sql):
    agent, _ = sql_agent

    assert agent._is_safe_sql(sql) is True


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM service_cases",
        "SELECT * FROM service_cases; DELETE FROM service_cases",
        "SELECT * FROM service_cases -- hidden statement",
        "WITH removed AS (DELETE FROM service_cases RETURNING *) SELECT * FROM removed",
        "SELECT * INTO archived_cases FROM service_cases",
        "SELECT * FROM service_cases FOR UPDATE",
        "SELECT pg_sleep(100)",
        "SELECT load_extension('unsafe')",
        "SELECT * FROM query_logs",
        "SELECT * FROM pg_catalog.pg_authid",
        "WITH query_logs AS (SELECT * FROM query_logs) SELECT * FROM query_logs",
        "WITH RECURSIVE totals AS (SELECT 1 UNION ALL SELECT 1 FROM totals) SELECT * FROM totals",
    ],
)
def test_safe_sql_rejects_mutations_comments_and_locks(sql_agent, sql):
    agent, _ = sql_agent

    assert agent._is_safe_sql(sql) is False


@pytest.mark.asyncio
async def test_execute_sql_enforces_hard_outer_row_limit(sql_agent):
    agent, session = sql_agent

    rows, error = await agent.execute_sql("SELECT * FROM service_cases LIMIT 100000")

    assert error is None
    assert rows == [{"id": 1}, {"id": 2}]
    executed_sql = str(session.execute.await_args.args[0])
    parameters = session.execute.await_args.args[1]
    assert (
        "SELECT * FROM (SELECT * FROM service_cases LIMIT 100000) AS generated_query"
        in executed_sql
    )
    assert parameters == {"max_rows": agent.MAX_ROWS}


async def test_failed_query_does_not_poison_transaction():
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with async_sessionmaker(engine)() as session:
            agent = SQLAgent(session)
            _, error = await agent.execute_sql("SELECT id FROM service_cases")
            assert error is not None
            rows, error = await agent.execute_sql("SELECT 1 AS count")
            assert error is None
            assert rows == [{"count": 1}]
    finally:
        await engine.dispose()


async def test_sql_execution_timeout(sql_agent, monkeypatch):
    import asyncio

    from app.config import settings

    agent, session = sql_agent
    monkeypatch.setattr(settings, "sql_timeout_seconds", 0.01)

    async def blocked(*args, **kwargs):
        await asyncio.Event().wait()

    session.execute.side_effect = blocked
    _, error = await agent.execute_sql("SELECT 1")
    assert error is not None
