"""Unit tests for SQL read-only enforcement."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.agents.sql_agent import SQLAgent

pytestmark = pytest.mark.unit


@pytest.fixture
def sql_agent() -> tuple[SQLAgent, AsyncMock]:
    session = AsyncMock()
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
