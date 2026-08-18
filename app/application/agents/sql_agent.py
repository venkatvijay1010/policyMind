"""
SQL Agent - handles service_cases data queries with self-correction.
"""

import re
from typing import Any, List, Optional, cast

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.domain.entities.models import QueryResult, QueryType
from app.infrastructure.llm.openai_client import get_openai_client

logger = structlog.get_logger()


# Schema information for SQL generation
SCHEMA_INFO = """
Tables:

1. service_cases (insurance service_cases)
   - id: INTEGER (PK)
   - case_ref: VARCHAR (unique synthetic case reference)
   - contract_id: INTEGER (FK to benefit_contracts)
   - participant_id: INTEGER (FK to participants)
   - funding_mode: VARCHAR ('DIRECT_BILLING', 'MEMBER_PAID')
   - care_setting: VARCHAR ('IPD', 'OPD', 'DAYCARE')
   - condition_code: VARCHAR (ICD-10 code)
   - condition_label: VARCHAR
   - service_category: VARCHAR
   - provider_label: VARCHAR
   - provider_city: VARCHAR
   - provider_region: VARCHAR
   - service_started_on: DATE
   - service_ended_on: DATE
   - requested_amount: DECIMAL (amount requested for the service case)
   - eligible_amount: DECIMAL (amount eligible under the synthetic contract)
   - fixed_share_applied: DECIMAL
   - percentage_share_applied: DECIMAL
   - payable_amount: DECIMAL
   - case_status: VARCHAR ('OPENED', 'IN_REVIEW', 'ELIGIBLE', 'RESOLVED', 'DECLINED')
   - submitted_on: DATE
   - resolved_on: DATE
   - decision_reason: TEXT
   - created_at: TIMESTAMP

2. participants (insured participants)
   - id: INTEGER (PK)
   - participant_ref: VARCHAR
   - contract_id: INTEGER (FK)
   - participant_label: VARCHAR
   - enrolment_role: VARCHAR ('SELF', 'SPOUSE', 'CHILD', 'PARENT')
   - gender: VARCHAR
   - birth_date: DATE
   - age: INTEGER
   - benefit_ceiling: DECIMAL
   - status: VARCHAR ('ACTIVE', 'INACTIVE')
   - city: VARCHAR
   - state: VARCHAR

3. benefit_contracts (insurance benefit_contracts)
   - id: INTEGER (PK)
   - contract_ref: VARCHAR
   - contract_title: VARCHAR
   - plan_category: VARCHAR
   - sponsor_label: VARCHAR (company name)
   - effective_from: DATE
   - effective_until: DATE
   - participant_count: INTEGER
   - aggregate_benefit_cap: DECIMAL
   - contribution_amount: DECIMAL

4. icd_codes (diagnosis codes reference)
   - code: VARCHAR (PK)
   - description: VARCHAR
   - category: VARCHAR
   - is_chronic: BOOLEAN
   - typical_hospitalization_days: INTEGER
   - typical_treatment_cost: DECIMAL
"""


class SQLExecutionError(RuntimeError):
    """Raised when generated SQL cannot be safely executed."""


_READ_ONLY_START = re.compile(r"^(?:select|with)\b", re.IGNORECASE)
_FORBIDDEN_SQL_KEYWORDS = re.compile(
    r"\b(?:alter|analyze|call|copy|create|delete|discard|do|drop|execute|grant|"
    r"insert|listen|lock|merge|notify|refresh|revoke|set|show|truncate|unlisten|"
    r"update|vacuum)\b",
    re.IGNORECASE,
)
_LOCKING_CLAUSE = re.compile(
    r"\bfor\s+(?:no\s+key\s+update|key\s+share|share|update)\b", re.IGNORECASE
)
_SELECT_INTO = re.compile(r"\bselect\b[\s\S]*\binto\b", re.IGNORECASE)


class SQLAgent:
    """
    SQL Agent for service_cases data queries.

    Features:
    - Natural language to SQL conversion
    - Self-correction on errors (max 2 retries)
    - Read-only query enforcement
    - Result explanation
    """

    MAX_RETRIES = 2
    MAX_ROWS = 100

    def __init__(self, session: AsyncSession):
        self.session = session
        self.llm = get_openai_client()

    def _normalize_read_only_sql(self, sql: str) -> Optional[str]:
        """
        Validate and normalize one read-only SQL statement.

        This is intentionally conservative. A database role with read-only grants and
        a statement timeout are still required production defenses.
        """
        normalized_sql = sql.strip()
        if normalized_sql.endswith(";"):
            normalized_sql = normalized_sql[:-1].rstrip()

        if not normalized_sql or not _READ_ONLY_START.match(normalized_sql):
            return None
        if (
            ";" in normalized_sql
            or "--" in normalized_sql
            or "/*" in normalized_sql
            or "*/" in normalized_sql
        ):
            return None
        if _FORBIDDEN_SQL_KEYWORDS.search(normalized_sql):
            return None
        if _LOCKING_CLAUSE.search(normalized_sql) or _SELECT_INTO.search(normalized_sql):
            return None

        return normalized_sql

    def _is_safe_sql(self, sql: str) -> bool:
        """Return whether SQL is a single non-locking read-only statement."""
        return self._normalize_read_only_sql(sql) is not None

    async def generate_sql(self, query: str, error_context: Optional[str] = None) -> dict[str, Any]:
        """
        Generate SQL from natural language.
        """
        result = await self.llm.generate_sql(
            query=query, schema_info=SCHEMA_INFO, error_context=error_context
        )

        return result

    async def execute_sql(self, sql: str) -> tuple[List[dict[str, Any]], Optional[str]]:
        """
        Execute SQL and return results or error.
        """
        normalized_sql = self._normalize_read_only_sql(sql)
        if normalized_sql is None:
            return [], "Query rejected: Only SELECT queries are allowed"

        try:
            # An outer cap also bounds LLM-provided LIMIT values without trying to
            # rewrite the generated statement's syntax.
            bounded_sql = f"SELECT * FROM ({normalized_sql}) AS generated_query LIMIT :max_rows"
            result = await self.session.execute(text(bounded_sql), {"max_rows": self.MAX_ROWS})
            rows = result.fetchall()
            columns = result.keys()

            # Convert to list of dicts
            data = [dict(zip(columns, row)) for row in rows]

            return data, None

        except Exception as exc:
            logger.error("SQL execution error", sql=normalized_sql[:500], error=str(exc))
            return [], str(exc)

    async def explain_results(self, query: str, sql: str, results: List[dict[str, Any]]) -> str:
        """
        Generate a natural language explanation of the results.
        """
        # Format results for the prompt
        if not results:
            results_str = "No results found."
        elif len(results) <= 10:
            results_str = str(results)
        else:
            results_str = f"First 10 of {len(results)} results:\n{str(results[:10])}"

        prompt = f"""The user asked: {query}

This SQL was executed:
{sql}

Results:
{results_str}

Provide a clear, concise answer to the user's question based on these results.
Include key numbers and insights. Format with bullet points if appropriate."""

        system_prompt = """You are a data analyst explaining query results.
Be precise with numbers. Highlight key insights.
If no results were found, explain what that means."""

        answer = await self.llm.generate(
            prompt=prompt, system_prompt=system_prompt, temperature=0.2
        )

        return cast(str, answer)

    async def answer(self, query: str) -> QueryResult:
        """
        Full SQL pipeline with self-correction.
        """
        import time

        start_time = time.time()

        last_error: Optional[str] = None
        sql: Optional[str] = None
        results: List[dict[str, Any]] = []

        # Try generating and executing SQL with retries
        for attempt in range(self.MAX_RETRIES + 1):
            # Generate SQL
            sql_result = await self.generate_sql(query=query, error_context=last_error)

            sql = str(sql_result.get("sql", ""))
            explanation = str(sql_result.get("explanation", ""))

            logger.info(
                "Generated SQL", attempt=attempt + 1, sql=sql[:200], explanation=explanation
            )

            # Execute SQL
            results, error = await self.execute_sql(sql)

            if error is None:
                # Success
                break
            else:
                last_error = error
                logger.warning("SQL execution failed, retrying", attempt=attempt + 1, error=error)

        if last_error is not None:
            raise SQLExecutionError("Unable to execute the generated query")

        answer = await self.explain_results(query, sql or "", results)

        latency_ms = int((time.time() - start_time) * 1000)

        return QueryResult(
            query=query,
            query_type=QueryType.RECORDS_SQL,
            answer=answer,
            citations=[],
            sql_query=sql,
            sql_result=results[:20] if results else None,  # Limit results in response
            latency_ms=latency_ms,
            model_used=settings.chat_model,
        )
