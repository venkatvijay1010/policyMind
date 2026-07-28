"""
SQL Agent - handles claims data queries with self-correction.
"""
from typing import List, Optional, Any
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities.models import QueryResult, QueryType
from app.infrastructure.llm.openai_client import get_openai_client
from app.config import settings

logger = structlog.get_logger()


# Schema information for SQL generation
SCHEMA_INFO = """
Tables:

1. claims (insurance claims)
   - id: INTEGER (PK)
   - claim_number: VARCHAR (unique claim identifier)
   - policy_id: INTEGER (FK to policies)
   - member_id: INTEGER (FK to members)
   - claim_type: VARCHAR ('CASHLESS', 'REIMBURSEMENT')
   - claim_category: VARCHAR ('IPD', 'OPD', 'DAYCARE')
   - diagnosis_code: VARCHAR (ICD-10 code)
   - diagnosis_description: VARCHAR
   - treatment_type: VARCHAR
   - hospital_name: VARCHAR
   - hospital_city: VARCHAR
   - hospital_state: VARCHAR
   - admission_date: DATE
   - discharge_date: DATE
   - claim_amount: DECIMAL (total claimed amount)
   - approved_amount: DECIMAL (approved by insurer)
   - deductible_applied: DECIMAL
   - copay_applied: DECIMAL
   - net_payable: DECIMAL
   - claim_status: VARCHAR ('REGISTERED', 'UNDER_PROCESS', 'APPROVED', 'SETTLED', 'REJECTED')
   - registration_date: DATE
   - settlement_date: DATE
   - rejection_reason: TEXT
   - created_at: TIMESTAMP

2. members (insured members)
   - id: INTEGER (PK)
   - member_id: VARCHAR
   - policy_id: INTEGER (FK)
   - member_name: VARCHAR
   - relationship: VARCHAR ('SELF', 'SPOUSE', 'CHILD', 'PARENT')
   - gender: VARCHAR
   - date_of_birth: DATE
   - age: INTEGER
   - sum_insured: DECIMAL
   - status: VARCHAR ('ACTIVE', 'INACTIVE')
   - city: VARCHAR
   - state: VARCHAR

3. policies (insurance policies)
   - id: INTEGER (PK)
   - policy_number: VARCHAR
   - policy_name: VARCHAR
   - product_type: VARCHAR
   - insured_name: VARCHAR (company name)
   - policy_start_date: DATE
   - policy_end_date: DATE
   - total_lives: INTEGER
   - total_sum_insured: DECIMAL
   - premium_amount: DECIMAL

4. icd_codes (diagnosis codes reference)
   - code: VARCHAR (PK)
   - description: VARCHAR
   - category: VARCHAR
   - is_chronic: BOOLEAN
   - typical_hospitalization_days: INTEGER
   - typical_treatment_cost: DECIMAL
"""


class SQLAgent:
    """
    SQL Agent for claims data queries.
    
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
    
    def _is_safe_sql(self, sql: str) -> bool:
        """
        Validate that SQL is read-only and safe.
        """
        sql_lower = sql.lower().strip()
        
        # Must start with SELECT or WITH (for CTEs)
        if not (sql_lower.startswith("select") or sql_lower.startswith("with")):
            return False
        
        # Banned keywords
        banned = ["insert", "update", "delete", "drop", "alter", "create", "truncate", "grant", "revoke"]
        for keyword in banned:
            if keyword in sql_lower:
                return False
        
        return True
    
    async def generate_sql(
        self,
        query: str,
        error_context: Optional[str] = None
    ) -> dict:
        """
        Generate SQL from natural language.
        """
        result = await self.llm.generate_sql(
            query=query,
            schema_info=SCHEMA_INFO,
            error_context=error_context
        )
        
        return result
    
    async def execute_sql(self, sql: str) -> tuple[List[dict], Optional[str]]:
        """
        Execute SQL and return results or error.
        """
        if not self._is_safe_sql(sql):
            return [], "Query rejected: Only SELECT queries are allowed"
        
        try:
            # Add LIMIT if not present
            sql_lower = sql.lower()
            if "limit" not in sql_lower:
                sql = f"{sql.rstrip(';')} LIMIT {self.MAX_ROWS}"
            
            result = await self.session.execute(text(sql))
            rows = result.fetchall()
            columns = result.keys()
            
            # Convert to list of dicts
            data = [dict(zip(columns, row)) for row in rows]
            
            return data, None
            
        except Exception as e:
            logger.error("SQL execution error", sql=sql[:500], error=str(e))
            return [], str(e)
    
    async def explain_results(
        self,
        query: str,
        sql: str,
        results: List[dict]
    ) -> str:
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
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2
        )
        
        return answer
    
    async def answer(self, query: str) -> QueryResult:
        """
        Full SQL pipeline with self-correction.
        """
        import time
        start_time = time.time()
        
        last_error = None
        sql = None
        results = []
        
        # Try generating and executing SQL with retries
        for attempt in range(self.MAX_RETRIES + 1):
            # Generate SQL
            sql_result = await self.generate_sql(
                query=query,
                error_context=last_error
            )
            
            sql = sql_result.get("sql", "")
            explanation = sql_result.get("explanation", "")
            
            logger.info(
                "Generated SQL",
                attempt=attempt + 1,
                sql=sql[:200],
                explanation=explanation
            )
            
            # Execute SQL
            results, error = await self.execute_sql(sql)
            
            if error is None:
                # Success
                break
            else:
                last_error = error
                logger.warning(
                    "SQL execution failed, retrying",
                    attempt=attempt + 1,
                    error=error
                )
        
        # Generate explanation
        if last_error and not results:
            answer = f"I couldn't execute the query due to an error: {last_error}"
        else:
            answer = await self.explain_results(query, sql, results)
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return QueryResult(
            query=query,
            query_type=QueryType.CLAIMS_SQL,
            answer=answer,
            citations=[],
            sql_query=sql,
            sql_result=results[:20] if results else None,  # Limit results in response
            latency_ms=latency_ms,
            model_used=settings.chat_model
        )
