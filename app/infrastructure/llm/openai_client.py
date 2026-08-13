"""
OpenAI client wrapper for LLM operations.
"""
import json
from typing import List, Optional, Any
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

from app.config import settings

logger = structlog.get_logger()


class OpenAIClient:
    """Async OpenAI client wrapper with retry logic."""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.chat_model = settings.chat_model
        self.embedding_model = settings.embedding_model
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        response_format: Optional[dict] = None
    ) -> str:
        """
        Generate a response from the LLM.
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        kwargs = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if response_format:
            kwargs["response_format"] = response_format
        
        logger.debug("Calling OpenAI", model=self.chat_model, prompt_length=len(prompt))
        
        response = await self.client.chat.completions.create(**kwargs)
        
        content = response.choices[0].message.content
        
        logger.debug(
            "OpenAI response received",
            model=self.chat_model,
            tokens=response.usage.total_tokens if response.usage else None
        )
        
        return content
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def generate_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000
    ) -> dict:
        """
        Generate a structured JSON response from the LLM.
        """
        content = await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON response", error=str(e), content=content[:500])
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")
    
    async def classify_query(self, query: str) -> dict:
        """
        Classify a query into document_qa, records_sql, or hybrid.
        """
        system_prompt = """You are a query classifier for an insurance policy Q&A system.

Classify the user's question into one of these categories:
- document_qa: Questions about policy coverage, exclusions, limits, terms, conditions, waiting periods
- records_sql: Questions about synthetic service-case data, statistics, amounts, counts, and trends
- hybrid: Questions that need both policy documents AND service_cases data

Return JSON with:
{
    "query_type": "document_qa" | "records_sql" | "hybrid",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}"""

        prompt = f"Classify this query: {query}"
        
        result = await self.generate_structured(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.0
        )
        
        return result
    
    async def generate_sql(
        self,
        query: str,
        schema_info: str,
        error_context: Optional[str] = None
    ) -> dict:
        """
        Generate SQL from natural language query.
        """
        system_prompt = """You are a SQL generator for an insurance service_cases database.

Generate PostgreSQL queries that are:
- Read-only (SELECT only, no INSERT/UPDATE/DELETE)
- Safe (no SQL injection, use parameterized patterns)
- Efficient (use appropriate indexes)

Available tables and columns:
{schema_info}

Return JSON with:
{{
    "sql": "SELECT ...",
    "explanation": "What this query does",
    "columns_returned": ["col1", "col2"]
}}"""

        prompt = f"Generate SQL for: {query}"
        
        if error_context:
            prompt += f"\n\nPrevious attempt failed with error: {error_context}\nPlease fix the query."
        
        result = await self.generate_structured(
            prompt=prompt,
            system_prompt=system_prompt.format(schema_info=schema_info),
            temperature=0.0
        )
        
        return result
    
    async def generate_answer(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate an answer based on retrieved context.
        """
        if not system_prompt:
            system_prompt = """You are an insurance policy expert assistant.

Answer the user's question based ONLY on the provided context.
- Be accurate and specific
- Cite the source sections when possible
- If the information is not in the context, say "I don't have information about this in the provided policy documents"
- Never make up information
- Use clear, professional language

Format your answer clearly with bullet points or numbered lists when appropriate."""

        prompt = f"""Context from policy documents:
{context}

User Question: {query}

Answer:"""

        return await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=1500
        )
    
    async def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1
    ) -> dict:
        """
        Alias for generate_structured for JSON output.
        """
        return await self.generate_structured(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature
        )


# Singleton instance
_client: Optional[OpenAIClient] = None


def get_openai_client() -> OpenAIClient:
    """Get or create OpenAI client singleton."""
    global _client
    if _client is None:
        _client = OpenAIClient()
    return _client
