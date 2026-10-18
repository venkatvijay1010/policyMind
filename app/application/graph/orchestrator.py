"""
LangGraph Orchestrator - State machine for query routing and execution.
"""

import asyncio
from time import perf_counter
from typing import Awaitable, Callable, NotRequired, Optional, TypedDict, cast
from uuid import uuid4

import structlog
from langgraph.graph import END, StateGraph
from openai import APIConnectionError, APIStatusError, RateLimitError
from sqlalchemy.ext.asyncio import AsyncSession

from app.application.agents.conversation_agent import ConversationAgent
from app.application.agents.hybrid_agent import HybridAgent
from app.application.agents.query_classifier import QueryClassifier, RoutingError
from app.application.agents.rag_agent import RAGAgent
from app.application.agents.sql_agent import SQLAgent
from app.config import settings
from app.domain.entities.models import ClassificationResult, QueryResult, QueryType
from app.domain.services.citation_builder import CitationBuilder

logger = structlog.get_logger()


class GraphExecutionError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class GraphState(TypedDict):
    """State that flows through the graph."""

    # Input
    query: str
    contract_id: Optional[int]
    search_method: str
    conversation: list[dict[str, str]]

    # Classification
    classification: Optional[ClassificationResult]
    query_type: Optional[QueryType]

    # Execution
    result: Optional[QueryResult]
    error: Optional[str]

    # Metadata
    current_node: str
    request_id: NotRequired[str]
    started_at: NotRequired[float]
    attempts: NotRequired[int]
    retryable: NotRequired[bool]
    error_code: NotRequired[Optional[str]]
    resolved_query: NotRequired[str]


class PolicyMindGraph:
    """
    LangGraph-based orchestrator for PolicyMind queries.

    Flow:
    1. classify -> Determine query type
    2. Route to the appropriate agent (chat/rag/sql/hybrid)
    3. Clarify missing intent or execute and validate the result

    Graph:
        classify -> clarify or agent -> retry / finalize / failure -> END
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.classifier = QueryClassifier()
        self.chat_agent = ConversationAgent()
        self.rag_agent = RAGAgent(session)
        self.sql_agent = SQLAgent(session)
        self.hybrid_agent = HybridAgent(session)

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""

        # Create graph with state schema
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("classify", self._classify_node)
        workflow.add_node("chat", self._chat_node)
        workflow.add_node("rag", self._rag_node)
        workflow.add_node("sql", self._sql_node)
        workflow.add_node("hybrid", self._hybrid_node)
        workflow.add_node("retry", self._retry_node)
        workflow.add_node("finalize", self._finalize_node)
        workflow.add_node("failure", self._failure_node)
        workflow.add_node("clarify", self._clarify_node)

        # Set entry point
        workflow.set_entry_point("classify")

        # Add conditional routing from classify node
        workflow.add_conditional_edges(
            "classify",
            self._route,
            {
                "chat": "chat",
                "rag": "rag",
                "sql": "sql",
                "hybrid": "hybrid",
                "clarify": "clarify",
                "failure": "failure",
            },
        )
        workflow.add_edge("clarify", "finalize")

        for node_name in ("chat", "rag", "sql", "hybrid"):
            workflow.add_conditional_edges(
                node_name,
                self._after_execution,
                {"retry": "retry", "finalize": "finalize", "failure": "failure"},
            )
        workflow.add_conditional_edges(
            "retry",
            self._route,
            {"chat": "chat", "rag": "rag", "sql": "sql", "hybrid": "hybrid", "failure": "failure"},
        )
        workflow.add_conditional_edges(
            "finalize",
            self._after_finalization,
            {"done": END, "failure": "failure"},
        )
        workflow.add_edge("failure", END)

        return workflow.compile()

    async def _classify_node(self, state: GraphState) -> dict:
        """Classification node - determines query type."""
        try:
            classification = await asyncio.wait_for(
                self.classifier.classify(state["query"], state.get("conversation", [])),
                timeout=settings.node_timeout_seconds,
            )

            logger.info(
                "Query classified",
                query_type=classification.query_type.value if classification.query_type else None,
                decision=classification.decision,
                request_id=state.get("request_id"),
            )

            return {
                "classification": classification,
                "query_type": classification.query_type,
                "current_node": "classify",
                "resolved_query": classification.resolved_query or state["query"],
            }
        except Exception as e:
            logger.warning(
                "Classification failed",
                error_type=type(e).__name__,
                request_id=state.get("request_id"),
            )
            return {
                "query_type": None,
                "error": "Unable to determine the required query tools.",
                "error_code": e.code if isinstance(e, RoutingError) else "routing_unavailable",
                "current_node": "classify",
            }

    def _clarify_node(self, state: GraphState) -> dict:
        classification = state.get("classification")
        if classification is None or not classification.clarification_question:
            return {
                "result": None,
                "error_code": "invalid_routing",
                "error": "Missing clarification question.",
            }
        return {
            "result": QueryResult(
                query=state["query"],
                query_type=QueryType.CLARIFICATION,
                answer=classification.clarification_question,
                model_used=settings.chat_model,
            ),
            "current_node": "clarify",
        }

    def _after_execution(self, state: GraphState) -> str:
        if state.get("result") is not None:
            return "finalize"
        if state.get("retryable") and state.get("attempts", 0) < settings.graph_max_retries:
            return "retry"
        return "failure"

    def _retry_node(self, state: GraphState) -> dict:
        return {"attempts": state.get("attempts", 0) + 1, "current_node": "retry"}

    def _after_finalization(self, state: GraphState) -> str:
        return "done" if state.get("result") is not None else "failure"

    def _finalize_node(self, state: GraphState) -> dict:
        result = state.get("result")
        if result is None or not result.answer.strip():
            return {
                "result": None,
                "error_code": "invalid_result",
                "error": "No answer was produced.",
            }
        if result.citations and not CitationBuilder.validate_answer(
            result.answer, result.citations
        ):
            return {
                "result": None,
                "error_code": "invalid_citations",
                "error": "The answer contains invalid source references.",
            }
        result.latency_ms = int((perf_counter() - state.get("started_at", perf_counter())) * 1000)
        result.query = state["query"]
        logger.info(
            "Query completed",
            request_id=state.get("request_id"),
            query_type=result.query_type.value,
            latency_ms=result.latency_ms,
            retries=state.get("attempts", 0),
        )
        return {"result": result, "error": None, "error_code": None, "current_node": "finalize"}

    def _failure_node(self, state: GraphState) -> dict:
        logger.warning(
            "Query failed",
            request_id=state.get("request_id"),
            error_code=state.get("error_code"),
            attempts=state.get("attempts", 0),
        )
        return {
            "result": None,
            "current_node": "failure",
            "error_code": state.get("error_code") or "execution_failed",
            "error": state.get("error") or "Query execution failed.",
        }

    async def _execute_node(
        self,
        name: str,
        state: GraphState,
        operation: Callable[[], Awaitable[QueryResult]],
    ) -> dict:
        started = perf_counter()
        try:
            result = await asyncio.wait_for(operation(), timeout=settings.node_timeout_seconds)
            return {
                "result": result,
                "error": None,
                "error_code": None,
                "retryable": False,
                "current_node": name,
            }
        except Exception as exc:
            retryable = isinstance(
                exc, (TimeoutError, ConnectionError, APIConnectionError, RateLimitError)
            )
            if isinstance(exc, APIStatusError):
                retryable = exc.status_code >= 500 or exc.status_code == 429
            code = "node_timeout" if isinstance(exc, TimeoutError) else "execution_failed"
            logger.warning(
                "Node failed",
                node=name,
                error_type=type(exc).__name__,
                request_id=state.get("request_id"),
            )
            return {
                "result": None,
                "error": "Unable to complete the selected query path.",
                "error_code": code,
                "retryable": retryable,
                "current_node": name,
            }
        finally:
            logger.info(
                "Node execution metric",
                node=name,
                request_id=state.get("request_id"),
                attempt=state.get("attempts", 0) + 1,
                duration_ms=int((perf_counter() - started) * 1000),
            )

    def _route(self, state: GraphState) -> str:
        """Route to appropriate agent based on classification."""
        if state.get("error_code") in {"invalid_routing", "routing_unavailable"}:
            return "failure"
        classification = state.get("classification")
        if classification is not None and classification.decision == "clarify":
            return "clarify"
        query_type = state.get("query_type")

        if query_type == QueryType.CHAT:
            return "chat"
        if query_type == QueryType.RECORDS_SQL:
            return "sql"
        elif query_type == QueryType.HYBRID:
            return "hybrid"
        elif query_type == QueryType.DOCUMENT_QA:
            return "rag"
        return "failure"

    async def _chat_node(self, state: GraphState) -> dict:
        """Conversation node - responds with the local LLM and no retrieval."""
        return await self._execute_node(
            "chat",
            state,
            lambda: self.chat_agent.answer(
                state.get("resolved_query", state["query"]),
                conversation=state.get("conversation", []),
            ),
        )

    async def _rag_node(self, state: GraphState) -> dict:
        """RAG node - executes document Q&A."""
        return await self._execute_node(
            "rag",
            state,
            lambda: self.rag_agent.answer(
                query=state.get("resolved_query", state["query"]),
                contract_id=state.get("contract_id"),
                search_method=state["search_method"],
            ),
        )

    async def _sql_node(self, state: GraphState) -> dict:
        """SQL node - executes service_cases data queries."""
        return await self._execute_node(
            "sql", state, lambda: self.sql_agent.answer(state.get("resolved_query", state["query"]))
        )

    async def _hybrid_node(self, state: GraphState) -> dict:
        """Hybrid node - executes combined RAG + SQL."""
        return await self._execute_node(
            "hybrid",
            state,
            lambda: self.hybrid_agent.answer(
                query=state.get("resolved_query", state["query"]),
                contract_id=state.get("contract_id"),
                search_method=state["search_method"],
            ),
        )

    async def invoke(
        self,
        query: str,
        contract_id: Optional[int] = None,
        search_method: str = "hybrid",
        conversation: Optional[list[dict[str, str]]] = None,
    ) -> QueryResult:
        """
        Execute the graph with the given query.
        """
        initial_state: GraphState = {
            "query": query,
            "contract_id": contract_id,
            "search_method": search_method,
            "conversation": conversation or [],
            "classification": None,
            "query_type": None,
            "result": None,
            "error": None,
            "current_node": "start",
            "request_id": uuid4().hex,
            "started_at": perf_counter(),
            "attempts": 0,
        }

        # Run the graph
        try:
            async with asyncio.timeout(settings.query_timeout_seconds):
                final_state = cast(GraphState, await self.graph.ainvoke(initial_state))
        except TimeoutError as exc:
            raise GraphExecutionError("query_timeout", "Query execution timed out.") from exc

        # Extract result
        result = final_state.get("result")
        if result is not None:
            return result

        # Propagate failure to the API layer instead of returning an HTTP 200 answer
        # that merely contains an infrastructure error message.
        error_msg = final_state.get("error", "Unknown error occurred")
        raise GraphExecutionError(
            final_state.get("error_code") or "execution_failed",
            error_msg or "Query execution failed.",
        )

    async def stream(
        self,
        query: str,
        contract_id: Optional[int] = None,
        search_method: str = "hybrid",
        conversation: Optional[list[dict[str, str]]] = None,
    ):
        """
        Stream the graph execution (for real-time updates).
        """
        initial_state: GraphState = {
            "query": query,
            "contract_id": contract_id,
            "search_method": search_method,
            "conversation": conversation or [],
            "classification": None,
            "query_type": None,
            "result": None,
            "error": None,
            "current_node": "start",
            "request_id": uuid4().hex,
            "started_at": perf_counter(),
            "attempts": 0,
        }

        try:
            async with asyncio.timeout(settings.query_timeout_seconds):
                async for event in self.graph.astream(initial_state, stream_mode="updates"):
                    yield event
        except TimeoutError as exc:
            raise GraphExecutionError("query_timeout", "Query execution timed out.") from exc


# Factory function for creating the graph
async def create_graph(session: AsyncSession) -> PolicyMindGraph:
    """Create a PolicyMindGraph instance."""
    return PolicyMindGraph(session)
