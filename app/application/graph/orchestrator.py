"""
LangGraph Orchestrator - State machine for query routing and execution.
"""

from typing import Optional, TypedDict, cast

import structlog
from langgraph.graph import END, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.application.agents.conversation_agent import ConversationAgent
from app.application.agents.hybrid_agent import HybridAgent
from app.application.agents.query_classifier import QueryClassifier
from app.application.agents.rag_agent import RAGAgent
from app.application.agents.sql_agent import SQLAgent
from app.domain.entities.models import ClassificationResult, QueryResult, QueryType

logger = structlog.get_logger()


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


class PolicyMindGraph:
    """
    LangGraph-based orchestrator for PolicyMind queries.

    Flow:
    1. classify -> Determine query type
    2. Route to the appropriate agent (chat/rag/sql/hybrid)
    3. Return result

    Graph:
        classify -> route -> [chat_node | rag_node | sql_node | hybrid_node] -> END
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

        # Set entry point
        workflow.set_entry_point("classify")

        # Add conditional routing from classify node
        workflow.add_conditional_edges(
            "classify",
            self._route,
            {"chat": "chat", "rag": "rag", "sql": "sql", "hybrid": "hybrid"},
        )

        # All agents go to END
        workflow.add_edge("chat", END)
        workflow.add_edge("rag", END)
        workflow.add_edge("sql", END)
        workflow.add_edge("hybrid", END)

        return workflow.compile()

    async def _classify_node(self, state: GraphState) -> dict:
        """Classification node - determines query type."""
        try:
            classification = await self.classifier.classify(
                state["query"], state.get("conversation", [])
            )

            logger.info(
                "Query classified",
                query_type=classification.query_type.value,
                confidence=classification.confidence,
                reasoning=classification.reasoning,
            )

            return {
                "classification": classification,
                "query_type": classification.query_type,
                "current_node": "classify",
            }
        except Exception as e:
            logger.error("Classification failed", error=str(e))
            # Default to document_qa
            return {
                "query_type": QueryType.DOCUMENT_QA,
                "error": str(e),
                "current_node": "classify",
            }

    def _route(self, state: GraphState) -> str:
        """Route to appropriate agent based on classification."""
        query_type = state.get("query_type", QueryType.DOCUMENT_QA)

        if query_type == QueryType.CHAT:
            return "chat"
        if query_type == QueryType.RECORDS_SQL:
            return "sql"
        elif query_type == QueryType.HYBRID:
            return "hybrid"
        else:
            return "rag"

    async def _chat_node(self, state: GraphState) -> dict:
        """Conversation node - responds with the local LLM and no retrieval."""
        try:
            result = await self.chat_agent.answer(
                state["query"], conversation=state.get("conversation", [])
            )
            return {"result": result, "current_node": "chat"}
        except Exception as e:
            logger.error("Conversation execution failed", error=str(e))
            return {"error": str(e), "current_node": "chat"}

    async def _rag_node(self, state: GraphState) -> dict:
        """RAG node - executes document Q&A."""
        try:
            result = await self.rag_agent.answer(
                query=state["query"],
                contract_id=state.get("contract_id"),
                search_method=state["search_method"],
            )
            return {"result": result, "current_node": "rag"}
        except Exception as e:
            logger.error("RAG execution failed", error=str(e))
            return {"error": str(e), "current_node": "rag"}

    async def _sql_node(self, state: GraphState) -> dict:
        """SQL node - executes service_cases data queries."""
        try:
            result = await self.sql_agent.answer(state["query"])
            return {"result": result, "current_node": "sql"}
        except Exception as e:
            logger.error("SQL execution failed", error=str(e))
            return {"error": str(e), "current_node": "sql"}

    async def _hybrid_node(self, state: GraphState) -> dict:
        """Hybrid node - executes combined RAG + SQL."""
        try:
            result = await self.hybrid_agent.answer(
                query=state["query"],
                contract_id=state.get("contract_id"),
                search_method=state["search_method"],
            )
            return {"result": result, "current_node": "hybrid"}
        except Exception as e:
            logger.error("Hybrid execution failed", error=str(e))
            return {"error": str(e), "current_node": "hybrid"}

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
        }

        # Run the graph
        final_state = cast(GraphState, await self.graph.ainvoke(initial_state))

        # Extract result
        result = final_state.get("result")
        if result is not None:
            return result

        # Propagate failure to the API layer instead of returning an HTTP 200 answer
        # that merely contains an infrastructure error message.
        error_msg = final_state.get("error", "Unknown error occurred")
        raise RuntimeError(f"PolicyMind graph execution failed: {error_msg}")

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
        }

        async for event in self.graph.astream(initial_state):
            yield event


# Factory function for creating the graph
async def create_graph(session: AsyncSession) -> PolicyMindGraph:
    """Create a PolicyMindGraph instance."""
    return PolicyMindGraph(session)
