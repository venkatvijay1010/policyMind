"""
Citation builder for grounding answers in source documents.
"""

import re
from typing import List

from app.domain.entities.models import Citation, DocumentChunk


class CitationBuilder:
    """
    Builds citations from retrieved document chunks.

    Every claim in the answer should be traceable to a source.
    This is critical for preventing hallucination.
    """

    @staticmethod
    def validate_answer(answer: str, citations: List[Citation]) -> bool:
        source_ids = {citation.source_id for citation in citations}
        markers = re.findall(r"\[Source\s+(\d+)\]", answer, flags=re.IGNORECASE)
        return (
            bool(markers)
            and len(source_ids) == len(citations)
            and all(int(marker) in source_ids for marker in markers)
            and all(citation.chunk_text.strip() for citation in citations)
        )

    @staticmethod
    def build_citations(
        chunks: List[DocumentChunk], max_snippet_length: int = 200
    ) -> List[Citation]:
        """
        Build citation objects from retrieved chunks.
        """
        citations = []

        for i, chunk in enumerate(chunks, 1):
            # Truncate snippet
            snippet = chunk.content[:max_snippet_length]
            if len(chunk.content) > max_snippet_length:
                snippet += "..."

            citation = Citation(
                source_id=i,
                contract_title=chunk.contract_title or f"Policy-{chunk.contract_id}",
                section=CitationBuilder._section_label(chunk, i),
                page=chunk.source_page,
                chunk_text=snippet,
                relevance_score=chunk.score or 0.0,
            )
            citations.append(citation)

        return citations

    @staticmethod
    def format_citations_for_prompt(citations: List[Citation]) -> str:
        """
        Format citations as context for the LLM prompt.
        """
        if not citations:
            return "No relevant sources found."

        formatted = []
        for i, citation in enumerate(citations, 1):
            source_info = f"[Source {i}]"
            if citation.section:
                source_info += f" Section: {citation.section}"
            if citation.page:
                source_info += f" (Page {citation.page})"

            formatted.append(f"{source_info}\n{citation.chunk_text}")

        return "\n\n".join(formatted)

    @staticmethod
    def format_citations_for_response(citations: List[Citation]) -> List[dict]:
        """
        Format citations for API response.
        """
        return [
            {
                "source": f"{citation.contract_title}",
                "section": citation.section or "General",
                "page": citation.page,
                "snippet": citation.chunk_text,
                "relevance": round(citation.relevance_score, 3)
                if citation.relevance_score
                else None,
            }
            for citation in citations
        ]

    @staticmethod
    def add_inline_citations(answer: str, citations: List[Citation]) -> str:
        """
        Preserve model-provided inline markers.

        The browser renders the complete, structured citation cards. Appending a
        second plain-text source list here made answers visually repetitive and
        separated page/section details from their evidence excerpts.
        """
        del citations
        return answer

    @staticmethod
    def _section_label(chunk: DocumentChunk, fallback_index: int) -> str:
        """Use persisted metadata, then recover a Markdown section for legacy chunks."""
        if chunk.topic_title:
            return chunk.topic_title
        heading = re.search(r"^\s*#{2,6}\s+(.+?)\s*#*\s*$", chunk.content, re.MULTILINE)
        if heading:
            return heading.group(1).strip()
        return f"Policy passage {chunk.passage_order + 1 if chunk.passage_order is not None else fallback_index}"
