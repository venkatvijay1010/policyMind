"""
Citation builder for grounding answers in source documents.
"""
from typing import List, Optional
from app.domain.entities.models import Citation, DocumentChunk


class CitationBuilder:
    """
    Builds citations from retrieved document chunks.
    
    Every claim in the answer should be traceable to a source.
    This is critical for preventing hallucination.
    """
    
    @staticmethod
    def build_citations(
        chunks: List[DocumentChunk],
        max_snippet_length: int = 200
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
                policy_name=f"Policy-{chunk.policy_id}",
                section=chunk.section_name,
                page=chunk.page_number,
                chunk_text=snippet,
                relevance_score=chunk.score or 0.0
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
                "source": f"{citation.policy_name}",
                "section": citation.section or "General",
                "page": citation.page,
                "snippet": citation.chunk_text,
                "relevance": round(citation.relevance_score, 3) if citation.relevance_score else None
            }
            for citation in citations
        ]
    
    @staticmethod
    def add_inline_citations(
        answer: str,
        citations: List[Citation]
    ) -> str:
        """
        Add inline citation markers to the answer.
        This is a simple version - production would use more sophisticated matching.
        """
        # For now, just append a sources section
        if not citations:
            return answer
        
        sources_section = "\n\n**Sources:**"
        for i, citation in enumerate(citations, 1):
            source_line = f"\n[{i}] {citation.policy_name}"
            if citation.section:
                source_line += f", {citation.section}"
            if citation.page:
                source_line += f" (Page {citation.page})"
            sources_section += source_line
        
        return answer + sources_section
