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
        
        for chunk in chunks:
            # Truncate snippet
            snippet = chunk.content[:max_snippet_length]
            if len(chunk.content) > max_snippet_length:
                snippet += "..."
            
            citation = Citation(
                policy_number=f"Policy-{chunk.policy_id}",  # Will be replaced with actual policy number
                section_type=chunk.section_type.value if chunk.section_type else None,
                section_name=chunk.section_name,
                page_number=chunk.page_number,
                snippet=snippet,
                relevance_score=chunk.score
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
            if citation.section_name:
                source_info += f" Section: {citation.section_name}"
            if citation.page_number:
                source_info += f" (Page {citation.page_number})"
            
            formatted.append(f"{source_info}\n{citation.snippet}")
        
        return "\n\n".join(formatted)
    
    @staticmethod
    def format_citations_for_response(citations: List[Citation]) -> List[dict]:
        """
        Format citations for API response.
        """
        return [
            {
                "source": f"{citation.policy_number}",
                "section": citation.section_name or "General",
                "page": citation.page_number,
                "snippet": citation.snippet,
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
            source_line = f"\n[{i}] {citation.policy_number}"
            if citation.section_name:
                source_line += f", {citation.section_name}"
            if citation.page_number:
                source_line += f" (Page {citation.page_number})"
            sources_section += source_line
        
        return answer + sources_section
