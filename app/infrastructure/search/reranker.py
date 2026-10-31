"""Fast, local reranking for a bounded set of retrieved policy chunks."""

import re
from typing import List

from app.domain.entities.models import DocumentChunk

_TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)
_STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "for",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


class LocalReranker:
    """Reorder search candidates using query coverage and their retrieval rank.

    This is intentionally deterministic and inexpensive: it avoids an extra
    model call while adding an explicit query-to-passage relevance pass after
    vector/BM25/RRF candidate generation.
    """

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [token.casefold() for token in _TOKEN_PATTERN.findall(text)]

    def rerank(self, query: str, chunks: List[DocumentChunk], top_k: int) -> List[DocumentChunk]:
        """Return at most ``top_k`` candidates in reranked order.

        Existing ``chunk.score`` values are not overwritten because they
        describe the original search method (including RRF for hybrid search).
        """
        if top_k <= 0 or not chunks:
            return []

        query_tokens = {
            token for token in self._tokens(query) if len(token) > 1 and token not in _STOP_WORDS
        }
        query_phrase = " ".join(self._tokens(query))

        scored: list[tuple[float, int, DocumentChunk]] = []
        for original_rank, chunk in enumerate(chunks):
            content_tokens = set(self._tokens(chunk.content))
            title_tokens = set(self._tokens(chunk.topic_title or ""))
            if query_tokens:
                coverage = len(query_tokens & content_tokens) / len(query_tokens)
                title_coverage = len(query_tokens & title_tokens) / len(query_tokens)
                normalized_content = " ".join(self._tokens(chunk.content))
                phrase_match = float(bool(query_phrase and query_phrase in normalized_content))
                lexical_score = 0.75 * coverage + 0.15 * phrase_match + 0.10 * title_coverage
            else:
                lexical_score = 0.0

            # Preserve the initial retriever's ordering as a semantic signal,
            # but let direct query coverage promote a better matching passage.
            rank_score = 1.0 / (original_rank + 1)
            final_score = 0.6 * lexical_score + 0.4 * rank_score
            scored.append((final_score, original_rank, chunk))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [chunk for _, _, chunk in scored[:top_k]]
