"""Embedding generation through the configured OpenAI-compatible provider."""

from typing import List, Optional, cast

import structlog
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

logger = structlog.get_logger()


class EmbeddingService:
    """Service for generating text embeddings."""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        self.model = settings.embedding_model
        self.dimension = settings.embedding_dimension

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        """
        response = await self.client.embeddings.create(
            model=self.model, input=text, encoding_format="float"
        )

        return cast(List[float], response.data[0].embedding)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        Ollama supports batch inputs through its OpenAI-compatible embeddings API.
        """
        if not texts:
            return []

        # A modest batch keeps memory use acceptable for local models too.
        batch_size = 100
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            logger.debug(
                "Generating embeddings batch",
                batch_start=i,
                batch_size=len(batch),
                total=len(texts),
            )

            response = await self.client.embeddings.create(
                model=self.model, input=batch, encoding_format="float"
            )

            # Sort by index to maintain order
            batch_embeddings = [
                cast(List[float], item.embedding)
                for item in sorted(response.data, key=lambda x: x.index)
            ]

            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        Same as embed_text but named for clarity.
        """
        return cast(List[float], await self.embed_text(query))

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Alias for embed_texts for backward compatibility.
        """
        return cast(List[List[float]], await self.embed_texts(texts))


# Singleton instance
_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service singleton."""
    global _service
    if _service is None:
        _service = EmbeddingService()
    return _service
