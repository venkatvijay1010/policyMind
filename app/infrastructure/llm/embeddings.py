"""
Embedding generation service using OpenAI.
"""
from typing import List, Optional
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

from app.config import settings

logger = structlog.get_logger()


class EmbeddingService:
    """Service for generating text embeddings."""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model
        self.dimension = settings.embedding_dimension
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        """
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float"
        )
        
        return response.data[0].embedding
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        OpenAI supports batching up to 2048 texts per request.
        """
        if not texts:
            return []
        
        # Batch in chunks of 100 to avoid rate limits
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            logger.debug(
                "Generating embeddings batch",
                batch_start=i,
                batch_size=len(batch),
                total=len(texts)
            )
            
            response = await self.client.embeddings.create(
                model=self.model,
                input=batch,
                encoding_format="float"
            )
            
            # Sort by index to maintain order
            batch_embeddings = [
                item.embedding 
                for item in sorted(response.data, key=lambda x: x.index)
            ]
            
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        Same as embed_text but named for clarity.
        """
        return await self.embed_text(query)


# Singleton instance
_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service singleton."""
    global _service
    if _service is None:
        _service = EmbeddingService()
    return _service
