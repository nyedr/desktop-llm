"""Embeddings service for memory operations."""

import logging
import asyncio
import numpy as np
from typing import List, Union, Optional, Dict
from sentence_transformers import SentenceTransformer
import torch

from app.services.model_service import ModelService
from app.utils.profiling import profile_operation

logger = logging.getLogger(__name__)

# Constants for optimization
EMBEDDING_CACHE_SIZE = 1000
BATCH_SIZE = 32
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 dimension
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingService:
    """Service for handling embeddings with caching and batching."""

    def __init__(self):
        """Initialize the embedding service."""
        self.model_service = ModelService()
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._embedding_semaphore = asyncio.Semaphore(10)

        # Initialize MiniLM model
        try:
            self.model = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2')
            self.model.to(DEVICE)
            logger.info(f"Initialized MiniLM model on {DEVICE}")
        except Exception as e:
            logger.error(f"Failed to initialize MiniLM model: {e}")
            self.model = None

    @staticmethod
    def _hash_text_for_cache(text: str) -> str:
        """Create a stable hash for text to use as cache key."""
        return str(hash(text.strip().lower()))

    def _get_cached_embedding(self, text_hash: str) -> Optional[np.ndarray]:
        """Get cached embedding using text hash as key."""
        return self._embedding_cache.get(text_hash)

    def _cache_embedding(self, text: str, embedding: np.ndarray) -> np.ndarray:
        """Cache an embedding with proper cache hit tracking."""
        text_hash = self._hash_text_for_cache(text)
        cached = self._get_cached_embedding(text_hash)
        if cached is not None:
            return cached

        # Update cache, maintaining size limit
        if len(self._embedding_cache) >= EMBEDDING_CACHE_SIZE:
            # Remove oldest item (first item in dict)
            self._embedding_cache.pop(next(iter(self._embedding_cache)))

        self._embedding_cache[text_hash] = embedding
        return embedding

    def _safe_quantize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Safely quantize embedding to prevent division by zero."""
        if embedding is None or embedding.size == 0:
            return np.zeros(EMBEDDING_DIM, dtype=np.uint8)

        try:
            # Ensure the embedding is normalized
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            # Scale to 0-255 range
            scaled = (embedding + 1) * 127.5  # Maps [-1,1] to [0,255]
            quantized = np.clip(np.round(scaled), 0, 255).astype(np.uint8)

            return quantized
        except Exception as e:
            logger.error(f"Error in quantization: {e}")
            return np.zeros(EMBEDDING_DIM, dtype=np.uint8)

    async def _get_embeddings_with_retry(
        self,
        batch: List[str],
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: int = 30
    ) -> List[np.ndarray]:
        """Get embeddings with retry logic and proper timing."""
        retry_count = 0
        last_error = None

        while retry_count < max_retries:
            try:
                async with profile_operation("generate_minilm_embeddings"):
                    # Convert to tensor and move to device
                    embeddings = self.model.encode(
                        batch,
                        batch_size=BATCH_SIZE,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        device=DEVICE
                    )
                    return [np.array(emb, dtype=np.float32) for emb in embeddings]

            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    logger.error(
                        f"Embedding generation failed after {max_retries} retries: {str(e)}")
                    raise
                logger.warning(
                    f"Embedding request failed, retrying ({retry_count}/{max_retries}): {str(e)}")
                last_error = e
                await asyncio.sleep(retry_delay * (2 ** (retry_count - 1)))

        raise last_error or Exception("Failed to get embeddings after retries")

    async def get_embeddings(
        self,
        texts: Union[str, List[str]]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Get embeddings for text(s) with caching and batching.

        Args:
            texts: Text or list of texts to embed
        """
        async with self._embedding_semaphore:
            try:
                # Handle single text input
                if isinstance(texts, str):
                    texts = [texts]
                    return_single = True
                else:
                    return_single = False

                all_embeddings = [None] * len(texts)

                # Process in batches
                for i in range(0, len(texts), BATCH_SIZE):
                    batch = texts[i:i + BATCH_SIZE]
                    batch_indices = range(i, min(i + BATCH_SIZE, len(texts)))

                    # Try cache first
                    uncached_texts = []
                    uncached_indices = []

                    for j, text in enumerate(batch):
                        text_hash = self._hash_text_for_cache(text)
                        cached_emb = self._get_cached_embedding(text_hash)
                        if cached_emb is not None:
                            all_embeddings[batch_indices[j]] = cached_emb
                        else:
                            uncached_texts.append(text)
                            uncached_indices.append(j)

                    # Get embeddings for uncached texts
                    if uncached_texts:
                        try:
                            new_embeddings = await self._get_embeddings_with_retry(
                                uncached_texts,
                                max_retries=3,
                                retry_delay=1.0,
                                timeout=30
                            )

                            # Cache and store new embeddings
                            for idx, (text, emb) in enumerate(zip(uncached_texts, new_embeddings)):
                                if emb is not None:
                                    quantized_emb = self._safe_quantize_embedding(
                                        emb)
                                    self._cache_embedding(text, quantized_emb)
                                    orig_idx = batch_indices[uncached_indices[idx]]
                                    all_embeddings[orig_idx] = quantized_emb

                        except Exception as e:
                            logger.error(
                                f"Error generating embeddings for batch: {e}")
                            # Fill failed embeddings with zeros
                            for idx in uncached_indices:
                                orig_idx = batch_indices[idx]
                                all_embeddings[orig_idx] = np.zeros(
                                    EMBEDDING_DIM, dtype=np.uint8)

                # Ensure no None values in results
                all_embeddings = [
                    emb if emb is not None else np.zeros(
                        EMBEDDING_DIM, dtype=np.uint8)
                    for emb in all_embeddings
                ]

                return all_embeddings[0] if return_single else all_embeddings

            except Exception as e:
                logger.error(
                    f"Error in embedding function: {str(e)}", exc_info=True)
                raise
