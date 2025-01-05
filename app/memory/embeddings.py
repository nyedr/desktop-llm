"""Embeddings service for memory operations."""

import logging
import asyncio
import time
import numpy as np
from typing import List, Union, Optional, Dict, Literal
from lightrag.llm import ollama_embedding
from sentence_transformers import SentenceTransformer
import torch

from app.services.model_service import ModelService

logger = logging.getLogger(__name__)

# Constants for optimization
EMBEDDING_CACHE_SIZE = 1000
BATCH_SIZE = 32

# Model configurations
MINILM_DIM = 384  # all-MiniLM-L6-v2 dimension
NOMIC_DIM = 768   # nomic-embed-text dimension
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ModelType = Literal["minilm", "nomic"]


class EmbeddingService:
    """Service for handling embeddings with caching and batching."""

    def __init__(self):
        """Initialize the embedding service."""
        self.model_service = ModelService()
        self._embedding_cache: Dict[str, Dict[str, np.ndarray]] = {
            "minilm": {},
            "nomic": {}
        }
        self._embedding_semaphore = asyncio.Semaphore(10)
        self.cache_hits = {"minilm": 0, "nomic": 0}
        self.cache_misses = {"minilm": 0, "nomic": 0}
        self.total_embedding_calls = {"minilm": 0, "nomic": 0}
        self.embedding_times: Dict[str, List[float]] = {
            "minilm": [], "nomic": []}

        # Initialize MiniLM model
        try:
            self.minilm_model = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2')
            self.minilm_model.to(DEVICE)
            logger.info(f"Initialized MiniLM model on {DEVICE}")
        except Exception as e:
            logger.error(f"Failed to initialize MiniLM model: {e}")
            self.minilm_model = None

    @staticmethod
    def _hash_text_for_cache(text: str, model_type: ModelType) -> str:
        """Create a stable hash for text to use as cache key."""
        return f"{model_type}_{hash(text.strip().lower())}"

    def _get_cached_embedding(self, text_hash: str, model_type: ModelType) -> Optional[np.ndarray]:
        """Get cached embedding using text hash as key."""
        return self._embedding_cache[model_type].get(text_hash)

    def _cache_embedding(self, text: str, embedding: np.ndarray, model_type: ModelType) -> np.ndarray:
        """Cache an embedding with proper cache hit tracking."""
        text_hash = self._hash_text_for_cache(text, model_type)
        cached = self._get_cached_embedding(text_hash, model_type)
        if cached is not None:
            self.cache_hits[model_type] += 1
            return cached

        # Update cache, maintaining size limit per model
        if len(self._embedding_cache[model_type]) >= EMBEDDING_CACHE_SIZE:
            # Remove oldest item (first item in dict)
            self._embedding_cache[model_type].pop(
                next(iter(self._embedding_cache[model_type])))

        self._embedding_cache[model_type][text_hash] = embedding
        self.cache_misses[model_type] += 1
        return embedding

    def _safe_quantize_embedding(self, embedding: np.ndarray, model_type: ModelType) -> np.ndarray:
        """Safely quantize embedding to prevent division by zero."""
        dim = MINILM_DIM if model_type == "minilm" else NOMIC_DIM
        if embedding is None or embedding.size == 0:
            return np.zeros(dim, dtype=np.uint8)

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
            return np.zeros(dim, dtype=np.uint8)

    async def _get_minilm_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings using MiniLM model."""
        try:
            # Convert to tensor and move to device
            embeddings = self.minilm_model.encode(
                texts,
                batch_size=BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                device=DEVICE
            )
            return [np.array(emb, dtype=np.float32) for emb in embeddings]
        except Exception as e:
            logger.error(f"Error getting MiniLM embeddings: {e}")
            return [np.zeros(MINILM_DIM, dtype=np.float32) for _ in texts]

    async def _get_embeddings_with_retry(
        self,
        batch: List[str],
        model_type: ModelType,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: int = 30
    ) -> List[List[float]]:
        """Get embeddings with retry logic and proper timing."""
        start_time = time.perf_counter()
        retry_count = 0
        last_error = None
        embedding_time = 0.0

        while retry_count < max_retries:
            try:
                # Get embeddings based on model type
                embedding_start = time.perf_counter()
                if model_type == "minilm":
                    embeddings = await self._get_minilm_embeddings(batch)
                else:  # nomic
                    embeddings = await asyncio.wait_for(
                        ollama_embedding(
                            batch, embed_model="nomic-embed-text"),
                        timeout=timeout
                    )
                embedding_time = time.perf_counter() - embedding_start

                # Process embeddings
                dim = MINILM_DIM if model_type == "minilm" else NOMIC_DIM
                processed_embeddings = []
                batch_size = min(32, len(embeddings))

                for i in range(0, len(embeddings), batch_size):
                    batch_embs = embeddings[i:i + batch_size]
                    batch_processed = []

                    for emb in batch_embs:
                        if emb is None:
                            logger.warning(
                                "Received None embedding, using zeros")
                            processed_emb = np.zeros(dim, dtype=np.float32)
                        else:
                            try:
                                processed_emb = np.array(emb, dtype=np.float32)
                                if processed_emb.size == 0:
                                    logger.warning(
                                        "Empty embedding received, using zeros")
                                    processed_emb = np.zeros(
                                        dim, dtype=np.float32)
                                elif processed_emb.size != dim:
                                    logger.warning(
                                        f"Invalid embedding dimension: {processed_emb.size}, reshaping/padding")
                                    if processed_emb.size > dim:
                                        processed_emb = processed_emb[:dim]
                                    else:
                                        temp = np.zeros(dim, dtype=np.float32)
                                        temp[:processed_emb.size] = processed_emb
                                        processed_emb = temp
                                processed_emb = processed_emb.reshape(dim)
                            except Exception as e:
                                logger.error(
                                    f"Error processing embedding: {e}")
                                processed_emb = np.zeros(dim, dtype=np.float32)

                        # Normalize
                        norm = np.linalg.norm(processed_emb)
                        if norm > 0:
                            processed_emb = processed_emb / norm

                        batch_processed.append(processed_emb)

                    processed_embeddings.extend(batch_processed)

                # Record timing and stats
                duration = time.perf_counter() - start_time
                if duration > 0:
                    self.embedding_times[model_type].append(duration)
                self.total_embedding_calls[model_type] += len(batch)

                # Log performance metrics
                logger.info(
                    f"Generated {len(processed_embeddings)} {model_type} embeddings in {duration:.3f}s "
                    f"(embedding_time={embedding_time:.3f}s, "
                    f"processing_time={(duration-embedding_time):.3f}s, "
                    f"throughput={len(processed_embeddings)/duration:.1f} embeddings/s)"
                )

                return processed_embeddings

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

    def _determine_model_type(self, text: str, force_model: Optional[ModelType] = None, is_file: bool = False) -> ModelType:
        """Determine which model to use based on content type and force_model parameter."""
        if force_model:
            return force_model
        return "nomic" if is_file else "minilm"

    async def get_embeddings(
        self,
        texts: Union[str, List[str]],
        force_model: Optional[ModelType] = None,
        is_file: bool = False
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Get embeddings for text(s) with caching and batching.

        Args:
            texts: Text or list of texts to embed
            force_model: Force a specific model to be used
            is_file: Whether the text is from a file document
        """
        async with self._embedding_semaphore:
            start_time = time.perf_counter()
            try:
                # Handle single text input
                if isinstance(texts, str):
                    texts = [texts]
                    return_single = True
                else:
                    return_single = False

                # Group texts by model type
                text_groups: Dict[ModelType, List[tuple[int, str]]] = {
                    "minilm": [], "nomic": []}
                for i, text in enumerate(texts):
                    model_type = self._determine_model_type(
                        text, force_model, is_file)
                    text_groups[model_type].append((i, text))

                # Process each group
                all_embeddings = [None] * len(texts)
                for model_type, text_group in text_groups.items():
                    if not text_group:
                        continue

                    indices, group_texts = zip(*text_group)

                    # Process in batches
                    for i in range(0, len(group_texts), BATCH_SIZE):
                        batch = list(group_texts[i:i + BATCH_SIZE])
                        batch_indices = indices[i:i + BATCH_SIZE]

                        # Try cache first
                        uncached_texts = []
                        uncached_indices = []

                        for j, text in enumerate(batch):
                            text_hash = self._hash_text_for_cache(
                                text, model_type)
                            cached_emb = self._get_cached_embedding(
                                text_hash, model_type)
                            if cached_emb is not None:
                                all_embeddings[batch_indices[j]] = cached_emb
                                self.cache_hits[model_type] += 1
                            else:
                                uncached_texts.append(text)
                                uncached_indices.append(j)
                                self.cache_misses[model_type] += 1

                        # Get embeddings for uncached texts
                        if uncached_texts:
                            try:
                                new_embeddings = await self._get_embeddings_with_retry(
                                    uncached_texts,
                                    model_type=model_type,
                                    max_retries=3,
                                    retry_delay=1.0,
                                    timeout=30
                                )

                                # Cache and store new embeddings
                                for idx, (text, emb) in enumerate(zip(uncached_texts, new_embeddings)):
                                    if emb is not None:
                                        quantized_emb = self._safe_quantize_embedding(
                                            emb, model_type)
                                        self._cache_embedding(
                                            text, quantized_emb, model_type)
                                        orig_idx = batch_indices[uncached_indices[idx]]
                                        all_embeddings[orig_idx] = quantized_emb

                            except Exception as e:
                                logger.error(
                                    f"Error generating {model_type} embeddings for batch: {e}")
                                # Fill failed embeddings with zeros
                                dim = MINILM_DIM if model_type == "minilm" else NOMIC_DIM
                                for idx in uncached_indices:
                                    orig_idx = batch_indices[idx]
                                    all_embeddings[orig_idx] = np.zeros(
                                        dim, dtype=np.uint8)

                # Log overall performance
                total_duration = time.perf_counter() - start_time
                logger.info(
                    f"Embedding generation complete:"
                    f"\n  Total time: {total_duration:.3f}s"
                    f"\n  Texts processed: {len(texts)}"
                    f"\n  MiniLM hits/misses: {self.cache_hits['minilm']}/{self.cache_misses['minilm']}"
                    f"\n  Nomic hits/misses: {self.cache_hits['nomic']}/{self.cache_misses['nomic']}"
                )

                # Ensure no None values in results
                all_embeddings = [
                    emb if emb is not None else np.zeros(
                        MINILM_DIM if self._determine_model_type(
                            texts[i], force_model) == "minilm" else NOMIC_DIM,
                        dtype=np.uint8
                    )
                    for i, emb in enumerate(all_embeddings)
                ]

                return all_embeddings[0] if return_single else all_embeddings

            except Exception as e:
                logger.error(
                    f"Error in embedding function: {str(e)}", exc_info=True)
                raise
