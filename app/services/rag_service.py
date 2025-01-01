from openai import OpenAI
import os
from typing import Optional, Any, List
import logging
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm import ollama_embedding
import numpy as np
from dotenv import load_dotenv
from pathlib import Path

logger = logging.getLogger(__name__)

# Load environment variables from project root .env file
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'
load_dotenv(env_path)
logger.debug(f"Loading environment variables from: {env_path}")
logger.debug(
    f"OpenRouter API Key present: {bool(os.getenv('OPENROUTER_API_KEY'))}")

API_BASE_URL = "https://openrouter.ai/api/v1"

API_MODEL_LIST = [
    # Free models from OpenRouter - 15 requests per minute and 200 requests per day
    "google/gemini-2.0-flash-thinking-exp:free",
    "google/gemini-2.0-flash-exp:free",
    "google/gemini-exp-1206:free",
    "google/gemini-exp-1121:free",
    "google/learnlm-1.5-pro-experimental:free",
    "google/gemini-exp-1114:free",

    # Paid models from OpenRouter
    "meta-llama/llama-3.2-3b-instruct"
]

DEFAULT_WORKING_DIR = "./lightrag_storage"
DEFAULT_MODEL = "meta-llama/llama-3.2-3b-instruct"
DEFAULT_EMBEDDING_DIM = 768  # nomic-embed-text embedding dimension
DEFAULT_MAX_TOKENS = 32768  # LightRAG's default max tokens
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"  # Ollama's embedding model


class OpenRouterError(Exception):
    """Base exception for OpenRouter API errors."""
    pass


class OpenRouterAPIKeyError(OpenRouterError):
    """Raised when there are issues with the OpenRouter API key."""
    pass


class OpenRouterModelError(OpenRouterError):
    """Raised when there are issues with model selection or availability."""
    pass


async def get_embeddings(texts: List[str], model: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    """
    Get embeddings for texts using Ollama's nomic-embed-text model.

    Args:
        texts: List of texts to get embeddings for
        model: Embedding model to use (defaults to nomic-embed-text)

    Returns:
        np.ndarray: Array of embeddings

    Raises:
        Exception: If embedding fails
    """
    try:
        return await ollama_embedding(texts, embed_model=model)
    except Exception as e:
        raise Exception(f"Failed to get embeddings: {str(e)}")


def create_lightrag(
    working_dir: str = DEFAULT_WORKING_DIR,
    model_name: str = DEFAULT_MODEL,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> LightRAG:
    """Create and configure a LightRAG instance with Ollama integration for embeddings."""
    # Create working directory if it doesn't exist
    os.makedirs(working_dir, exist_ok=True)

    # Create embedding function using Ollama's nomic-embed-text
    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=max_tokens,
        func=lambda texts: get_embeddings(texts, model=embedding_model)
    )

    # Check for OpenRouter API key only if using an OpenRouter model
    if any(api_model in model_name for api_model in API_MODEL_LIST):
        if not os.getenv("OPENROUTER_API_KEY"):
            raise OpenRouterAPIKeyError(
                "OPENROUTER_API_KEY environment variable must be set when using OpenRouter models")

    # Initialize LightRAG with configuration from documentation
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=openrouter_llm_complete,
        llm_model_name=model_name,
        llm_model_max_token_size=max_tokens,
        embedding_func=embedding_func,
        enable_llm_cache=True,
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        embedding_batch_num=32,
        embedding_func_max_async=16,
        llm_model_max_async=16,
        embedding_cache_config={
            "enabled": True,
            "similarity_threshold": 0.95,
            "use_llm_check": False
        }
    )

    logger.info(
        f"Initialized LightRAG with model {model_name} and {embedding_model} embeddings in directory {working_dir}")
    return rag


async def openrouter_llm_complete(
    prompt: str,
    hashing_kv: Optional[Any] = None,
    model_name: str = "meta-llama/llama-3.2-3b-instruct",
    **kwargs
) -> str:
    """Get completions from OpenRouter API with caching support."""
    # Validate API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise OpenRouterAPIKeyError(
            "OPENROUTER_API_KEY environment variable not set")

    # Check cache if available
    if hashing_kv is not None and hasattr(hashing_kv, 'get'):
        cache_key = f"openrouter_llm_{model_name}_{prompt}"
        cached_response = hashing_kv.get(cache_key)
        if cached_response:
            logger.debug(f"Cache hit for prompt with model {model_name}")
            return cached_response

    try:
        # Initialize OpenAI client
        client = OpenAI(base_url=API_BASE_URL, api_key=api_key)

        # Filter out unsupported kwargs
        supported_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ['temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty', 'stop']
        }

        # Get completion
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            **supported_kwargs
        )

        response_text = response.choices[0].message.content

        # Cache response if available
        if hashing_kv is not None and hasattr(hashing_kv, 'set'):
            cache_key = f"openrouter_llm_{model_name}_{prompt}"
            hashing_kv.set(cache_key, response_text)
            logger.debug(f"Cached response for prompt with model {model_name}")

        return response_text

    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower():
            raise OpenRouterAPIKeyError(f"Invalid API key: {error_msg}")
        elif "model" in error_msg.lower():
            raise OpenRouterModelError(f"Model error: {error_msg}")
        else:
            raise OpenRouterError(f"OpenRouter API error: {error_msg}")
