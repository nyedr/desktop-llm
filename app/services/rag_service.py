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
from app.memory.lightrag.config import (
    LIGHTRAG_DATA_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MAX_TOKENS
)

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

DEFAULT_MODEL = "meta-llama/llama-3.2-3b-instruct"


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
    Get embeddings for texts using the configured embedding model.

    Args:
        texts: List of texts to get embeddings for
        model: Embedding model to use (defaults to config.DEFAULT_EMBEDDING_MODEL)

    Returns:
        np.ndarray: Array of embeddings

    Raises:
        Exception: If embedding fails
    """
    try:
        embeddings = await ollama_embedding(texts, embed_model=model)
        # Convert to numpy array if not already
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        return embeddings
    except Exception as e:
        raise Exception(f"Failed to get embeddings: {str(e)}")


def create_lightrag(
    working_dir: str = LIGHTRAG_DATA_DIR,
    model_name: str = DEFAULT_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> LightRAG:
    """Create and configure a LightRAG instance with configured embedding model."""
    # Create working directory if it doesn't exist
    os.makedirs(working_dir, exist_ok=True)

    # Create embedding function using configured model
    embedding_func = EmbeddingFunc(
        embedding_dim=DEFAULT_EMBEDDING_DIM,
        max_token_size=DEFAULT_MAX_TOKENS,
        func=lambda texts: get_embeddings(texts, model=embedding_model)
    )

    # Check for OpenRouter API key only if using an OpenRouter model
    if any(api_model in model_name for api_model in API_MODEL_LIST):
        if not os.getenv("OPENROUTER_API_KEY"):
            raise OpenRouterAPIKeyError(
                "OPENROUTER_API_KEY environment variable must be set when using OpenRouter models")

    # Initialize LightRAG with documented configuration
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=openrouter_llm_complete,
        llm_model_name=model_name,
        embedding_func=embedding_func,
        # Core parameters from documentation
        chunk_token_size=500,
        chunk_overlap_token_size=100,
        entity_extract_max_gleaning=5,
        entity_summary_to_max_tokens=2000,
        # Processing parameters
        embedding_batch_num=32,
        embedding_func_max_async=16,
        llm_model_max_async=16,
        # Enable caching with LLM verification
        enable_llm_cache=True,
        embedding_cache_config={
            "enabled": True,
            "similarity_threshold": 0.95,
            "use_llm_check": True
        },
        # Simple addon params
        addon_params={
            "insert_batch_size": 10,
            "memory_queue_size": 100,  # Increased queue size
            "process_memory_interval": 0.1  # Faster processing interval
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
    if hashing_kv is not None:
        try:
            cache_key = f"openrouter_llm_{model_name}_{prompt}"
            if hasattr(hashing_kv, 'aget'):
                cached_response = await hashing_kv.aget(cache_key)
            elif hasattr(hashing_kv, 'get_value'):
                cached_response = hashing_kv.get_value(cache_key)
            else:
                cached_response = None

            if cached_response:
                logger.info(f"Cache hit for prompt with model {model_name}")
                return cached_response
            logger.debug(f"Cache miss for prompt with model {model_name}")
        except Exception as e:
            logger.warning(f"Error accessing cache: {e}")

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
        logger.info(f"LLM response: {response_text}")

        # Cache response if available
        if hashing_kv is not None:
            try:
                cache_key = f"openrouter_llm_{model_name}_{prompt}"
                if hasattr(hashing_kv, 'aset'):
                    await hashing_kv.aset(cache_key, response_text)
                elif hasattr(hashing_kv, 'set_value'):
                    hashing_kv.set_value(cache_key, response_text)
                logger.info(
                    f"Cached response for prompt with model {model_name}")
            except Exception as e:
                logger.warning(f"Error caching response: {e}")

        return response_text

    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower():
            raise OpenRouterAPIKeyError(f"Invalid API key: {error_msg}")
        elif "model" in error_msg.lower():
            raise OpenRouterModelError(f"Model error: {error_msg}")
        else:
            raise OpenRouterError(f"OpenRouter API error: {error_msg}")
