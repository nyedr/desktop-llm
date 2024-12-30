from openai import OpenAI
import os
from typing import Optional, Any, List, Dict
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
    **kwargs
) -> LightRAG:
    """
    Create and configure a LightRAG instance with Ollama integration for embeddings.

    Args:
        working_dir: Directory for LightRAG storage
        model_name: OpenRouter model to use
        embedding_dim: Dimension of embeddings (768 for nomic-embed-text)
        max_tokens: Maximum tokens for context window
        embedding_model: Model to use for embeddings
        **kwargs: Additional arguments passed to LightRAG initialization

    Returns:
        LightRAG: Configured LightRAG instance

    Raises:
        OpenRouterAPIKeyError: If OPENROUTER_API_KEY is not set and using OpenRouter models
    """
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

    # Initialize LightRAG with comprehensive configuration
    addon_params = {
        # Document processing settings
        "example_number": -1,  # Process all examples
        "language": "English",
        "entity_types": ["organization", "person", "geo", "event", "date", "numeric", "time", "quantity"],
        "insert_batch_size": 20,  # Batch size for document processing

        # Async processing configuration
        "async_mode": True,
        "max_async_tasks": 16,  # Maximum concurrent async operations

        # Entity extraction settings
        "entity_confidence": 0.7,  # Confidence threshold for entity extraction
        "relationship_confidence": 0.8,  # Confidence threshold for relationships
        "entity_extract_max_gleaning": 1,  # Number of loops in entity extraction

        # Document chunking settings
        "chunk_token_size": 1200,  # Maximum tokens per chunk
        "chunk_overlap_token_size": 100,  # Overlap between chunks

        # Embedding cache configuration
        "embedding_cache_config": {
            "enabled": True,
            "similarity_threshold": 0.85,
            "use_llm_check": False
        },

        # Storage and retrieval settings
        "source_id_prefix": "doc-",  # Prefix for document IDs
        "max_token_for_text_unit": 4000,  # Tokens for original chunks
        "max_token_for_global_context": 4000,  # Tokens for relationship descriptions
        "max_token_for_local_context": 4000,  # Tokens for entity descriptions

        # Knowledge graph settings
        "node_embedding_algorithm": "node2vec",
        "node2vec_params": {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3
        }
    }

    # Initialize LightRAG with correct parameter names
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=openrouter_llm_complete,  # Function for LLM generation
        llm_model_name=model_name,  # Model name for generation
        llm_model_max_token_size=max_tokens,  # Maximum token size for LLM generation
        llm_model_max_async=4,  # Maximum concurrent async LLM processes
        embedding_func=embedding_func,  # Function to generate embedding vectors
        embedding_batch_num=32,  # Maximum batch size for embedding processes
        embedding_func_max_async=16,  # Maximum concurrent async embedding processes
        addon_params=addon_params,  # Additional configuration parameters
        enable_llm_cache=True  # Enable LLM response caching
    )

    logger.info(
        f"Initialized LightRAG with model {model_name} and {embedding_model} embeddings in directory {working_dir}")
    return rag


async def openrouter_llm_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict] = None,
    keyword_extraction: bool = False,
    hashing_kv: Optional[Any] = None,
    model_name: str = "meta-llama/llama-3.2-3b-instruct",
    **kwargs
) -> str:
    """
    Function to get completions from OpenRouter API with caching support.
    Using synchronous API to avoid compatibility issues.

    Args:
        prompt: The input prompt for the model
        system_prompt: Optional system prompt to set context
        history_messages: Optional list of previous messages for context
        keyword_extraction: Whether this is a keyword extraction request
        hashing_kv: Optional key-value store for caching responses
        model_name: The model to use from OpenRouter
        **kwargs: Additional arguments passed to the completion API

    Returns:
        str: The model's response text

    Raises:
        OpenRouterAPIKeyError: If the API key is missing or invalid
        OpenRouterModelError: If there are issues with the model
        OpenRouterError: For other API-related errors
    """
    # Validate API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise OpenRouterAPIKeyError(
            "OPENROUTER_API_KEY environment variable not set")

    # Check cache first if hashing_kv is provided
    if hashing_kv is not None:
        try:
            cache_key = f"openrouter_llm_{model_name}_{prompt}"
            if keyword_extraction:
                cache_key += "_kw"
            cached_response = hashing_kv.get(
                cache_key) if hasattr(hashing_kv, 'get') else None
            if cached_response:
                logger.debug(f"Cache hit for prompt with model {model_name}")
                return cached_response
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {str(e)}")

    try:
        # Initialize OpenAI client with OpenRouter base URL
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=api_key
        )

        # Prepare messages list
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        # Add history messages if provided
        if history_messages:
            messages.extend([{
                "role": msg.get("role", "user"),
                "content": msg["content"]
            } for msg in history_messages if isinstance(msg, dict) and "content" in msg])

        # Add keyword extraction system prompt if needed
        if keyword_extraction:
            messages.append({
                "role": "system",
                "content": "Extract key entities and relationships from the following text. Return them in a structured format."
            })

        # Add the current prompt
        messages.append({
            "role": "user",
            "content": prompt
        })

        # Make the API call
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            **kwargs
        )

        response_text = response.choices[0].message.content

        # Cache the response if hashing_kv is provided
        if hashing_kv is not None:
            try:
                cache_key = f"openrouter_llm_{model_name}_{prompt}"
                if keyword_extraction:
                    cache_key += "_kw"
                if hasattr(hashing_kv, 'set'):
                    hashing_kv.set(cache_key, response_text)
                logger.debug(
                    f"Cached response for prompt with model {model_name}")
            except Exception as e:
                logger.warning(f"Cache storage failed: {str(e)}")

        return response_text

    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower():
            raise OpenRouterAPIKeyError(f"Invalid API key: {error_msg}")
        elif "model" in error_msg.lower():
            raise OpenRouterModelError(f"Model error: {error_msg}")
        else:
            raise OpenRouterError(f"OpenRouter API error: {error_msg}")
