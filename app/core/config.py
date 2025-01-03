from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, HttpUrl
from pathlib import Path
import os

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


class LLMConfig(BaseModel):
    """LLM-specific configuration settings."""
    model: str = Field(default="deepseek/deepseek-chat")
    base_url: Optional[HttpUrl] = Field(default="https://openrouter.ai/api/v1")
    api_key: Optional[str] = Field(default="")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(default=4096)
    stream: bool = Field(default=True)
    timeout: int = Field(default=300, gt=0)
    request_timeout: int = Field(default=30, gt=0)
    enable_tools: bool = Field(default=True)
    tools: Optional[List[Dict[str, Any]]] = None
    rate_limit: str = Field(default="60/minute")
    tokenizer_model: str = Field(default="gpt2")


class LoggingConfig(BaseModel):
    """Logging-specific configuration settings."""
    level: str = Field(
        default="DEBUG",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )
    format: str = Field(
        default='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
        description="JSON logging format"
    )
    request_id_header: str = Field(default="X-Request-ID")


class FunctionConfig(BaseModel):
    """Function-specific configuration settings."""
    config_path: Path = Field(default=PROJECT_ROOT / "functions/config.json")
    async_loading: bool = Field(default=True)
    version_required: bool = Field(default=True)
    dependency_check: bool = Field(default=True)
    max_concurrent_calls: int = Field(default=10, gt=0)
    execution_timeout: int = Field(default=30)
    enable_model_filter: bool = Field(default=False)
    model_filter_list: List[str] = Field(default_factory=list)


class MemoryConfig(BaseModel):
    """Memory and LightRAG-specific configuration settings."""
    data_dir: Path = Field(default=PROJECT_ROOT / "lightrag_data")
    queue_process_delay: float = Field(default=0.1)
    queue_error_retry_delay: float = Field(default=1.0)
    cleanup_interval: int = Field(default=3600)  # 1 hour
    optimization_interval: int = Field(default=7200)  # 2 hours
    monitoring_interval: int = Field(default=300)  # 5 minutes
    default_retention_days: int = Field(default=30)
    default_embedding_model: str = Field(default="nomic-embed-text")
    default_embedding_dim: int = Field(default=384)
    chunk_size: int = Field(default=512)
    max_chunk_tokens: int = Field(default=1024)
    openai_embedding_model: str = Field(default="text-embedding-ada-002")


class AppConfig(BaseModel):
    """Application-wide configuration settings."""
    # Sub-configs
    llm: LLMConfig = Field(default_factory=LLMConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    functions: FunctionConfig = Field(default_factory=FunctionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    # Environment
    environment: str = Field(default="development",
                             pattern="^(development|production|testing)$")
    workspace_dir: Path = Field(default=PROJECT_ROOT)

    # SSE Settings
    sse_ping_interval: int = Field(default=15)
    sse_retry_timeout: int = Field(default=5000)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def load_config() -> AppConfig:
    """Load application configuration from environment variables."""
    env_settings = {
        "llm": {
            "model": os.getenv("DEFAULT_MODEL", "deepseek/deepseek-chat"),
            "base_url": os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "temperature": float(os.getenv("MODEL_TEMPERATURE", "0.7")),
            "max_tokens": int(os.getenv("MAX_TOKENS", "4096")),
            "timeout": int(os.getenv("GENERATION_REQUEST_TIMEOUT", "300")),
            "request_timeout": int(os.getenv("MODEL_REQUEST_TIMEOUT", "30")),
            "enable_tools": os.getenv("FUNCTION_CALLS_ENABLED", "true").lower() == "true",
            "rate_limit": os.getenv("RATE_LIMIT", "60/minute"),
            "tokenizer_model": os.getenv("TOKENIZER_MODEL", "gpt2")
        },
        "logging": {
            "level": os.getenv("LOG_LEVEL", "DEBUG"),
        },
        "functions": {
            "execution_timeout": int(os.getenv("FUNCTION_EXECUTION_TIMEOUT", "30")),
            "enable_model_filter": os.getenv("ENABLE_MODEL_FILTER", "false").lower() == "true",
            "model_filter_list": eval(os.getenv("MODEL_FILTER_LIST", "[]"))
        },
        "memory": {
            "data_dir": os.getenv("LIGHTRAG_DATA_DIR", str(PROJECT_ROOT / "lightrag_data")),
            "queue_process_delay": float(os.getenv("MEMORY_QUEUE_PROCESS_DELAY", "0.1")),
            "queue_error_retry_delay": float(os.getenv("MEMORY_QUEUE_ERROR_RETRY_DELAY", "1.0")),
            "cleanup_interval": int(os.getenv("CLEANUP_INTERVAL", "3600")),
            "optimization_interval": int(os.getenv("OPTIMIZATION_INTERVAL", "7200")),
            "monitoring_interval": int(os.getenv("MONITORING_INTERVAL", "300")),
            "default_retention_days": int(os.getenv("DEFAULT_RETENTION_DAYS", "30")),
            "default_embedding_model": os.getenv("DEFAULT_EMBEDDING_MODEL", "nomic-embed-text"),
            "default_embedding_dim": int(os.getenv("DEFAULT_EMBEDDING_DIM", "384")),
            "chunk_size": int(os.getenv("CHUNK_SIZE", "512")),
            "max_chunk_tokens": int(os.getenv("MAX_CHUNK_TOKENS", "1024"))
        },
        "sse_ping_interval": int(os.getenv("SSE_PING_INTERVAL", "15")),
        "sse_retry_timeout": int(os.getenv("SSE_RETRY_TIMEOUT", "5000"))
    }

    return AppConfig.parse_obj(env_settings)


# Create a global config instance
config = load_config()
