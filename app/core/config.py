from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class AppConfig(BaseSettings):
    """Application-wide configuration settings."""
    OLLAMA_BASE_URLS: List[str] = Field(default=["http://localhost:11434"])
    OPENAI_API_KEY: str = Field(default="")
    DEFAULT_MODEL: str = Field(default="granite3.1-8b-24k:latest")
    MODEL_TEMPERATURE: float = Field(default=0.7)
    MAX_TOKENS: int = Field(default=4096)
    MODEL_REQUEST_TIMEOUT: int = Field(default=30)
    DEFAULT_TEMPERATURE: float = Field(default=0.7)
    RATE_LIMIT: str = Field(default="60/minute")
    FUNCTION_CALLS_ENABLED: bool = Field(default=True)
    FUNCTION_EXECUTION_TIMEOUT: int = Field(default=30)
    ENABLE_MODEL_FILTER: bool = Field(default=False)
    MODEL_FILTER_LIST: List[str] = Field(default_factory=list)
    GENERATION_REQUEST_TIMEOUT: int = Field(default=300)
    LOG_LEVEL: str = Field(default="DEBUG")
    # Seconds between SSE ping messages
    SSE_PING_INTERVAL: int = Field(default=15)
    # Milliseconds to wait before retrying connection
    SSE_RETRY_TIMEOUT: int = Field(default=5000)
    TOKENIZER_MODEL: str = Field(default="gpt2")

    # Chroma Settings
    CHROMA_PERSIST_DIRECTORY: str = Field(default="chroma_data")
    CHROMA_COLLECTION_NAME: str = Field(default="desktop_llm_memory")
    CHROMA_EMBEDDING_MODEL: str = Field(default="multi-qa-mpnet-base-dot-v1")

    # MCP Settings
    MCP_SERVER_FILESYSTEM_PATH: str = Field(
        default="./src/filesystem/dist/index.js")
    MCP_SERVER_FILESYSTEM_COMMAND: str = Field(default="node")
    WORKSPACE_DIR: str = Field(default="./data")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


# Create a global config instance
config = AppConfig()
