from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, HttpUrl
from pathlib import Path

class LLMConfig(BaseModel):
    model: str
    provider: str  # "ollama", "openai", etc.
    base_url: Optional[HttpUrl] = Field(None, description="Base URL for LLM API (e.g., Ollama)")
    api_key: Optional[str] = Field(None, description="API key for services like OpenAI")
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, gt=0)
    stream: bool = False
    timeout: int = Field(300, gt=0, description="Timeout in seconds")
    enable_tools: bool = Field(True, description="Enable function calling/tools")
    tools: Optional[List[Dict[str, Any]]] = None

class LoggingConfig(BaseModel):
    level: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    format: str = Field(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
        description="JSON logging format"
    )
    request_id_header: str = "X-Request-ID"

class FunctionConfig(BaseModel):
    config_path: Path = Field("functions/config.json")
    async_loading: bool = True
    version_required: bool = True
    dependency_check: bool = True
    max_concurrent_calls: int = Field(10, gt=0)

class AppConfig(BaseModel):
    llm: LLMConfig
    logging: LoggingConfig = LoggingConfig()
    functions: FunctionConfig = FunctionConfig()
    workspace_dir: Path = Field(default_factory=lambda: Path.cwd())
    environment: str = Field("development", pattern="^(development|production|testing)$")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load application configuration from environment variables and/or config file."""
    # First load from environment variables
    config = AppConfig.parse_obj({})
    
    # If config_path provided, update with file-based config
    if config_path and Path(config_path).exists():
        config = AppConfig.parse_file(config_path)
    
    return config
