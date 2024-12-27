# Desktop LLM Architecture

## Overview

Desktop LLM is a FastAPI-based application that provides a robust interface for interacting with local and remote language models. The application supports streaming chat completions, function calling, and a modular system for processing inputs and outputs, with integrated long-term memory and file system capabilities.

## Core Components

### Configuration (`app/core/config.py`)

- Manages application-wide settings using Pydantic BaseSettings
- Key configurations:
  - Multiple Ollama base URLs support
  - Model parameters (temperature, max tokens)
  - Rate limiting
  - Function execution settings
  - SSE (Server-Sent Events) configurations
  - Chroma and MCP settings
  - Workspace directory configuration

### Service Layer

#### Model Service (`app/services/model_service.py`)

- Handles interactions with language models
- Features:
  - Model discovery and caching from multiple providers (Ollama, OpenAI)
  - Streaming chat completions
  - Tool/function calling integration
  - Response chunking and word-by-word streaming
- Uses AsyncClient for non-blocking operations

#### Function Service (`app/services/function_service.py`)

- Manages the function registry and execution
- Supports multiple function types:
  - Tools: Callable by LLMs for external actions
  - Filters: Process input/output streams
  - Pipelines: Complex multi-step workflows

#### Agent Service (`app/services/agent.py`)

- High-level orchestration of model and function interactions
- Manages conversation state and tool execution
- Provides unified interface for chat and completion endpoints

#### Chroma Service (`app/services/chroma_service.py`)

- Manages vector database operations
- Features:
  - Persistent storage of embeddings
  - Semantic search capabilities
  - Memory management with metadata
  - Document retrieval and storage

#### MCP Service (`app/services/mcp_service.py`)

- Manages Model Context Protocol integrations
- Features:
  - File system operations
  - Tool discovery and execution
  - Plugin management
  - Session handling

#### LangChain Service (`app/services/langchain_service.py`)

- Integrates LangChain capabilities
- Features:
  - Vector store integration with Chroma
  - Memory querying and retrieval
  - File system integration with MCP
  - Custom embedding strategies

### API Layer

#### Chat Router (`app/routers/chat.py`)

- Main endpoint: `/api/v1/chat/stream`
- Features:
  - Server-Sent Events for streaming responses
  - Function/tool integration
  - Pipeline processing
  - Input/output filtering
  - Error handling and logging

#### Health Router (`app/routers/health.py`)

- System health monitoring
- Reports:
  - Available models
  - Registered functions
  - System metrics (memory, disk)
  - Component status
  - Service states

### Function System

#### Base Classes (`app/functions/base.py`)

- Abstract base classes for function types:
  - `BaseFunction`: Common attributes
  - `Filter`: Input/output processing
  - `Tool`: LLM-callable functions
  - `Pipeline`: Multi-step processors
- Built-in error handling and validation

#### Function Registry (`app/functions/registry.py`)

- Central registry for all function types
- Features:
  - Dynamic function discovery
  - Dependency checking
  - Configuration loading
  - Type-safe registration

#### Function Types

1. Tools (`app/functions/types/tools/`)

   - File system tools
   - Web scraping tools
   - Calculator
   - Weather tools

2. Filters (`app/functions/types/filters/`)

   - Text modification
   - Content filtering

3. Pipelines (`app/functions/types/pipelines/`)
   - Multi-step processing
   - Complex workflows

### Service Locator (`app/core/service_locator.py`)

- Manages service dependencies
- Prevents circular dependencies
- Provides global access to services
- Supports runtime service registration

## Data Flow

1. Request Processing

   ```
   Client Request
   → FastAPI Router
   → Request Validation (Pydantic)
   → Agent Service
   → Model Service
   → LLM Provider (Ollama/OpenAI)
   ```

2. Response Processing

   ```
   LLM Response
   → Model Service (chunking)
   → Tool Execution (if needed)
   → Filter Processing
   → SSE Stream
   → Client
   ```

3. Memory Operations
   ```
   File/Content
   → MCP Service
   → LangChain Service
   → Chroma Service (vectorization)
   → Persistent Storage
   ```

## Key Features

### Streaming

- Word-by-word streaming for smooth UI updates
- Configurable chunk sizes
- Keep-alive ping mechanism
- Error handling with immediate client notification

### Tool Integration

- Dynamic tool discovery and registration
- Schema validation
- Asynchronous execution
- Result streaming

### Model Management

- Multiple provider support
- Model caching
- Health checking
- Automatic retries

### Long-term Memory

- Vector-based storage with Chroma
- Semantic search capabilities
- File system integration
- Metadata filtering

## Dependencies

- FastAPI: Web framework
- Pydantic: Data validation
- aiohttp: Async HTTP client
- SSE-Starlette: Server-Sent Events
- Ollama: Local model interface
- OpenAI (optional): Remote model access
- Chroma: Vector database
- LangChain: LLM framework
- SentenceTransformers: Embeddings
- MCP: Model Context Protocol

## Configuration

Key environment variables:

```
OLLAMA_BASE_URLS=["http://localhost:11434"]
DEFAULT_MODEL="qwen2.5-coder:14b-instruct-q4_K_M"
MODEL_TEMPERATURE=0.7
MAX_TOKENS=4096
FUNCTION_CALLS_ENABLED=true
LOG_LEVEL=DEBUG
CHROMA_PERSIST_DIRECTORY=chroma_data
CHROMA_COLLECTION_NAME=desktop_llm_memory
MCP_SERVER_FILESYSTEM_PATH=./src/filesystem/dist/index.js
WORKSPACE_DIR=./data
```

## API Endpoints

### Chat

- `POST /api/v1/chat/stream`
  - Streaming chat completions
  - Function calling
  - Pipeline processing

### Health

- `GET /api/v1/health`
  - System status
  - Component health
  - Resource metrics

### Memory

- `POST /api/v1/memory/add`

  - Add content to memory
  - File processing
  - Directory indexing

- `GET /api/v1/memory/query`
  - Semantic search
  - File retrieval
  - Metadata filtering
