# Desktop LLM Project Rules

## Project Context

(Rule: ProjectContext)
Desktop LLM is a FastAPI-based application that provides a robust interface for interacting with local and remote language models. The application focuses on streaming chat completions, function calling, and modular system processing with integrated long-term memory and file system capabilities.

## Tech Stack & Architecture

(Rule: TechStack)

- Primary Language: Python
- Framework: FastAPI
- Key Dependencies:
  - Pydantic: Data validation
  - aiohttp: Async HTTP client
  - SSE-Starlette: Server-Sent Events
  - OpenAI Compatible Models: Remote model access
  - LightRAG: Memory management and retrieval system
  - SentenceTransformers: Embeddings with optimized storage
  - MCP: Model Context Protocol with extended capabilities

(Rule: Architecture)

- Service-oriented architecture with clear separation of concerns
- Modular design with distinct service layers:
  - Model Service: Handles LLM interactions with streaming support
  - Function Service: Manages function registry and execution with transaction handling
  - Agent Service: High-level orchestration with memory integration
  - Memory Service: LightRAG-based memory operations and storage
  - MCP Service: Model Context Protocol integrations with extended tool support
- Asynchronous operations for non-blocking performance
- Event-driven streaming responses with optimized memory usage
- LightRAG-based memory system with hybrid storage
- Multi-level memory hierarchy with graph-based relationships

## Code Style & Structure

(Rule: CodeStyle)

- Follow PEP 8 conventions for Python code
- Use type hints for all function parameters and return values
- Implement async/await patterns for I/O operations
- Keep functions focused and single-purpose
- Use descriptive variable names that reflect their purpose
- Document complex logic with clear comments
- Use Pydantic models for data validation and serialization
- Ensure code is modular and easy to understand, with clear separation of concerns.

## Repository Structure

(Rule: RepoStructure)

```
project-root/
├── app/
│   ├── context/
│   │   └── llm_context.py
│   ├── core/
│   │   ├── config.py
│   │   ├── mcp_config.py
│   │   ├── prompts.py
│   │   └── service_locator.py
│   ├── dependencies/
│   │   └── providers.py
│   ├── functions/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── base.py
│   │   ├── chat_helper.py
│   │   ├── executor.py
│   │   ├── registry.py
│   │   ├── types/
│   │   │   ├── __init__.py
│   │   │   ├── filters/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── parameter_normalizer.py
│   │   │   │   └── text_modifier.py
│   │   │   ├── pipelines/
│   │   │   │   ├── __init__.py
│   │   │   │   └── multi_step.py
│   │   │   └── tools/
│   │   │       ├── __init__.py
│   │   │       ├── calculator.py
│   │   │       ├── memory_tool.py
│   │   │       ├── weather_tools.py
│   │   │       └── web_scrape_tool.py
│   │   └── utils.py
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── datastore.py
│   │   ├── embeddings.py
│   │   ├── ingestion.py
│   │   └── manager.py
│   ├── models/
│   │   ├── agent.py
│   │   ├── chat.py
│   │   ├── completion.py
│   │   ├── function_base.py
│   │   ├── function.py
│   │   ├── memory.py
│   │   └── model.py
│   ├── routers/
│   │   ├── chat.py
│   │   ├── completion.py
│   │   ├── functions.py
│   │   └── health.py
│   ├── services/
│   │   ├── assistant.py
│   │   ├── base.py
│   │   ├── context_service.py
│   │   ├── function_service.py
│   │   ├── mcp_service.py
│   │   ├── model_service.py
│   │   └── monitoring.py
│   ├── utils/
│   │   ├── chat_messages.py
│   │   ├── chat_setup.py
│   │   ├── chat_tools.py
│   │   ├── filters.py
│   │   ├── profiling.py
│   │   ├── tool_formatter.py
│   │   └── utils.py
│   └── main.py
├── config.json
├── config.py
├── environment.yml
├── logger.py
├── pyproject.toml
├── requirements.txt
├── run.py
├── setup.py
├── stories/
│   ├── ARCHITECTURE.md
│   ├── chroma_revamp.md
│   ├── Context_management_revamp.md
│   ├── function_development.md
│   ├── functions_revamp.md
│   ├── lightrag.md
│   ├── memory_pipeline.md
│   ├── rules.md
│   └── todos.md
└── streaming/
    ├── __init__.py
    ├── processor.py
    ├── sse.py
    ├── stream_config.py
    └── stream_processor.py
```

## Naming Conventions

(Rule: NamingConventions)

- Python files: snake_case (e.g., `model_service.py`)
- Classes: PascalCase (e.g., `ModelService`)
- Functions/methods: snake_case (e.g., `get_model_response`)
- Variables: snake_case (e.g., `response_stream`)
- Constants: UPPER_SNAKE_CASE (e.g., `MAX_TOKENS`)
- API endpoints: kebab-case (e.g., `/api/v1/chat-stream`)

## Error Handling & Logging

(Rule: ErrorHandling)

- Use custom exception classes for specific error cases
- Implement proper error boundaries in API endpoints
- Provide meaningful error messages for debugging
- Handle async operation errors appropriately
- Implement parameter normalization for function inputs
- Use structured error types from `app/functions/base.py`

(Rule: Logging)

- Use structured logging with appropriate log levels
- Include relevant context in log messages
- Configure logging based on environment
- Implement proper error tracking and monitoring

## Testing

(Rule: Testing)

- Write unit tests for all services and functions
- Use pytest as the testing framework
- Implement integration tests for API endpoints
- Mock external dependencies in tests
- Maintain high test coverage for critical components
- Test all function types (Tools, Filters, Pipelines)
- Implement end-to-end flow tests
- Include performance benchmarks

## Security & Compliance

(Rule: Security)

- Store sensitive data in environment variables
- Implement proper rate limiting
- Validate all input data using Pydantic models
- Handle API keys and credentials securely
- Implement proper CORS policies
- Ensure secure file system operations
- Implement proper access controls for memory operations

## Version Control & Workflow

(Rule: GitWorkflow)

- Use feature branches for development
- Write descriptive commit messages
- Review code changes through pull requests
- Keep the main branch stable
- Document breaking changes

## Documentation

(Rule: Documentation)

- Maintain clear and up-to-date README files
- Document API endpoints with OpenAPI/Swagger
- Include docstrings for classes and functions
- Keep architecture documentation current
- Document configuration options
- Provide clear guidelines for function development
- Document memory management strategies

## Performance

(Rule: Performance)

- Use async operations for I/O-bound tasks
- Implement proper caching strategies
- Optimize database queries and vector operations
- Monitor memory usage with large language models
- Configure appropriate timeouts for external services
- Implement efficient streaming mechanisms
- Use proper chunking for responses

## Function System Guidelines

(Rule: FunctionSystem)

- Function Types:

  - Tools: LLM-callable functions for external actions
  - Filters: Process input/output streams
  - Pipelines: Complex multi-step workflows

- Registration & Discovery:

  - Singleton registry pattern for managing functions
  - Automatic discovery of functions in specified directories
  - Configuration-based loading of functions
  - Dependency checking before registration
  - Dynamic function creation for runtime registration

- Execution Flow:

  - Parameter normalization and validation
  - Error handling with custom exception classes
  - Structured ToolResponse format for consistent results
  - Profiling of function execution
  - Batch processing of multiple tool calls

- Implementation Requirements:

  - Use proper base classes from `app/functions/base.py`
  - Register functions using the `@register_function` decorator
  - Implement proper parameter validation
  - Handle function execution errors gracefully
  - Support streaming responses where appropriate
  - Use proper utility functions from `app/functions/utils.py`

- Function Schema:

  - name: Unique identifier
  - type: filter, pipe, or action
  - parameters: Structured input requirements
  - description: Clear purpose explanation
  - dependencies: Required external resources
  - valves: Dynamic configuration options

- Service Integration:
  - FunctionService manages execution and lifecycle
  - Provides OpenAI-compatible function schemas
  - Handles both static and dynamic function registration
  - Maintains comprehensive logging and monitoring

## Memory Management

(Rule: MemoryManagement)

- LightRAG-based memory system with hybrid storage:
  - NanoVectorDB for vector storage
  - NetworkX for graph-based relationships
  - JSON-based key-value storage for metadata
- Optimized embedding strategies:
  - MiniLM embeddings for efficient semantic search
  - Batch processing with configurable batch sizes
  - Embedding cache for improved performance
- Comprehensive metadata management:
  - Automatic metadata generation
  - Custom metadata fields support
  - Indexed fields for efficient filtering
- File system integration:
  - Automatic file ingestion and chunking
  - Support for multiple file formats
  - Configurable chunk sizes and overlaps
- Advanced memory operations:
  - Entity-based search with graph relationships
  - Context-aware memory retrieval
  - Multi-level memory hierarchy
- Performance optimization:
  - Asynchronous operations with configurable concurrency
  - LLM and embedding caching
  - Query timeouts and retries
- Monitoring and maintenance:
  - Automatic memory cleanup
  - Query profiling and optimization
  - Error handling and recovery
