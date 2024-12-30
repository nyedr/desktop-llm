# Memory Pipeline Documentation

This document outlines the complete pipeline from user prompt to response in the LightRAG memory system, including descriptions, file references, and code snippets.

## 1. Input Processing

- User prompt is received through the chat interface
- Prompt is validated and preprocessed
- System determines if prompt requires memory retrieval
- Filters and pipelines are applied if specified

**File:** `app/routers/chat.py`

```python
async def setup_chat_components(
    request_id: str,
    chat_request: ChatRequest,
    function_service: FunctionService
) -> tuple[Optional[List[Dict]], List[Filter], Optional[Any]]:
    """Setup function schemas, filters, and pipeline for chat."""
    # Get available functions
    function_schemas = None
    if chat_request.enable_tools:
        function_schemas = function_service.get_function_schemas()

    # Get requested filters
    filters = []
    if chat_request.filters:
        for filter_name in chat_request.filters:
            filter_class = function_service.get_function(filter_name)
            if filter_class and filter_class.model_fields['type'].default == FunctionType.FILTER:
                filter_instance = filter_class()
                filters.append(filter_instance)

    # Get requested pipeline
    pipeline = None
    if chat_request.pipeline:
        pipeline_class = function_service.get_function(chat_request.pipeline)
        if pipeline_class and pipeline_class.model_fields['type'].default == FunctionType.PIPELINE:
            pipeline = pipeline_class()

    return function_schemas, filters, pipeline

async def process_memory_context(
    messages: List[StrictChatMessage],
    langchain_service: LangChainService,
    chat_request: ChatRequest,
    request_id: str
) -> List[StrictChatMessage]:
    """Process memory context in the background without blocking."""
    try:
        # Only process user and assistant messages
        filtered_messages = [
            msg for msg in messages
            if (isinstance(msg, dict) and msg.get("role") in ["user", "assistant"]) or
               (hasattr(msg, "role") and msg.role in ["user", "assistant"])
        ]

        # Process conversation with proper metadata
        processed_messages = await langchain_service.process_conversation(
            messages=filtered_messages,
            memory_type=chat_request.memory_type,
            conversation_id=chat_request.conversation_id,
            enable_summarization=chat_request.enable_summarization,
            metadata_filter=chat_request.memory_filter,
            top_k=chat_request.top_k_memories
        )

        # Preserve system messages in the final output
        system_messages = [
            msg for msg in messages
            if (isinstance(msg, dict) and msg.get("role") == "system") or
               (hasattr(msg, "role") and msg.role == "system")
        ]

        return system_messages + processed_messages
    except Exception as e:
        logger.warning(f"[{request_id}] Failed to add memory context: {e}")
        return messages
```

## 2. Embedding Generation

- Prompt text is passed to the embedding function
- Using Ollama's nomic-embed-text model (768 dimensions)
- Embedding is generated asynchronously
- Embedding vector is normalized

**File:** `app/services/rag_service.py`

```python
async def get_embeddings(texts: List[str], model: str = "nomic-embed-text") -> np.ndarray:
    try:
        return await ollama_embedding(texts, embed_model=model)
    except Exception as e:
        raise Exception(f"Failed to get embeddings: {str(e)}")
```

## 3. Memory Storage & Retrieval

### Storage Functions

**File:** `app/memory/lightrag/datastore.py`

```python
def store_memory(
    self,
    text: str,
    embedding: np.ndarray,
    metadata: Dict[str, Any]
) -> str:
    """Store a memory with associated text, embedding, and metadata.
    Returns the unique ID of the stored memory.
    Full documents are stored once in the datastore with their metadata."""

def store_chunk(
    self,
    chunk: str,
    embedding: np.ndarray,
    metadata: Dict[str, Any],
    parent_doc_id: str
) -> str:
    """Store a chunk with associated text, embedding, and metadata.
    Maintains reference to parent document through parent_doc_id.
    Returns the unique ID of the stored chunk."""

def update_memory(
    self,
    memory_id: str,
    new_text: Optional[str] = None,
    new_embedding: Optional[np.ndarray] = None,
    new_metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """Update an existing memory with new text, embedding, or metadata.
    Returns True if update was successful."""

def delete_memory(self, memory_id: str) -> bool:
    """Delete a memory by its ID.
    Returns True if deletion was successful."""
```

### Retrieval Functions

**File:** `app/memory/lightrag/datastore.py`

```python
def search_memories(
    self,
    query_embedding: np.ndarray,
    top_k: int = 5,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Search memories using vector similarity and optional metadata filters.
    Returns top_k most relevant memories."""

def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a specific memory by its ID.
    Returns memory data or None if not found."""

def get_related_memories(
    self,
    memory_id: str,
    relationship_type: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """Retrieve memories related to a specific memory by relationship type.
    Returns up to limit related memories."""
```

### Metadata Management

**File:** `app/memory/lightrag/datastore.py`

```python
def add_metadata(
    self,
    memory_id: str,
    metadata: Dict[str, Any]
) -> bool:
    """Add additional metadata to an existing memory.
    Returns True if metadata was added successfully."""

def update_metadata(
    self,
    memory_id: str,
    key: str,
    value: Any
) -> bool:
    """Update specific metadata key for a memory.
    Returns True if update was successful."""

def remove_metadata(
    self,
    memory_id: str,
    key: str
) -> bool:
    """Remove specific metadata key from a memory.
    Returns True if removal was successful."""
```

## 4. Context Construction

- Retrieved chunks are ranked by relevance
- Context window is constructed considering:
  - Max token limit (32768)
  - Chunk overlap (100 tokens)
  - Entity relationships
- System prompt is added for context

**File:** `app/services/rag_service.py`

```python
addon_params = {
    "chunk_token_size": 1200,
    "chunk_overlap_token_size": 100,
    "max_token_for_text_unit": 4000,
    "max_token_for_global_context": 4000,
    "max_token_for_local_context": 4000
}
```

## 5. LLM Generation

- Constructed context is sent to OpenRouter API
- Using configured model (default: meta-llama/llama-3.2-3b-instruct)
- Response is generated with:
  - Temperature: 0.7
  - Max tokens: 4000
  - Top-p: 0.9

**File:** `app/services/rag_service.py`

```python
async def openrouter_llm_complete(prompt: str, model_name: str = "meta-llama/llama-3.2-3b-instruct") -> str:
    client = OpenAI(base_url="https://openrouter.ai/api/v1")
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=4000,
        top_p=0.9
    )
    return response.choices[0].message.content
```

## 6. Response Processing

- LLM response is parsed and validated
- Entities and relationships are extracted
- Response is formatted for output
- Relevant metadata is attached

**File:** `app/memory/lightrag/manager.py`

```python
def process_response(response: str) -> Dict:
    entities = extract_entities(response)
    relationships = extract_relationships(response)
    return {
        "response": format_response(response),
        "metadata": {
            "entities": entities,
            "relationships": relationships
        }
    }
```

## 7. Memory Update

- New conversation context is stored
- Entities and relationships are updated
- Embeddings are cached for future use
- Memory cleanup is performed based on retention policies

**File:** `app/memory/lightrag/datastore.py`

```python
def cleanup_entities_by_policy(self, policy_name: str) -> int:
    cutoff = datetime.now()
    if policy_name == "short_term":
        cutoff -= timedelta(days=SHORT_TERM_RETENTION_DAYS)
    elif policy_name == "long_term":
        cutoff -= timedelta(days=LONG_TERM_RETENTION_DAYS)
    else:  # default
        cutoff -= timedelta(days=DEFAULT_RETENTION_DAYS)
```

## Performance Considerations

- Async processing for I/O operations
- Batch processing for embeddings
- Caching of frequent queries
- Optimized similarity search

**File:** `app/services/rag_service.py`

```python
rag = LightRAG(
    working_dir=working_dir,
    llm_model_func=openrouter_llm_complete,
    llm_model_name=model_name,
    llm_model_max_token_size=max_tokens,
    llm_model_max_async=4,
    embedding_func=embedding_func,
    embedding_batch_num=32,
    embedding_func_max_async=16,
    enable_llm_cache=True
)
```

## Error Handling

- API key validation
- Model availability checks
- Rate limiting
- Fallback mechanisms

**File:** `app/services/rag_service.py`

```python
class OpenRouterError(Exception):
    """Base exception for OpenRouter API errors."""
    pass

class OpenRouterAPIKeyError(OpenRouterError):
    """Raised when there are issues with the OpenRouter API key."""
    pass

class OpenRouterModelError(OpenRouterError):
    """Raised when there are issues with model selection or availability."""
    pass
```
