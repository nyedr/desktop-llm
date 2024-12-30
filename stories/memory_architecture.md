# Memory Architecture

## Overview

Desktop LLM implements a sophisticated dual-layer memory system using ChromaDB as its vector store backend. The system is designed to provide efficient, context-aware memory management with support for both ephemeral (temporary) and persistent (model) memory storage.

## Core Components

### 1. Storage Layer

#### ChromaDB Vector Store

- **Backend**: SQLite-based persistent storage
- **Collections**:
  - `EPHEMERAL`: Temporary conversation history and context
  - `MODEL_MEMORY`: Long-term persistent memory storage
- **Embedding Model**: SentenceTransformer for vector embeddings
- **Vector Similarity**: Cosine similarity for memory retrieval

### 2. Memory Types

#### Ephemeral Memory

- **Purpose**: Short-term storage for conversation context
- **Contents**:
  - Active conversation messages
  - Conversation summaries
  - Temporary context information
- **Lifecycle**: Cleared between sessions or on explicit cleanup

#### Model Memory

- **Purpose**: Long-term knowledge retention
- **Contents**:
  - User preferences and information
  - Important conversation outcomes
  - Reusable knowledge
- **Lifecycle**: Persists across sessions

### 3. Memory Management Services

#### ChromaService

- Vector storage operations
- Memory CRUD operations
- Chunking and reassembly
- Duplicate detection
- Metadata management

#### LangChainService

- Memory integration with LLM
- Conversation processing
- Text summarization
- Context retrieval

#### LLMContextManager

- Context window management
- Memory retrieval orchestration
- Prompt engineering
- Token counting and optimization

## Memory Operations

### 1. Storage Operations

#### Adding Memories

```python
async def add_memory(
    text: Union[str, List[str]],
    collection: MemoryType,
    metadata: Optional[Dict[str, Any]] = None,
    max_chunk_tokens: Optional[int] = 512,
    conversation_id: Optional[str] = None
) -> Optional[Union[str, List[str]]]
```

- Automatic chunking for long texts
- Duplicate detection
- Metadata enrichment
- Vector embedding generation

#### Retrieving Memories

```python
async def retrieve_memories(
    query: str,
    collection: MemoryType,
    top_k: int = 4,
    score_threshold: float = 0.5,
    metadata_filter: Optional[Dict[str, Any]] = None,
    conversation_id: Optional[str] = None,
    reassemble_chunks: bool = True
) -> List[Dict[str, Any]]
```

- Semantic search using vector similarity
- Chunk reassembly
- Relevance scoring
- Metadata filtering

### 2. Context Management

#### Token Management

- Dynamic context window sizing
- Token counting with model-specific tokenizers
- Automatic summarization on overflow
- Priority-based content selection

#### Memory Prioritization

1. System context and instructions
2. Recent conversation history
3. Relevant long-term memories
4. Background context

### 3. Summarization

#### Conversation Summarization

- Automatic summarization on context overflow
- Periodic conversation state capture
- Important information extraction
- Storage in ephemeral collection

#### Summary Types

1. **Conversation Summaries**

   - Recent interaction context
   - Key decisions and outcomes
   - User preferences and information

2. **Knowledge Summaries**
   - Reusable information
   - Common patterns
   - User-specific knowledge

## Memory Lifecycle

### 1. Creation

1. Text preprocessing and sanitization
2. Metadata generation
3. Embedding calculation
4. Chunking (if needed)
5. Storage in appropriate collection

### 2. Retrieval

1. Query vector generation
2. Similarity search
3. Metadata filtering
4. Chunk reassembly
5. Relevance scoring

### 3. Updates

1. Content validation
2. Metadata enrichment
3. Vector recalculation
4. Storage update

### 4. Deletion

1. Memory validation
2. Collection cleanup
3. Associated data removal

## Best Practices

### 1. Memory Storage

- Store user preferences in model memory
- Keep conversation context in ephemeral memory
- Use appropriate chunk sizes (default: 512 tokens)
- Include comprehensive metadata

### 2. Memory Retrieval

- Use specific queries for better relevance
- Apply appropriate metadata filters
- Set reasonable similarity thresholds
- Consider context window limitations

### 3. Performance Optimization

- Batch similar operations
- Use efficient chunk sizes
- Implement proper indexing
- Regular maintenance and cleanup

## Security Considerations

### 1. Data Protection

- Sanitize input text
- Validate metadata
- Secure storage backend
- Access control implementation

### 2. Privacy

- User data separation
- Conversation isolation
- Metadata filtering
- Secure deletion support

## Future Improvements

1. **Enhanced Summarization**

   - Improved algorithms
   - Better information extraction
   - More efficient storage

2. **Advanced Retrieval**

   - Hybrid search methods
   - Better relevance scoring
   - Improved chunk handling

3. **Optimization**

   - Better embedding models
   - More efficient storage
   - Improved performance

4. **Features**
   - Multi-modal memory support
   - Enhanced privacy controls
   - Better memory organization
