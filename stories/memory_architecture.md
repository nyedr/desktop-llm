# Combined Application and Memory System Guide

## Updated Repository Structure

```
project-root/
├── app/
│   ├── context/
│   │   └── llm_context.py
│   ├── core/
│   │   ├── config.py
│   │   ├── mcp_config.py
│   │   └── service_locator.py
│   ├── dependencies/
│   │   └── providers.py
│   ├── entities/
│   │   ├── __init__.py
│   │   ├── extractor.py          # spaCy-based entity and relation extraction
│   │   ├── models.py             # Data models for entities and relationships
│   │   └── relations.py          # Rules and logic for relationships
│   ├── functions/
│   │   ├── __init__.py
│   │   ├── base.py
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
│   │   ├── manager.py            # Handles memory CRUD operations
│   │   ├── vector_store.py       # ChromaDB operations for vector memory
│   │   └── relational_store.py   # Relational database operations for entities and metadata
│   ├── models/
│   │   ├── chat.py
│   │   ├── completion.py
│   │   ├── entity_models.py      # Models for extracted entities
│   │   ├── function_models.py
│   │   ├── memory.py
│   │   └── model.py
│   ├── routers/
│   │   ├── chat.py
│   │   ├── completion.py
│   │   ├── functions.py
│   │   └── health.py
│   ├── services/
│   │   ├── agent.py
│   │   ├── base.py
│   │   ├── chroma_service.py
│   │   ├── entity_service.py     # Entity extraction and relation management service
│   │   ├── function_service.py
│   │   ├── langchain_service.py
│   │   ├── mcp_service.py
│   │   ├── model_service.py
│   │   └── monitoring.py
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
│   ├── entity_relation_system.md # Documentation for entity and relationship handling
│   ├── function_development.md
│   ├── functions_revamp.md
│   └── todos.md
└── streaming/
    ├── __init__.py
    ├── processor.py
    ├── sse.py
    ├── stream_config.py
    └── stream_processor.py
```

## Redesigned Memory System Architecture

### Overview

The redesigned memory system provides efficient, context-aware memory management by leveraging Chroma as its vector store backend, relational databases for structured data, and spaCy for NLP processing. This system consolidates all memory into a single collection, storing all data long-term until pruned based on relevance or storage capacity constraints, ensuring scalability and reliability.

### Core Components

#### 1. Storage Layer

##### ChromaDB Vector Store

- **Backend**: SQLite-based persistent storage.
- **Single Collection**: Unified storage for all memory, designed for long-term retention.
- **Embedding Model**: SentenceTransformer for vector embeddings.
- **Similarity Metric**: Cosine similarity for efficient memory retrieval.

##### Relational Database

- **Purpose**: Structured storage for relational data, metadata, and long-term associations.
- **Technologies**: PostgreSQL or MySQL.
- **Design**:
  - Schema optimized for context-based querying.
  - Indexing for efficient relational lookups.

#### 2. Memory Management Services

##### ChromaService

- **Responsibilities**:
  - CRUD operations for the unified memory collection.
  - Chunking and reassembly of data.
  - Duplicate detection.
  - Metadata enrichment and retrieval.

##### RelationalService

- **Responsibilities**:
  - Relational data storage and retrieval.
  - Schema management for structured data.
  - Metadata filtering for enhanced retrieval accuracy.
  - **Entity and Relation Creation**:
    - Automatically extract and store entities (e.g., names, dates, topics) using spaCy.
    - Define relationships between entities by analyzing dependency trees and contextual cues using spaCy. Relationships are modeled explicitly (e.g., "User -> Prefers -> Product") and stored as directed edges in the relational database. "User -> Prefers -> Product").
    - Maintain entity-linking mappings for efficient retrieval.

##### NLPProcessor (spaCy)

- **Responsibilities**:
  - Query preprocessing (tokenization, lemmatization, etc.).
  - Extraction of intents and entities.
  - Metadata augmentation for stored data.
  - **Entity and Relation Extraction**:
    - Identify entities and their attributes from input text.
    - Detect relationships through dependency parsing, leveraging spaCy's syntactic analysis capabilities to identify dependency arcs between words. Custom rules can be added to recognize domain-specific relationships by analyzing co-occurrence patterns, part-of-speech tags, and contextual embeddings. These relationships are then stored in the relational database as directed edges to enable efficient retrieval and query augmentation.

## Memory Operations

### Storage Operations

#### Adding Memories

```python
async def add_memory(
    text: Union[str, List[str]],
    metadata: Optional[Dict[str, Any]] = None,
    max_chunk_tokens: Optional[int] = 512
) -> Optional[str]:
```

- **Features**:
  - Automatic text chunking for optimized storage.
  - Embedding generation using SentenceTransformer.
  - Metadata integration for enriched context.
  - **Entity and Relation Creation**:
    - Extract entities and attributes using spaCy.
    - Automatically associate relevant relationships in the relational database.

#### Retrieving Memories

```python
async def retrieve_memories(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.5,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
```

- **Features**:
  - Semantic search via cosine similarity.
  - Metadata-based filtering.
  - Dynamic relevance scoring.
  - **Relation-Based Augmentation**:
    - Retrieve data associated with entities and relationships relevant to the query.

### Retrieval and Context Management

#### Query Workflow

1. **Preprocessing**:
   - Parse query using spaCy.
   - Generate embedding vectors for semantic search.
2. **Retrieval**:
   - Search Chroma for similar vector-based content.
   - Query the relational database using SQL queries dynamically constructed based on the required entities, attributes, and relationships. Leverage indexed fields for efficient lookups and apply metadata filters to narrow results. Support complex queries by joining tables representing entities and relationships, ensuring context-aware data retrieval tailored to user queries.
3. **Augmentation**:
   - Merge and rank results using a scoring system based on relevance and importance. This process incorporates relational context by querying the relational database for linked entities and their attributes.
   - Entity-linking is performed dynamically to identify and prioritize the most contextually relevant relationships, ensuring the final ranked results are tailored to the user's query.

## Best Practices

### Storage Optimization

- Limit chunk sizes to 512 tokens for efficient retrieval.
- Enrich memory with descriptive metadata.
- Regularly prune irrelevant data.
- Maintain accurate entity mappings and relationships.

### Retrieval Optimization

- Use clear and specific queries for high relevance.
- Leverage metadata filters to narrow search scope.
- Set appropriate thresholds for similarity scores.
- Integrate relationship-based filtering for enhanced context.

### Performance Enhancements

- Implement caching for frequent operations.
- Use parallel processing for batch tasks.
- Maintain comprehensive logging for debugging and analytics.\

A contextually rich, long-term memory system—one that feels human-like, provides fast retrieval, and includes relational or additional context for richer answers.&#x20;
