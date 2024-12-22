Current step: 5

# Implementing File System Management, Persistent Memory, and Semantic Search in `desktop-llm` with MCP, LangChain, and Chroma (Server-Side)

## Overview

This guide details the server-side integration of **Model Context Protocol (MCP)**, **LangChain**, and **Chroma** into the `desktop-llm` application. This integration will enhance the application with:

1. **File System Management:** Using the MCP File System plugin for local file system interaction (search, read, write).
2. **Persistent Memory:** Implementing long-term memory storage and retrieval with **Chroma** as the vector database.
3. **Advanced Semantic Search:** Leveraging Chroma's capabilities for enhanced relevance-based retrieval with custom embedding strategies.
4. **Future Scalability:** Maintaining a modular design to easily add more MCP plugins (e.g., web scraping, API tools) in the future.

**Note:** This guide focuses on the server-side implementation using Python. Client-side integration with technologies like React, TypeScript, and Electron will be addressed separately.

## Implementation Steps

### Step 1: Set Up the Environment

- **Install Necessary Packages:**

  ```bash
  pip install langchain langchain-community chromadb sentence-transformers langchain-mcp
  ```

  - `langchain`: Core framework for building LLM applications.
  - `langchain-community`: Contains community integrations for LangChain.
  - `chromadb`: Vector database for storing and searching embeddings.
  - `sentence-transformers`: For generating sentence embeddings.
  - `langchain-mcp`: Provides the `MCPToolkit` for integrating with MCP servers.

- **Install the MCP File System Plugin:**

  We will be using the local `mcp-server-filesystem` plugin.

  1. **Navigate to the `filesystem` directory:**

     ```bash
     cd /src/filesystem
     ```

  2. **Install dependencies:**

     ```bash
     npm install
     ```

  3. **Build the plugin:**

     ```bash
     npm run build
     ```

     The compiled `index.js` file will be located in `/src/filesystem/dist`.

- **Directory Structure:**

  Ensure your project has the following directory structure (or adjust paths accordingly in the configuration):

  ```
  desktop-llm/
  ├── app/             # Your existing application code
  │   ├── core/
  │   ├── dependencies/
  │   ├── functions/
  │   ├── models/
  │   ├── routers/
  │   ├── services/
  │   └── ...
  ├── data/            # Data directory for file system interactions (used by mcp-server-filesystem)
  │   └── example.txt  # Example file for testing
  ├── mcp/             # MCP plugin configurations
  │   └── server-filesystem/ # Compiled filesystem plugin
  │       └── dist/
  │           └── index.js    # Compiled plugin
  ├── src/             # MCP server source code
      └── filesystem/  # Filesystem plugin source
          ├── package.json
          ├── index.ts
          ├── Dockerfile
          ├── tsconfig.json
          └── README.md
  ├── .env             # Environment variables
  ├── .env.example
  ├── .gitignore
  ├── config.json
  ├── config.py
  ├── environment.yml
  ├── logger.py
  ├── pyproject.toml
  ├── requirements-dev.txt
  ├── requirements.txt
  ├── setup.py
  └── ...
  ```

- **Environment Variables:**

  Update your `.env` file:

  ```
  # ... existing environment variables ...

  # Chroma Settings
  CHROMA_PERSIST_DIRECTORY=chroma_data
  CHROMA_COLLECTION_NAME=desktop_llm_memory

  # Model Context Protocol Settings
  MCP_SERVER_FILESYSTEM_PATH=./src/filesystem/dist/index.js
  MCP_SERVER_FILESYSTEM_COMMAND=node
  WORKSPACE_DIR=./data # Or the directory you want the filesystem plugin to access
  ```

  - `CHROMA_PERSIST_DIRECTORY`: Directory for Chroma to store its database files.
  - `CHROMA_COLLECTION_NAME`: Name of the collection within Chroma to store our data.
  - `MCP_SERVER_FILESYSTEM_PATH`: Points to the compiled `index.js` of the `mcp-server-filesystem`.
  - `MCP_SERVER_FILESYSTEM_COMMAND`: Command used to run the server (Node.js for this plugin).
  - `WORKSPACE_DIR`: The directory the filesystem server will be allowed to access.

### Step 2: Initialize MCP and File System Plugin

First, implement the TypeScript components for the filesystem plugin:

- **File:** `src/filesystem/file-operations.ts`

```typescript
export interface FileInfo {
  type: "file" | "directory";
  size: number;
  permissions: string;
  created: string;
  modified: string;
  accessed: string;
}

export async function getFileStats(filePath: string): Promise<FileInfo> {
  // Implementation for retrieving file information
  // Should return FileInfo object with file metadata
}

export async function searchFiles(
  directory: string,
  pattern: string,
  excludePatterns: string[] = []
): Promise<string[]> {
  // Implementation for finding files matching patterns
  // Should support glob patterns and exclusions
}

export async function applyFileEdits(
  filePath: string,
  edits: Array<{ oldText: string; newText: string }>,
  dryRun = false
): Promise<string> {
  // Implementation for modifying files
  // Should return a unified diff of changes
}
```

- **File:** `src/filesystem/__tests__/file-operations.test.ts`

```typescript
describe("File Operations", () => {
  describe("getFileStats", () => {
    // Tests for file information retrieval
  });

  describe("searchFiles", () => {
    // Tests for file pattern matching
  });

  describe("applyFileEdits", () => {
    // Tests for file modification
  });
});
```

- **File:** `src/filesystem/__tests__/path-utils.test.ts`

```typescript
describe("Path Utilities", () => {
  describe("validatePath", () => {
    // Tests for path validation
    // Should include tests for:
    // - Paths within allowed directories
    // - Symlinks
    // - Parent directory validation
    // - Unauthorized access attempts
  });
});
```

- **Required Dependencies:**
  Add to `src/filesystem/package.json`:

  ```json
  {
    "dependencies": {
      "glob": "^7.2.0",
      "minimatch": "^9.0.3",
      "diff": "^5.1.0"
    },
    "devDependencies": {
      "@types/glob": "^7.2.0",
      "@types/diff": "^5.0.9",
      "@types/jest": "^29.5.0",
      "@types/node": "^20.0.0",
      "jest": "^29.5.0",
      "ts-jest": "^29.1.0",
      "typescript": "^5.0.0"
    }
  }
  ```

- **Implementation Notes:**
  1. The `FileInfo` interface should match the expected output format for file metadata.
  2. File operations should handle Windows and Unix paths correctly.
  3. Path validation should be strict to prevent unauthorized access.
  4. Tests should cover edge cases like symlinks and non-existent files.
  5. All file operations should be asynchronous.

Then, implement the Python components for integrating with the application:

- **File:** `app/services/mcp_service.py`

```python
import asyncio
import pathlib
import logging
from typing import Optional

from langchain_community.tools.mcp import MCPToolkit
from mcp import ClientSession, StdioServerParameters, stdio_client

from app.core.config import config

logger = logging.getLogger(__name__)

class MCPService:
    def __init__(self):
        self.toolkit: Optional[MCPToolkit] = None
        self.session: Optional[ClientSession] = None

    async def initialize(self):
        """Initializes the MCP toolkit and establishes a session."""
        try:
            logger.info("Initializing MCP Service...")
            server_params = StdioServerParameters(
                command=config.MCP_SERVER_FILESYSTEM_COMMAND,
                args=[config.MCP_SERVER_FILESYSTEM_PATH, str(pathlib.Path(config.WORKSPACE_DIR))],
            )
            logger.debug(f"MCP server parameters: {server_params}")

            async with stdio_client(server_params) as (read, write):
                self.session = ClientSession(read, write)
                async with self.session as session:
                    self.toolkit = MCPToolkit(session=session)
                    await self.toolkit.ainitialize()
                    logger.info("MCP Service initialized successfully.")
                    return self.toolkit

        except Exception as e:
            logger.error(f"Failed to initialize MCP Service: {e}")
            raise

    async def get_tools(self):
        """Retrieves the tools from the MCP toolkit."""
        if not self.toolkit:
            await self.initialize()
        return self.toolkit.get_tools()

    async def close_session(self):
        """Closes the MCP session."""
        if self.session:
            await self.session.close()
            self.session = None
            self.toolkit = None
            logger.info("MCP session closed.")
```

- **Update `app/core/config.py` to include new environment variables:**

```python
class AppConfig(BaseSettings):
    """Application-wide configuration settings."""
    # ... other settings ...
    CHROMA_PERSIST_DIRECTORY: str = Field(default="chroma_data")
    CHROMA_COLLECTION_NAME: str = Field(default="desktop_llm_memory")
    MCP_SERVER_FILESYSTEM_PATH: str = Field(default="./src/filesystem/dist/index.js")
    MCP_SERVER_FILESYSTEM_COMMAND: str = Field(default="node")
    WORKSPACE_DIR: str = Field(default="./data")
    # ... other settings ...
```

- **Update `app/dependencies/providers.py`:**

```python
# ... other imports ...
from app.services.mcp_service import MCPService

class Providers:
    # ... other providers ...
    _mcp_service: Optional[MCPService] = None

    # ... other methods ...

    @classmethod
    def get_mcp_service(cls) -> MCPService:
        """Get or create MCP service instance."""
        if cls._mcp_service is None:
            cls._mcp_service = MCPService()
        return cls._mcp_service

# ... other dependency functions ...

def get_mcp_service(request: Request) -> MCPService:
    """Get MCP service instance."""
    return Providers.get_mcp_service()
```

- **Update `app/main.py` to initialize `MCPService`:**

```python
# ... other imports ...
from app.dependencies.providers import Providers
from app.core.config import config as app_config # Alias to avoid conflict with function name
# ... other code ...

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for FastAPI application."""
    # ... other initialization code ...

    try:
        # ... other initializations ...
        mcp_service = Providers.get_mcp_service()
        await mcp_service.initialize()

        # ... rest of your startup code ...
        yield

    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        raise
    finally:
        # ... other shutdown code ...
        if mcp_service:
            await mcp_service.close_session()
        logger.info("Application shutdown complete")
```

### Step 3: Set Up Chroma for Persistent Memory

First, ensure you have the required dependencies:

```bash
pip install chromadb langchain_community
```

- **Create a Chroma Service:**
  - File: `app/services/chroma_service.py`

```python
import chromadb
from chromadb.config import Settings
import logging
from typing import List, Dict, Any, Optional
from chromadb.api.models.Collection import Collection
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

from app.core.config import config

logger = logging.getLogger(__name__)

class ChromaService:
    def __init__(self):
        self.client = None
        self.collection = None
        self.embeddings = None
        self.vector_store = None

    async def initialize(self):
        """Initialize the Chroma client and collection."""
        try:
            logger.info("Initializing Chroma Service...")

            # Initialize Chroma client with persistence
            self.client = chromadb.PersistentClient(
                path=config.CHROMA_PERSIST_DIRECTORY,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Initialize embeddings
            self.embeddings = OpenAIEmbeddings()

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=config.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )

            # Initialize LangChain vector store
            self.vector_store = Chroma(
                client=self.client,
                collection_name=config.CHROMA_COLLECTION_NAME,
                embedding_function=self.embeddings
            )

            logger.info("Chroma Service initialized successfully")
            return self.vector_store

        except Exception as e:
            logger.error(f"Failed to initialize Chroma Service: {e}")
            raise

    async def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Add texts and their metadata to the vector store."""
        try:
            if not self.vector_store:
                await self.initialize()

            return await self.vector_store.aadd_texts(texts=texts, metadatas=metadatas)

        except Exception as e:
            logger.error(f"Failed to add texts to Chroma: {e}")
            raise

    async def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents in the vector store."""
        try:
            if not self.vector_store:
                await self.initialize()

            return await self.vector_store.asimilarity_search(query, k=k)

        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            raise

    async def delete_collection(self):
        """Delete the current collection."""
        try:
            if self.client and self.collection:
                self.client.delete_collection(config.CHROMA_COLLECTION_NAME)
                self.collection = None
                self.vector_store = None
                logger.info(f"Deleted collection: {config.CHROMA_COLLECTION_NAME}")

        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise
```

- **Create tests for the Chroma Service:**
  - File: `tests/services/test_chroma_service.py`

```python
import pytest
from unittest.mock import Mock, patch
from app.services.chroma_service import ChromaService

@pytest.fixture
async def chroma_service():
    service = ChromaService()
    await service.initialize()
    yield service
    await service.delete_collection()

@pytest.mark.asyncio
async def test_add_and_search(chroma_service):
    # Test data
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Pack my box with five dozen liquor jugs"
    ]
    metadatas = [
        {"source": "test1.txt", "page": 1},
        {"source": "test2.txt", "page": 1}
    ]

    # Add texts
    ids = await chroma_service.add_texts(texts, metadatas)
    assert len(ids) == 2

    # Search
    results = await chroma_service.similarity_search("fox jumps", k=1)
    assert len(results) == 1
    assert "fox" in results[0].page_content.lower()
```

- **Update the providers to include ChromaService:**
  - File: `app/dependencies/providers.py`

```python
from app.services.chroma_service import ChromaService

class Providers:
    _chroma_service: Optional[ChromaService] = None

    @classmethod
    def get_chroma_service(cls) -> ChromaService:
        """Get or create Chroma service instance."""
        if cls._chroma_service is None:
            cls._chroma_service = ChromaService()
        return cls._chroma_service

def get_chroma_service(request: Request) -> ChromaService:
    """Get Chroma service instance."""
    return Providers.get_chroma_service()
```

- **Update the application startup to initialize ChromaService:**
  - File: `app/main.py`

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for FastAPI application."""
    try:
        # Initialize services
        mcp_service = Providers.get_mcp_service()
        chroma_service = Providers.get_chroma_service()

        await mcp_service.initialize()
        await chroma_service.initialize()

        yield

    finally:
        if mcp_service:
            await mcp_service.close_session()
        logger.info("Application shutdown complete")
```

- **Create an API endpoint to test the integration:**
  - File: `app/routers/memory.py`

```python
from fastapi import APIRouter, Depends, HTTPException
from app.services.chroma_service import ChromaService
from app.dependencies.providers import get_chroma_service

router = APIRouter()

@router.post("/memory/add")
async def add_to_memory(
    text: str,
    metadata: dict,
    chroma_service: ChromaService = Depends(get_chroma_service)
):
    """Add text to the vector store."""
    try:
        ids = await chroma_service.add_texts([text], [metadata])
        return {"message": "Text added successfully", "ids": ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/memory/search")
async def search_memory(
    query: str,
    k: int = 4,
    chroma_service: ChromaService = Depends(get_chroma_service)
):
    """Search for similar texts in the vector store."""
    try:
        results = await chroma_service.similarity_search(query, k=k)
        return {
            "results": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

- **Implementation Notes:**

  1. The Chroma service uses OpenAI embeddings by default. Make sure you have set the `OPENAI_API_KEY` environment variable.
  2. The vector store is persisted to disk in the directory specified by `CHROMA_PERSIST_DIRECTORY`.
  3. The service supports both synchronous and asynchronous operations through LangChain's async interfaces.
  4. The collection name is configurable through the `CHROMA_COLLECTION_NAME` environment variable.
  5. Error handling includes logging and appropriate exception propagation.
  6. The test suite provides examples of basic usage patterns.

- **Usage Example:**

```python
# Initialize services
chroma_service = ChromaService()
await chroma_service.initialize()

# Add some text with metadata
text = "Important information to remember"
metadata = {"source": "user_input", "timestamp": "2024-01-20T12:00:00Z"}
await chroma_service.add_texts([text], [metadata])

# Search for similar content
results = await chroma_service.similarity_search("important info", k=1)
for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
```

This completes the Chroma integration, providing persistent vector storage for the application's memory system. The next step will focus on integrating this with the LangChain memory components.

### Step 4: Integrate Chroma into LangChain

- **Create a LangChain Service:**
  - File: `app/services/langchain_service.py`

The LangChain service will integrate the Chroma vector store with the MCP file system and provide high-level memory operations. Key components include:

1. **Dependencies and Configuration:**

   ```python
   from langchain.chains import RetrievalQA
   from langchain.chat_models import ChatOpenAI
   from langchain.vectorstores import Chroma
   from langchain_community.embeddings import HuggingFaceEmbeddings
   from langchain_core.documents import Document
   ```

2. **Service Initialization:**

   - Initialize with both ChromaService and MCPService
   - Set up the vector store with HuggingFace embeddings
   - Configure the retrieval chain with Ollama model
   - Use the same collection and persistence settings as ChromaService

3. **Core Functionality:**

   - Memory querying with semantic search
   - File content storage and retrieval
   - Directory processing for bulk file operations
   - Metadata-based filtering

4. **Required Methods:**

   ```python
   async def query_memory(self, query: str) -> Dict[str, Any]:
       """Queries the Chroma database via LangChain and retrieves relevant memories."""

   async def save_file_to_memory(self, file_path: str) -> None:
       """Reads a file using MCP and saves its content to Chroma."""

   async def query_files_in_memory(self, query: str, top_k: int = 5) -> List[str]:
       """Queries Chroma for memories associated with files and returns the file paths."""

   async def process_directory(self, directory_path: str, file_extensions: Optional[List[str]] = None) -> None:
       """Processes all files in a directory and saves them to memory."""
   ```

5. **Integration Points:**

   - Use the ChromaService for vector storage operations
   - Use the MCPService for file system operations
   - Integrate with the application's LLM configuration
   - Handle both synchronous and asynchronous operations

6. **Error Handling:**

   - Proper logging of all operations
   - Graceful handling of file system errors
   - Vector store operation error management
   - LLM interaction error handling

7. **Configuration Requirements:**

   ```python
   # In app/core/config.py
   class AppConfig(BaseSettings):
       DEFAULT_MODEL: str = "mistral"  # or your chosen model
       MODEL_TEMPERATURE: float = 0.7
       OLLAMA_BASE_URLS: List[str]
       CHROMA_PERSIST_DIRECTORY: str
       CHROMA_COLLECTION_NAME: str
       CHROMA_EMBEDDING_MODEL: str = "multi-qa-mpnet-base-dot-v1"
   ```

8. **Provider Integration:**

   ```python
   # In app/dependencies/providers.py
   class Providers:
       _langchain_service: Optional[LangChainService] = None

       @classmethod
       def get_langchain_service(cls) -> LangChainService:
           if cls._langchain_service is None:
               cls._langchain_service = LangChainService()
           return cls._langchain_service
   ```

9. **Application Startup:**

   ```python
   # In app/main.py lifespan context
   langchain_service = Providers.get_langchain_service()
   # Initialize after ChromaService and MCPService
   ```

10. **Testing Requirements:**

    - Mock configurations for testing
    - Test file operations with temporary files
    - Test vector store operations
    - Test LLM interactions
    - Test error handling scenarios

11. **Dependencies:**
    ```yaml
    # In environment.yml
    dependencies:
      - langchain==0.1.9
      - langchain-community==0.0.24
      - langchain-core==0.3.28
      - chromadb==0.4.20
      - sentence-transformers==2.2.2
      - pytest-asyncio==0.24.0
    ```

This integration provides a high-level interface for memory operations, combining:

- Vector storage (Chroma) for semantic search
- File system operations (MCP) for content management
- LLM capabilities for enhanced retrieval and processing
- Proper error handling and logging
- Asynchronous operation support
- Comprehensive testing coverage

The service should be initialized after both ChromaService and MCPService, as it depends on both for its operations. All operations should be properly logged and should handle errors gracefully, providing meaningful error messages and appropriate error propagation.

### Step 5: Advanced Semantic Search with Chroma

The provided `ChromaService` already includes methods for advanced semantic search, including filtering by metadata and using custom embedding models.

### Step 6: Link File System and Persistent Memory

The `LangChainService` includes methods to link the file system and persistent memory:

- **`save_file_to_memory()`:** Reads a file using the MCP File System plugin and saves its content to Chroma.
- **`query_files_in_memory()`:** Queries Chroma for memories associated with files and returns the file paths.
- **`process_directory()`:** Processes all files in a directory (optionally filtering by file extension) and saves them to memory using `save_file_to_memory()`.

### Step 7: How to Add New MCP Plugins in the Future

**Integrating Additional MCP Plugins**

To expand the capabilities of your `desktop-llm` application, you can integrate additional MCP plugins. Here's a general guide on how to do this:

1. **Plugin Installation**:

   - Install the Node.js package for the desired MCP plugin, or if developing a new plugin, ensure it's accessible in your project structure.

2. **MCPService Update**:

   - Modify the `initialize()` method in `app/services/mcp_service.py` to load the new plugin.
   - Create a new `ClientSession` for each plugin, ensuring they run as separate processes.
   - Initialize new toolkits for each plugin and integrate them into the `MCPToolkit`.

3. **LangChain Integration (Optional)**:

   - Update `app/services/langchain_service.py` if the new plugin provides data that interacts with Chroma or requires LangChain's functionalities.
   - Add methods to interact with the new tools, retrieve data, and manage it within Chroma as needed.

4. **API Endpoints Update (Optional)**:

   - Modify `app/routers/chat.py` to expose new functionalities through API endpoints, if necessary.

5. **Testing**:
   - Write unit and integration tests to validate the new plugin's integration.
   - Perform end-to-end tests to ensure seamless operation with existing features.

#### Example: Adding a New Plugin

Suppose you want to add a new plugin named `mcp-server-webscraper`.

1. **Installation**:

   - Install the plugin using npm or add it to your project's `src` directory.

2. **Update `MCPService`**:

```python
async def initialize(self):
    # ... existing initialization code ...

    # Initialize Web Scraper plugin
    webscraper_server_params = StdioServerParameters(
        command="npx", # Assuming it's a Node.js based server
        args=["-y", "@modelcontextprotocol/server-webscraper"],
    )
    logger.debug(f"MCP web scraper server parameters: {webscraper_server_params}")

    async with stdio_client(webscraper_server_params) as (ws_read, ws_write):
        webscraper_session = ClientSession(ws_read, ws_write)
        async with webscraper_session as session:
            webscraper_toolkit = MCPToolkit(session=session)
            await webscraper_toolkit.ainitialize()
            logger.info("MCP Web Scraper Service initialized successfully.")

    # Combine toolkits
    self.toolkit = MCPToolkit(tools={
        **{tool.name: tool for tool in fs_toolkit.get_tools()},
        **{tool.name: tool for tool in webscraper_toolkit.get_tools()},
        # ... add more toolkits as needed ...
    })

# ... other methods ...

async def scrape_website(self, url: str) -> str:
    """Scrapes a website using the web scraper tool."""
    tools = await self.get_tools()
    scrape_tool = next((tool for tool in tools if "scrape_website" in tool.name), None)

    if scrape_tool:
        try:
            result = await scrape_tool.ainvoke({"url": url})
            return result["content"][0]["text"]
        except Exception as e:
            logger.error(f"Failed to scrape website: {e}")
            return ""
    else:
        logger.error("Web scraping tool not found.")
        return ""
```

3. **Update `LangChainService` (if needed):**

```python
async def scrape_and_save_website(self, url: str) -> None:
    """Scrapes a website and saves the content to Chroma."""
    scraped_content = await self.mcp_service.scrape_website(url)
    if scraped_content:
        metadata = {"source": "web", "url": url}
        self.chroma_service.add_memory(scraped_content, metadata)
        logger.info(f"Saved scraped content from {url} to memory.")
```

4. **Update API Endpoints (if needed):**

```python
@router.post("/chat/website/scrape")
async def scrape_and_save_website(
    request: Request,
    url: str,
    langchain_service: LangChainService = Depends(get_langchain_service),
):
    """Scrapes a website and saves the content to the memory store."""
    try:
        await langchain_service.scrape_and_save_website(url)
        return {"status": "success", "message": f"Website '{url}' scraped and saved to memory."}
    except Exception as e:
        logger.error(f"Error scraping website: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to scrape website: {e}")
```

### Step 8: Create API Endpoints

Update `app/routers/chat.py` to expose the new functionality:

```python
"""Chat router."""
# ... other imports ...
from app.dependencies.providers import (
    get_agent,
    get_model_service,
    get_function_service,
    get_chroma_service,
    get_langchain_service,
    get_mcp_service
)
from app.services.langchain_service import LangChainService
from app.services.mcp_service import MCPService
from app.services.chroma_service import ChromaService
# ... other imports ...

# ... other routes ...

@router.post("/chat/memory/add")
async def add_to_memory(
    request: Request,
    memory_text: str,
    metadata: Optional[Dict[str, Any]] = None,
    chroma_service: ChromaService = Depends(get_chroma_service)
):
    """Adds a text snippet to the memory store."""
    try:
        chroma_service.add_memory(memory_text, metadata)
        return {"status": "success", "message": "Memory added successfully."}
    except Exception as e:
        logger.error(f"Error adding to memory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add to memory: {e}")

@router.post("/chat/memory/query")
async def query_memory(
    request: Request,
    query: str,
    langchain_service: LangChainService = Depends(get_langchain_service)
):
    """Queries the memory store."""
    try:
        response = await langchain_service.query_memory(query)
        return {"status": "success", "response": response}
    except Exception as e:
        logger.error(f"Error querying memory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to query memory: {e}")

@router.post("/chat/file/save")
async def save_file_to_memory(
    request: Request,
    file_path: str,
    langchain_service: LangChainService = Depends(get_langchain_service)
):
    """Saves the contents of a file to the memory store."""
    try:
        await langchain_service.save_file_to_memory(file_path)
        return {"status": "success", "message": f"File '{file_path}' saved to memory."}
    except Exception as e:
        logger.error(f"Error saving file to memory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file to memory: {e}")

@router.get("/chat/files/query")
async def query_files_in_memory(
    request: Request,
    query: str,
    langchain_service: LangChainService = Depends(get_langchain_service)
):
    """Queries the memory store for relevant files."""
    try:
        file_paths = await langchain_service.query_files_in_memory(query)
        return {"status": "success", "file_paths": file_paths}
    except Exception as e:
        logger.error(f"Error querying files in memory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to query files in memory: {e}")

@router.post("/chat/directory/process")
async def process_directory(
    request: Request,
    directory_path: str,
    file_extensions: Optional[List[str]] = None,
    langchain_service: LangChainService = Depends(get_langchain_service)
):
    """Processes all files in a directory and saves them to memory."""
    try:
        await langchain_service.process_directory(directory_path, file_extensions)
        return {"status": "success", "message": f"Directory '{directory_path}' processed and saved to memory."}
    except Exception as e:
        logger.error(f"Error processing directory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process directory: {e}")
```

### Step 9: Testing

**Unit Tests:**

- **File:** `tests/services/test_chroma_service.py`

```python
import pytest
from app.services.chroma_service import ChromaService
from unittest.mock import patch, MagicMock

# Mock config for testing
class MockConfig:
    CHROMA_PERSIST_DIRECTORY = "test_chroma_data"
    CHROMA_COLLECTION_NAME = "test_collection"

@pytest.fixture
def chroma_service():
    with patch("app.services.chroma_service.config", MockConfig):
        service = ChromaService()
        yield service
        service.clear_collection()  # Clean up after each test

@pytest.mark.asyncio
async def test_add_and_retrieve_memory(chroma_service):
    chroma_service.add_memory("Test memory", {"source": "test"})
    memories = chroma_service.retrieve_memories("Test")
    assert len(memories) == 1
    assert memories[0]["document"] == "Test memory"

@pytest.mark.asyncio
async def test_retrieve_with_metadata(chroma_service):
    chroma_service.add_memory("Memory 1", {"project": "A"})
    chroma_service.add_memory("Memory 2", {"project": "B"})
    memories = chroma_service.retrieve_with_metadata("Memory", {"project": "A"})
    assert len(memories) == 1
    assert memories[0]["document"] == "Memory 1"

# Add more tests for other methods like add_memories_batch, update_memory, etc.
```

- **File:** `tests/api/test_chat_integration.py`

```python
import pytest
from httpx import AsyncClient
from app.main import app
from app.models.chat import ChatRequest, ChatMessage
from app.dependencies.providers import Providers

@pytest.fixture(scope="module")
def event_loop():
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="module")
async def async_test_client():
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client

@pytest.mark.asyncio
async def test_add_and_query_memory_integration(async_test_client):
    # Ensure Chroma service is available for this test
    chroma_service = Providers.get_chroma_service()
    assert chroma_service is not None, "Chroma service not initialized"

    # 1. Add a memory
    add_response = await async_test_client.post(
        "/api/v1/chat/memory/add",
        json={"memory_text": "Project X is on track.", "metadata": {"project": "X"}}
    )
    assert add_response.status_code == 200

    # 2. Query the memory
    query_response = await async_test_client.post(
        "/api/v1/chat/memory/query",
        json={"query": "What is the status of Project X?"}
    )
    assert query_response.status_code == 200
    assert "Project X is on track" in query_response.json()["response"]["result"]
```

**End-to-End Tests:**

- **File:** `tests/api/test_chat_e2e.py`

```python
import pytest
import os
from httpx import AsyncClient
from app.main import app
from app.dependencies.providers import Providers

@pytest.fixture(scope="module")
def event_loop():
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="module")
async def async_test_client():
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client

@pytest.mark.asyncio
async def test_save_and_query_file_e2e(async_test_client, tmp_path):
    # Ensure Chroma and MCP services are available for this test
    chroma_service = Providers.get_chroma_service()
    mcp_service = Providers.get_mcp_service()
    await mcp_service.initialize()
    assert chroma_service is not None, "Chroma service not initialized"
    assert mcp_service is not None, "MCP service not initialized"

    # Create a temporary file for testing
    test_file = tmp_path / "test_file.txt"
    test_file_content = "This is a test file for Project Y."
    test_file.write_text(test_file_content)

    # 1. Save the file to memory
    save_response = await async_test_client.post(
        "/api/v1/chat/file/save",
        json={"file_path": str(test_file)}
    )
    assert save_response.status_code == 200

    # 2. Query for the file in memory
    query_response = await async_test_client.get(
        "/api/v1/chat/files/query",
        params={"query": "Project Y"}
    )
    assert query_response.status_code == 200
    assert str(test_file) in query_response.json()["file_paths"]

    # 3. Query the memory to get information related to the file content
    memory_query_response = await async_test_client.post(
        "/api/v1/chat/memory/query",
        json={"query": "What is the content of the test file?"}
    )
    assert memory_query_response.status_code == 200
    assert test_file_content in memory_query_response.json()["response"]["result"]
```

**Run Tests:**

```bash
pytest
```

### Step 10: Documentation

- **README.md:** Update your project's `README.md` with instructions on:
  - Setting up the environment (installing dependencies, configuring environment variables).
  - Running the application with the new integrations.
  - Using the new API endpoints.
  - How the MCP, LangChain, and Chroma components work together.
- **Code Comments:** Add clear and concise comments to the new code, explaining the purpose of each class, method, and significant code block.
- **API Documentation:** If you're using a tool like Swagger or OpenAPI, update your API documentation to include the new endpoints.

### Deployment

- **Dependencies:** Ensure your deployment environment has all necessary packages installed (from `requirements.txt`).
- **Node.js:** You'll need Node.js and npm installed on your server to run the MCP plugins.
- **Environment Variables:** Set the environment variables (Chroma paths, MCP paths) in your deployment configuration.
- **Persistence:** Make sure the `CHROMA_PERSIST_DIRECTORY` is a persistent volume so that your Chroma data is not lost when the application restarts.
