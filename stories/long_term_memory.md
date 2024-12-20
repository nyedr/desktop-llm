**Project Goal**

To augment the existing application with robust file system interaction, persistent memory, and advanced semantic search capabilities using MCP, LangChain, and Chroma, while maintaining a modular design for future expansion.

**Core Components**

- **Model Context Protocol (MCP):** A protocol for standardizing interactions between AI agents and external tools/plugins. We'll use its File System plugin.
- **LangChain:** A framework for building applications with large language models (LLMs). It will orchestrate the interactions between the LLM, MCP, and Chroma.
- **Chroma:** A vector database for storing and retrieving embeddings, enabling semantic search and long-term memory.

**Implementation Steps**

**Step 1: Set Up the Environment**

- **Install Necessary Packages:**

  ```bash
  pip install langchain langchain-community chromadb sentence-transformers
  ```

  - `langchain`: Core LangChain framework.
  - `langchain-community`: Contains community integrations for LangChain, including `langchain_mcp`.
  - `chromadb`: The Chroma vector database.
  - `sentence-transformers`: For generating sentence embeddings.

- **Directory Structure:**

  We'll need a directory for testing the file system interactions. Here's a suggested project structure (including the existing `app` directory):

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
  ├── data/            # New: Data directory for file system interactions
  │   └── example.txt
  ├── mcp/             # New: MCP plugin configurations
  ├── .env
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

- **Environment Variables**
  Add the following to your existing `.env` file or create one based on `.env.example`:

  ```
  # ... existing environment variables ...

  # Chroma Settings
  CHROMA_PERSIST_DIRECTORY=chroma_data
  CHROMA_COLLECTION_NAME=desktop_llm_memory

  # Model Context Protocol Settings
  MCP_SERVER_PATH=./mcp/server-filesystem
  ```

  - `CHROMA_PERSIST_DIRECTORY`: Where Chroma will store its database files.
  - `CHROMA_COLLECTION_NAME`: The name of the collection within Chroma to store our data.
  - `MCP_SERVER_PATH`: Points to where you will keep your MCP plugin configurations.

- **Verification of MCP Server Plugins:**

  1. **Install Node.js and npm:** If you don't have them already, download and install them from the official Node.js website.
  2. **Install the MCP File System Server:**

     ```bash
     mkdir -p ./mcp/server-filesystem
     cd ./mcp/server-filesystem
     npm init -y
     npm install @modelcontextprotocol/server-filesystem
     ```

     Verify that the installation was successful by checking for a `node_modules` directory and a `package.json` file in `./mcp/server-filesystem`.

  3. **Run a Simple Test:**
     To ensure the MCP File System server is working, you can run a simple test using `npx`:

     ```bash
     npx -y @modelcontextprotocol/server-filesystem ./data
     ```

     This command should start the MCP server, and it should output something like:

     ```
     ready - started server on 0.0.0.0:3000, url: http://localhost:3000
     ```

     Press `Ctrl+C` to stop the server after testing.

**Step 2: Initialize MCP and File System Plugin**

- **File:** `app/services/mcp_service.py` (Create a new service file)

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
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", str(pathlib.Path(config.MCP_SERVER_PATH).parent)],
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

# Example usage (you'll integrate this into your main application flow)
async def main():
    mcp_service = MCPService()
    try:
        await mcp_service.initialize()
        tools = await mcp_service.get_tools()
        for tool in tools:
            print(f" - {tool.name}: {tool.description}")

        # Use the tools here (example: read a file)
        # file_read_tool = next((tool for tool in tools if "read" in tool.name), None)
        # if file_read_tool:
        #     file_content = await file_read_tool.invoke({"path": "data/example.txt"})
        #     print(f"Content of example.txt:\n{file_content}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await mcp_service.close_session()

if __name__ == "__main__":
    asyncio.run(main())
```

- **Explanation:**

  - `MCPService` class: Manages the MCP toolkit and session.
  - `initialize()`: Sets up the MCP File System plugin by defining `server_params` to point to the plugin's installation location. It then creates a `ClientSession` and initializes the `MCPToolkit`.
  - `get_tools()`: Retrieves the available tools (e.g., file search, read, write) from the initialized toolkit.
  - `close_session()`: Closes the MCP session when it is no longer needed.
  - `main()`: Example of how to use `MCPService`.

- **Update** `app/dependencies/providers.py`:

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

- **Update** `app/main.py` to initialize `MCPService`:

```python
# ... other imports ...
from app.dependencies.providers import Providers
# ... other code ...

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
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
        await mcp_service.close_session()
        logger.info("Application shutdown complete")
```

**Step 3: Set Up Chroma for Persistent Memory**

- **Create a Chroma Service:**
  - File: `app/services/chroma_service.py` (Create a new service file)

```python
import chromadb
from chromadb.config import Settings
import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

from app.core.config import config

logger = logging.getLogger(__name__)

class ChromaService:
    def __init__(self):
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=config.CHROMA_PERSIST_DIRECTORY
        ))
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME
        )
        self.embedding_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
        logger.info(f"Chroma service initialized with persist directory: {config.CHROMA_PERSIST_DIRECTORY}")
        logger.info(f"Collection '{config.CHROMA_COLLECTION_NAME}' ready.")

    def add_memory(self, memory_text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Adds a memory (text and optional metadata) to the Chroma collection."""
        embedding = self.embedding_model.encode(memory_text).tolist()
        memory_id = str(hash(memory_text))  # Simple hash as a unique ID

        try:
            self.collection.add(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[memory_text],
                metadatas=[metadata or {}]
            )
            logger.info(f"Added memory with ID: {memory_id}")
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")

    def retrieve_memories(self, query: str, top_k: int = 5, where_filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieves memories from Chroma based on a query."""
        query_embedding = self.embedding_model.encode(query).tolist()

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filters
            )
            logger.info(f"Retrieved {len(results['documents'][0])} memories for query: '{query}'")
            # Returning a list of dictionaries with ids, documents and metadatas
            return [{"id": id, "document": doc, "metadata": meta} for id, doc, meta in zip(results["ids"][0], results["documents"][0], results["metadatas"][0])]
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []

    def clear_collection(self) -> None:
        """Clears all data from the collection."""
        try:
            self.collection.delete()
            logger.info(f"Collection '{self.collection.name}' cleared.")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")

    def list_collections(self) -> List[str]:
        """Lists all collections."""
        try:
            collections = self.client.list_collections()
            return [collection.name for collection in collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

# Example usage
async def main():
    chroma_service = ChromaService()

    # Add some memories
    chroma_service.add_memory("The weather today is sunny.", {"source": "weather_report", "date": "2023-12-20"})
    chroma_service.add_memory("Project Alpha's deadline is approaching.", {"project": "Alpha", "status": "urgent"})

    # Retrieve memories
    relevant_memories = chroma_service.retrieve_memories("What is the status of Project Alpha?")
    for memory in relevant_memories:
        print(memory)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

- **Explanation:**

  - `ChromaService`: Manages interactions with the Chroma database.
  - `__init__()`: Initializes the Chroma client, gets or creates the specified collection, and loads the sentence embedding model.
  - `add_memory()`: Adds a text memory and associated metadata to the collection, creating an embedding for the text.
  - `retrieve_memories()`: Queries the collection for memories similar to the given query, using the embedding model to create a query embedding. It also accepts optional metadata filters.
  - `clear_collection()`: Deletes all data from the collection. Use with caution!
  - `list_collections()`: Lists the names of all collections in the Chroma database.

- **Update** `app/dependencies/providers.py`:

```python
# ... other imports ...
from app.services.chroma_service import ChromaService

class Providers:
    # ... other providers ...
    _chroma_service: Optional[ChromaService] = None

    # ... other methods ...

    @classmethod
    def get_chroma_service(cls) -> ChromaService:
        """Get or create Chroma service instance."""
        if cls._chroma_service is None:
            cls._chroma_service = ChromaService()
        return cls._chroma_service

# ... other dependency functions ...

def get_chroma_service(request: Request) -> ChromaService:
    """Get Chroma service instance."""
    return Providers.get_chroma_service()
```

- **Update** `app/main.py` to initialize `ChromaService` (though we won't use it directly in `lifespan` yet):

```python
# ... other imports ...
from app.dependencies.providers import Providers
# ... other code ...

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # ... other initialization code ...

    try:
        # ... other initializations ...
        chroma_service = Providers.get_chroma_service() # Initialize Chroma Service
        # ... rest of your startup code ...
        yield

    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        raise
    finally:
        # ... other shutdown code ...
        logger.info("Application shutdown complete")
```

**Step 4: Integrate Chroma into LangChain**

- **Create a LangChain Service:**
  - File: `app/services/langchain_service.py` (Create a new service file)

```python
import logging
from typing import Dict, Any

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.core.config import config
from app.dependencies.providers import Providers
from app.services.chroma_service import ChromaService

logger = logging.getLogger(__name__)

class LangChainService:
    def __init__(self):
        self.chroma_service: ChromaService = Providers.get_chroma_service()
        self.vectorstore = Chroma(
            collection_name=config.CHROMA_COLLECTION_NAME,
            embedding_function=HuggingFaceEmbeddings(model_name="multi-qa-mpnet-base-dot-v1"),
            client=self.chroma_service.client,
            persist_directory=config.CHROMA_PERSIST_DIRECTORY
        )
        self.retriever = self.vectorstore.as_retriever()
        self.llm = ChatOpenAI(model=config.DEFAULT_MODEL, temperature=config.MODEL_TEMPERATURE) # Using Ollama Model
        self.retrieval_qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )
        logger.info("LangChain service initialized.")

    async def query_memory(self, query: str) -> Dict[str, Any]:
        """Queries the Chroma database via LangChain and retrieves relevant memories."""
        try:
            response = self.retrieval_qa.invoke(query)
            logger.info(f"Retrieved response for query: '{query}'")
            return response
        except Exception as e:
            logger.error(f"Failed to query memory: {e}")
            return {"error": str(e)}

# Example usage
async def main():
    langchain_service = LangChainService()
    query = "What is the status of Project Alpha?"
    response = await langchain_service.query_memory(query)
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

- **Explanation:**

  - `LangChainService`: A service to interact with Chroma using LangChain's abstractions.
  - `__init__()`:
    - Gets the `ChromaService` instance.
    - Initializes a `Chroma` vector store using the existing collection and embedding function.
    - Creates a `retriever` from the vector store.
    - Initializes an LLM (using `ChatOpenAI` with settings from `config`).
    - Sets up a `RetrievalQA` chain, which combines the retriever and LLM for question answering.
  - `query_memory()`: Takes a query, uses the `RetrievalQA` chain to retrieve relevant documents from Chroma and generates a response using the LLM.

- **Update** `app/dependencies/providers.py`:

```python
# ... other imports ...
from app.services.langchain_service import LangChainService

class Providers:
    # ... other providers ...
    _langchain_service: Optional[LangChainService] = None

    # ... other methods ...

    @classmethod
    def get_langchain_service(cls) -> LangChainService:
        """Get or create LangChain service instance."""
        if cls._langchain_service is None:
            cls._langchain_service = LangChainService()
        return cls._langchain_service

# ... other dependency functions ...

def get_langchain_service(request: Request) -> LangChainService:
    """Get LangChain service instance."""
    return Providers.get_langchain_service()
```

- **Update** `app/main.py` to initialize `LangChainService` (we'll use it later when creating API endpoints):

```python
# ... other imports ...
from app.dependencies.providers import Providers
# ... other code ...

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # ... other initialization code ...

    try:
        # ... other initializations ...
        langchain_service = Providers.get_langchain_service()  # Initialize LangChain Service
        # ... rest of your startup code ...
        yield

    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        raise
    finally:
        # ... other shutdown code ...
        logger.info("Application shutdown complete")
```

**Step 5: Advanced Semantic Search with Chroma**

This step focuses on enhancing the search capabilities within the `ChromaService`.

- **Update `app/services/chroma_service.py`:**

```python
import chromadb
from chromadb.config import Settings
import logging
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer

from app.core.config import config

logger = logging.getLogger(__name__)

class ChromaService:
    # ... (rest of the existing ChromaService code) ...

    def update_embedding_model(self, model_name: str):
        """Updates the embedding model used by the service."""
        try:
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Embedding model updated to: {model_name}")
        except Exception as e:
            logger.error(f"Failed to update embedding model: {e}")
            raise

    def retrieve_with_metadata(self, query: str, where_filters: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieves memories from Chroma based on a query and metadata filters."""
        query_embedding = self.embedding_model.encode(query).tolist()

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filters
            )
            logger.info(f"Retrieved {len(results['documents'][0])} memories for query: '{query}' with filters")
            # Returning a list of dictionaries with ids, documents and metadatas
            return [{"id": id, "document": doc, "metadata": meta} for id, doc, meta in zip(results["ids"][0], results["documents"][0], results["metadatas"][0])]
        except Exception as e:
            logger.error(f"Failed to retrieve memories with filters: {e}")
            return []

    def add_memories_batch(self, memories: List[Dict[str, Any]]) -> None:
        """Adds a batch of memories to the Chroma collection."""
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for memory in memories:
            memory_text = memory.get("document")
            metadata = memory.get("metadata", {})
            if not memory_text:
                logger.warning("Skipping memory with no document text.")
                continue

            embedding = self.embedding_model.encode(memory_text).tolist()
            memory_id = str(hash(memory_text))  # Simple hash as a unique ID

            ids.append(memory_id)
            embeddings.append(embedding)
            documents.append(memory_text)
            metadatas.append(metadata)

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Added {len(ids)} memories in batch.")
        except Exception as e:
            logger.error(f"Failed to add memories in batch: {e}")

    def update_memory(self, memory_id: str, new_text: Optional[str] = None, new_metadata: Optional[Dict[str, Any]] = None) -> None:
        """Updates an existing memory in the Chroma collection."""
        try:
            # Fetch the existing memory
            existing_memory = self.collection.get(ids=[memory_id], include=["metadatas", "embeddings", "documents"])
            if not existing_memory:
                raise ValueError(f"Memory with ID '{memory_id}' not found.")

            # Update text and embedding if new text is provided
            if new_text is not None:
                new_embedding = self.embedding_model.encode(new_text).tolist()
                existing_memory["embeddings"] = [new_embedding]
                existing_memory["documents"] = [new_text]

            # Update metadata if new metadata is provided
            if new_metadata is not None:
                # Assuming you want to merge with existing metadata
                if existing_memory["metadatas"] and existing_memory["metadatas"][0]:
                    updated_metadata = {**existing_memory["metadatas"][0], **new_metadata}
                else:
                    updated_metadata = new_metadata
                existing_memory["metadatas"] = [updated_metadata]

            # Update the collection
            self.collection.update(
                ids=[memory_id],
                embeddings=existing_memory["embeddings"],
                documents=existing_memory["documents"],
                metadatas=existing_memory["metadatas"]
            )
            logger.info(f"Updated memory with ID: {memory_id}")

        except Exception as e:
            logger.error(f"Failed to update memory: {e}")

    def delete_memory(self, memory_id: str) -> None:
        """Deletes a memory from the Chroma collection by its ID."""
        try:
            self.collection.delete(ids=[memory_id])
            logger.info(f"Deleted memory with ID: {memory_id}")
        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")

    def count_memories(self) -> int:
        """Returns the total number of memories in the collection."""
        try:
            count = self.collection.count()
            logger.info(f"Total number of memories in collection: {count}")
            return count
        except Exception as e:
            logger.error(f"Failed to count memories: {e}")
            return 0

    def get_memory_by_id(self, memory_id: str, include: Optional[List[str]] = None) -> Dict[str, Any]:
        """Retrieves a single memory by its ID."""
        if include is None:
            include = ["metadatas", "documents"]

        try:
            result = self.collection.get(ids=[memory_id], include=include)
            if result:
                # Extract the first item from each list since we're querying by a single ID
                return {
                    "id": result["ids"][0],
                    "document": result["documents"][0] if "documents" in result else None,
                    "metadata": result["metadatas"][0] if "metadatas" in result else None,
                    "embedding": result["embeddings"][0] if "embeddings" in result else None
                }
            else:
                logger.warning(f"Memory with ID '{memory_id}' not found.")
                return {}
        except Exception as e:
            logger.error(f"Failed to retrieve memory by ID: {e}")
            return {}

# Example usage (you can add this to your main function for testing):
    # ... other ChromaService methods ...

# Add to the main() function in chroma_service.py for demonstration:
async def main():
    # ... previous main() code ...
    # Retrieve memories with metadata filtering
    filtered_memories = chroma_service.retrieve_with_metadata("What is the status of projects?", {"project": "Alpha"})
    print("Filtered Memories:", filtered_memories)

    # Add memories in batch
    batch_memories = [
        {"document": "Batch memory 1.", "metadata": {"batch": "1"}},
        {"document": "Batch memory 2.", "metadata": {"batch": "1"}}
    ]
    chroma_service.add_memories_batch(batch_memories)

    # Update a memory
    chroma_service.update_memory("123", new_text="Updated memory text.", new_metadata={"updated": True})

    # Delete a memory
    chroma_service.delete_memory("456")

    # Count memories
    total_memories = chroma_service.count_memories()
    print(f"Total memories: {total_memories}")

    # Get memory by ID
    memory = chroma_service.get_memory_by_id("some_id")
    print(f"Memory by ID: {memory}")

if __name__ == "__main__":
    asyncio.run(main())

```

- **Explanation of Changes:**

  - `update_embedding_model()`: Allows you to change the embedding model dynamically. Useful for testing or switching to a more suitable model later.
  - `retrieve_with_metadata()`: Performs semantic search with metadata filtering. You provide a query and a dictionary of metadata filters (e.g., `{"project": "Alpha", "status": "In Progress"}`).
  - `add_memories_batch()`: Allows you to add multiple memories to Chroma in a single, more efficient operation. This is useful when ingesting a large amount of data initially or when processing a batch of new information.
  - `update_memory()`: Allows modification of the text and/or metadata of an existing memory in the Chroma collection. When updating the text, the corresponding embedding is also recalculated and updated.
  - `delete_memory()`: Enables the removal of a specific memory from the collection using its unique ID.
  - `count_memories()`: Provides a quick way to check the total number of memories stored in the collection, which can be useful for monitoring and debugging.
  - `get_memory_by_id()`: Fetches a single memory using its unique identifier. You can specify what data to include in the result (e.g., metadata, document text, embedding).

- **Incremental Indexing:**
  - The `add_memory()` and `add_memories_batch()` methods can be used for incremental indexing. You would need to keep track of which files have already been added to Chroma (e.g., using a separate database or file to store processed file paths/hashes).
  - When new files are added or existing files are modified, you can add or update only those specific files in Chroma.

**Step 6: Link File System and Persistent Memory**

- **Update `app/services/langchain_service.py`:**

```python
import logging
from typing import Dict, Any, List, Optional

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from app.core.config import config
from app.dependencies.providers import Providers
from app.services.chroma_service import ChromaService
from app.services.mcp_service import MCPService

logger = logging.getLogger(__name__)

class LangChainService:
    def __init__(self):
        self.chroma_service: ChromaService = Providers.get_chroma_service()
        self.mcp_service: MCPService = Providers.get_mcp_service()  # Get MCP service
        self.vectorstore = Chroma(
            collection_name=config.CHROMA_COLLECTION_NAME,
            embedding_function=HuggingFaceEmbeddings(model_name="multi-qa-mpnet-base-dot-v1"),
            client=self.chroma_service.client,
            persist_directory=config.CHROMA_PERSIST_DIRECTORY
        )
        self.retriever = self.vectorstore.as_retriever()
        self.llm = ChatOpenAI(model=config.DEFAULT_MODEL, temperature=config.MODEL_TEMPERATURE)
        self.retrieval_qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )
        logger.info("LangChain service initialized.")

    async def query_memory(self, query: str) -> Dict[str, Any]:
        # ... existing method code ...

    async def save_file_to_memory(self, file_path: str) -> None:
        """Reads a file using MCP and saves its content to Chroma."""
        tools = await self.mcp_service.get_tools()
        file_read_tool = next((tool for tool in tools if "read" in tool.name), None)

        if file_read_tool:
            try:
                file_content = await file_read_tool.ainvoke({"path": file_path})
                metadata = {"source": "file_system", "file_path": file_path}
                # Add to Chroma using the add_memory method (with metadata)
                self.chroma_service.add_memory(file_content, metadata)
                logger.info(f"Saved file content to memory: {file_path}")
            except Exception as e:
                logger.error(f"Failed to save file to memory: {e}")
        else:
            logger.error("File read tool not found.")

    async def query_files_in_memory(self, query: str, top_k: int = 5) -> List[str]:
        """Queries Chroma for memories associated with files and returns the file paths."""
        try:
            # Use the retrieve_memories method of ChromaService
            memories = self.chroma_service.retrieve_memories(query, top_k=top_k)

            # Extract file paths from the metadata of retrieved memories
            file_paths = [
                memory["metadata"].get("file_path")
                for memory in memories
                if memory["metadata"] and "file_path" in memory["metadata"]
            ]
            logger.info(f"Retrieved {len(file_paths)} file paths for query: '{query}'")
            return file_paths
        except Exception as e:
            logger.error(f"Failed to query files in memory: {e}")
            return []

    async def process_directory(self, directory_path: str, file_extensions: Optional[List[str]] = None) -> None:
        """Processes all files in a directory, optionally filtering by file extension, and saves them to memory."""
        tools = await self.mcp_service.get_tools()
        file_search_tool = next((tool for tool in tools if "search" in tool.name), None)

        if file_search_tool:
            try:
                search_results = await file_search_tool.ainvoke({"path": directory_path, "query": "*"})
                files_to_process = [
                    os.path.join(directory_path, result["path"])
                    for result in search_results
                    if "path" in result and
                    (file_extensions is None or any(result["path"].endswith(ext) for ext in file_extensions))
                ]

                for file_path in files_to_process:
                    await self.save_file_to_memory(file_path)

                logger.info(f"Processed {len(files_to_process)} files from directory: {directory_path}")

            except Exception as e:
                logger.error(f"Failed to process directory: {e}")
        else:
            logger.error("File search tool not found.")

# Example usage (you can add this to your main function for testing):
# ... other LangChainService methods ...

# Add to the main() function in langchain_service.py for demonstration:
async def main():
    langchain_service = LangChainService()

    # Save a specific file to memory
    await langchain_service.save_file_to_memory("data/example.txt")

    # Query for files in memory
    file_paths = await langchain_service.query_files_in_memory("example file")
    print("Relevant file paths:", file_paths)

    # Process all files in a directory
    await langchain_service.process_directory("data")

    # Example of querying memory (existing example)
    query = "What is the status of Project Alpha?"
    response = await langchain_service.query_memory(query)
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

- **Explanation of Changes:**

  - `save_file_to_memory()`:
    - Uses the MCP File System plugin's "read" tool to get the file content.
    - Calls the `ChromaService`'s `add_memory()` method to store the content and metadata (including the `file_path`) in Chroma.
  - `query_files_in_memory()`:
    - Queries Chroma using `chroma_service.retrieve_memories()`.
    - Filters the results to extract the `file_path` from the metadata of each memory.
    - Returns a list of file paths that are relevant to the query.
  - `process_directory()`:
    - Uses the MCP File System plugin's "search" tool to find files within a directory.
    - Optionally filters files by extension.
    - Calls `save_file_to_memory()` for each file found.

- **Important:** Make sure you have a file named `example.txt` (or adjust the path in the `main()` function) in your `data/` directory for the example to work correctly. You can put any text content in it.

**Step 7: Add New MCP Plugins (Future Scalability)**

- **General Approach:**

  1. **Install the Plugin:** Install the Node.js package for the desired MCP plugin (e.g., a web scraper plugin, a database connector, etc.).
  2. **Configure the Plugin:** Create a configuration file (similar to `server-filesystem`) for the new plugin in your `mcp/` directory.
  3. **Update `MCPService`:**
     - Modify the `initialize()` method of `MCPService` to load the new plugin, similar to how the File System plugin is loaded. You might need to create a new `ClientSession` for each plugin if they need to run as separate processes.
     - Add new methods to `MCPService` to interact with the tools provided by the new plugin.
  4. **Integrate with LangChain:**
     - If the new plugin provides data that can be stored in Chroma, update the `LangChainService` to add methods to interact with the new tools, retrieve the data, and store it in Chroma as needed.
     - Update your API endpoints (which we'll create in the next step) to expose the new functionalities.

- **Example: Adding a Hypothetical Web Scraping Plugin:**

  1. **Install:**

     ```bash
     mkdir -p ./mcp/server-webscraper
     cd ./mcp/server-webscraper
     npm init -y
     npm install @modelcontextprotocol/server-webscraper # Hypothetical package name
     ```

  2. **Configure (mcp/server-webscraper/index.js):**

     ```javascript
     // Hypothetical configuration for a web scraper plugin
     const { startServer } = require("@modelcontextprotocol/server-webscraper");

     startServer({
       port: 3001, // Use a different port if necessary
       // ... other configuration options for the web scraper ...
     });
     ```

  3. **Update `MCPService`:**

     ```python
     # ... other imports ...

     class MCPService:
         # ... existing code ...

         async def initialize(self):
             """Initializes the MCP toolkits and establishes sessions."""
             try:
                 # ... existing File System plugin initialization ...

                 # Initialize Web Scraper plugin
                 webscraper_server_params = StdioServerParameters(
                     command="node",
                     args=[str(pathlib.Path(config.MCP_SERVER_PATH).parent / "server-webscraper" / "index.js")],
                 )
                 logger.debug(f"MCP web scraper server parameters: {webscraper_server_params}")

                 async with stdio_client(webscraper_server_params) as (ws_read, ws_write):
                     self.webscraper_session = ClientSession(ws_read, ws_write)
                     async with self.webscraper_session as ws_session:
                         self.webscraper_toolkit = MCPToolkit(session=ws_session)
                         await self.webscraper_toolkit.ainitialize()
                         logger.info("MCP Web Scraper Service initialized successfully.")

                 # ... rest of initialization ...

             except Exception as e:
                 logger.error(f"Failed to initialize MCP Service: {e}")
                 raise

         async def get_webscraper_tools(self):
             """Retrieves the tools from the MCP web scraper toolkit."""
             if not self.webscraper_toolkit:
                 await self.initialize()
             return self.webscraper_toolkit.get_tools()

         # ... other methods ...

         async def close_session(self):
             """Closes all MCP sessions."""
             # ... existing session closing code ...
             if self.webscraper_session:
                 await self.webscraper_session.close()
                 self.webscraper_session = None
                 self.webscraper_toolkit = None
                 logger.info("MCP web scraper session closed.")

     # ... other code ...
     ```

  4. **Integrate with LangChain (update `LangChainService`):**

     ```python
     # ... other imports ...

     class LangChainService:
         # ... existing code ...

         async def scrape_website_and_save(self, url: str) -> None:
             """Scrapes a website using the MCP web scraper plugin and saves the content to Chroma."""
             tools = await self.mcp_service.get_webscraper_tools()
             scrape_tool = next((tool for tool in tools if "scrape" in tool.name), None) # Assuming the tool is named something like "scrape_website"

             if scrape_tool:
                 try:
                     scraped_content = await scrape_tool.ainvoke({"url": url})
                     metadata = {"source": "web", "url": url, "date_scraped": datetime.now().isoformat()}
                     self.chroma_service.add_memory(scraped_content, metadata)
                     logger.info(f"Scraped website content saved to memory: {url}")
                 except Exception as e:
                     logger.error(f"Failed to scrape and save website content: {e}")
             else:
                 logger.error("Web scraping tool not found.")

     # ... other methods ...
     ```

**Step 8: Create API Endpoints**

Now, we'll create API endpoints to expose the functionality of our services to the frontend or other clients.

- **Update `app/routers/chat.py`:**

```python
"""Chat router."""
# ... other imports ...
from app.dependencies.providers import get_agent, get_model_service, get_function_service, get_chroma_service, get_langchain_service, get_mcp_service
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

- **Explanation:**
  - `add_to_memory()`: Adds a text snippet and optional metadata directly to Chroma.
  - `query_memory()`: Uses the `LangChainService` to query Chroma (semantically) and get a response from the LLM.
  - `save_file_to_memory()`: Uses `LangChainService` to read a file (via MCP) and save its content to Chroma.
  - `query_files_in_memory()`: Uses `LangChainService` to find file paths in Chroma that are relevant to a query.
  - `process_directory()`: Uses `LangChainService` to process a directory (via MCP) and save file contents to Chroma.

**Step 9: Testing**

- **Unit Tests:** Write unit tests for each of the new service methods (`ChromaService`, `MCPService`, `LangChainService`) to ensure they work in isolation. You can use `pytest` and `unittest.mock` to mock dependencies.
- **Integration Tests:** Write integration tests to test the interactions between the services, especially the flow from API endpoints to `LangChainService` to `ChromaService` and `MCPService`.
- **End-to-End Tests:** Use a tool like `httpx` or `requests` to test the API endpoints, simulating user interactions.

**Example Unit Test (using `pytest`):**

- File: `tests/services/test_chroma_service.py`

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

**Example Integration Test (using `pytest` and `httpx`):**

- File: `tests/api/test_chat_integration.py` (You can create a new file or add to your existing `test_chat.py`)

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

**Example End-to-End Test (using `httpx`):**

- File: `tests/api/test_chat_e2e.py`

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

**Step 10: Documentation**

- **README.md:** Update your project's `README.md` with instructions on how to set up and use the new features (MCP, Chroma, LangChain integration, new API endpoints).
- **Code Comments:** Add clear and concise comments to the new code, explaining the purpose of each class, method, and significant code block.
- **API Documentation:** If you're using a tool like Swagger or OpenAPI, update your API documentation to include the new endpoints.

**Deployment Considerations**

- **Dependencies:** Ensure your deployment environment has all the necessary packages installed (from `requirements.txt`).
- **Node.js:** You'll need Node.js and npm installed on your server to run the MCP plugins.
- **Environment Variables:** Set the environment variables (Chroma, MCP paths) in your deployment configuration.
- **Persistence:** Make sure the `CHROMA_PERSIST_DIRECTORY` is a persistent volume so that your Chroma data is not lost when the application restarts.
- **Scalability:** For larger deployments, consider using a more robust Chroma setup (e.g., a Chroma server instead of the DuckDB backend) and potentially a dedicated server for the MCP plugins.

**Error Handling and Logging**

- Throughout the code, I've included `try-except` blocks to handle potential errors gracefully.
- Use the `logger` to log important events, warnings, and errors at appropriate levels (DEBUG, INFO, WARNING, ERROR).
- In the API endpoints, return informative error messages to the client in case of failures.

**Future Enhancements**

- **More MCP Plugins:** Integrate other MCP plugins (e.g., for web search, database access, etc.) to expand the application's capabilities.
- **Advanced LangChain Features:** Explore more advanced LangChain features like agents, different chain types, and custom prompts to create more sophisticated interactions.
- **Caching:** Implement caching mechanisms to improve performance, especially for repeated queries to Chroma or the LLM.
- **Security:** If you're handling sensitive data, implement appropriate security measures (authentication, authorization, data encryption).
- **User Interface:** Develop a user-friendly interface to interact with the new features.

This detailed guide should provide a solid foundation for integrating MCP, LangChain, and Chroma into your application. Remember to adapt the code and configurations to your specific needs and environment. If you have any more questions, feel free to ask!
