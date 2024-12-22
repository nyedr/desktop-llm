# Desktop LLM Todos

This **todo list** outlines the tasks required to integrate **Model Context Protocol (MCP)**, **LangChain**, and **Chroma** into the `desktop-llm` application. Tasks are organized in **implementation order**, with detailed instructions and **code examples** drawn from the provided guide.

## 1. Environment Setup

### 1.1 Install Necessary Python Packages

- [ ] **Add required packages** to `requirements.txt` or `pyproject.toml`:
  ```bash
  pip install langchain langchain-community chromadb sentence-transformers
  ```
  - `langchain`: Core framework for building LLM applications.
  - `langchain-community`: Community integrations (including MCP support).
  - `chromadb`: Vector database for embeddings.
  - `sentence-transformers`: For generating sentence embeddings.

### 1.2 Install and Build MCP Server Plugins

- [ ] **Navigate** to the `mcp-server-filesystem` plugin directory (e.g., `/src/filesystem`) and run:
  ```bash
  npm install
  npm run build
  ```
  > **Note:** Repeat these steps for any other MCP plugins you want to add.

### 1.3 Validate Project Structure

- [ ] **Ensure** your project has a structure similar to:
  ```
  desktop-llm/
  ├── app/
  ├── data/
  ├── mcp/
  │   └── server-filesystem/
  │       └── dist/
  ├── .env
  ├── config.json
  ├── config.py
  ├── requirements.txt
  └── ...
  ```
  > Adjust paths as needed in the configuration files.

### 1.4 Set Environment Variables

- [ ] **Update** your `.env`:

  ```bash
  # Chroma
  CHROMA_PERSIST_DIRECTORY=chroma_data
  CHROMA_COLLECTION_NAME=desktop_llm_memory

  # MCP
  MCP_SERVER_FILESYSTEM_PATH=./mcp/server-filesystem/dist/index.js
  MCP_SERVER_FILESYSTEM_COMMAND=node
  ```

  - Points to the **compiled** filesystem plugin.
  - Directory for **Chroma** (`chroma_data`) must be persistent.

---

## 2. MCP Integration

### 2.1 Create MCP Service

- [ ] **Add** `MCPService` in `app/services/mcp_service.py`:

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
                  args=[
                      config.MCP_SERVER_FILESYSTEM_PATH,
                      str(pathlib.Path(config.WORKSPACE_DIR))
                  ],
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

- [ ] **Integrate** MCP in `app/dependencies/providers.py`:

  ```python
  # from app.services.mcp_service import MCPService

  class Providers:
      _mcp_service: Optional[MCPService] = None

      @classmethod
      def get_mcp_service(cls) -> MCPService:
          if cls._mcp_service is None:
              cls._mcp_service = MCPService()
          return cls._mcp_service

  def get_mcp_service(request: Request) -> MCPService:
      return Providers.get_mcp_service()
  ```

### 2.2 Initialize MCP During App Startup

- [ ] **Initialize** MCP in `app/main.py`:

  ```python
  # from app.dependencies.providers import Providers

  @asynccontextmanager
  async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
      try:
          # ... other inits ...
          mcp_service = Providers.get_mcp_service()
          await mcp_service.initialize()

          yield
      except Exception as e:
          logger.error(f"Error during startup: {e}")
          raise
      finally:
          if mcp_service:
              await mcp_service.close_session()
          logger.info("Application shutdown complete")
  ```

---

## 3. Chroma Integration

### 3.1 Create Chroma Service

- [ ] **Add** `ChromaService` in `app/services/chroma_service.py`:

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
          """Adds a memory to Chroma."""
          # Implementation...
  ```

- [ ] **Implement** additional methods (e.g., `retrieve_memories`, `update_memory`, `delete_memory`, etc.) exactly as in the guide:

  ```python
      def retrieve_memories(self, query: str, top_k: int = 5, where_filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
          """Retrieves memories from Chroma based on a query."""
          # Implementation...

      def update_memory(self, memory_id: str, new_text: Optional[str] = None, ...):
          # Implementation...

      # etc.
  ```

### 3.2 Integrate Chroma in `app/dependencies/providers.py`

- [ ] **Add** ChromaService to Providers:

  ```python
  # from app.services.chroma_service import ChromaService

  class Providers:
      _chroma_service: Optional[ChromaService] = None

      @classmethod
      def get_chroma_service(cls) -> ChromaService:
          if cls._chroma_service is None:
              cls._chroma_service = ChromaService()
          return cls._chroma_service
  ```

### 3.3 Initialize Chroma During App Startup

- [ ] **Initialize** Chroma in `app/main.py`:
  ```python
  @asynccontextmanager
  async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
      try:
          chroma_service = Providers.get_chroma_service()
          # ... other init ...
          yield
      except Exception as e:
          logger.error(f"Error during startup: {e}", exc_info=True)
          raise
      finally:
          logger.info("Application shutdown complete")
  ```

---

## 4. LangChain Integration

### 4.1 Create LangChain Service

- [ ] **Add** `LangChainService` in `app/services/langchain_service.py`:

  ```python
  import logging
  from typing import Dict, Any, List, Optional

  from langchain.chains import RetrievalQA
  from langchain.chat_models import ChatOpenAI
  from langchain.vectorstores import Chroma
  from langchain_community.embeddings import HuggingFaceEmbeddings

  from app.core.config import config
  from app.dependencies.providers import Providers
  from app.services.chroma_service import ChromaService
  from app.services.mcp_service import MCPService

  logger = logging.getLogger(__name__)

  class LangChainService:
      def __init__(self):
          self.chroma_service: ChromaService = Providers.get_chroma_service()
          self.mcp_service: MCPService = Providers.get_mcp_service()
          self.vectorstore = Chroma(
              collection_name=config.CHROMA_COLLECTION_NAME,
              embedding_function=HuggingFaceEmbeddings("multi-qa-mpnet-base-dot-v1"),
              client=self.chroma_service.client,
              persist_directory=config.CHROMA_PERSIST_DIRECTORY,
          )
          self.retriever = self.vectorstore.as_retriever()
          self.llm = ChatOpenAI(
              model=config.DEFAULT_MODEL,
              temperature=config.MODEL_TEMPERATURE,
              openai_api_base=config.OLLAMA_BASE_URLS[0],
              openai_api_key="EMPTY"  # Hack for LangChain usage
          )
          self.retrieval_qa = RetrievalQA.from_chain_type(
              llm=self.llm,
              chain_type="stuff",
              retriever=self.retriever,
              return_source_documents=True,
          )
          logger.info("LangChain service initialized.")

      async def query_memory(self, query: str) -> Dict[str, Any]:
          """Queries the Chroma database via LangChain."""
          # Implementation...
  ```

### 4.2 Link MCP File System and Chroma

- [ ] **Implement** methods in `LangChainService` that use MCP to read files and Chroma to store them:

  ```python
      async def save_file_to_memory(self, file_path: str) -> None:
          """Reads a file using MCP and saves its content to Chroma."""
          tools = await self.mcp_service.get_tools()
          file_read_tool = next((t for t in tools if "read_file" in t.name), None)
          if file_read_tool:
              try:
                  file_content = await file_read_tool.ainvoke({"path": file_path})
                  metadata = {"source": "file_system", "file_path": file_path}
                  self.chroma_service.add_memory(file_content, metadata)
              except Exception as e:
                  logger.error(f"Failed to save file: {e}")
  ```

  ```python
      async def query_files_in_memory(self, query: str, top_k: int = 5) -> List[str]:
          """Queries Chroma for file content matching a given query."""
          memories = self.chroma_service.retrieve_memories(query, top_k=top_k)
          return [
              m["metadata"].get("file_path")
              for m in memories
              if m["metadata"] and "file_path" in m["metadata"]
          ]
  ```

### 4.3 Initialize LangChain During App Startup

- [ ] **Initialize** LangChain in `app/main.py`:
  ```python
  @asynccontextmanager
  async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
      try:
          langchain_service = Providers.get_langchain_service()
          # ... other init ...
          yield
      except Exception as e:
          logger.error(f"Error during startup: {e}", exc_info=True)
          raise
      finally:
          logger.info("Application shutdown complete")
  ```

---

## 5. Advanced Semantic Search with Chroma

### 5.1 Update and Filter by Metadata

- [ ] **Confirm** `ChromaService` has:
  - `retrieve_with_metadata()`
  - `add_memories_batch()`
  - `update_memory()`
  - `delete_memory()`
  - `count_memories()`
  - `get_memory_by_id()`

### 5.2 Incremental File Indexing

- [ ] **Create** a method or reference in `LangChainService` to track which files have already been added to Chroma, so you can avoid re-processing or handle updates with `update_memory()`.

---

## 6. MCP Plugin Expansion

### 6.1 Add New Plugins

- [ ] **Install** new Node.js-based MCP plugins.
- [ ] **Configure** them similarly to the filesystem plugin in `MCPService.initialize()`.
- [ ] **Expose** plugin tools in `LangChainService` (if needed for file reading, web scraping, etc.).

---

## 7. API Endpoints

### 7.1 Add Memory Endpoints

- [ ] **Add** routes in `app/routers/chat.py`:

  ```python
  @router.post("/chat/memory/add")
  async def add_to_memory(...):
      chroma_service.add_memory(memory_text, metadata)
      return {"status": "success"}

  @router.post("/chat/memory/query")
  async def query_memory(...):
      response = await langchain_service.query_memory(query)
      return {"status": "success", "response": response}
  ```

### 7.2 File and Directory Endpoints

- [ ] **Add** endpoints to read and query files:

  ```python
  @router.post("/chat/file/save")
  async def save_file_to_memory(file_path: str, ...):
      await langchain_service.save_file_to_memory(file_path)
      return {"status": "success"}

  @router.get("/chat/files/query")
  async def query_files_in_memory(query: str, ...):
      file_paths = await langchain_service.query_files_in_memory(query)
      return {"status": "success", "file_paths": file_paths}
  ```

- [ ] **Add** directory processing:
  ```python
  @router.post("/chat/directory/process")
  async def process_directory(directory_path: str, file_extensions: Optional[List[str]] = None):
      await langchain_service.process_directory(directory_path, file_extensions)
      return {"status": "success"}
  ```

---

## 8. Testing

### 8.1 Unit Tests for ChromaService

- [ ] **Create** `tests/services/test_chroma_service.py`:

  ```python
  import pytest
  from app.services.chroma_service import ChromaService

  @pytest.mark.asyncio
  async def test_add_and_retrieve_memory(chroma_service):
      chroma_service.add_memory("Test memory", {"source": "test"})
      results = chroma_service.retrieve_memories("Test")
      assert len(results) == 1
  ```

### 8.2 Integration Tests for API Endpoints

- [ ] **Create** `tests/api/test_chat_integration.py`:

  ```python
  import pytest
  from httpx import AsyncClient
  from app.main import app

  @pytest.mark.asyncio
  async def test_add_query_memory_integration():
      async with AsyncClient(app=app, base_url="http://testserver") as client:
          add_resp = await client.post("/api/v1/chat/memory/add",
               json={"memory_text": "Project X is on track.", "metadata": {"project": "X"}})
          assert add_resp.status_code == 200
  ```

### 8.3 End-to-End Tests

- [ ] **Create** `tests/api/test_chat_e2e.py`:

  ```python
  import pytest
  from httpx import AsyncClient
  from app.main import app
  from app.dependencies.providers import Providers

  @pytest.mark.asyncio
  async def test_save_and_query_file_e2e(tmp_path):
      async with AsyncClient(app=app, base_url="http://testserver") as client:
          # Create a test file in tmp_path...
          # Save and query it...
  ```

---

## 9. Documentation

### 9.1 Update README

- [ ] **Explain** the new features (MCP, Chroma, LangChain) and usage steps.
- [ ] **Document** environment variables in `.env.example`.
- [ ] **Add** known limitations and references.

### 9.2 Code Comments

- [ ] **Ensure** each new class and method has docstrings explaining usage.

### 9.3 API Documentation

- [ ] **Confirm** new routes (`/chat/memory`, `/chat/file/save`, etc.) are visible in FastAPI docs (e.g., `/docs`).

---

## 10. Deployment

### 10.1 Persistent Storage

- [ ] **Ensure** `CHROMA_PERSIST_DIRECTORY` is a mounted volume or persistent location in production.

### 10.2 Node.js Runtime

- [ ] **Verify** Node.js environment exists to run MCP plugins.

### 10.3 Scaling Considerations

- [ ] **Consider** using a dedicated Chroma server for large-scale usage.
- [ ] **Potentially** run MCP plugins separately if needed (e.g., Docker containers).

---

## 11. Future Enhancements

### 11.1 Performance Optimization

- [ ] **Optimize** streaming performance for large file reads.
- [ ] **Consider** alternative embeddings or indexing strategies in Chroma.
- [ ] **Implement** caching or advanced indexing for repeated queries.

### 11.2 Additional MCP Plugins

- [ ] **Explore** web scraping or other data retrieval plugins.
- [ ] **Implement** real-time streaming from external APIs.

---

**Completed Steps** can be marked as `[x]` once finished. This **detailed todo** provides the roadmap for integrating **MCP**, **Chroma**, and **LangChain** into your `desktop-llm` application. Each section correlates with the code examples and best practices from the original implementation guide.
