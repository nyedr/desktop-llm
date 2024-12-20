# MCP, LangChain, and Chroma Integration ToDo

This ToDo list details the steps required to implement the Model Context Protocol (MCP) integration, LangChain orchestration, and Chroma vector database for advanced semantic search and persistent memory in the existing application.

## Project Goal

Augment the existing application with:

- **File System Interaction:** Enable reading and searching files using MCP.
- **Persistent Memory (Chroma):** Store and retrieve embeddings of text content for semantic search.
- **LangChain Integration:** Combine MCP and Chroma with LangChain for advanced retrieval and LLM-based question answering.

## Core Components

- **MCP (Model Context Protocol):** Standardized AI-agent-to-tool/plugins protocol.
- **LangChain:** Framework for building LLM-powered applications.
- **Chroma:** Vector database for storing embeddings and supporting semantic search.

---

## Steps Overview

1. **Set Up Environment**
2. **Initialize MCP and File System Plugin**
3. **Set Up Chroma for Persistent Memory**
4. **Integrate Chroma into LangChain**
5. **Enhance Chroma Service with Advanced Semantic Search**
6. **Link File System and Persistent Memory using MCP & LangChain**
7. **Add (Optional) Additional MCP Plugins for Future Scalability**
8. **Create and Update API Endpoints**
9. **Testing and Validation**
10. **Documentation and Deployment Considerations**

---

## Detailed Tasks

### Step 1: Set Up Environment

- [ ] **Install Required Python Packages:**  
       Run:
  ```bash
  pip install langchain langchain-community chromadb sentence-transformers
  ```

This installs:

- `langchain`
- `langchain-community` (for MCP integration)
- `chromadb`
- `sentence-transformers` for embeddings

- [ ] **Ensure Directory Structure** matches:

  ```
  desktop-llm/
  ├── app/
  │   ├── core/
  │   ├── dependencies/
  │   ├── functions/
  │   ├── models/
  │   ├── routers/
  │   ├── services/
  │   └── ...
  ├── data/
  │   └── example.txt
  ├── mcp/
  ├── .env
  ├── config.py
  ├── ...
  ```

- [ ] **Add Environment Variables** to `.env`:

  ```
  CHROMA_PERSIST_DIRECTORY=chroma_data
  CHROMA_COLLECTION_NAME=desktop_llm_memory
  MCP_SERVER_PATH=./mcp/server-filesystem
  ```

- [ ] **Verify MCP Server Plugins:**
  - Install Node.js and npm.
  - In `./mcp/server-filesystem`:
    ```bash
    npm init -y
    npm install @modelcontextprotocol/server-filesystem
    ```
  - Test:
    ```bash
    npx -y @modelcontextprotocol/server-filesystem ./data
    ```
    Confirm server starts up, then `Ctrl+C` to stop.

### Step 2: Initialize MCP and File System Plugin

- [ ] **Create `MCPService` in `app/services/mcp_service.py`:**
      Implement `MCPService` class with:

  - `initialize()` to start MCP File System plugin via `stdio_client`.
  - `get_tools()` to retrieve MCP tools.
  - `close_session()` to shut down MCP session.

- [ ] **Integrate `MCPService` with Providers:**
  - In `app/dependencies/providers.py`, add a static method to get `MCPService` instance.
  - Update `app/main.py` `lifespan` function to initialize `MCPService` on startup and close on shutdown.

### Step 3: Set Up Chroma for Persistent Memory

- [ ] **Create `ChromaService` in `app/services/chroma_service.py`:**
      Implement:

  - `__init__()` to set up `Chroma` client and collection.
  - `add_memory(text, metadata)` to add a single memory.
  - `retrieve_memories(query, top_k, where_filters)` for querying embeddings.
  - `clear_collection()` and `list_collections()` for maintenance tasks.

- [ ] **Integrate `ChromaService` with Providers:**
  - In `app/dependencies/providers.py`, add `get_chroma_service()`.
  - In `app/main.py` `lifespan`, initialize `ChromaService`.

### Step 4: Integrate Chroma into LangChain

- [ ] **Create `LangChainService` in `app/services/langchain_service.py`:**

  - Initialize `Chroma` vector store with HuggingFace embeddings.
  - Create a retriever from Chroma store.
  - Set up a `RetrievalQA` chain with `ChatOpenAI`.
  - Implement `query_memory(query)` to run semantic queries against Chroma via LangChain.

- [ ] **Integrate `LangChainService` with Providers:**
  - In `app/dependencies/providers.py`, add `get_langchain_service()`.
  - Initialize `LangChainService` in `app/main.py` `lifespan`.

### Step 5: Advanced Semantic Search Enhancements

- [ ] **Enhance `ChromaService`:**
  - Add methods like `retrieve_with_metadata()`, `add_memories_batch()`, `update_memory()`, `delete_memory()`, `count_memories()`, and `get_memory_by_id()` for more granular control and flexibility.
  - Implement `update_embedding_model()` to switch embedding models if needed.

### Step 6: Link File System and Persistent Memory

- [ ] **Update `LangChainService` to use MCP Tools:**
  - Implement `save_file_to_memory(file_path)`:
    - Use MCP "read" tool to get file content.
    - Add the file content as a memory to Chroma with metadata (source=filesystem).
  - Implement `query_files_in_memory(query)`:
    - Retrieve memories from Chroma and return associated file paths.
  - Implement `process_directory(directory_path, file_extensions)`:
    - Use MCP "search" tool to find files in a directory.
    - For each file, `save_file_to_memory()`.

### Step 7: Add New MCP Plugins (Optional/Future)

- [ ] **Extend `MCPService` for Additional Plugins:**
  - For a hypothetical web scraper plugin, install it in `./mcp/server-webscraper`.
  - Initialize it similarly to the filesystem plugin.
  - Add `get_webscraper_tools()` and integrate them into `LangChainService` if needed.

### Step 8: Create/Update API Endpoints

- [ ] **Update `app/routers/chat.py`:**
  - Add endpoint `/chat/memory/add` to add arbitrary text to memory.
  - Add endpoint `/chat/memory/query` to query stored memories and get LLM answers.
  - Add endpoint `/chat/file/save` to store a specific file’s content in memory.
  - Add endpoint `/chat/files/query` to retrieve file paths from memory.
  - Add endpoint `/chat/directory/process` to index all files in a directory.

### Step 9: Testing and Validation

- [ ] **Write Unit Tests:**

  - For `ChromaService`, `MCPService`, `LangChainService` using `pytest`.
  - Mock external calls where needed.

- [ ] **Write Integration Tests:**

  - Test API endpoints to ensure end-to-end flows (e.g., add memory, then query it).

- [ ] **End-to-End Tests:**
  - Use `httpx` or `requests` to simulate realistic usage scenarios.

### Step 10: Documentation and Deployment Considerations

- [ ] **README.md:**

  - Document setup instructions, environment variable requirements, and usage patterns.

- [ ] **API Documentation:**

  - If using OpenAPI/Swagger, update with new endpoints.

- [ ] **Deployment:**
  - Ensure Node.js and npm are available where MCP plugins run.
  - Make sure `CHROMA_PERSIST_DIRECTORY` is persisted.
  - Scale as needed (consider a Chroma server, caching, etc.).

### Future Enhancements (Optional):

- [ ] **More MCP Plugins (web, DB, etc.)**
- [ ] **Advanced LangChain Features (agents, custom prompts)**
- [ ] **Caching and Performance Optimization**
- [ ] **Security (auth, encryption)**
- [ ] **UI/UX Improvements**

---

**By following the above steps in order, you will implement a robust system integrating MCP, LangChain, and Chroma into the existing application.**
