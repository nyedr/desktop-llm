# Development Plan: LLM Request Context Manager

## Goal

Create a context manager in a new file that enhances LLM requests by injecting semantically relevant memories from ChromaDB, specifically designed for managing conversation history and user-provided memories (no file storage).

## Key Requirements

1. **Context Manager (`LLMContextManager`)**:
   - Handles setup and teardown of resources for LLM requests.
   - Retrieves relevant memories from Chroma based on the current conversation.
   - Makes relevant memories available to the LLM request.
   - **No file storage** or retrieval from the file system will be handled by this manager or Chroma.
2. **Chroma Integration**:
   - Utilize existing `retrieve_memories` and `retrieve_with_metadata` in `ChromaService`.
   - Ensure Chroma is used solely for conversation history and user-added memories.
3. **LangChain Integration**:
   - LangChain should be used to assist with structuring the memory retrieval process. It won't be used for file storage or file-related operations.
4. **Error Handling**:
   - Gracefully handle cases where no relevant memories are found.
   - Log errors appropriately and provide informative error messages.

## Implementation Steps

### Step 1: Create the LLM Context Manager

**File:** `app/context/llm_context.py`

```python
from typing import Dict, Any, List
import logging
from app.services.chroma_service import ChromaService
from app.services.langchain_service import LangChainService

logger = logging.getLogger(__name__)

class LLMContextManager:
    """Context manager for handling LLM requests with memory retrieval."""

    def __init__(
        self,
        chroma_service: ChromaService,
        langchain_service: LangChainService,
        conversation_history: List[Dict[str, Any]],
        metadata_filter: Dict[str, Any] = None,
        top_k: int = 5
    ):
        self.chroma_service = chroma_service
        self.langchain_service = langchain_service
        self.conversation_history = conversation_history
        self.metadata_filter = metadata_filter
        self.top_k = top_k
        self.context_data = None

    async def __aenter__(self):
        """Retrieve relevant memories and prepare context data."""
        try:
            # Combine conversation history into a single query
            query = " ".join([msg["content"] for msg in self.conversation_history])
            if not query:
                logger.info("No conversation history provided.")
                self.context_data = []
                return self

            logger.info(f"Retrieving memories for query: {query}")

            if self.metadata_filter:
                memories = await self.chroma_service.retrieve_with_metadata(
                    query, self.metadata_filter, self.top_k
                )
            else:
                memories = await self.chroma_service.retrieve_memories(query, self.top_k)

            if memories:
                self.context_data = [
                    {
                        "content": memory["document"],
                        "metadata": memory["metadata"]
                    }
                    for memory in memories
                ]
            else:
                logger.info("No relevant memories found.")
                self.context_data = []

            return self

        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            self.context_data = []
            return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources if needed."""
        if exc_type:
            logger.error(f"An error occurred within the LLM context: {exc_type.__name__}: {exc_val}")
        # No specific cleanup needed here, but could add resource management if necessary
        pass

    def get_context_data(self):
        """Get the retrieved context data."""
        return self.context_data
```

### Step 2: Modify `app/dependencies/providers.py`

- Add a method to get the LangChain service if it hasn't already been added. This will be used for any functions that might require LangChain's functionality.

```python
@classmethod
def get_langchain_service(cls) -> LangChainService:
    """Get or create LangChain service instance."""
    if cls._langchain_service is None:
        cls._langchain_service = LangChainService()
    return cls._langchain_service
```

### Step 3: Update `app/services/langchain_service.py`

- Modify the LangChain service to focus on conversation and memory management:
- Remove the file related functionality.

```python
import logging
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

from app.core.config import config
from app.services.chroma_service import ChromaService
from app.services.mcp_service import MCPService

logger = logging.getLogger(__name__)

class LangChainService:
    """Service for integrating LangChain with Chroma and MCP."""

    def __init__(self):
        """Initialize the LangChain service."""
        self.chroma_service = None
        self.mcp_service = None
        self.retriever = None
        self.llm = None
        self.embeddings = None

    async def initialize(self, chroma_service: ChromaService, mcp_service: MCPService):
        """Initialize the service with required dependencies."""
        try:
            logger.info("Initializing LangChain Service...")
            self.chroma_service = chroma_service
            self.mcp_service = mcp_service

            # Initialize HuggingFace embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.CHROMA_EMBEDDING_MODEL
            )

            # Initialize Ollama LLM
            self.llm = Ollama(
                base_url=config.OLLAMA_BASE_URLS[0],
                model=config.DEFAULT_MODEL,
                temperature=config.MODEL_TEMPERATURE
            )

            # Initialize retriever
            self.retriever = Chroma(
                client=self.chroma_service.client,
                collection_name=config.CHROMA_COLLECTION_NAME,
                embedding_function=self.embeddings
            ).as_retriever()

            logger.info("LangChain Service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize LangChain Service: {e}")
            raise

    async def query_memory(self, query: str, **kwargs) -> Dict[str, Any]:
        """Query the memory store with a question."""
        try:
            if not self.retriever or not self.llm:
                raise ValueError("LangChain Service not initialized")

            # Use the context manager to retrieve memories
            async with LLMContextManager(self.chroma_service, self, [{"role": "user", "content": query}]) as context_manager:
                retrieved_docs = context_manager.get_context_data()

            if retrieved_docs is None:
                logger.info("No documents retrieved for query.")
                return {"result": "", "source_documents": []}

            # Format the retrieved documents into a string to be added to the prompt
            formatted_context = "\n".join(
                [f"Content: {doc['content']}\nSource: {doc['metadata']['source']}" for doc in retrieved_docs])

            # Create a prompt template
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Answer the user's question based on the following context:\n{context}"),
                ("user", "{question}")
            ])

            # Chain the prompt, llm, and output parser
            chain = prompt_template | self.llm | StrOutputParser()

            # Invoke the chain with the query and context
            response = await chain.ainvoke({"question": query, "context": formatted_context})

            return {
                "result": response,
                "source_documents": [Document(page_content=doc['content'], metadata=doc['metadata']) for doc in retrieved_docs]
            }

        except Exception as e:
            logger.error(f"Failed to query memory: {e}")
            raise
```

### Step 4: Modify the Agent to Use the Context Manager

**File:** `app/services/agent.py`

```python
# ... other imports ...
from app.context.llm_context import LLMContextManager
from app.dependencies.providers import Providers
from app.core.config import config
# ... rest of the class ...

    async def chat(
        self,
        messages: List[Union[Dict[str, Any], ChatMessage]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = True,
        tools: Optional[List[Dict[str, Any]]] = None,
        enable_tools: bool = False
    ) -> AsyncGenerator[str, None]:
        """Generate chat completions with memory context."""
        try:
            if enable_tools and tools:
                logger.debug(
                    f"Tools enabled for chat. Available tools: {[t['function']['name'] for t in tools]}")
            else:
                logger.debug("No tools enabled for chat")

            # Convert messages to list if needed
            if not isinstance(messages, list):
                messages = [messages]

            # Get the LangChain service from the providers
            langchain_service = Providers.get_langchain_service()

            # Use the LLMContextManager to retrieve memories
            async with LLMContextManager(
                chroma_service=self.chroma_service,
                langchain_service=langchain_service,
                conversation_history=messages
            ) as context_manager:
                context_data = context_manager.get_context_data()

                # Append the context data to the messages
                if context_data:
                    context_messages = [
                        {"role": "system", "content": f"Context: {item['content']}"}
                        for item in context_data
                    ]
                    messages = context_messages + messages

                # Get response from model service
                async for response in self.model_service.chat(
                    messages=messages,
                    model=model or self.model,
                    temperature=temperature or self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                    stream=stream,
                    tools=tools,
                    enable_tools=enable_tools,
                    function_service=self.function_service
                ):
                    if response:
                        yield response

        except Exception as e:
            logger.error(f"Error in chat: {e}", exc_info=True)
            raise
```

### Step 5: Update the `chat` route to Utilize the Context Manager:

**File:** `app/routers/chat.py`

```python
# ... other imports ...
from app.context.llm_context import LLMContextManager
# ... other code ...

async def stream_chat_response(
    request: Request,
    chat_request: ChatRequest,
    agent: Agent = Depends(get_agent),
    model_service: ModelService = Depends(get_model_service),
    function_service: FunctionService = Depends(get_function_service),
    chroma_service: ChromaService = Depends(get_chroma_service),
    langchain_service: LangChainService = Depends(get_langchain_service),
    is_test: bool = False
) -> AsyncGenerator[dict, None]:
    """Generate streaming chat response."""
    request_id = str(id(request))
    logger.info(f"[{request_id}] Starting chat stream")
    tool_call_in_progress = False

    try:
        # ... other code ...

        # Apply inlet filters in priority order
        data = {"messages": chat_request.messages}
        sorted_filters = sorted(filters, key=lambda f: f.priority or 0)
        for filter_func in sorted_filters:
            logger.debug(
                f"[{request_id}] Applying inlet filter: {filter_func.name} (priority: {filter_func.priority})")
            try:
                data = await filter_func.inlet(data)
                logger.debug(
                    f"[{request_id}] Filter {filter_func.name} inlet result: {data}")
            except Exception as e:
                logger.error(
                    f"[{request_id}] Error in filter {filter_func.name} inlet: {e}")
        chat_request.messages = data["messages"]

        # Use LLMContextManager to retrieve relevant memories
        async with LLMContextManager(
            chroma_service=chroma_service,
            langchain_service=langchain_service,
            conversation_history=chat_request.messages,
            metadata_filter=None  # Add metadata filter if needed
        ) as context_manager:
            context_data = context_manager.get_context_data()

            # Prepare messages with context
            messages_with_context = chat_request.messages.copy()
            if context_data:
                formatted_context = "\n".join(
                    [f"Relevant memory: {item['content']}" for item in context_data]
                )
                messages_with_context.insert(0, ChatMessage(role="system", content=f"Context: {formatted_context}"))

            # Apply pipeline
            if pipeline:
                logger.debug(f"[{request_id}] Applying pipeline: {pipeline.name}")
                try:
                    pipeline_data = await pipeline.pipe({"messages": messages_with_context})
                    logger.debug(f"[{request_id}] Pipeline result: {pipeline_data}")
                    if "messages" in pipeline_data:
                        messages_with_context = pipeline_data["messages"]
                    if "summary" in pipeline_data:
                        yield {
                            "event": "pipeline",
                            "data": json.dumps({"summary": pipeline_data["summary"]})
                        }

                        if isinstance(pipeline_data["summary"], dict):
                            for key, value in pipeline_data["summary"].items():
                                if isinstance(value, list):
                                    for item in value:
                                        yield {
                                            "event": "pipeline",
                                            "data": json.dumps({"type": key, "content": item})
                                        }
                                        await asyncio.sleep(0.1)
                except Exception as e:
                    logger.error(f"[{request_id}] Error in pipeline execution: {e}")

            # Stream chat completion with context
            async for response in agent.chat(
                messages=messages_with_context,
                model=chat_request.model or config.DEFAULT_MODEL,
                temperature=chat_request.temperature or config.MODEL_TEMPERATURE,
                max_tokens=chat_request.max_tokens or config.MAX_TOKENS,
                stream=True,
                tools=function_schemas,
                enable_tools=chat_request.enable_tools
            ):
                # ... rest of the streaming logic ...

    except Exception as e:
        logger.error(f"[{request_id}] Error in chat stream: {e}", exc_info=True)
        yield {"event": "error", "data": json.dumps({"error": str(e)})}
```

### Step 6: Update Tests

- Update your test files (e.g., `tests/services/test_langchain_service.py`, `tests/api/test_chat.py`) to use the new context manager. You'll need to mock `LLMContextManager` where necessary.
- Add tests specifically for the context manager's functionality.

### Step 7: Update `app/main.py` to Initialize `LangChainService`

- Ensure that `LangChainService` is initialized properly in the lifespan context manager:

```python
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for FastAPI application."""
    service_states = {
        'mcp': ServiceState(),
        'chroma': ServiceState(),
        'langchain': ServiceState()
    }
    app.state.service_states = service_states

    try:
        logger.info("Starting service initialization...")

        mcp_success = await initialize_service(
            "MCP service",
            Providers.get_mcp_service,
            service_states['mcp']
        )

        chroma_success = await initialize_service(
            "Chroma service",
            Providers.get_chroma_service,
            service_states['chroma']
        )

        # Initialize LangChain service if dependencies are available
        if chroma_success:
            langchain_state = service_states['langchain']
            try:
                langchain_state.set_status(ServiceStatus.INITIALIZING)
                langchain_service = Providers.get_langchain_service()
                langchain_state.service = langchain_service

                # Only use MCP service if it's available
                mcp_service = service_states['mcp'].service if mcp_success else None

                await langchain_service.initialize(
                    service_states['chroma'].service,
                    mcp_service
                )
                langchain_state.set_status(ServiceStatus.READY)
                logger.info(
                    f"LangChain service initialized successfully {ServiceStatus.READY.value}")
            except Exception as e:
                langchain_state.set_status(ServiceStatus.FAILED, e)
                logger.error(
                    f"LangChain service initialization failed {ServiceStatus.FAILED.value}: {e}", exc_info=True)
```
