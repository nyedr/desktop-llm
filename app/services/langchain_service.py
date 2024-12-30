"""Service for integrating LangChain with Chroma and memory operations."""
import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING, Union
from datetime import datetime
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain

from app.core.config import config
from app.memory.lightrag.manager import EnhancedLightRAGManager
from app.services.mcp_service import MCPService
from app.models.memory import MemoryType
from app.context.llm_context import LLMContextManager

# Forward references for type checking
if TYPE_CHECKING:
    from app.models.chat import StrictChatMessage

logger = logging.getLogger(__name__)


class LangChainService:
    """Service for integrating LangChain with Chroma and memory operations."""

    def __init__(self):
        """Initialize the LangChain service."""
        logger.info("Creating new LangChainService instance")
        self.manager = None
        self.mcp_service = None
        self.retriever = None
        self.llm = None
        self.embeddings = None
        self.summarize_chain = None
        self._initialized = False

    async def initialize(self, manager: EnhancedLightRAGManager, mcp_service: MCPService):
        """Initialize the service with required dependencies.

        This method sets up all necessary components for LangChain integration:
        1. Embeddings model for vector operations
        2. LLM for text generation and summarization
        3. Retriever for semantic search
        4. Summarization chain for context management

        Args:
            manager: EnhancedLightRAGManager for memory operations
            mcp_service: Service for model context protocol operations

        Raises:
            RuntimeError: If initialization fails or service is already initialized
        """
        try:
            logger.info("Starting LangChainService initialization...")
            if self._initialized:
                logger.info("LangChainService already initialized")
                return

            logger.info("Initializing core services...")
            await self._initialize_services(manager, mcp_service)

            logger.info("Initializing embeddings...")
            await self._initialize_embeddings()

            logger.info("Initializing LLM...")
            await self._initialize_llm()

            logger.info("Initializing retriever...")
            await self._initialize_retriever()

            logger.info("Initializing chains...")
            await self._initialize_chains()

            self._initialized = True
            logger.info(
                "LangChainService initialization completed successfully")
        except Exception as e:
            logger.error(
                f"Failed to initialize LangChainService: {str(e)}", exc_info=True)
            raise

    async def _initialize_services(self, manager: EnhancedLightRAGManager, mcp_service: MCPService):
        """Initialize service dependencies.

        Args:
            manager: EnhancedLightRAGManager for memory operations
            mcp_service: Service for model context protocol operations
        """
        try:
            logger.debug("Setting up manager and MCP service...")
            self.manager = manager
            self.mcp_service = mcp_service
            logger.debug("Core services initialized successfully")
        except Exception as e:
            logger.error(
                f"Failed to initialize core services: {str(e)}", exc_info=True)
            raise

    async def _initialize_embeddings(self):
        """Initialize embeddings model."""
        try:
            logger.debug("Setting up HuggingFace embeddings...")
            self.embeddings = HuggingFaceEmbeddings()
            logger.debug("Embeddings initialized successfully")
        except Exception as e:
            logger.error(
                f"Failed to initialize embeddings: {str(e)}", exc_info=True)
            raise

    async def _initialize_llm(self):
        """Initialize language model."""
        try:
            logger.debug("Setting up Ollama LLM...")
            self.llm = Ollama(
                base_url=config.OLLAMA_BASE_URLS[0],
                model=config.DEFAULT_MODEL,
                temperature=config.MODEL_TEMPERATURE
            )
            logger.debug(f"LLM initialized with model: {config.DEFAULT_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}", exc_info=True)
            raise

    async def _initialize_retriever(self):
        """Initialize retriever for semantic search."""
        try:
            logger.debug("Setting up Chroma retriever...")
            self.retriever = Chroma(
                embedding_function=self.embeddings).as_retriever()
            logger.debug("Retriever initialized successfully")
        except Exception as e:
            logger.error(
                f"Failed to initialize retriever: {str(e)}", exc_info=True)
            raise

    async def _initialize_chains(self):
        """Initialize LangChain chains."""
        try:
            logger.debug("Setting up summarization chain...")
            self.summarize_chain = load_summarize_chain(
                self.llm, chain_type="stuff")
            logger.debug("Chains initialized successfully")
        except Exception as e:
            logger.error(
                f"Failed to initialize chains: {str(e)}", exc_info=True)
            raise

    async def cleanup(self):
        """Cleanup resources."""
        try:
            if not self._initialized:
                return

            # Clean up any resources that need to be released
            self.manager = None
            self.mcp_service = None
            self.retriever = None
            self.llm = None
            self.embeddings = None
            self.summarize_chain = None
            self._initialized = False
            logger.info("LangChain Service cleaned up successfully")

        except Exception as e:
            logger.error(
                f"Error during LangChain Service cleanup: {e}", exc_info=True)
            raise

    def _get_message_value(self, message: Union[Dict[str, Any], 'StrictChatMessage'], key: str, default: Any = None) -> Any:
        """Safely get a value from either a dict or StrictChatMessage object."""
        if isinstance(message, dict):
            return message.get(key, default)
        return getattr(message, key, default)

    async def query_memory(
        self,
        query: str,
        context: Optional[List[Union[Dict[str, Any],
                                     'StrictChatMessage']]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Query the memory store with a question.

        Args:
            query: The question to ask
            context: Optional context messages to include
            **kwargs: Additional arguments for the query

        Returns:
            Dictionary containing the response and source documents
        """
        try:
            if not self._initialized:
                raise RuntimeError("LangChain Service not initialized")

            # Use the context manager to retrieve memories if no context provided
            if not context:
                async with LLMContextManager(
                    self.manager,
                    self,
                    [{"role": "user", "content": query}],
                    metadata_filter=kwargs.get("metadata_filter"),
                    top_k=kwargs.get("top_k", 5)
                ) as context_manager:
                    context = context_manager.get_context_messages()

            # Format the context into a string
            formatted_context = "\n".join([
                f"{self._get_message_value(msg, 'role', 'unknown')}: {self._get_message_value(msg, 'content', '')}"
                for msg in (context or [])
            ])

            # Create a prompt template with systematic structure
            prompt_template = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a helpful assistant. Use the following context to answer the question:\n\nContext:\n{context}"),
                ("user",
                 "Question: {question}\n\nProvide a clear and concise answer based on the context above.")
            ])

            # Chain the prompt, llm, and output parser
            chain = prompt_template | self.llm | StrOutputParser()

            # Invoke the chain with query and context
            response = await chain.ainvoke({
                "question": query,
                "context": formatted_context
            })

            # Extract source documents from context
            source_documents = []
            for msg in (context or []):
                role = self._get_message_value(msg, "role")
                content = self._get_message_value(msg, "content", "")
                if role == "system" and "Memory from" in content:
                    source_documents.append(
                        Document(
                            page_content=content,
                            metadata={"type": "memory"}
                        )
                    )

            return {
                "result": response,
                "source_documents": source_documents
            }

        except Exception as e:
            logger.error(f"Failed to query memory: {e}", exc_info=True)
            raise

    async def process_conversation(
        self,
        messages: List[Union[Dict[str, Any], 'StrictChatMessage']],
        memory_type: MemoryType = MemoryType.EPHEMERAL,
        conversation_id: Optional[str] = None,
        store_messages: bool = True,
        enable_summarization: bool = False,
        metadata_filter: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Process conversation history and retrieve relevant memories.

        Args:
            messages: List of conversation messages
            memory_type: Type of memory to use (ephemeral or model)
            conversation_id: Optional conversation ID for this conversation
            store_messages: Whether to store individual messages in memory (always ephemeral)
            enable_summarization: Whether to enable automatic summarization on context overflow
            metadata_filter: Optional filter for memory retrieval
            top_k: Number of memories to retrieve

        Returns:
            List of messages with context prepended
        """
        try:
            if not self._initialized:
                raise RuntimeError("LangChain Service not initialized")

            # Store messages in ephemeral memory if enabled
            if store_messages:
                for message in messages:
                    content = self._get_message_value(message, "content", "")
                    role = self._get_message_value(message, "role", "unknown")
                    name = self._get_message_value(message, "name")

                    # Always store messages in ephemeral collection
                    await self.manager.add_memory(
                        text=content,
                        collection=MemoryType.EPHEMERAL,  # Force ephemeral for messages
                        metadata={
                            "role": role,
                            "name": name,
                            "type": "message",
                            "conversation_id": conversation_id,
                            "timestamp": datetime.now().isoformat()
                        }
                    )

            # Process with context manager
            async with LLMContextManager(
                self.manager,
                self,
                messages,
                memory_type=memory_type,  # This is for retrieving memories
                conversation_id=conversation_id,
                # Controls summarization on context overflow
                enable_summarization=enable_summarization,
                metadata_filter=metadata_filter,
                top_k=top_k
            ) as context_manager:
                return context_manager.get_context_messages()

        except Exception as e:
            logger.error(f"Failed to process conversation: {e}", exc_info=True)
            return messages

    async def summarize_text(self, text: str) -> Optional[str]:
        """Summarize a piece of text using LangChain's summarization chain.

        Args:
            text: Text to summarize

        Returns:
            Summarized text or None if summarization fails
        """
        try:
            if not self._initialized:
                raise RuntimeError("LangChain Service not initialized")

            # Split text into documents for map-reduce summarization
            docs = [Document(page_content=text)]
            summary = await self.summarize_chain.arun(docs)
            return summary.strip()

        except Exception as e:
            logger.error(f"Failed to summarize text: {e}", exc_info=True)
            return None

    async def store_conversation_summary(
        self,
        messages: List[Union[Dict[str, Any], 'StrictChatMessage']],
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Summarize and store a conversation in memory.

        Args:
            messages: List of conversation messages
            conversation_id: Optional conversation ID
            metadata: Optional metadata for the summary

        Returns:
            ID of the stored summary or None if operation fails
        """
        try:
            if not self._initialized:
                raise RuntimeError("LangChain Service not initialized")

            # Combine messages into a single text
            conversation_text = "\n".join([
                f"{self._get_message_value(msg, 'role', 'unknown')}: {self._get_message_value(msg, 'content', '')}"
                for msg in messages
            ])

            # Generate summary
            summary = await self.summarize_text(conversation_text)
            if not summary:
                return None

            # Add metadata
            full_metadata = {
                "type": "conversation_summary",
                "timestamp": datetime.now().isoformat(),
                "message_count": len(messages),
                "conversation_id": conversation_id,
                **(metadata or {})
            }

            # Store in ephemeral collection
            summary_id = await self.manager.add_memory(
                text=summary,
                collection=MemoryType.EPHEMERAL,  # Use enum instead of string
                metadata=full_metadata
            )
            logger.info(f"Stored conversation summary with ID: {summary_id}")
            return summary_id

        except Exception as e:
            logger.error(
                f"Failed to store conversation summary: {e}", exc_info=True)
            return None
