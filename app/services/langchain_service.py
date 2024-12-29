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
from app.services.chroma_service import ChromaService
from app.services.mcp_service import MCPService
from app.context.llm_context import LLMContextManager  # Import at runtime

# Forward references for type checking
if TYPE_CHECKING:
    from app.models.chat import StrictChatMessage

logger = logging.getLogger(__name__)


class LangChainService:
    """Service for integrating LangChain with Chroma and memory operations."""

    def __init__(self):
        """Initialize the LangChain service."""
        self.chroma_service = None
        self.mcp_service = None
        self.retriever = None
        self.llm = None
        self.embeddings = None
        self.summarize_chain = None
        self._initialized = False

    async def initialize(self, chroma_service: ChromaService, mcp_service: MCPService):
        """Initialize the service with required dependencies."""
        try:
            if self._initialized:
                logger.warning("LangChain Service already initialized")
                return

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

            # Initialize summarization chain
            self.summarize_chain = load_summarize_chain(
                llm=self.llm,
                chain_type="map_reduce",
                verbose=True
            )

            self._initialized = True
            logger.info("LangChain Service initialized successfully")

        except Exception as e:
            logger.error(
                f"Failed to initialize LangChain Service: {e}", exc_info=True)
            raise

    async def cleanup(self):
        """Cleanup resources."""
        try:
            if not self._initialized:
                return

            # Clean up any resources that need to be released
            self.chroma_service = None
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
                    self.chroma_service,
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
        metadata_filter: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Process conversation history and retrieve relevant memories.

        Args:
            messages: List of conversation messages
            metadata_filter: Optional filter for memory retrieval
            top_k: Number of memories to retrieve

        Returns:
            List of messages with context prepended
        """
        try:
            if not self._initialized:
                raise RuntimeError("LangChain Service not initialized")

            async with LLMContextManager(
                self.chroma_service,
                self,
                messages,
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
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Summarize and store a conversation in Chroma.

        Args:
            messages: List of conversation messages
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
                **(metadata or {})
            }

            # Check for duplicates before storing
            existing_memories = await self.chroma_service.retrieve_memories(
                summary,
                top_k=3,
                score_threshold=0.95
            )

            if existing_memories:
                logger.debug("Found similar existing memory, skipping storage")
                return None

            # Store in Chroma
            summary_id = await self.chroma_service.add_memory(
                summary,
                metadata=full_metadata
            )
            logger.info(f"Stored conversation summary with ID: {summary_id}")
            return summary_id

        except Exception as e:
            logger.error(
                f"Failed to store conversation summary: {e}", exc_info=True)
            return None
