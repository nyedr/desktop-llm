import logging
from typing import List, Dict, Any, Optional
from langchain.schema.retriever import BaseRetriever
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.llms import Ollama

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

    async def query_memory(self, query: str) -> Dict[str, Any]:
        """Query the memory store with a question."""
        try:
            if not self.retriever or not self.llm:
                raise ValueError("LangChain Service not initialized")

            docs = await self.retriever.aget_relevant_documents(query)
            response = await self.llm.ainvoke({
                "query": query,
                "docs": docs
            })
            return {
                "result": response["result"],
                "source_documents": docs
            }
        except Exception as e:
            logger.error(f"Failed to query memory: {e}")
            raise

    async def save_file_to_memory(self, file_path: str) -> None:
        """Read a file using MCP and save its content to Chroma."""
        try:
            if not self.mcp_service or not self.chroma_service:
                raise ValueError("Services not initialized")

            # Get file tools from MCP
            tools = await self.mcp_service.get_tools()
            read_tool = next(
                (tool for tool in tools if "read_file" in tool.name), None)

            if not read_tool:
                raise ValueError("Read file tool not found")

            # Read file content using MCP
            result = await read_tool.ainvoke({
                "relative_workspace_path": file_path,
                "should_read_entire_file": True,
                "start_line_one_indexed": 1,
                "end_line_one_indexed_inclusive": 999999  # Large number to read entire file
            })

            if not result or "content" not in result:
                raise ValueError(f"Failed to read file: {file_path}")

            # Save to Chroma with metadata
            await self.chroma_service.add_memory(
                result["content"],
                {"source": "file", "file_path": file_path}
            )

            logger.info(f"Saved file {file_path} to memory")

        except Exception as e:
            logger.error(f"Failed to save file to memory: {e}")
            raise

    async def query_files_in_memory(self, query: str, top_k: int = 5) -> List[str]:
        """Query the memory store for relevant files."""
        try:
            if not self.chroma_service:
                raise ValueError("ChromaService not initialized")

            # Query memories with metadata filter for files
            results = await self.chroma_service.retrieve_with_metadata(
                query,
                {"source": "file"},
                top_k=top_k
            )

            # Extract file paths from results
            file_paths = [
                result["metadata"]["file_path"]
                for result in results
                if "file_path" in result.get("metadata", {})
            ]

            return file_paths

        except Exception as e:
            logger.error(f"Failed to query files in memory: {e}")
            raise

    async def process_directory(
        self,
        directory_path: str,
        file_extensions: Optional[List[str]] = None
    ) -> None:
        """Process all files in a directory and save them to memory."""
        try:
            if not self.mcp_service:
                raise ValueError("MCPService not initialized")

            # Get directory listing tools from MCP
            tools = await self.mcp_service.get_tools()
            list_tool = next(
                (tool for tool in tools if "list_dir" in tool.name), None)

            if not list_tool:
                raise ValueError("List directory tool not found")

            # List directory contents
            result = await list_tool.ainvoke({
                "relative_workspace_path": directory_path
            })

            if not result or "entries" not in result:
                raise ValueError(f"Failed to list directory: {directory_path}")

            # Filter files by extension if specified
            files = [
                entry["name"]
                for entry in result["entries"]
                if entry["type"] == "file" and
                (not file_extensions or any(
                    entry["name"].endswith(ext) for ext in file_extensions
                ))
            ]

            # Process each file
            for file_name in files:
                file_path = f"{directory_path}/{file_name}"
                await self.save_file_to_memory(file_path)

            logger.info(
                f"Processed {len(files)} files from directory {directory_path}"
            )

        except Exception as e:
            logger.error(f"Failed to process directory: {e}")
            raise
