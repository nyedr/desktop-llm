import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from langchain.schema.retriever import BaseRetriever
from app.services.langchain_service import LangChainService
from app.services.chroma_service import ChromaService
from app.services.mcp_service import MCPService

# Test data
TEST_QUERY = "test query"
TEST_FILE_PATH = "test/file.txt"
TEST_FILE_CONTENT = "This is a test file content"
TEST_DIRECTORY = "test/directory"
TEST_FILES = [
    {"name": "file1.txt", "type": "file"},
    {"name": "file2.md", "type": "file"},
    {"name": "subdir", "type": "directory"}
]


class MockRetriever(BaseRetriever):
    """Mock retriever for testing."""

    def __init__(self):
        super().__init__()
        self._mock_aget_relevant_documents = AsyncMock()

    async def _aget_relevant_documents(self, query):
        return await self._mock_aget_relevant_documents(query)

    def _get_relevant_documents(self, query):
        raise NotImplementedError()


@pytest.fixture
async def mock_chroma_service():
    """Create a mock ChromaService."""
    service = Mock(spec=ChromaService)
    service.client = Mock()
    service.add_memory = AsyncMock()
    service.retrieve_with_metadata = AsyncMock()
    return service


@pytest.fixture
async def mock_mcp_service():
    """Create a mock MCPService."""
    service = Mock(spec=MCPService)
    mock_tools = [
        Mock(name="read_file"),
        Mock(name="list_dir")
    ]
    # Make the list of tools iterable
    mock_tools[0].name = "read_file"
    mock_tools[1].name = "list_dir"
    service.get_tools = AsyncMock(return_value=mock_tools)
    return service


@pytest.fixture
async def langchain_service(mock_chroma_service, mock_mcp_service):
    """Create a LangChainService instance with mocked dependencies."""
    service = LangChainService()
    with patch('langchain.embeddings.HuggingFaceEmbeddings'), \
            patch('langchain.llms.Ollama') as mock_llm, \
            patch('langchain.vectorstores.Chroma') as mock_chroma:
        # Create a mock retriever
        mock_retriever = MockRetriever()
        mock_chroma.return_value.as_retriever.return_value = mock_retriever
        # Set up the LLM mock
        mock_llm.return_value.ainvoke = AsyncMock(return_value="Test result")
        await service.initialize(mock_chroma_service, mock_mcp_service)
        yield service


@pytest.mark.asyncio
async def test_initialize(mock_chroma_service, mock_mcp_service):
    """Test service initialization."""
    service = LangChainService()
    with patch('langchain.embeddings.HuggingFaceEmbeddings'), \
            patch('langchain.llms.Ollama'), \
            patch('langchain.vectorstores.Chroma') as mock_chroma:
        # Create a mock retriever
        mock_retriever = MockRetriever()
        mock_chroma.return_value.as_retriever.return_value = mock_retriever
        await service.initialize(mock_chroma_service, mock_mcp_service)

    assert service.chroma_service == mock_chroma_service
    assert service.mcp_service == mock_mcp_service
    assert service.embeddings is not None
    assert service.llm is not None
    assert service.retriever is not None


@pytest.mark.asyncio
async def test_query_memory(langchain_service):
    """Test querying the memory store."""
    # Mock retriever response
    mock_docs = [Mock(page_content="doc1"), Mock(page_content="doc2")]
    langchain_service.retriever._aget_relevant_documents = AsyncMock(
        return_value=mock_docs)

    # Mock LLM response
    mock_response = {
        "result": "Test result",
        "source_documents": mock_docs
    }
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    langchain_service.llm = mock_llm

    # Test query
    result = await langchain_service.query_memory("test query")
    assert result == mock_response
    # Assert call was made with any run_manager
    call_args = langchain_service.retriever._aget_relevant_documents.call_args
    assert call_args[0][0] == "test query"  # First positional argument
    langchain_service.llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_save_file_to_memory(langchain_service, mock_mcp_service):
    """Test saving a file to memory."""
    # Mock MCP read tool
    mock_read_tool = Mock(name="read_file")
    mock_read_tool.name = "read_file"
    mock_read_tool.ainvoke = AsyncMock(
        return_value={"content": TEST_FILE_CONTENT})
    mock_mcp_service.get_tools = AsyncMock(return_value=[mock_read_tool])
    langchain_service.mcp_service = mock_mcp_service

    # Test save file
    await langchain_service.save_file_to_memory(TEST_FILE_PATH)

    # Verify MCP tool was called correctly
    mock_read_tool.ainvoke.assert_called_once_with({
        "relative_workspace_path": TEST_FILE_PATH,
        "should_read_entire_file": True,
        "start_line_one_indexed": 1,
        "end_line_one_indexed_inclusive": 999999
    })

    # Verify content was saved to Chroma
    langchain_service.chroma_service.add_memory.assert_called_once_with(
        TEST_FILE_CONTENT,
        {"source": "file", "file_path": TEST_FILE_PATH}
    )


@pytest.mark.asyncio
async def test_query_files_in_memory(langchain_service):
    """Test querying files in memory."""
    # Mock Chroma response
    mock_results = [
        {
            "document": "doc1",
            "metadata": {"source": "file", "file_path": "path1.txt"}
        },
        {
            "document": "doc2",
            "metadata": {"source": "file", "file_path": "path2.txt"}
        }
    ]
    langchain_service.chroma_service.retrieve_with_metadata.return_value = mock_results

    # Test query
    file_paths = await langchain_service.query_files_in_memory(TEST_QUERY)

    assert file_paths == ["path1.txt", "path2.txt"]
    langchain_service.chroma_service.retrieve_with_metadata.assert_called_once_with(
        TEST_QUERY,
        {"source": "file"},
        top_k=5
    )


@pytest.mark.asyncio
async def test_process_directory(langchain_service, mock_mcp_service):
    """Test processing a directory."""
    # Mock MCP list tool
    mock_list_tool = Mock(name="list_dir")
    mock_list_tool.name = "list_dir"
    mock_list_tool.ainvoke = AsyncMock(return_value={"entries": TEST_FILES})
    mock_mcp_service.get_tools = AsyncMock(return_value=[mock_list_tool])
    langchain_service.mcp_service = mock_mcp_service

    # Mock save_file_to_memory
    langchain_service.save_file_to_memory = AsyncMock()

    # Test process directory with file extension filter
    await langchain_service.process_directory(TEST_DIRECTORY, [".txt"])

    # Verify directory was listed
    mock_list_tool.ainvoke.assert_called_once_with({
        "relative_workspace_path": TEST_DIRECTORY
    })

    # Verify only .txt files were processed
    assert langchain_service.save_file_to_memory.call_count == 1
    langchain_service.save_file_to_memory.assert_called_with(
        f"{TEST_DIRECTORY}/file1.txt"
    )


@pytest.mark.asyncio
async def test_error_handling(langchain_service):
    """Test error handling in various scenarios."""
    # Test uninitialized service
    langchain_service.retriever = None
    with pytest.raises(ValueError, match="LangChain Service not initialized"):
        await langchain_service.query_memory(TEST_QUERY)

    # Test missing MCP tool
    mock_mcp_service = Mock(spec=MCPService)
    mock_mcp_service.get_tools = AsyncMock(return_value=[])
    langchain_service.mcp_service = mock_mcp_service
    with pytest.raises(ValueError, match="Read file tool not found"):
        await langchain_service.save_file_to_memory(TEST_FILE_PATH)

    # Test invalid directory listing
    mock_list_tool = Mock(name="list_dir")
    mock_list_tool.name = "list_dir"
    mock_list_tool.ainvoke = AsyncMock(return_value={})  # Missing 'entries'
    mock_mcp_service.get_tools = AsyncMock(return_value=[mock_list_tool])
    with pytest.raises(ValueError, match="Failed to list directory"):
        await langchain_service.process_directory(TEST_DIRECTORY)
