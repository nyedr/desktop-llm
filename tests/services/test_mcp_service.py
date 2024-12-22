import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.services.mcp_service import MCPService

# Test data
TEST_TOOLS = [
    Mock(name="read_file"),
    Mock(name="list_dir"),
    Mock(name="grep_search")
]


@pytest.fixture
async def mock_toolkit():
    """Create a mock MCPToolkit."""
    toolkit = Mock()
    toolkit.initialize = AsyncMock()
    toolkit.get_tools = Mock(return_value=TEST_TOOLS)
    return toolkit


@pytest.fixture
async def mock_session():
    """Create a mock ClientSession."""
    session = Mock()
    session.close = AsyncMock()
    return session


@pytest.fixture
async def mock_stdio_client():
    """Create a mock stdio_client context manager."""
    class MockStdioClient:
        async def __aenter__(self):
            return Mock(), Mock()

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    def mock_client(*args, **kwargs):
        return MockStdioClient()

    return mock_client


@pytest.fixture
async def mcp_service(mock_toolkit, mock_session, mock_stdio_client):
    """Create an MCPService instance with mocked dependencies."""
    with patch('app.services.mcp_service.MCPToolkit', return_value=mock_toolkit), \
            patch('app.services.mcp_service.ClientSession', return_value=mock_session), \
            patch('app.services.mcp_service.stdio_client', side_effect=mock_stdio_client):
        service = MCPService()
        yield service
        await service.close_session()


@pytest.mark.asyncio
async def test_initialize(mcp_service, mock_toolkit):
    """Test service initialization."""
    # Initialize service
    toolkit = await mcp_service.initialize()

    assert toolkit is not None
    assert mcp_service.toolkit is not None
    assert mcp_service.session is not None
    mock_toolkit.initialize.assert_called_once()


@pytest.mark.asyncio
async def test_get_tools(mcp_service, mock_toolkit):
    """Test retrieving tools from the toolkit."""
    # Get tools without initialization
    tools = await mcp_service.get_tools()

    assert tools == TEST_TOOLS
    assert mcp_service.toolkit is not None
    mock_toolkit.get_tools.assert_called_once()


@pytest.mark.asyncio
async def test_get_tools_initialized(mcp_service, mock_toolkit):
    """Test retrieving tools when service is already initialized."""
    # Initialize first
    await mcp_service.initialize()

    # Get tools
    tools = await mcp_service.get_tools()

    assert tools == TEST_TOOLS
    mock_toolkit.get_tools.assert_called_once()


@pytest.mark.asyncio
async def test_close_session(mcp_service, mock_session):
    """Test closing the session."""
    # Initialize service
    await mcp_service.initialize()

    # Close session
    await mcp_service.close_session()

    assert mcp_service.session is None
    assert mcp_service.toolkit is None
    mock_session.close.assert_called_once()


@pytest.mark.asyncio
async def test_close_session_not_initialized(mock_toolkit, mock_session, mock_stdio_client):
    """Test closing session when service is not initialized."""
    with patch('app.services.mcp_service.MCPToolkit', return_value=mock_toolkit), \
            patch('app.services.mcp_service.ClientSession', return_value=mock_session), \
            patch('app.services.mcp_service.stdio_client', side_effect=mock_stdio_client):
        service = MCPService()
        # Close session without initialization
        await service.close_session()

        assert service.session is None
        assert service.toolkit is None
        mock_session.close.assert_not_called()


@pytest.mark.asyncio
async def test_error_handling(mcp_service, mock_toolkit):
    """Test error handling during initialization."""
    # Make toolkit initialization fail
    mock_toolkit.initialize.side_effect = Exception("Test error")

    # Attempt to initialize
    with pytest.raises(Exception, match="Test error"):
        await mcp_service.initialize()

    # Toolkit is created but not initialized
    assert mcp_service.toolkit is not None
    assert mcp_service.session is not None  # Session is created
