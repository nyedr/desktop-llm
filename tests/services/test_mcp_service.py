import pytest
from unittest.mock import Mock, patch, AsyncMock
from app.services.mcp_service import MCPService, MCPInitializationError
import asyncio

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
    toolkit.ainitialize = AsyncMock()
    toolkit.get_tools = Mock(return_value=TEST_TOOLS)
    return toolkit


@pytest.fixture
async def mock_session():
    """Create a mock ClientSession."""
    session = Mock()
    session.__aexit__ = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.wait_for_notification = AsyncMock(return_value={
        "serverInfo": {
            "pid": 1234,
            "version": "v16.0.0",
            "startTime": "2024-01-01T00:00:00.000Z"
        }
    })
    session.request = AsyncMock(return_value={
        "status": "ok",
        "server_info": {
            "pid": 1234,
            "version": "v16.0.0",
            "uptime": 0,
            "startTime": "2024-01-01T00:00:00.000Z",
            "activeConnections": 1
        }
    })
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
async def test_initialization_success(mcp_service, mock_session, mock_toolkit):
    """Test successful initialization of the MCP service."""
    await mcp_service.initialize()

    # Verify initialization sequence
    mock_session.wait_for_notification.assert_called_once_with(
        "initialized", timeout=10.0)
    mock_session.request.assert_called_once_with("ping", {})
    mock_toolkit.ainitialize.assert_called_once()
    assert mcp_service._initialized is True


@pytest.mark.asyncio
async def test_initialization_timeout(mcp_service, mock_session):
    """Test initialization timeout handling."""
    mock_session.wait_for_notification.side_effect = asyncio.TimeoutError()

    with pytest.raises(MCPInitializationError, match="Server initialization timed out"):
        await mcp_service.initialize()

    assert mcp_service._initialized is False


@pytest.mark.asyncio
async def test_initialization_ping_failure(mcp_service, mock_session):
    """Test handling of ping failure during initialization."""
    mock_session.request.return_value = {"status": "error"}

    with pytest.raises(MCPInitializationError, match="Server ping failed"):
        await mcp_service.initialize()

    assert mcp_service._initialized is False


@pytest.mark.asyncio
async def test_get_tools(mcp_service, mock_toolkit):
    """Test retrieving tools from the service."""
    await mcp_service.initialize()
    tools = await mcp_service.get_tools()

    assert tools == TEST_TOOLS
    mock_toolkit.get_tools.assert_called_once()


@pytest.mark.asyncio
async def test_close_session(mcp_service, mock_session):
    """Test proper cleanup during session closure."""
    await mcp_service.initialize()
    await mcp_service.close_session()

    mock_session.__aexit__.assert_called_once()
    assert mcp_service.session is None
    assert mcp_service.toolkit is None
    assert mcp_service._initialized is False
