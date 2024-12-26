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
    session = AsyncMock()
    session.__aexit__ = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.initialize = AsyncMock()
    session.close = AsyncMock()
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
async def mock_node_process():
    """Create a mock Node.js process."""
    process = AsyncMock()
    process.pid = 1234
    process.returncode = None
    process.wait = AsyncMock()
    process.terminate = Mock()
    process.kill = Mock()
    return process


@pytest.fixture
async def mcp_service(mock_toolkit, mock_session, mock_stdio_client, mock_node_process):
    """Create an MCPService instance with mocked dependencies."""
    with patch('app.services.mcp_service.MCPToolkit', return_value=mock_toolkit), \
            patch('app.services.mcp_service.ClientSession', return_value=mock_session), \
            patch('app.services.mcp_service.stdio_client', side_effect=mock_stdio_client):
        service = MCPService()
        service._node_process = mock_node_process
        yield service
        await service.terminate()


@pytest.mark.asyncio
async def test_initialization_success(mcp_service, mock_session, mock_toolkit):
    """Test successful initialization of the MCP service."""
    result = await mcp_service.initialize()

    # Verify initialization sequence
    mock_session.initialize.assert_called_once()
    assert result is True
    assert mcp_service._initialized is True


@pytest.mark.asyncio
async def test_initialization_timeout(mcp_service, mock_session):
    """Test initialization timeout handling."""
    # Set up the session to raise TimeoutError
    mock_session.initialize.side_effect = asyncio.TimeoutError()

    # Initialize should return False and log the error
    result = await mcp_service.initialize()
    assert result is False
    assert mcp_service._initialized is False


@pytest.mark.asyncio
async def test_initialization_ping_failure(mcp_service, mock_session):
    """Test handling of ping failure during initialization."""
    # Set up the session to fail initialization
    mock_session.initialize.side_effect = Exception("Failed to initialize")

    # Initialize should return False and log the error
    result = await mcp_service.initialize()
    assert result is False
    assert mcp_service._initialized is False


@pytest.mark.asyncio
async def test_get_tools(mcp_service, mock_toolkit):
    """Test retrieving tools from the service."""
    # Set up the toolkit with tools
    mcp_service.toolkit = mock_toolkit

    # Get tools should return the tools
    tools = await mcp_service.get_tools()

    assert tools == TEST_TOOLS
    mock_toolkit.get_tools.assert_called_once()


@pytest.mark.asyncio
async def test_terminate(mcp_service, mock_session, mock_node_process):
    """Test proper cleanup during service termination."""
    # Reset the mock to clear any previous calls
    mock_node_process.terminate.reset_mock()

    # Reset the node process to test termination
    mcp_service._node_process = None

    await mcp_service.initialize()
    # Set the node process after initialization
    mcp_service._node_process = mock_node_process

    await mcp_service.terminate()

    mock_session.close.assert_called_once()
    assert mcp_service.session is None
    assert mcp_service._initialized is False
    mock_node_process.terminate.assert_called_once()
