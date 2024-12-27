"""Model Context Protocol (MCP) Service.

This module provides the MCPService class for managing MCP server interactions.
It handles server lifecycle, communication, and error handling.

Important:
- All JSON-RPC communication must follow version 2.0 spec
- Initialization must complete before any operations
- Always use proper cleanup sequence to avoid resource leaks
- Handle task cancellation gracefully using anyio
"""

import asyncio
import logging
import shutil
from typing import Optional, List

from mcp.client.stdio import (
    stdio_client,
    StdioServerParameters,
)
from mcp.client.session import ClientSession
from langchain_mcp import MCPToolkit
from langchain_core.tools.base import BaseTool

from app.services.base import BaseService
from app.core.mcp_config import get_server_args, validate_server_installation

logger = logging.getLogger(__name__)


class MCPServiceError(Exception):
    """Base exception class for MCP service errors."""
    pass


class MCPInitializationError(MCPServiceError):
    """Raised when initialization of the MCP service fails."""
    pass


class MCPService(BaseService):
    """Model Context Protocol (MCP) Service."""

    def __init__(self):
        """Initialize the MCP service."""
        super().__init__()
        self.toolkit: Optional[MCPToolkit] = None
        self.session: Optional[ClientSession] = None
        self._stdio_client = None
        self._initialized = False
        self._initialization_timeout = 5.0  # 5 seconds timeout

    async def initialize(self) -> None:
        """Initialize the MCP service with improved error handling."""
        try:
            # Verify Node.js is available
            node_cmd = shutil.which("node")
            if not node_cmd:
                raise MCPInitializationError(
                    "Node.js not found in PATH. Please install Node.js and ensure it's in your PATH."
                )
            logger.debug(f"Using Node.js from: {node_cmd}")

            # Verify server is installed
            validate_server_installation("filesystem")

            await self._start_server()
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize MCP service: {e}")
            await self.terminate()
            raise MCPInitializationError(str(e))

    async def terminate(self):
        """Terminate the MCP service and clean up resources."""
        try:
            if self.session:
                try:
                    await self.session.close()
                except Exception as e:
                    logger.error(f"Error closing session: {e}")
                self.session = None

            if self._stdio_client:
                try:
                    await self._stdio_client.__aexit__(None, None, None)
                except Exception as e:
                    logger.error(f"Error closing stdio client: {e}")
                self._stdio_client = None

            self._initialized = False
            logger.info("MCP service terminated successfully")
        except Exception as e:
            logger.error(f"Error during MCP service termination: {e}")

    async def _start_server(self):
        """Start the Node.js server using globally installed package."""
        try:
            # Get server arguments from config
            args = get_server_args("filesystem")

            # Create server parameters using global npm installation
            server_params = StdioServerParameters(
                command="node",
                args=args
            )

            # Log the exact command being used
            logger.debug(
                f"Starting Node.js server with command: {server_params.command}")
            logger.debug(f"Server arguments: {server_params.args}")

            # Create stdio client and session
            self._stdio_client = stdio_client(server_params)
            read_stream, write_stream = await self._stdio_client.__aenter__()
            self.session = ClientSession(read_stream, write_stream)
            await self.session.__aenter__()

            # Initialize MCPToolkit (which will handle session initialization)
            self.toolkit = MCPToolkit(session=self.session)
            await asyncio.wait_for(
                self.toolkit.initialize(),
                timeout=self._initialization_timeout
            )

            logger.info("Server and MCPToolkit initialized successfully")

        except asyncio.TimeoutError:
            logger.error("Server initialization timed out")
            raise MCPInitializationError("Server initialization timed out")
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise MCPInitializationError(f"Failed to start server: {e}")

    @property
    def is_initialized(self):
        """Check if the service is initialized."""
        return self._initialized

    async def get_tools(self) -> List[BaseTool]:
        """
        Get the LangChain Tools from MCPToolkit.
        If the service isn't initialized, it will initialize first.

        Returns:
            List[BaseTool]: List of LangChain tools provided by the MCP server

        Raises:
            MCPServiceError: If initialization fails or toolkit is not initialized
        """
        if not self._initialized:
            await self.initialize()

        if not self.toolkit:
            raise MCPServiceError("MCPToolkit not initialized")

        return self.toolkit.get_tools()

    async def close_session(self):
        """Close the session and terminate the service."""
        await self.terminate()
