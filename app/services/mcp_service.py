"""Model Context Protocol (MCP) Service.

This module provides the MCPService class for managing MCP server interactions.
It handles server lifecycle, communication, and error handling.

Known Issues and Solutions:
1. JSONRPCRequest creation:
   - Issue: The MCP library internally creates request objects, causing conflicts
   - Solution: Pass method and params directly instead of creating request objects
2. Session handling:
   - Issue: Session cleanup can fail if not properly managed
   - Solution: Use context managers and proper cleanup sequence
3. Task cancellation:
   - Issue: Tasks can be cancelled during initialization
   - Solution: Use anyio task groups and handle cancellation gracefully
4. Stdio client lifecycle:
   - Issue: Stdio client can be closed prematurely
   - Solution: Use async context managers properly and maintain references

Important:
- All JSON-RPC communication must follow version 2.0 spec
- Initialization must complete before any operations
- Always use proper cleanup sequence to avoid resource leaks
- Handle task cancellation gracefully using anyio
"""

import asyncio
import logging
import os
import time
from typing import Optional, List, Dict, Any, Tuple, TypeVar, Type
from contextlib import asynccontextmanager

import anyio
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
from mcp.shared.session import (
    JSONRPCRequest,
    JSONRPCResponse,
    JSONRPCError,
)
from langchain_mcp import MCPToolkit

from app.services.base import BaseService

logger = logging.getLogger(__name__)

# Type variable for response types
ResponseT = TypeVar('ResponseT')


class MCPServiceError(Exception):
    """Base exception class for MCP service errors."""
    pass


class MCPInitializationError(MCPServiceError):
    """Raised when initialization of the MCP service fails."""
    pass


class MCPCommunicationError(MCPServiceError):
    """Raised when communication with the MCP server fails."""
    pass


class MCPService(BaseService):
    """Model Context Protocol (MCP) Service.

    This service manages communication with the Node.js MCP server, which provides
    filesystem access and other tools through a JSON-RPC protocol.

    The service lifecycle:
    1. Initialization - Validates paths and starts the Node.js server
    2. Server Session Creation - Establishes stdio communication and tests connectivity
    3. Operation - Handles tool requests and responses
    4. Shutdown - Gracefully closes connections and stops the server

    Error Handling:
    - MCPInitializationError: Raised for any initialization failures
    - MCPCommunicationError: Raised for communication issues after initialization
    - MCPToolError: Raised for errors during tool execution
    """

    def __init__(self):
        """Initialize the MCP service."""
        super().__init__()
        self.toolkit: Optional[MCPToolkit] = None
        self.session: Optional[ClientSession] = None
        self._stdio_client = None

    async def initialize(self):
        """Initialize the MCP toolkit and establish a session."""
        try:
            logger.info("Initializing MCP Service...")
            # Get the absolute path to the Node.js server
            node_server_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "src",
                    "filesystem",
                    "dist",
                    "index.js"
                )
            )

            # Get the workspace path
            workspace_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "data"
                )
            )

            # Create server parameters
            server_params = StdioServerParameters(
                command="node",
                args=[node_server_path, workspace_path]
            )
            logger.debug(f"MCP server parameters: {server_params}")

            # Create stdio client and session
            self._stdio_client = stdio_client(server_params)
            read_stream, write_stream = await self._stdio_client.__aenter__()
            self.session = ClientSession(read_stream, write_stream)

            # Initialize toolkit
            self.toolkit = MCPToolkit(session=self.session)
            await self.toolkit.initialize()
            logger.info("MCP Service initialized successfully")
            return self.toolkit

        except Exception as e:
            logger.error(f"Failed to initialize MCP Service: {e}")
            raise

    async def get_tools(self):
        """Retrieve tools from the MCP toolkit."""
        if not self.toolkit:
            await self.initialize()
        return self.toolkit.get_tools()

    async def close_session(self):
        """Close the MCP session."""
        if self.session:
            await self.session.close()
            if self._stdio_client:
                await self._stdio_client.__aexit__(None, None, None)
            self.session = None
            self.toolkit = None
            logger.info("MCP session closed.")
