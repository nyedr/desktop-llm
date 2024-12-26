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
import signal
import sys
from typing import Optional
from pathlib import Path

from mcp.client.stdio import (
    stdio_client,
    StdioServerParameters,
)
from mcp.client.session import ClientSession
from langchain_mcp import MCPToolkit

from app.services.base import BaseService

logger = logging.getLogger(__name__)

# Define the filesystem server directory relative to the project root
FILESYSTEM_SERVER_DIR = Path(
    __file__).parent.parent.parent / "src" / "filesystem"


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
        self._node_process = None
        self._max_retries = 2
        self._retry_delay = 1.0  # seconds
        self._initialization_timeout = 5.0  # 5 seconds timeout
        self._task_group = None

    async def _terminate_node_server(self):
        """Terminate the Node.js server process if it's running."""
        try:
            if self._node_process and hasattr(self._node_process, 'returncode') and self._node_process.returncode is None:
                logger.info(
                    f"Terminating Node.js server (PID: {self._node_process.pid})")
                if sys.platform == 'win32':
                    self._node_process.terminate()
                else:
                    os.kill(self._node_process.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(self._node_process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Node.js server did not terminate gracefully, forcing kill")
                    if sys.platform == 'win32':
                        self._node_process.kill()
                    else:
                        os.kill(self._node_process.pid, signal.SIGKILL)
        except Exception as e:
            logger.error(f"Error terminating Node.js server: {e}")

    async def initialize(self):
        """Initialize the MCP service with improved error handling."""
        try:
            await self._start_server()
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize MCP service: {e}")
            await self.terminate()
            return False

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

            if self._node_process:
                await self._terminate_node_server()

            self._initialized = False
            logger.info("MCP service terminated successfully")
        except Exception as e:
            logger.error(f"Error during MCP service termination: {e}")

    async def _start_server(self):
        """Start the Node.js server with retries."""
        for attempt in range(self._max_retries):
            try:
                if self._node_process:
                    await self._terminate_node_server()

                # Create server parameters
                server_params = StdioServerParameters(
                    command="node",
                    args=[str(FILESYSTEM_SERVER_DIR / "dist" / "index.js")],
                )

                logger.info(
                    f"Starting Node.js server (attempt {attempt + 1}/{self._max_retries})")

                # Create stdio client and session
                self._stdio_client = stdio_client(server_params)
                read_stream, write_stream = await self._stdio_client.__aenter__()
                self.session = ClientSession(read_stream, write_stream)
                await self.session.__aenter__()

                # Initialize session with timeout
                await asyncio.wait_for(
                    self.session.initialize(),
                    timeout=self._initialization_timeout
                )
                logger.info("Server initialized successfully")
                return

            except asyncio.TimeoutError:
                logger.error("Server initialization timed out")
                await self.terminate()
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay)
                    continue
                raise MCPInitializationError("Server initialization timed out")

            except Exception as e:
                logger.warning(
                    f"Server start attempt {attempt + 1} failed: {e}")
                await self.terminate()
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay)
                else:
                    raise MCPInitializationError(
                        f"Failed to start server after retries: {e}")

    @property
    def is_initialized(self):
        """Check if the service is initialized."""
        return self._initialized

    async def get_tools(self):
        """Retrieve tools from the MCP toolkit."""
        if not self.toolkit:
            await self.initialize()
        return self.toolkit.get_tools()
