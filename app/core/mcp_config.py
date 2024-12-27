"""Configuration for MCP servers and tools.

This module provides configuration for MCP servers, handling different Node.js installations
(NVM, global NPM, or Windows MSI installer) and providing robust path detection.
"""

import os
import logging
import shutil
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def find_global_node_modules() -> Path:
    """
    Find the global node_modules directory by checking common installation paths.
    Prioritizes NVM if available, falls back to global npm location.
    """
    # User's home directory
    home = Path(os.path.expanduser("~"))

    # Common paths for node_modules (in order of preference)
    possible_paths = [
        # NVM installation (if using NVM for Windows)
        home / "AppData" / "Roaming" / "nvm" / "v20.14.0" / "node_modules",
        # Global NPM installation (Windows default)
        home / "AppData" / "Roaming" / "npm" / "node_modules",
        # Alternate NPM locations
        Path("C:/Program Files/nodejs/node_modules"),
        Path("C:/Program Files (x86)/nodejs/node_modules")
    ]

    # Check if Node.js is available
    node_cmd = shutil.which("node")
    if not node_cmd:
        logger.warning("Node.js not found in PATH. Please install Node.js")
    else:
        logger.debug(f"Found Node.js at: {node_cmd}")

    # Find first existing path
    for path in possible_paths:
        if path.exists():
            logger.debug(f"Found global node_modules at: {path}")
            return path

    # Default to npm global if nothing found (will be validated when used)
    default_path = possible_paths[1]  # Global NPM installation path
    logger.warning(
        f"No existing node_modules found. Using default path: {default_path}"
    )
    return default_path


# Global node_modules path
NODE_MODULES_PATH = find_global_node_modules()

# Server configurations
MCP_SERVERS = {
    "filesystem": {
        "package": "@modelcontextprotocol/server-filesystem",
        "entry_point": "dist/index.js",
        "allowed_paths": [
            os.path.expanduser("~"),
            os.path.expanduser("~/Desktop"),
            os.path.expanduser("~/Documents"),
            os.path.expanduser("~/Downloads")
        ],
        "required": True  # If True, will raise error if not found
    }
}


def validate_server_installation(server_name: str) -> bool:
    """
    Validate that a server package is properly installed.

    Args:
        server_name: Name of the server from MCP_SERVERS

    Returns:
        bool: True if server is properly installed

    Raises:
        ValueError: If server_name is unknown
        RuntimeError: If required server is not installed
    """
    if server_name not in MCP_SERVERS:
        raise ValueError(f"Unknown MCP server: {server_name}")

    server_config = MCP_SERVERS[server_name]
    server_path = NODE_MODULES_PATH / server_config["package"]

    is_installed = server_path.exists()
    if not is_installed:
        msg = f"MCP server '{server_name}' not found at {server_path}"
        if server_config.get("required", False):
            raise RuntimeError(
                f"{msg}. Please install with: npm install -g {server_config['package']}")
        else:
            logger.warning(f"{msg}")

    return is_installed


def get_server_path(server_name: str) -> Path:
    """
    Get the full path to a server's entry point.

    Args:
        server_name: Name of the server from MCP_SERVERS

    Returns:
        Path: Full path to the server's entry point

    Raises:
        ValueError: If server_name is unknown
        RuntimeError: If required server is not installed
    """
    if not validate_server_installation(server_name):
        raise RuntimeError(f"Server '{server_name}' is not properly installed")

    server_config = MCP_SERVERS[server_name]
    return NODE_MODULES_PATH / server_config["package"] / server_config["entry_point"]


def get_server_args(server_name: str) -> List[str]:
    """
    Get the command line arguments for a server.

    Args:
        server_name: Name of the server from MCP_SERVERS

    Returns:
        List[str]: Command line arguments including server path and allowed paths

    Raises:
        ValueError: If server_name is unknown
        RuntimeError: If required server is not installed
    """
    server_path = get_server_path(server_name)
    server_config = MCP_SERVERS[server_name]

    return [str(server_path)] + server_config.get("allowed_paths", [])
