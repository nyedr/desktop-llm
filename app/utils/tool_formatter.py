"""Utility functions for formatting tools for chat."""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def format_tools_for_chat(tools: List[Dict[str, Any]], request_id: str) -> List[Dict[str, Any]]:
    """Format tools for chat API request.

    Args:
        tools: List of tool configurations
        request_id: Request ID for logging

    Returns:
        List of formatted tools ready for API request
    """
    formatted_tools = []
    for tool in tools:
        # Handle case where tool is already formatted
        if isinstance(tool, dict) and tool.get("type") == "function":
            formatted_tools.append(tool)
            continue

        # Handle case where tool is a function object
        if hasattr(tool, "name"):
            formatted_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": getattr(tool, "description", ""),
                    "parameters": getattr(tool, "parameters", {})
                }
            }
        # Handle case where tool is a dict with function info
        elif isinstance(tool, dict) and "function" in tool:
            formatted_tool = {
                "type": "function",
                "function": tool["function"]
            }
        # Handle case where tool is a dict with direct properties
        elif isinstance(tool, dict):
            formatted_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {})
                }
            }
        else:
            logger.warning(
                f"[{request_id}] Skipping invalid tool format: {tool}")
            continue

        if not formatted_tool["function"]["name"]:
            logger.warning(
                f"[{request_id}] Skipping tool without name: {tool}")
            continue

        formatted_tools.append(formatted_tool)

    return formatted_tools
