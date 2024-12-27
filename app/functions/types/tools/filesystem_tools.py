"""Filesystem tools for interacting with the local filesystem through MCP."""

import os
from pathlib import Path
from typing import Dict, Any, Literal, List
import logging
from pydantic import Field

from app.functions.base import Tool, FunctionType, register_function, InputValidationError, ExecutionError
from app.core.service_locator import get_service_locator

logger = logging.getLogger(__name__)


class BaseFileSystemTool(Tool):
    """Base class for filesystem tools."""
    type: Literal[FunctionType.TOOL] = Field(
        default=FunctionType.TOOL, description="Tool type")

    def _normalize_path(self, path: str) -> str:
        """Normalize a path to use forward slashes and absolute paths."""
        try:
            # Convert to Path object and resolve to absolute path
            normalized = str(Path(path).resolve())
            # Convert Windows backslashes to forward slashes
            normalized = normalized.replace("\\", "/")
            logger.debug(f"Normalized path: {path} -> {normalized}")
            return normalized
        except Exception as e:
            logger.error(f"Error normalizing path {path}: {e}")
            raise InputValidationError(f"Invalid path: {path}")

    async def _execute_with_mcp(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a method through the MCP service."""
        try:
            # Get the global MCP service
            mcp_service = get_service_locator().get_mcp_service()
            if not mcp_service.is_initialized:
                await mcp_service.initialize()

            # Get MCP tools
            tools = await mcp_service.get_tools()
            tool = next((t for t in tools if method in t.name), None)
            if not tool:
                raise ValueError(f"Tool {method} not found in MCP service")

            # Log the tool details and parameters
            logger.debug(f"Found MCP tool: {tool.name}")
            logger.debug(f"Calling tool with params: {params}")

            # Pass parameters as a single input argument
            result = await tool.ainvoke(input=params)
            return result
        except KeyError:
            raise RuntimeError(
                "MCP service not registered. Ensure the service is properly initialized.")
        except Exception as e:
            logger.error(f"Error executing MCP method {method}: {e}")
            raise


@register_function(
    func_type=FunctionType.TOOL,
    name="list_directory",
    description="List directory contents with [FILE] or [DIR] prefixes"
)
class ListDirectoryTool(BaseFileSystemTool):
    """Tool for listing directory contents."""
    type: Literal[FunctionType.TOOL] = Field(
        default=FunctionType.TOOL, description="Tool type")
    name: str = Field(default="list_directory",
                      description="Name of the list directory tool")
    description: str = Field(default="List directory contents with [FILE] or [DIR] prefixes",
                             description="Description of what the tool does")
    parameters: Dict[str, Any] = Field(
        default={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory to list"
                }
            },
            "required": ["path"]
        },
        description="Parameters schema for the list directory tool"
    )

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the list directory command."""
        try:
            if not args:
                logger.error("No arguments provided to list_directory")
                raise InputValidationError("No arguments provided")

            if "path" not in args:
                logger.error("Path argument missing in list_directory")
                raise InputValidationError("Path argument is required")

            path = self._normalize_path(args["path"])
            logger.info(f"Listing directory: {path}")
            logger.debug(f"Full arguments: {args}")

            result = await self._execute_with_mcp("list_directory", {"path": path})
            logger.debug(f"MCP result: {result}")
            return result

        except InputValidationError:
            raise
        except Exception as e:
            logger.error(f"Error listing directory: {e}", exc_info=True)
            raise ExecutionError(f"Error listing directory: {str(e)}")


@register_function(
    func_type=FunctionType.TOOL,
    name="read_file",
    description="Read complete contents of a file"
)
class ReadFileTool(BaseFileSystemTool):
    """Tool for reading file contents."""
    type: Literal[FunctionType.TOOL] = Field(
        default=FunctionType.TOOL, description="Tool type")
    name: str = Field(default="read_file",
                      description="Name of the read file tool")
    description: str = Field(default="Read complete contents of a file with UTF-8 encoding",
                             description="Description of what the tool does")
    parameters: Dict[str, Any] = Field(
        default={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            },
            "required": ["path"]
        },
        description="Parameters schema for the read file tool"
    )

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the read file command."""
        try:
            if not args:
                logger.error("No arguments provided to read_file")
                raise InputValidationError("No arguments provided")

            if "path" not in args:
                logger.error("Path argument missing in read_file")
                raise InputValidationError("Path argument is required")

            path = self._normalize_path(args["path"])
            logger.info(f"Reading file: {path}")
            logger.debug(f"Full arguments: {args}")

            result = await self._execute_with_mcp("read_file", {"path": path})
            logger.debug(f"MCP result: {result}")
            return result

        except InputValidationError:
            raise
        except Exception as e:
            logger.error(f"Error reading file: {e}", exc_info=True)
            raise ExecutionError(f"Error reading file: {str(e)}")


@register_function(
    func_type=FunctionType.TOOL,
    name="read_multiple_files",
    description="Read multiple files simultaneously"
)
class ReadMultipleFilesTool(BaseFileSystemTool):
    """Tool for reading multiple files simultaneously."""
    type: Literal[FunctionType.TOOL] = Field(
        default=FunctionType.TOOL, description="Tool type")
    name: str = Field(default="read_multiple_files",
                      description="Name of the read multiple files tool")
    description: str = Field(default="Read multiple files simultaneously. Failed reads won't stop the entire operation",
                             description="Description of what the tool does")
    parameters: Dict[str, Any] = Field(
        default={
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file paths to read"
                }
            },
            "required": ["paths"]
        },
        description="Parameters schema for the read multiple files tool"
    )

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the read multiple files command."""
        try:
            if not args:
                logger.error("No arguments provided to read_multiple_files")
                raise InputValidationError("No arguments provided")

            if "paths" not in args:
                logger.error("Paths argument missing in read_multiple_files")
                raise InputValidationError("Paths argument is required")

            # Normalize all paths
            normalized_paths = [self._normalize_path(
                path) for path in args["paths"]]
            logger.info(f"Reading files: {normalized_paths}")
            logger.debug(f"Full arguments: {args}")

            result = await self._execute_with_mcp("read_multiple_files", {"paths": normalized_paths})
            logger.debug(f"MCP result: {result}")
            return result

        except InputValidationError:
            raise
        except Exception as e:
            logger.error(f"Error reading multiple files: {e}", exc_info=True)
            raise ExecutionError(f"Error reading multiple files: {str(e)}")


@register_function(
    func_type=FunctionType.TOOL,
    name="write_file",
    description="Create new file or overwrite existing"
)
class WriteFileTool(BaseFileSystemTool):
    """Tool for writing file contents."""
    type: Literal[FunctionType.TOOL] = Field(
        default=FunctionType.TOOL, description="Tool type")
    name: str = Field(default="write_file",
                      description="Name of the write file tool")
    description: str = Field(default="Create new file or overwrite existing (exercise caution with this)",
                             description="Description of what the tool does")
    parameters: Dict[str, Any] = Field(
        default={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File location"
                },
                "content": {
                    "type": "string",
                    "description": "File content"
                }
            },
            "required": ["path", "content"]
        },
        description="Parameters schema for the write file tool"
    )

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the write file command."""
        try:
            if not args:
                logger.error("No arguments provided to write_file")
                raise InputValidationError("No arguments provided")

            if "path" not in args:
                logger.error("Path argument missing in write_file")
                raise InputValidationError("Path argument is required")

            if "content" not in args:
                logger.error("Content argument missing in write_file")
                raise InputValidationError("Content argument is required")

            path = self._normalize_path(args["path"])
            logger.info(f"Writing to file: {path}")
            logger.debug(f"Full arguments: {args}")

            result = await self._execute_with_mcp("write_file", {
                "path": path,
                "content": args["content"]
            })
            logger.debug(f"MCP result: {result}")
            return result

        except InputValidationError:
            raise
        except Exception as e:
            logger.error(f"Error writing file: {e}", exc_info=True)
            raise ExecutionError(f"Error writing file: {str(e)}")


@register_function(
    func_type=FunctionType.TOOL,
    name="edit_file",
    description="Make selective edits using advanced pattern matching and formatting"
)
class EditFileTool(BaseFileSystemTool):
    """Tool for editing files with advanced pattern matching."""
    type: Literal[FunctionType.TOOL] = Field(
        default=FunctionType.TOOL, description="Tool type")
    name: str = Field(default="edit_file",
                      description="Name of the edit file tool")
    description: str = Field(
        default="Make selective edits using advanced pattern matching and formatting",
        description="Description of what the tool does"
    )
    parameters: Dict[str, Any] = Field(
        default={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File to edit"
                },
                "edits": {
                    "type": "array",
                    "description": "List of edit operations"
                },
                "oldText": {
                    "type": "string",
                    "description": "Text to search for (can be substring)"
                },
                "newText": {
                    "type": "string",
                    "description": "Text to replace with"
                },
                "dryRun": {
                    "type": "boolean",
                    "description": "Preview changes without applying",
                    "default": False
                },
                "options": {
                    "type": "object",
                    "description": "Optional formatting settings",
                    "properties": {
                        "preserveIndentation": {
                            "type": "boolean",
                            "default": True
                        },
                        "normalizeWhitespace": {
                            "type": "boolean",
                            "default": True
                        },
                        "partialMatch": {
                            "type": "boolean",
                            "default": True
                        }
                    }
                }
            },
            "required": ["path", "oldText", "newText"]
        },
        description="Parameters schema for the edit file tool"
    )

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the edit file command."""
        try:
            if not args:
                logger.error("No arguments provided to edit_file")
                raise InputValidationError("No arguments provided")

            required_args = ["path", "oldText", "newText"]
            for arg in required_args:
                if arg not in args:
                    logger.error(f"{arg} argument missing in edit_file")
                    raise InputValidationError(f"{arg} argument is required")

            path = self._normalize_path(args["path"])
            logger.info(f"Editing file: {path}")
            logger.debug(f"Full arguments: {args}")

            result = await self._execute_with_mcp("edit_file", args)
            logger.debug(f"MCP result: {result}")
            return result

        except InputValidationError:
            raise
        except Exception as e:
            logger.error(f"Error editing file: {e}", exc_info=True)
            raise ExecutionError(f"Error editing file: {str(e)}")


@register_function(
    func_type=FunctionType.TOOL,
    name="create_directory",
    description="Create new directory or ensure it exists"
)
class CreateDirectoryTool(BaseFileSystemTool):
    """Tool for creating directories."""
    type: Literal[FunctionType.TOOL] = Field(
        default=FunctionType.TOOL, description="Tool type")
    name: str = Field(default="create_directory",
                      description="Name of the create directory tool")
    description: str = Field(
        default="Create new directory or ensure it exists. Creates parent directories if needed",
        description="Description of what the tool does"
    )
    parameters: Dict[str, Any] = Field(
        default={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to create"
                }
            },
            "required": ["path"]
        },
        description="Parameters schema for the create directory tool"
    )

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the create directory command."""
        try:
            if not args:
                logger.error("No arguments provided to create_directory")
                raise InputValidationError("No arguments provided")

            if "path" not in args:
                logger.error("Path argument missing in create_directory")
                raise InputValidationError("Path argument is required")

            path = self._normalize_path(args["path"])
            logger.info(f"Creating directory: {path}")
            logger.debug(f"Full arguments: {args}")

            result = await self._execute_with_mcp("create_directory", {"path": path})
            logger.debug(f"MCP result: {result}")
            return result

        except InputValidationError:
            raise
        except Exception as e:
            logger.error(f"Error creating directory: {e}", exc_info=True)
            raise ExecutionError(f"Error creating directory: {str(e)}")


@register_function(
    func_type=FunctionType.TOOL,
    name="move_file",
    description="Move or rename files and directories"
)
class MoveFileTool(BaseFileSystemTool):
    """Tool for moving or renaming files."""
    type: Literal[FunctionType.TOOL] = Field(
        default=FunctionType.TOOL, description="Tool type")
    name: str = Field(default="move_file",
                      description="Name of the move file tool")
    description: str = Field(
        default="Move or rename files and directories. Fails if destination exists",
        description="Description of what the tool does"
    )
    parameters: Dict[str, Any] = Field(
        default={
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Source path"
                },
                "destination": {
                    "type": "string",
                    "description": "Destination path"
                }
            },
            "required": ["source", "destination"]
        },
        description="Parameters schema for the move file tool"
    )

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the move file command."""
        try:
            if not args:
                logger.error("No arguments provided to move_file")
                raise InputValidationError("No arguments provided")

            if "source" not in args:
                logger.error("Source argument missing in move_file")
                raise InputValidationError("Source argument is required")

            if "destination" not in args:
                logger.error("Destination argument missing in move_file")
                raise InputValidationError("Destination argument is required")

            source = self._normalize_path(args["source"])
            destination = self._normalize_path(args["destination"])
            logger.info(f"Moving file from {source} to {destination}")
            logger.debug(f"Full arguments: {args}")

            result = await self._execute_with_mcp("move_file", {
                "source": source,
                "destination": destination
            })
            logger.debug(f"MCP result: {result}")
            return result

        except InputValidationError:
            raise
        except Exception as e:
            logger.error(f"Error moving file: {e}", exc_info=True)
            raise ExecutionError(f"Error moving file: {str(e)}")


@register_function(
    func_type=FunctionType.TOOL,
    name="search_files",
    description="Recursively search for files/directories"
)
class SearchFilesTool(BaseFileSystemTool):
    """Tool for searching files."""
    type: Literal[FunctionType.TOOL] = Field(
        default=FunctionType.TOOL, description="Tool type")
    name: str = Field(default="search_files",
                      description="Name of the search files tool")
    description: str = Field(
        default="Recursively search for files/directories with case-insensitive matching",
        description="Description of what the tool does"
    )
    parameters: Dict[str, Any] = Field(
        default={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Starting directory"
                },
                "pattern": {
                    "type": "string",
                    "description": "Search pattern"
                },
                "excludePatterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exclude any patterns. Glob formats are supported.",
                    "default": []
                }
            },
            "required": ["path", "pattern"]
        },
        description="Parameters schema for the search files tool"
    )

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the search files command."""
        try:
            if not args:
                logger.error("No arguments provided to search_files")
                raise InputValidationError("No arguments provided")

            required_args = ["path", "pattern"]
            for arg in required_args:
                if arg not in args:
                    logger.error(f"{arg} argument missing in search_files")
                    raise InputValidationError(f"{arg} argument is required")

            path = self._normalize_path(args["path"])
            logger.info(
                f"Searching files in {path} with pattern: {args['pattern']}")
            logger.debug(f"Full arguments: {args}")

            result = await self._execute_with_mcp("search_files", {
                "path": path,
                "pattern": args["pattern"],
                "excludePatterns": args.get("excludePatterns", [])
            })
            logger.debug(f"MCP result: {result}")
            return result

        except InputValidationError:
            raise
        except Exception as e:
            logger.error(f"Error searching files: {e}", exc_info=True)
            raise ExecutionError(f"Error searching files: {str(e)}")


@register_function(
    func_type=FunctionType.TOOL,
    name="get_file_info",
    description="Get detailed file/directory metadata"
)
class GetFileInfoTool(BaseFileSystemTool):
    """Tool for getting file information."""
    type: Literal[FunctionType.TOOL] = Field(
        default=FunctionType.TOOL, description="Tool type")
    name: str = Field(default="get_file_info",
                      description="Name of the get file info tool")
    description: str = Field(
        default="Get detailed file/directory metadata including size, times, type, and permissions",
        description="Description of what the tool does"
    )
    parameters: Dict[str, Any] = Field(
        default={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to get info for"
                }
            },
            "required": ["path"]
        },
        description="Parameters schema for the get file info tool"
    )

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the get file info command."""
        try:
            if not args:
                logger.error("No arguments provided to get_file_info")
                raise InputValidationError("No arguments provided")

            if "path" not in args:
                logger.error("Path argument missing in get_file_info")
                raise InputValidationError("Path argument is required")

            path = self._normalize_path(args["path"])
            logger.info(f"Getting file info for: {path}")
            logger.debug(f"Full arguments: {args}")

            result = await self._execute_with_mcp("get_file_info", {"path": path})
            logger.debug(f"MCP result: {result}")
            return result

        except InputValidationError:
            raise
        except Exception as e:
            logger.error(f"Error getting file info: {e}", exc_info=True)
            raise ExecutionError(f"Error getting file info: {str(e)}")


@register_function(
    func_type=FunctionType.TOOL,
    name="list_allowed_directories",
    description="List all directories the server is allowed to access"
)
class ListAllowedDirectoriesTool(BaseFileSystemTool):
    """Tool for listing allowed directories."""
    type: Literal[FunctionType.TOOL] = Field(
        default=FunctionType.TOOL, description="Tool type")
    name: str = Field(default="list_allowed_directories",
                      description="Name of the list allowed directories tool")
    description: str = Field(
        default="List all directories that this server can read/write from",
        description="Description of what the tool does"
    )
    parameters: Dict[str, Any] = Field(
        default={
            "type": "object",
            "properties": {},
            "required": []
        },
        description="Parameters schema for the list allowed directories tool"
    )

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the list allowed directories command."""
        try:
            logger.info("Listing allowed directories")
            logger.debug(f"Full arguments: {args}")

            result = await self._execute_with_mcp("list_allowed_directories", {})
            logger.debug(f"MCP result: {result}")
            return result

        except Exception as e:
            logger.error(
                f"Error listing allowed directories: {e}", exc_info=True)
            raise ExecutionError(
                f"Error listing allowed directories: {str(e)}")
