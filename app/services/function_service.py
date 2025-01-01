"""Function service module."""

import logging
from typing import Dict, Any, Optional, List, Type
from app.functions import function_registry, executor
from app.models.function import BaseFunction, Tool, FunctionType, ToolResponse
import json

logger = logging.getLogger(__name__)


class FunctionService:
    """Service for managing and executing functions."""

    def __init__(self):
        logger.info("[INIT] Initializing FunctionService")
        self.registry = function_registry
        self.executor = executor
        logger.info(
            "[INIT] FunctionService initialized with registry and executor")

    def register_function(
        self,
        name: str,
        description: Optional[str] = None,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        function_class: Optional[type] = None
    ) -> None:
        """Register a function for execution."""
        logger.info(f"[REGISTER] Starting function registration: {name}")

        try:
            if function_class:
                logger.info(
                    f"[REGISTER] Registering existing class: {function_class.__name__}")
                self.registry.register(function_class)
            else:
                logger.info("[REGISTER] Creating dynamic Tool class")
                # Create a dynamic Tool class
                tool_name = name  # Capture outer scope variables
                tool_description = description or ""
                tool_parameters = input_schema or {
                    "type": "object",
                    "properties": {},
                    "required": []
                }

                class DynamicTool(Tool):
                    type = FunctionType.TOOL
                    name = tool_name
                    description = tool_description
                    parameters = tool_parameters

                    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
                        return args  # Placeholder implementation

                logger.info(f"[REGISTER] Registering dynamic tool: {name}")
                self.registry.register(DynamicTool)

            logger.info(f"[REGISTER] Successfully registered function: {name}")
            logger.debug(
                f"[REGISTER] Current functions: {list(self.registry._functions.keys())}")

        except Exception as e:
            logger.error(
                f"[REGISTER] Error registering function {name}: {e}", exc_info=True)
            raise

    def list_functions(self) -> List[Dict[str, Any]]:
        """List all registered functions."""
        logger.info("[LIST] Getting list of all registered functions")
        functions = self.registry.list_functions()
        logger.debug(f"[LIST] Found {len(functions)} functions")
        logger.debug(f"[LIST] Functions by type:")
        for func in functions:
            logger.debug(f"  - {func['name']}: {func['type']}")
        return functions

    def get_function_schemas(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible function schemas for all registered functions."""
        logger.info("[SCHEMAS] Getting function schemas")
        schemas = []

        functions = self.registry.list_functions()
        logger.debug(f"[SCHEMAS] Processing {len(functions)} functions")

        for func in functions:
            try:
                # Include tools in function schemas
                if func["type"] == FunctionType.TOOL:
                    logger.info(
                        f"[SCHEMAS] Processing tool function: {func['name']}")
                    schema = {
                        "name": func["name"],
                        "description": func["description"],
                        "parameters": func.get("parameters", {
                            "type": "object",
                            "properties": {},
                            "required": []
                        })
                    }
                    schemas.append({
                        "type": "function",
                        "function": schema
                    })
                    logger.info(
                        f"[SCHEMAS] Added schema for tool: {func['name']}")
                else:
                    logger.debug(
                        f"[SCHEMAS] Skipping non-tool function: {func['name']} (type: {func['type']})")
            except Exception as e:
                logger.error(
                    f"[SCHEMAS] Error processing function schema for {func.get('name', 'unknown')}: {e}", exc_info=True)

        return schemas

    async def execute_function(self, function_name: str, function_args: Dict[str, Any]) -> ToolResponse:
        """Execute a function by name with given arguments.

        Args:
            function_name: Name of the function to execute
            function_args: Arguments to pass to the function

        Returns:
            ToolResponse with the result or error
        """
        logger.info(f"[EXECUTE] Executing function: {function_name}")

        try:
            function_class = self.get_function(function_name)
            if not function_class:
                return ToolResponse(
                    success=False,
                    result=None,
                    error=f"Function {function_name} not found",
                    tool_name=function_name
                )

            if function_class.model_fields['type'].default != FunctionType.TOOL:
                return ToolResponse(
                    success=False,
                    result=None,
                    error=f"Function {function_name} is not a tool",
                    tool_name=function_name
                )

            function_instance = function_class()
            try:
                result = await function_instance.execute(args=function_args)
            except TypeError as e:
                # Handle case where execute() doesn't accept args parameter
                try:
                    result = await function_instance.execute(**function_args)
                except Exception as inner_e:
                    return ToolResponse(
                        success=False,
                        result=None,
                        error=str(inner_e),
                        tool_name=function_name
                    )

            # If result is already a ToolResponse, return it
            if isinstance(result, ToolResponse):
                # Log the result as a dict for JSON serialization
                log_data = {
                    "success": result.success,
                    "result": result.result,
                    "error": result.error,
                    "tool_name": result.tool_name,
                    "metadata": result.metadata
                }
                logger.debug(
                    f"[EXECUTE] Result: {json.dumps(log_data, indent=2)}")
                return result

            # Convert raw result to ToolResponse
            tool_response = ToolResponse(
                success=True,
                result=result,
                error=None,
                tool_name=function_name
            )
            logger.debug(
                f"[EXECUTE] Raw result converted: {json.dumps({'result': result}, indent=2)}")
            return tool_response

        except Exception as e:
            logger.error(
                f"[EXECUTE] Error executing function {function_name}: {e}")
            return ToolResponse(
                success=False,
                result=None,
                error=str(e),
                tool_name=function_name
            )

    async def handle_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Handle multiple tool calls in sequence."""
        logger.info(f"[TOOLS] Handling {len(tool_calls)} tool calls")
        logger.debug(f"[TOOLS] Tool calls: {json.dumps(tool_calls, indent=2)}")

        try:
            results = await self.executor.handle_tool_calls(tool_calls)
            logger.info(
                f"[TOOLS] Successfully handled {len(results)} tool calls")
            logger.debug(f"[TOOLS] Results: {json.dumps(results, indent=2)}")
            return results
        except Exception as e:
            logger.error(
                f"[TOOLS] Error handling tool calls: {e}", exc_info=True)
            raise

    def get_function(self, name: str) -> Optional[Type[BaseFunction]]:
        """Get a function class by name."""
        logger.debug(f"[GET] Getting function: {name}")
        try:
            func = self.registry.get_function(name)
            if func:
                logger.debug(
                    f"[GET] Found function {name} of type {func.model_fields['type'].default}")
            else:
                logger.warning(f"[GET] Function {name} not found in registry")
            return func
        except Exception as e:
            logger.error(
                f"[GET] Error getting function {name}: {e}", exc_info=True)
            return None
