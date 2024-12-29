"""Function executor for running registered functions."""

import logging
import json
from typing import Dict, Any, List, Union
from jsonschema import validate, ValidationError as JsonSchemaValidationError

from app.functions.base import (
    FunctionError,
    ValidationError,
    ExecutionError,
    FunctionNotFoundError,
    InputValidationError,
    Tool,
    Filter,
    Pipeline,
    FunctionResponse,
    ToolResponse,
    FilterResponse,
    PipelineResponse,
    FunctionResult
)
from app.functions.registry import function_registry

logger = logging.getLogger(__name__)


class FunctionExecutor:
    """Executes registered functions with validation and error handling."""

    def __init__(self):
        self.registry = function_registry
        functions = function_registry.list_functions()
        logger.info(
            f"FunctionExecutor initialized with {len(functions)} functions")

    async def execute(self, name: str, params: Dict[str, Any]) -> FunctionResult:
        """Execute a function by name with the given parameters.

        Args:
            name: Name of the function to execute
            params: Parameters to pass to the function

        Returns:
            Function execution result of type ToolResponse, FilterResponse, or PipelineResponse

        Raises:
            FunctionError: If any error occurs during execution
        """
        try:
            logger.info(f"Executing function: {name}")
            logger.debug(
                f"Function parameters: {json.dumps(params, indent=2)}")

            # Get function class
            func_class = self.registry.get_function(name)
            if not func_class:
                raise FunctionNotFoundError(
                    f"Function not found: {name}", function_name=name)

            # Create function instance
            func = func_class()

            # Validate input parameters if schema exists
            if hasattr(func, 'parameters'):
                logger.debug(f"Validating input parameters for {name}")
                try:
                    validate(instance=params, schema=func.parameters)
                except JsonSchemaValidationError as e:
                    raise InputValidationError(
                        f"Invalid input parameters for {name}: {str(e)}",
                        function_name=name,
                        invalid_params=list(e.path)
                    )

            # Execute the function based on its type
            logger.debug(f"Executing function {name}")
            try:
                if isinstance(func, Tool):
                    result = await func.execute(params)
                    if not isinstance(result, ToolResponse):
                        raise ExecutionError(
                            f"Tool {name} returned invalid response type: {type(result)}",
                            function_name=name
                        )
                elif isinstance(func, Filter):
                    result = await func.execute(params)
                    if not isinstance(result, FilterResponse):
                        raise ExecutionError(
                            f"Filter {name} returned invalid response type: {type(result)}",
                            function_name=name
                        )
                elif isinstance(func, Pipeline):
                    result = await func.execute(params)
                    if not isinstance(result, PipelineResponse):
                        raise ExecutionError(
                            f"Pipeline {name} returned invalid response type: {type(result)}",
                            function_name=name
                        )
                else:
                    raise ExecutionError(
                        f"Unknown function type for {name}: {type(func)}",
                        function_name=name
                    )

                logger.debug(f"Function {name} executed successfully")
                return result

            except Exception as e:
                if isinstance(e, FunctionError):
                    raise
                raise ExecutionError(
                    f"Error executing function {name}: {str(e)}",
                    function_name=name,
                    original_error=e
                )

        except FunctionError:
            # Re-raise known function errors
            raise
        except Exception as e:
            # Wrap unknown errors
            logger.error(
                f"Unexpected error executing {name}: {str(e)}", exc_info=True)
            raise ExecutionError(
                f"Unexpected error executing {name}: {str(e)}",
                function_name=name,
                original_error=e
            )

    async def handle_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Handle multiple tool calls in sequence.

        Args:
            tool_calls: List of tool call specifications

        Returns:
            List of results from each tool call
        """
        try:
            logger.info(f"Handling {len(tool_calls)} tool calls")
            logger.debug(f"Tool calls: {json.dumps(tool_calls, indent=2)}")

            if not tool_calls:
                return []

            results = []
            for idx, tool_call in enumerate(tool_calls):
                try:
                    logger.info(
                        f"Processing tool call {idx + 1}/{len(tool_calls)}")
                    logger.debug(
                        f"Tool call details: {json.dumps(tool_call, indent=2)}")

                    # Validate tool call format
                    if not isinstance(tool_call, dict):
                        raise ValidationError(
                            f"Invalid tool call format: expected dict, got {type(tool_call)}")

                    # Extract tool call details
                    function_name = tool_call.get("name")
                    if not function_name:
                        raise ValidationError(
                            "Tool call missing function name")

                    arguments = tool_call.get("arguments", {})
                    if not isinstance(arguments, dict):
                        raise ValidationError(
                            f"Invalid arguments format: expected dict, got {type(arguments)}")

                    # Execute the function
                    result = await self.execute(function_name, arguments)

                    # Format the result based on response type
                    formatted_result = {
                        "tool_call_id": tool_call.get("id"),
                        "name": function_name,
                    }

                    if isinstance(result, (ToolResponse, FilterResponse, PipelineResponse)):
                        if result.success:
                            if isinstance(result, ToolResponse):
                                formatted_result["result"] = result.result
                            elif isinstance(result, FilterResponse):
                                formatted_result["result"] = result.modified_data
                            else:  # PipelineResponse
                                formatted_result["result"] = result.results
                        else:
                            formatted_result["error"] = {
                                "type": "ExecutionError",
                                "message": result.error or "Unknown error"
                            }
                    else:
                        formatted_result["error"] = {
                            "type": "InvalidResponseType",
                            "message": f"Function returned invalid response type: {type(result)}"
                        }

                    results.append(formatted_result)
                    logger.debug(f"Tool call {idx + 1} completed successfully")

                except FunctionError as e:
                    # Handle known function errors
                    error_result = {
                        "tool_call_id": tool_call.get("id"),
                        "name": tool_call.get("name"),
                        "error": {
                            "type": e.__class__.__name__,
                            "message": str(e)
                        }
                    }
                    results.append(error_result)
                    logger.error(f"Error in tool call {idx + 1}: {str(e)}")

                except Exception as e:
                    # Handle unexpected errors
                    error_result = {
                        "tool_call_id": tool_call.get("id"),
                        "name": tool_call.get("name"),
                        "error": {
                            "type": "UnexpectedError",
                            "message": str(e)
                        }
                    }
                    results.append(error_result)
                    logger.error(
                        f"Unexpected error in tool call {idx + 1}: {str(e)}", exc_info=True)

            logger.info(f"Completed processing {len(tool_calls)} tool calls")
            return results

        except Exception as e:
            logger.error(f"Error handling tool calls: {str(e)}", exc_info=True)
            raise ExecutionError(f"Error handling tool calls: {str(e)}")


# Global executor instance
executor = FunctionExecutor()
