"""Function management endpoints."""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Request, Depends, HTTPException
from pydantic import BaseModel, Field
import importlib
import json
from pathlib import Path

from app.dependencies.providers import Providers
from app.services.function_service import FunctionService
from app.functions.registry import function_registry

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/functions",
    tags=["functions"],
    responses={
        400: {"description": "Bad request - Invalid input parameters"},
        404: {"description": "Function not found"},
        500: {"description": "Internal server error"},
        429: {"description": "Too many requests - Rate limit exceeded"},
    }
)


class RegisterFunctionRequest(BaseModel):
    """Request model for registering a function."""
    name: str = Field(..., description="Name of the function")
    module_path: str = Field(...,
                             description="Python module path where the function is located")
    function_name: str = Field(...,
                               description="Name of the function in the module")
    description: str = Field(
        None, description="Description of what the function does")
    input_schema: Dict[str, Any] = Field(
        None, description="JSON schema for function inputs")
    output_schema: Dict[str, Any] = Field(
        None, description="JSON schema for function outputs")


class ExecuteFunctionRequest(BaseModel):
    """Request model for executing a function."""
    name: str = Field(...,
                      description="Name of the registered function to execute")
    arguments: Dict[str, Any] = Field(...,
                                      description="Arguments to pass to the function")
    timeout: int = Field(30, description="Execution timeout in seconds")


class FunctionResponse(BaseModel):
    """Response model for function data."""
    name: str = Field(..., description="Name of the function")
    type: str = Field(...,
                      description="Type of the function (tool/filter/pipeline)")
    description: Optional[str] = Field(
        None, description="Function description")
    parameters: Optional[Dict[str, Any]] = Field(
        None, description="Function parameters schema")


@router.post("/register",
             response_model=dict,
             summary="Register Function",
             description="""
    Register a new function in the system.
    
    This endpoint allows you to register a new function that can be called by the language model.
    The function must be defined in a Python module and follow the required interface.
    
    Features:
    - Dynamic function registration
    - Input/output schema validation
    - Automatic function discovery
    - Configuration persistence
    """,
             response_description="Registration status and confirmation",
             responses={
                 200: {
                     "description": "Function registered successfully",
                     "content": {
                         "application/json": {
                             "example": {
                                 "status": "success",
                                 "message": "Function example_function registered successfully"
                             }
                         }
                     }
                 }
             }
             )
async def register_function(request: RegisterFunctionRequest):
    """Register a new function.

    Args:
        request: The registration request containing function details

    Returns:
        A dictionary containing registration status and confirmation message

    Raises:
        HTTPException: If registration fails or there are validation errors
    """
    try:
        # Load current config
        config_path = Path("app/functions/config.json")
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = {"functions": []}

        # Convert request to dict and add to config
        function_config = request.model_dump()

        # Check if function already exists
        existing_idx = next(
            (i for i, f in enumerate(
                config["functions"]) if f["name"] == request.name),
            None
        )

        if existing_idx is not None:
            # Update existing function
            config["functions"][existing_idx] = function_config
            logger.info(
                f"Updated existing function configuration: {request.name}")
        else:
            # Add new function
            config["functions"].append(function_config)
            logger.info(f"Added new function configuration: {request.name}")

        # Save updated config
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Register the function
        try:
            module = importlib.import_module(request.module_path)
            if hasattr(module, request.function_name):
                func_class = getattr(module, request.function_name)

                # Update the class's model fields with the config values
                if hasattr(func_class, 'model_fields'):
                    # Update parameters schema
                    if 'parameters' in func_class.model_fields:
                        func_class.model_fields['parameters'].default = request.parameters

                    # Update description
                    if 'description' in func_class.model_fields:
                        func_class.model_fields['description'].default = request.description

                    # Update output schema if provided
                    if request.output_schema and hasattr(func_class, 'output_schema'):
                        func_class.output_schema = request.output_schema

                function_registry.register(func_class)
                logger.info(
                    f"Successfully registered function: {request.name}")
                return {"status": "success", "message": f"Function {request.name} registered successfully"}
            else:
                raise ValueError(
                    f"Function {request.function_name} not found in module {request.module_path}")
        except Exception as e:
            logger.error(f"Error registering function: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error registering function: {str(e)}"
            )

    except Exception as e:
        logger.error(f"Error in register_function: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error registering function: {str(e)}"
        )


@router.delete("/{name}",
               summary="Unregister Function",
               description="""
    Remove a registered function from the system.
    
    This endpoint unregisters a function, making it unavailable for future calls.
    The function's configuration will be removed from the system.
    """,
               response_description="Unregistration status and confirmation",
               responses={
                   200: {
                       "description": "Function unregistered successfully",
                       "content": {
                           "application/json": {
                               "example": {
                                   "status": "success",
                                   "message": "Function example_function unregistered successfully"
                               }
                           }
                       }
                   }
               }
               )
async def unregister_function(
    name: str,
    function_service: FunctionService = Depends(Providers.get_function_service)
) -> Dict[str, Any]:
    """Unregister a function.

    Args:
        name: The name of the function to unregister
        function_service: The function service for managing functions

    Returns:
        A dictionary containing unregistration status and confirmation message

    Raises:
        HTTPException: If the function is not found or there are errors during unregistration
    """
    logger.info(f"Unregistering function: {name}")
    try:
        if function_service.unregister_function(name):
            logger.info(f"Function {name} unregistered successfully")
            return {"status": "success", "message": f"Function {name} unregistered successfully"}
        logger.warning(f"Function {name} not found")
        raise HTTPException(
            status_code=404, detail=f"Function {name} not found")
    except Exception as e:
        logger.error(f"Error unregistering function: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("",
            response_model=List[FunctionResponse],
            summary="List Functions",
            description="""
    List all registered functions in the system.
    
    This endpoint returns information about all registered functions, including their:
    - Name and type
    - Description
    - Input/output parameters
    - Configuration
    """,
            response_description="List of registered function details",
            responses={
                200: {
                    "description": "List of functions retrieved successfully",
                    "content": {
                        "application/json": {
                            "example": [
                                {
                                    "name": "example_function",
                                    "type": "tool",
                                    "description": "An example function",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "input": {"type": "string"}
                                        }
                                    }
                                }
                            ]
                        }
                    }
                }
            }
            )
async def list_functions(
    request: Request,
    function_service: FunctionService = Depends(Providers.get_function_service)
) -> List[Dict[str, Any]]:
    """List all registered functions.

    Args:
        request: The FastAPI request object
        function_service: The function service for managing functions

    Returns:
        A list of dictionaries containing function details

    Raises:
        HTTPException: If there are errors retrieving the function list
    """
    logger.info(
        f"[{request.state.request_id}] Listing all registered functions")
    try:
        logger.debug(
            f"[{request.state.request_id}] Getting function service instance")
        functions = function_service.list_functions()
        logger.info(
            f"[{request.state.request_id}] Found {len(functions)} registered functions")
        logger.debug(f"[{request.state.request_id}] Functions: {functions}")
        return functions
    except Exception as e:
        logger.error(
            f"[{request.state.request_id}] Error listing functions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute",
             summary="Execute Function",
             description="""
    Execute a registered function with provided arguments.
    
    This endpoint allows you to execute any registered function by providing:
    - Function name
    - Required arguments
    - Optional timeout
    
    The function will be executed asynchronously and its result returned.
    """,
             response_description="Function execution result",
             responses={
                 200: {
                     "description": "Function executed successfully",
                     "content": {
                         "application/json": {
                             "example": {
                                 "status": "success",
                                 "result": {"output": "Function result"}
                             }
                         }
                     }
                 }
             }
             )
async def execute_function(
    request: ExecuteFunctionRequest,
    function_service: FunctionService = Depends(Providers.get_function_service)
) -> Dict[str, Any]:
    """Execute a registered function.

    Args:
        request: The execution request containing function name and arguments
        function_service: The function service for executing functions

    Returns:
        A dictionary containing the execution status and result

    Raises:
        HTTPException: If the function execution fails or times out
    """
    logger.info(f"Executing function: {request.name}")
    try:
        result = await function_service.execute_function(
            function_name=request.name,
            arguments=request.arguments,
            timeout=request.timeout
        )
        logger.info(f"Function {request.name} executed successfully")
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error executing function: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
