"""Function management endpoints."""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Request, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
import importlib
import json
from pathlib import Path

from app.dependencies.providers import get_function_service
from app.services.function_service import FunctionService
from app.models.function_models import RegisterFunctionRequest
from app.functions.registry import function_registry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/functions", tags=["functions"])


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
    name: str
    type: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


@router.post("/register", response_model=dict)
async def register_function(request: RegisterFunctionRequest):
    """Register a new function."""
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


@router.delete("/{name}")
async def unregister_function(
    name: str,
    function_service: FunctionService = Depends(get_function_service)
) -> Dict[str, Any]:
    """Unregister a function."""
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


@router.get("", response_model=List[FunctionResponse])
async def list_functions(
    request: Request,
    function_service: FunctionService = Depends(get_function_service)
) -> List[Dict[str, Any]]:
    """List all registered functions."""
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


@router.post("/execute")
async def execute_function(
    request: ExecuteFunctionRequest,
    function_service: FunctionService = Depends(get_function_service)
) -> Dict[str, Any]:
    """Execute a registered function."""
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
