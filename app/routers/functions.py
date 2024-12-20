"""Function management endpoints."""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Request, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from app.dependencies.providers import get_function_service
from app.services.function_service import FunctionService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/functions", tags=["functions"])

class RegisterFunctionRequest(BaseModel):
    """Request model for registering a function."""
    name: str = Field(..., description="Name of the function")
    module_path: str = Field(..., description="Python module path where the function is located")
    function_name: str = Field(..., description="Name of the function in the module")
    description: str = Field(None, description="Description of what the function does")
    input_schema: Dict[str, Any] = Field(None, description="JSON schema for function inputs")
    output_schema: Dict[str, Any] = Field(None, description="JSON schema for function outputs")

class ExecuteFunctionRequest(BaseModel):
    """Request model for executing a function."""
    name: str = Field(..., description="Name of the registered function to execute")
    arguments: Dict[str, Any] = Field(..., description="Arguments to pass to the function")
    timeout: int = Field(30, description="Execution timeout in seconds")

@router.post("/register")
async def register_function(
    request: RegisterFunctionRequest,
    function_service: FunctionService = Depends(get_function_service)
) -> Dict[str, Any]:
    """Register a function for execution."""
    logger.info(f"Registering function: {request.name}")
    try:
        function_service.register_function(
            name=request.name,
            module_path=request.module_path,
            function_name=request.function_name,
            description=request.description,
            input_schema=request.input_schema,
            output_schema=request.output_schema
        )
        logger.info(f"Function {request.name} registered successfully")
        return {"status": "success", "message": f"Function {request.name} registered successfully"}
    except Exception as e:
        logger.error(f"Error registering function: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

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
        raise HTTPException(status_code=404, detail=f"Function {name} not found")
    except Exception as e:
        logger.error(f"Error unregistering function: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("")
async def list_functions(
    request: Request,
    function_service: FunctionService = Depends(get_function_service)
) -> List[Dict[str, Any]]:
    """List all registered functions."""
    logger.info(f"[{request.state.request_id}] Listing all registered functions")
    try:
        logger.debug(f"[{request.state.request_id}] Getting function service instance")
        functions = function_service.list_functions()
        logger.info(f"[{request.state.request_id}] Found {len(functions)} registered functions")
        logger.debug(f"[{request.state.request_id}] Functions: {functions}")
        return functions
    except Exception as e:
        logger.error(f"[{request.state.request_id}] Error listing functions: {e}", exc_info=True)
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
