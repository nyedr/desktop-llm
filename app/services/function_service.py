"""Function service module."""

import logging
from typing import Dict, Any, Optional, List, Type
from app.functions import registry, executor
from app.functions.base import BaseFunction, Tool, FunctionType

logger = logging.getLogger(__name__)

class FunctionService:
    """Service for managing and executing functions."""
    
    def __init__(self):
        self.registry = registry
        self.executor = executor
        logger.info("Initializing FunctionService")
        
    def register_function(
        self,
        name: str,
        description: Optional[str] = None,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        function_class: Optional[type] = None
    ) -> None:
        """Register a function for execution."""
        logger.info(f"Registering function: {name}")
        
        try:
            if function_class:
                # If a class is provided directly, register it
                self.registry.register(function_class)
            else:
                # Create a dynamic Tool class
                class DynamicTool(Tool):
                    type = FunctionType.PIPE
                    name = name
                    description = description or ""
                    parameters = input_schema or {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                    
                    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
                        return args  # Placeholder implementation
                
                self.registry.register(DynamicTool)
            
            logger.info(f"Successfully registered function: {name}")
            logger.debug(f"Function schema - Input: {input_schema}, Output: {output_schema}")
            
        except Exception as e:
            logger.error(f"Error registering function {name}: {e}", exc_info=True)
            raise
            
    def list_functions(self) -> List[Dict[str, Any]]:
        """List all registered functions."""
        return self.registry.list_functions()
            
    def get_function_schemas(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible function schemas for all registered functions."""
        logger.debug("Getting function schemas for all registered functions")
        schemas = []
        
        for func in self.registry.list_functions():
            if func["type"] == FunctionType.PIPE:
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
            
        logger.debug(f"Found {len(schemas)} function schemas")
        return schemas
            
    async def execute_function(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a registered function."""
        try:
            result = await self.executor.execute(name, args)
            return result
        except Exception as e:
            logger.error(f"Error executing function {name}: {e}", exc_info=True)
            raise
            
    async def handle_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Handle multiple tool calls in sequence."""
        try:
            return await self.executor.handle_tool_calls(tool_calls)
        except Exception as e:
            logger.error(f"Error handling tool calls: {e}", exc_info=True)
            raise

    def get_function(self, name: str) -> Optional[Type[BaseFunction]]:
        """Get a function class by name.
        
        Args:
            name: Name of the function to get
            
        Returns:
            Function class if found, None otherwise
        """
        try:
            return self.registry.get_function(name)
        except Exception as e:
            logger.error(f"Error getting function {name}: {e}", exc_info=True)
            return None
