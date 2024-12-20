"""Function registry for managing and discovering functions."""

import asyncio
import importlib
import logging
import json
from typing import Dict, Type, List, Optional, Any
from pathlib import Path

from app.functions.base import (
    BaseFunction,
    Filter,
    Tool,
    Pipeline,
    FunctionType,
    FunctionError,
    FunctionNotFoundError,
    ModuleImportError
)

logger = logging.getLogger(__name__)

class FunctionRegistry:
    """Registry for managing all available functions."""
    
    _instance = None
    _functions: Dict[str, Type[BaseFunction]] = {}
    _dependency_cache: Dict[str, bool] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FunctionRegistry, cls).__new__(cls)
            cls._instance._functions = {}
            cls._instance._dependency_cache = {}
        return cls._instance

    def register(self, function_class: Type[BaseFunction]) -> None:
        """Register a function class.
        
        Args:
            function_class: Class to register
        """
        if 'name' not in function_class.model_fields:
            raise ValueError(f"Function class {function_class.__name__} must have a name field")
        
        # Get default name from the field
        name = function_class.model_fields['name'].default
        if not name:
            raise ValueError(f"Function class {function_class.__name__} must have a default name")
        
        self._functions[name] = function_class
        logger.info(f"Registered function: {name}")
    
    def get_function(self, name: str) -> Optional[Type[BaseFunction]]:
        """Get a function class by name.
        
        Args:
            name: Name of the function
            
        Returns:
            Function class if found, None otherwise
        """
        return self._functions.get(name)
    
    def get_functions_by_type(self, func_type: FunctionType) -> List[Type[BaseFunction]]:
        """Get all functions of a specific type.
        
        Args:
            func_type: Type of functions to get
            
        Returns:
            List of function classes
        """
        return [
            func for func in self._functions.values()
            if func.type == func_type
        ]
    
    def list_functions(self) -> List[Dict[str, Any]]:
        """List all registered functions.
        
        Returns:
            List of function metadata
        """
        return [{
            "name": func.model_fields['name'].default,
            "description": func.model_fields['description'].default,
            "type": func.model_fields['type'].default,
            "parameters": func.model_fields['parameters'].default if 'parameters' in func.model_fields else None
        } for func in self._functions.values()]

    async def check_dependencies(self, dependencies: List[str]) -> bool:
        """Check if all dependencies are available.
        
        Args:
            dependencies: List of package names to check
            
        Returns:
            True if all dependencies are satisfied
        """
        for dep in dependencies:
            if dep in self._dependency_cache:
                if not self._dependency_cache[dep]:
                    return False
                continue

            try:
                importlib.import_module(dep)
                self._dependency_cache[dep] = True
            except ImportError:
                self._dependency_cache[dep] = False
                return False
        return True

    async def discover_functions(self, directory: Path) -> None:
        """Discover and load functions from a directory.
        
        Args:
            directory: Directory to search for functions
        """
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return

        for file_path in directory.rglob("*.py"):
            if file_path.name.startswith("_"):
                continue

            module_name = str(file_path.relative_to(directory.parent)).replace("/", ".").replace("\\", ".")[:-3]
            
            try:
                module = importlib.import_module(module_name)
                
                # Look for classes that inherit from our base classes
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, BaseFunction) and 
                        attr not in (BaseFunction, Filter, Tool, Pipeline)):
                        self.register(attr)
                        
            except Exception as e:
                logger.error(f"Error loading module {module_name}: {e}")

    async def load_from_config(self, config_path: Path) -> None:
        """Load function configuration from a JSON file.
        
        Args:
            config_path: Path to the config file
        """
        try:
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            with open(config_path) as f:
                config = json.load(f)

            if not isinstance(config, dict) or "functions" not in config:
                raise ValueError("Invalid config format")

            for func_config in config["functions"]:
                # Check required fields
                required = ["name", "type", "description"]
                if not all(field in func_config for field in required):
                    logger.error(f"Missing required fields in function config: {func_config}")
                    continue

                # Check dependencies
                dependencies = func_config.get("dependencies", [])
                if not await self.check_dependencies(dependencies):
                    logger.error(f"Dependencies not satisfied for function {func_config['name']}")
                    continue

                # Register the function if it has a module path
                if "module_path" in func_config:
                    try:
                        module = importlib.import_module(func_config["module_path"])
                        if hasattr(module, func_config["name"]):
                            self.register(getattr(module, func_config["name"]))
                    except Exception as e:
                        logger.error(f"Error loading function {func_config['name']}: {e}")

        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            raise

# Global registry instance
registry = FunctionRegistry()
