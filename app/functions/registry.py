"""Function registry for managing and discovering functions."""

import asyncio
import importlib
import logging
import json
import os
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


def pydantic_field_exists(func_cls, field_name: str) -> bool:
    """Check if a Pydantic model has a given field (works in v1 or v2)."""
    if hasattr(func_cls, "model_fields"):  # Pydantic v2
        return field_name in func_cls.model_fields
    else:  # Pydantic v1
        return field_name in func_cls.__fields__


def get_field_default(func_cls, field_name: str):
    """Retrieve a field's default (works in v1 or v2)."""
    if hasattr(func_cls, "model_fields"):  # v2
        return func_cls.model_fields[field_name].default
    else:  # v1
        return func_cls.__fields__[field_name].default


def set_field_default(func_cls, field_name: str, value):
    """Set a field's default (works in v1 or v2)."""
    if hasattr(func_cls, "model_fields"):  # v2
        func_cls.model_fields[field_name].default = value
    else:  # v1
        func_cls.__fields__[field_name].default = value
        # Also update field_info for v1
        func_cls.__fields__[field_name].field_info.default = value


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
            logger.info("Created new FunctionRegistry instance")
            logger.debug("Initial functions: {}")
        return cls._instance

    def register(self, function_class: Type[BaseFunction]) -> None:
        """Register a function class.

        Args:
            function_class: Class to register
        """
        logger.info(
            f"[REGISTER] Starting registration for class: {function_class.__name__}")
        logger.debug(
            f"[REGISTER] Current functions before registration: {list(self._functions.keys())}")

        try:
            # Check if class has required fields
            if not (hasattr(function_class, 'model_fields') or hasattr(function_class, '__fields__')):
                logger.error(
                    f"[REGISTER] Class {function_class.__name__} has no model_fields or __fields__")
                raise ValueError(
                    f"Function class {function_class.__name__} must be a Pydantic model")

            if not pydantic_field_exists(function_class, 'name'):
                logger.error(
                    f"[REGISTER] Class {function_class.__name__} missing name field")
                logger.debug(
                    f"[REGISTER] Available fields: {list(function_class.model_fields.keys() if hasattr(function_class, 'model_fields') else function_class.__fields__.keys())}")
                raise ValueError(
                    f"Function class {function_class.__name__} must have a name field")

            # Get default name from the field
            name = get_field_default(function_class, 'name')
            logger.debug(f"[REGISTER] Got name from field: {name}")

            if not name:
                logger.error(
                    f"[REGISTER] Class {function_class.__name__} has no default name")
                raise ValueError(
                    f"Function class {function_class.__name__} must have a default name")

            # Get function type
            if not pydantic_field_exists(function_class, 'type'):
                logger.error(
                    f"[REGISTER] Class {function_class.__name__} missing type field")
                raise ValueError(
                    f"Function class {function_class.__name__} must have a type field")

            func_type = get_field_default(function_class, 'type')
            logger.info(
                f"[REGISTER] Registering function: {name} (type: {func_type}, class: {function_class.__name__})")

            # Log class details
            logger.debug(f"[REGISTER] Class details:")
            logger.debug(f"  - Name: {name}")
            logger.debug(f"  - Type: {func_type}")
            logger.debug(f"  - Base classes: {function_class.__bases__}")
            logger.debug(f"  - Module: {function_class.__module__}")
            logger.debug(
                f"  - Fields: {list(function_class.model_fields.keys() if hasattr(function_class, 'model_fields') else function_class.__fields__.keys())}")

            # Register the function
            self._functions[name] = function_class
            logger.info(
                f"[REGISTER] Successfully registered function: {name} (type: {func_type})")
            logger.info(
                f"[REGISTER] Current registered functions: {list(self._functions.keys())}")

        except Exception as e:
            logger.error(
                f"[REGISTER] Failed to register {function_class.__name__}: {str(e)}", exc_info=True)
            raise

    def get_function(self, name: str) -> Optional[Type[BaseFunction]]:
        """Get a function class by name."""
        logger.debug(f"[GET] Attempting to get function: {name}")
        func = self._functions.get(name)
        if func:
            logger.debug(
                f"[GET] Found function {name} of type {func.model_fields['type'].default}")
        else:
            logger.debug(f"[GET] Function {name} not found")
        return func

    def get_functions_by_type(self, func_type: FunctionType) -> List[Type[BaseFunction]]:
        """Get all functions of a specific type."""
        logger.debug(
            f"[GET_BY_TYPE] Looking for functions of type: {func_type}")
        functions = [
            func for func in self._functions.values()
            if func.model_fields['type'].default == func_type
        ]
        logger.debug(
            f"[GET_BY_TYPE] Found {len(functions)} functions of type {func_type}")
        return functions

    def list_functions(self) -> List[Dict[str, Any]]:
        """List all registered functions."""
        logger.info("[LIST] Listing all registered functions")
        logger.debug(f"[LIST] Raw functions dict: {self._functions}")

        functions = []
        for func in self._functions.values():
            try:
                # Get field values using the helper function that handles Pydantic v1/v2 differences
                func_type = get_field_default(func, 'type')
                name = get_field_default(func, 'name')
                description = get_field_default(func, 'description')

                # Skip if required fields are undefined
                if func_type is None or name is None:
                    logger.warning(
                        f"Skipping function with undefined required fields: {func}")
                    continue

                # Build function data with only defined values
                func_data = {
                    "name": str(name),  # Ensure name is a string
                    "type": func_type.value if isinstance(func_type, FunctionType) else str(func_type),
                }

                # Add description if defined
                if description is not None:
                    func_data["description"] = str(description)

                # Add parameters if they exist and are defined
                if pydantic_field_exists(func, 'parameters'):
                    parameters = get_field_default(func, 'parameters')
                    if parameters is not None:
                        func_data["parameters"] = parameters

                functions.append(func_data)
                logger.debug(f"[LIST] Added function: {func_data}")
            except Exception as e:
                logger.error(
                    f"[LIST] Error getting metadata for function {func}: {e}", exc_info=True)

        logger.info(f"[LIST] Found {len(functions)} functions")
        return functions

    async def discover_functions(self, directory: Path) -> None:
        """Discover and load functions from a directory."""
        logger.info(
            f"[DISCOVER] Starting function discovery in directory: {directory}")
        logger.debug(
            f"[DISCOVER] Current functions before discovery: {list(self._functions.keys())}")

        if not directory.exists():
            logger.warning(f"[DISCOVER] Directory does not exist: {directory}")
            return

        # Add the directory to Python path temporarily
        import sys
        # Go up to the workspace root
        parent_dir = str(directory.parent.parent.parent)
        logger.info(f"[DISCOVER] Adding to Python path: {parent_dir}")
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            logger.debug(f"[DISCOVER] Updated Python path: {sys.path}")

        try:
            for file_path in directory.rglob("*.py"):
                if file_path.name.startswith("_"):
                    logger.debug(f"[DISCOVER] Skipping file: {file_path}")
                    continue

                try:
                    # Construct module name from the directory structure
                    module_path = file_path.relative_to(
                        directory.parent.parent.parent)
                    module_name = str(module_path).replace(
                        "/", ".").replace("\\", ".")[:-3]
                    logger.info(
                        f"[DISCOVER] Processing module: {module_name} from file: {file_path}")

                    module = importlib.import_module(module_name)
                    logger.info(
                        f"[DISCOVER] Successfully imported module: {module_name}")
                    logger.debug(
                        f"[DISCOVER] Module attributes: {dir(module)}")

                    # Look for classes that inherit from our base classes
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        logger.debug(
                            f"[DISCOVER] Checking attribute: {attr_name} in module {module_name}")

                        if not isinstance(attr, type):
                            continue

                        try:
                            if issubclass(attr, BaseFunction):
                                logger.info(
                                    f"[DISCOVER] Found BaseFunction subclass: {attr_name}")
                                logger.debug(
                                    f"[DISCOVER] Class details: {attr.__module__}.{attr.__name__}")

                                if attr not in (BaseFunction, Filter, Tool, Pipeline):
                                    logger.info(
                                        f"[DISCOVER] Registering function class: {attr_name}")
                                    self.register(attr)
                                else:
                                    logger.debug(
                                        f"[DISCOVER] Skipping base class: {attr_name}")
                        except TypeError:
                            logger.debug(
                                f"[DISCOVER] {attr_name} is not a class, skipping")

                except Exception as e:
                    logger.error(
                        f"[DISCOVER] Error processing {file_path}: {str(e)}", exc_info=True)

        finally:
            # Remove the directory from Python path
            if parent_dir in sys.path:
                sys.path.remove(parent_dir)
                logger.info(
                    f"[DISCOVER] Removed from Python path: {parent_dir}")
                logger.debug(f"[DISCOVER] Final Python path: {sys.path}")

        logger.info(
            f"[DISCOVER] Discovery complete. Current functions: {list(self._functions.keys())}")

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

    async def load_from_config(self, config_path: Path) -> None:
        """Load function configuration from a JSON file.

        Args:
            config_path: Path to the config file
        """
        try:
            if not config_path.exists():
                logger.error(f"[CONFIG] Config file not found: {config_path}")
                raise FileNotFoundError(
                    f"Config file not found: {config_path}")

            logger.info(
                f"[CONFIG] Loading functions from config: {config_path}")
            with open(config_path) as f:
                config = json.load(f)

            if not isinstance(config, dict) or "functions" not in config:
                logger.error(
                    "[CONFIG] Invalid config format - missing 'functions' key")
                raise ValueError(
                    "Invalid config format - missing 'functions' key")

            # Add config directory's parent to Python path temporarily
            import sys
            config_parent = str(config_path.parent.parent.parent)
            logger.info(f"[CONFIG] Adding to Python path: {config_parent}")
            logger.debug(f"[CONFIG] Current working directory: {os.getcwd()}")
            logger.debug(
                f"[CONFIG] Absolute config path: {config_path.absolute()}")

            # Store the original Python path
            sys_path_copy = sys.path.copy()
            logger.debug("[CONFIG] Original Python path:")
            for path in sys_path_copy:
                logger.debug(f"  - {path}")

            # Add the config parent directory at the start if not already there
            if config_parent not in sys.path:
                sys.path.insert(0, config_parent)
                logger.info(f"[CONFIG] Added {config_parent} to Python path")

            logger.debug("[CONFIG] Updated Python path:")
            for path in sys.path:
                logger.debug(f"  - {path}")

            try:
                logger.info(
                    f"[CONFIG] Found {len(config['functions'])} function definitions")
                for func_config in config["functions"]:
                    try:
                        logger.debug(
                            f"[CONFIG] Processing function config: {json.dumps(func_config, indent=2)}")

                        # Check required fields
                        required = ["name", "module_path",
                                    "function_name", "type", "parameters"]
                        missing = [
                            field for field in required if field not in func_config]
                        if missing:
                            logger.error(
                                f"[CONFIG] Missing required fields in function config: {missing}")
                            logger.debug(
                                f"[CONFIG] Function config: {json.dumps(func_config, indent=2)}")
                            continue

                        if not func_config.get("enabled", True):
                            logger.info(
                                f"[CONFIG] Skipping disabled function: {func_config['name']}")
                            continue

                        # Check dependencies
                        dependencies = func_config.get("dependencies", [])
                        if not await self.check_dependencies(dependencies):
                            logger.error(
                                f"[CONFIG] Dependencies not satisfied for function {func_config['name']}: {dependencies}")
                            continue

                        # Import and register the function
                        try:
                            logger.info(
                                f"[CONFIG] Importing module {func_config['module_path']}")

                            # Handle relative imports by providing the package argument
                            if func_config["module_path"].startswith("."):
                                logger.debug(
                                    "[CONFIG] Handling relative import")
                                logger.debug(
                                    f"[CONFIG] Module path: {func_config['module_path']}")
                                logger.debug(
                                    f"[CONFIG] Package: app.functions")

                                # Try to verify the module exists before importing
                                module_path = func_config["module_path"]
                                package = "app.functions"

                                # For relative imports, we need to keep the relative path
                                # but ensure we're using the correct package
                                try:
                                    logger.debug(
                                        "[CONFIG] Attempting relative import")
                                    module = importlib.import_module(
                                        module_path, package=package)
                                    logger.info(
                                        "[CONFIG] Relative import successful")
                                except ImportError as e:
                                    logger.debug(
                                        f"[CONFIG] Relative import failed: {e}")
                                    # If relative import fails, try absolute import
                                    absolute_module_path = package + \
                                        module_path[1:]
                                    logger.debug(
                                        f"[CONFIG] Trying absolute import: {absolute_module_path}")
                                    module = importlib.import_module(
                                        absolute_module_path)
                                    logger.info(
                                        "[CONFIG] Absolute import successful")
                            else:
                                module = importlib.import_module(
                                    func_config["module_path"])
                                logger.info(
                                    "[CONFIG] Absolute import successful")

                            logger.debug(
                                f"[CONFIG] Module contents: {dir(module)}")

                            if hasattr(module, func_config["function_name"]):
                                func_class = getattr(
                                    module, func_config["function_name"])

                                # Update the class's model fields with the config values
                                if hasattr(func_class, 'model_fields') or hasattr(func_class, '__fields__'):
                                    # Update parameters schema
                                    if pydantic_field_exists(func_class, 'parameters'):
                                        set_field_default(
                                            func_class, 'parameters', func_config['parameters'])

                                    # Update description
                                    if pydantic_field_exists(func_class, 'description') and 'description' in func_config:
                                        set_field_default(
                                            func_class, 'description', func_config['description'])

                                    # Update output schema if provided
                                    if 'output_schema' in func_config and hasattr(func_class, 'output_schema'):
                                        func_class.output_schema = func_config['output_schema']

                                logger.info(
                                    f"[CONFIG] Registering function class: {func_config['function_name']}")
                                logger.debug(
                                    f"[CONFIG] Function parameters: {func_config['parameters']}")
                                if 'output_schema' in func_config:
                                    logger.debug(
                                        f"[CONFIG] Function output schema: {func_config['output_schema']}")

                                self.register(func_class)
                            else:
                                logger.error(
                                    f"[CONFIG] Function {func_config['function_name']} not found in module {func_config['module_path']}")
                                logger.debug(
                                    f"[CONFIG] Available module attributes: {dir(module)}")
                                raise AttributeError(
                                    f"Function {func_config['function_name']} not found in module {func_config['module_path']}")
                        except ImportError as e:
                            logger.error(
                                f"[CONFIG] Error importing module {func_config['module_path']}: {e}", exc_info=True)
                            raise ModuleImportError(
                                f"Error importing module {func_config['module_path']}: {e}") from e
                        except Exception as e:
                            logger.error(
                                f"[CONFIG] Error registering function {func_config['name']}: {e}", exc_info=True)
                            raise FunctionError(
                                f"Error registering function {func_config['name']}: {e}") from e

                    except Exception as e:
                        logger.error(
                            f"[CONFIG] Error processing function config: {e}", exc_info=True)
                        raise FunctionError(
                            f"Error processing function config: {e}") from e

            finally:
                # Restore the original Python path
                sys.path = sys_path_copy
                logger.info("[CONFIG] Restored original Python path")
                logger.debug("[CONFIG] Final Python path:")
                for path in sys.path:
                    logger.debug(f"  - {path}")

        except Exception as e:
            logger.error(
                f"[CONFIG] Error loading config file: {e}", exc_info=True)
            raise


# Global registry instance
function_registry = FunctionRegistry()

__all__ = ['FunctionRegistry', 'function_registry']
