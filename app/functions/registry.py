"""Function registry for managing and discovering functions."""

import importlib
import logging
import json
import sys
from typing import Dict, Type, List, Optional, Any
from pathlib import Path

from app.functions.base import (
    BaseFunction,
    Filter,
    Tool,
    Pipeline,
    FunctionType,
    ModuleImportError,
    ValidationError,
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
        return cls._instance

    def _validate_function_class(self, function_class: Type[BaseFunction]) -> None:
        """Validate a function class meets all requirements."""
        try:
            # Check if class has required fields
            if not (hasattr(function_class, 'model_fields') or hasattr(function_class, '__fields__')):
                raise ValidationError(
                    f"Function class {function_class.__name__} must be a Pydantic model")

            # Validate required fields
            required_fields = ['name', 'type', 'description', 'parameters']
            for field in required_fields:
                if not pydantic_field_exists(function_class, field):
                    raise ValidationError(
                        f"Function class {function_class.__name__} missing required field: {field}")

            # Validate function type
            func_type = get_field_default(function_class, 'type')
            if not isinstance(func_type, FunctionType):
                raise ValidationError(
                    f"Invalid function type for {function_class.__name__}: {func_type}")

            # Validate class inheritance
            if func_type == FunctionType.TOOL and not issubclass(function_class, Tool):
                raise ValidationError(
                    f"Tool function {function_class.__name__} must inherit from Tool")
            elif func_type == FunctionType.FILTER and not issubclass(function_class, Filter):
                raise ValidationError(
                    f"Filter function {function_class.__name__} must inherit from Filter")
            elif func_type == FunctionType.PIPELINE and not issubclass(function_class, Pipeline):
                raise ValidationError(
                    f"Pipeline function {function_class.__name__} must inherit from Pipeline")

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Function validation failed: {str(e)}")

    def register(self, function_class: Type[BaseFunction]) -> None:
        """Register a function class."""
        try:
            # Validate the function class
            self._validate_function_class(function_class)

            name = get_field_default(function_class, 'name')
            func_type = get_field_default(function_class, 'type')

            # Register the function
            self._functions[name] = function_class
            logger.debug(f"Registered function: {name} (type: {func_type})")
            self._log_function_summary()

        except Exception as e:
            logger.error(
                f"Failed to register {function_class.__name__}: {str(e)}")
            raise

    def _log_function_summary(self):
        """Log a summary of all registered functions."""
        if not self._functions:
            logger.info("No functions registered")
            return

        summary = "\nRegistered Functions Summary:"
        by_type = {}
        for func in self._functions.values():
            func_type = get_field_default(func, 'type')
            if func_type not in by_type:
                by_type[func_type] = []
            by_type[func_type].append(get_field_default(func, 'name'))

        for func_type, funcs in by_type.items():
            summary += f"\n{func_type}:"
            for func in sorted(funcs):
                summary += f"\n  - {func}"

        logger.info(summary)

    def get_function(self, name: str) -> Optional[Type[BaseFunction]]:
        """Get a function class by name."""
        return self._functions.get(name)

    def get_functions_by_type(self, func_type: FunctionType) -> List[Type[BaseFunction]]:
        """Get all functions of a specific type."""
        return [
            func for func in self._functions.values()
            if func.model_fields['type'].default == func_type
        ]

    def list_functions(self) -> List[Dict[str, Any]]:
        """List all registered functions."""
        functions = []
        for func in self._functions.values():
            try:
                func_type = get_field_default(func, 'type')
                name = get_field_default(func, 'name')
                description = get_field_default(func, 'description')

                if func_type is None or name is None:
                    logger.warning(
                        f"Skipping function with undefined required fields: {func}")
                    continue

                func_data = {
                    "name": str(name),
                    "type": func_type.value if isinstance(func_type, FunctionType) else str(func_type),
                }

                if description is not None:
                    func_data["description"] = str(description)

                if pydantic_field_exists(func, 'parameters'):
                    parameters = get_field_default(func, 'parameters')
                    if parameters is not None:
                        func_data["parameters"] = parameters

                functions.append(func_data)
            except Exception as e:
                logger.error(
                    f"Error getting metadata for function {func}: {e}")

        return functions

    async def discover_functions(self, directory: Path) -> None:
        """Discover and load functions from a directory."""
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return

        parent_dir = str(directory.parent.parent.parent)

        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        try:
            logger.info(f"Discovering functions in {directory}")
            for file_path in directory.rglob("*.py"):
                if file_path.name.startswith("_"):
                    continue

                try:
                    module_path = file_path.relative_to(
                        directory.parent.parent.parent)
                    module_name = str(module_path).replace(
                        "/", ".").replace("\\", ".")[:-3]

                    module = importlib.import_module(module_name)
                    logger.debug(f"Processing module: {module_name}")

                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if not isinstance(attr, type):
                            continue

                        try:
                            if issubclass(attr, BaseFunction) and attr not in (BaseFunction, Filter, Tool, Pipeline):
                                self.register(attr)
                        except TypeError:
                            continue

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")

        finally:
            if parent_dir in sys.path:
                sys.path.remove(parent_dir)
            self._log_function_summary()

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
        """Load function configuration from a JSON file."""
        try:
            with open(config_path) as f:
                config = json.load(f)

            if not isinstance(config, dict) or "functions" not in config:
                raise ValueError(
                    "Invalid config format - missing 'functions' key")

            config_parent = str(config_path.parent)
            sys_path_copy = sys.path.copy()

            if config_parent not in sys.path:
                sys.path.insert(0, config_parent)

            try:
                logger.info(
                    f"Loading {len(config['functions'])} functions from config")
                for func_config in config["functions"]:
                    try:
                        # Check required fields
                        required = ["name", "module_path",
                                    "function_name", "type", "parameters"]
                        missing = [
                            field for field in required if field not in func_config]
                        if missing:
                            logger.error(
                                f"Missing required fields in function config: {missing}")
                            continue

                        if not func_config.get("enabled", True):
                            continue

                        # Check dependencies
                        dependencies = func_config.get("dependencies", [])
                        if not await self.check_dependencies(dependencies):
                            logger.error(
                                f"Dependencies not satisfied for function {func_config['name']}")
                            continue

                        # Import and register the function
                        try:
                            module_path = func_config["module_path"]
                            package = "app.functions" if module_path.startswith(
                                ".") else None

                            try:
                                module = importlib.import_module(
                                    module_path, package=package)
                            except ImportError as e:
                                if package:
                                    absolute_module_path = package + \
                                        module_path[1:]
                                    module = importlib.import_module(
                                        absolute_module_path)

                            if hasattr(module, func_config["function_name"]):
                                func_class = getattr(
                                    module, func_config["function_name"])

                                # Update the class's model fields with the config values
                                if hasattr(func_class, 'model_fields') or hasattr(func_class, '__fields__'):
                                    if pydantic_field_exists(func_class, 'parameters'):
                                        set_field_default(
                                            func_class, 'parameters', func_config['parameters'])
                                    if pydantic_field_exists(func_class, 'description') and 'description' in func_config:
                                        set_field_default(
                                            func_class, 'description', func_config['description'])
                                    if 'output_schema' in func_config and hasattr(func_class, 'output_schema'):
                                        func_class.output_schema = func_config['output_schema']

                                self.register(func_class)
                            else:
                                raise AttributeError(
                                    f"Function {func_config['function_name']} not found in module {func_config['module_path']}")

                        except ImportError as e:
                            raise ModuleImportError(
                                f"Error importing module {func_config['module_path']}: {e}") from e

                    except Exception as e:
                        logger.error(
                            f"Failed to register function {func_config.get('name', 'unknown')}: {str(e)}")
                        continue

            finally:
                sys.path = sys_path_copy
                self._log_function_summary()

        except Exception as e:
            logger.error(f"Failed to load config file: {str(e)}")
            raise


# Global registry instance
function_registry = FunctionRegistry()

__all__ = ['FunctionRegistry', 'function_registry']
