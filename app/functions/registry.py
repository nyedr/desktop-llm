"""Function registry for managing and discovering functions."""

import logging
from typing import Dict, Type, List, Optional, Any

from app.models.function import ValidationError
from app.models.function_base import (
    BaseFunction,
    Filter,
    Pipeline,
    Tool,
    FunctionType,
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


# Global registry instance
function_registry = FunctionRegistry()

__all__ = ['FunctionRegistry', 'function_registry']
