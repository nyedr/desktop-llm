"""Parameter normalization filter for function calls."""

import logging
from typing import Dict, Any, ClassVar, List, Union
from app.models.function import Filter, FunctionType, register_function

logger = logging.getLogger(__name__)


@register_function(
    func_type=FunctionType.FILTER,
    name="parameter_normalizer",
    description="Normalizes function parameters before execution",
    priority=1  # High priority to run before other filters
)
class ParameterNormalizerFilter(Filter):
    """Filter that normalizes function parameters before they reach the function."""

    COMMON_NORMALIZATIONS: ClassVar[Dict[str, Dict[Union[str, bool], List[str]]]] = {
        "temperature_units": {
            "celsius": ["Celsius", "CELSIUS", "C", "c"],
            "fahrenheit": ["Fahrenheit", "FAHRENHEIT", "F", "f"]
        },
        "boolean_values": {
            True: ["true", "True", "TRUE", "1", "yes", "Yes", "YES"],
            False: ["false", "False", "FALSE", "0", "no", "No", "NO"]
        }
    }

    async def inlet(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming data before it reaches the LLM.

        Args:
            data: Dictionary containing messages and tool calls

        Returns:
            Modified data with normalized parameters
        """
        try:
            # Check for tool calls in the data
            if "tool_calls" in data:
                tool_calls = data["tool_calls"]
                for tool_call in tool_calls:
                    if "function" in tool_call:
                        # Normalize function arguments
                        args = tool_call["function"].get("arguments", {})
                        normalized_args = self._normalize_parameters(
                            args,
                            tool_call["function"].get("name")
                        )
                        tool_call["function"]["arguments"] = normalized_args

                        logger.debug(
                            f"Normalized parameters for {tool_call['function'].get('name')}: "
                            f"{args} -> {normalized_args}"
                        )

            return data

        except Exception as e:
            logger.error(
                f"Error in parameter normalization: {e}", exc_info=True)
            # Return original data if normalization fails
            return data

    async def outlet(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process outgoing data (pass-through).

        Args:
            data: Response data

        Returns:
            Unmodified response data
        """
        return data

    def _normalize_parameters(self, args: Dict[str, Any], function_name: str) -> Dict[str, Any]:
        """Normalize function parameters based on common patterns.

        Args:
            args: Original function arguments
            function_name: Name of the function being called

        Returns:
            Normalized arguments
        """
        normalized = args.copy()

        # Normalize temperature units
        if "unit" in normalized:
            unit_value = str(normalized["unit"]).lower()
            # Check against temperature unit normalizations
            for standard, variants in self.COMMON_NORMALIZATIONS["temperature_units"].items():
                if unit_value in [v.lower() for v in variants]:
                    normalized["unit"] = standard
                    break

        # Normalize boolean values
        for key, value in normalized.items():
            if isinstance(value, str):
                # Check against boolean normalizations
                for standard, variants in self.COMMON_NORMALIZATIONS["boolean_values"].items():
                    if value.lower() in [v.lower() for v in variants]:
                        normalized[key] = standard
                        break

        # Function-specific normalizations
        if function_name == "get_current_weather":
            self._normalize_weather_params(normalized)

        return normalized

    def _normalize_weather_params(self, args: Dict[str, Any]) -> None:
        """Normalize weather-specific parameters.

        Args:
            args: Weather function arguments to normalize
        """
        # Normalize location names (basic example)
        if "location" in args:
            location = args["location"]
            # Remove extra spaces and standardize format
            location = " ".join(location.split())
            # Capitalize major words
            args["location"] = location.title()
