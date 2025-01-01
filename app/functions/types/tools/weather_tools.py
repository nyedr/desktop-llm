"""Weather-related tools."""

import logging
from typing import ClassVar, Dict, Any, List
from app.models.function import InputValidationError, Tool, FunctionType, register_function

logger = logging.getLogger(__name__)


@register_function(
    func_type=FunctionType.TOOL,
    name="get_current_weather",
    description="Get current weather information for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "Location to get weather for"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit",
                "default": "fahrenheit"
            }
        },
        "required": ["location"]
    }
)
class WeatherTool(Tool):
    """Tool for getting weather information."""

    # Parameter normalization mappings
    UNIT_MAPPINGS: ClassVar[Dict[str, List[str]]] = {
        "celsius": ["Celsius", "CELSIUS", "C", "c", "centigrade", "metric"],
        "fahrenheit": ["Fahrenheit", "FAHRENHEIT", "F", "f", "imperial"]
    }

    def normalize_parameters(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize weather tool parameters.

        Args:
            args: Original tool arguments

        Returns:
            Normalized arguments
        """
        normalized = args.copy()

        # Normalize temperature unit
        if "unit" in normalized:
            unit = str(normalized["unit"]).lower()
            for standard, variants in self.UNIT_MAPPINGS.items():
                if unit in [v.lower() for v in variants]:
                    normalized["unit"] = standard
                    break

        # Normalize location
        if "location" in normalized:
            location = normalized["location"]
            # Remove extra spaces and standardize format
            location = " ".join(location.split())
            # Capitalize major words
            normalized["location"] = location.title()

        return normalized

    def _fix_validation_error(self, error: InputValidationError, args: Dict[str, Any]) -> Dict[str, Any]:
        """Fix common validation errors in weather parameters.

        Args:
            error: The validation error
            args: Current arguments

        Returns:
            Fixed arguments
        """
        fixed = args.copy()

        # Check if the error is about the unit parameter
        if "unit" in error.details.get("invalid_params", []):
            # Default to fahrenheit if unit is invalid
            fixed["unit"] = "fahrenheit"
            logger.info(
                f"Fixed invalid unit in weather parameters: {args['unit']} -> fahrenheit")

        return fixed

    async def _execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the weather tool with the given arguments.

        Args:
            args: Normalized tool arguments

        Returns:
            Weather information

        Raises:
            ExecutionError: If weather data cannot be retrieved
        """
        try:
            # Your existing weather API call implementation
            # This is a placeholder - replace with actual API call
            return {
                "location": args["location"],
                "temperature": 72,
                "unit": args.get("unit", "fahrenheit"),
                "conditions": "sunny",
                "humidity": 45,
                "wind_speed": 8
            }

        except Exception as e:
            logger.error(f"Error getting weather data: {e}", exc_info=True)
            raise
