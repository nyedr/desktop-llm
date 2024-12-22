"""Tools for weather-related operations."""

from typing import Dict, Any
from app.functions.base import Tool, FunctionType
from pydantic import Field


class WeatherTool(Tool):
    """Tool for getting current weather information."""

    name: str = Field(
        default="get_current_weather",
        description="Get current weather information for a location"
    )
    description: str = Field(
        default="Get current weather information for a location",
        description="Brief description of the tool's purpose"
    )
    type: FunctionType = Field(
        default=FunctionType.TOOL,
        description="Type of function"
    )
    config: Dict[str, Any] = Field(
        default={"default_unit": "fahrenheit"},
        description="Configuration for the weather tool"
    )
    parameters: Dict[str, Any] = Field(
        default={
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
        },
        description="Parameters for the weather tool"
    )

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get current weather for a location.

        Args:
            args: Dictionary containing location and optional unit

        Returns:
            Current weather information
        """
        location = args["location"]
        unit = args.get("unit", self.config["default_unit"])

        # TODO: Implement actual weather API call
        weather = {
            "location": location,
            "temperature": 72 if unit == "fahrenheit" else 22,
            "unit": unit,
            "conditions": "sunny",
            "humidity": 45,
            "wind_speed": 8
        }

        return weather
