"""Tools for weather-related operations."""

import json
from typing import Dict, Any
from app.functions import Tool, register_function, FunctionType
from pydantic import Field

@register_function(
    func_type=FunctionType.PIPE,
    name="get_current_weather",
    description="Get current weather information for a location",
    priority=None,
    config={"default_unit": "fahrenheit"}
)
class WeatherTool(Tool):
    """Tool for getting current weather information."""
    
    name: str = "get_current_weather"
    description: str = "Get current weather information for a location"
    type: FunctionType = FunctionType.PIPE
    config: Dict[str, Any] = {"default_unit": "fahrenheit"}

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
