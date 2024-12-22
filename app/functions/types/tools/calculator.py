"""Example tool that performs calculations."""

from typing import Dict, Any, Literal
import operator
from app.functions.base import Tool, FunctionType, register_function
from pydantic import Field


@register_function(
    func_type=FunctionType.TOOL,
    name="calculator",
    description="Performs basic arithmetic calculations",
    priority=None,
    config={"max_precision": 10}
)
class CalculatorTool(Tool):
    """Tool for performing basic arithmetic operations."""

    name: str = Field(default="calculator",
                      description="Name of the calculator tool")
    description: str = Field(
        default="Performs basic arithmetic calculations",
        description="Description of the calculator tool"
    )
    type: Literal[FunctionType.TOOL] = Field(
        default=FunctionType.TOOL, description="Tool type")

    parameters: Dict[str, Any] = Field(
        default={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform"
                },
                "a": {
                    "type": "number",
                    "description": "First operand"
                },
                "b": {
                    "type": "number",
                    "description": "Second operand"
                }
            },
            "required": ["operation", "a", "b"]
        },
        description="Parameters for the calculator tool"
    )

    _operations = {
        "add": operator.add,
        "subtract": operator.sub,
        "multiply": operator.mul,
        "divide": operator.truediv
    }

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the calculation.

        Args:
            args: Dictionary containing operation and operands

        Returns:
            Result of the calculation
        """
        operation = args["operation"]
        a = args["a"]
        b = args["b"]

        if operation not in self._operations:
            raise ValueError(f"Invalid operation: {operation}")

        if operation == "divide" and b == 0:
            raise ValueError("Division by zero")

        result = self._operations[operation](a, b)

        # Round to max precision if needed
        max_precision = self.config.get("max_precision")
        if max_precision is not None:
            result = round(result, max_precision)

        return {
            "result": result,
            "operation": operation,
            "a": a,
            "b": b
        }
