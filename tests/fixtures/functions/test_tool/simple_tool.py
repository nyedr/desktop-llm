from app.models.function_base import Tool, FunctionType


class SimpleTool(Tool):
    name: str = "simple_tool"
    type: FunctionType = FunctionType.TOOL
    description: str = "A simple test tool"
    parameters: dict = {}
