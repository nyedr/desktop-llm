from typing import Dict, Any, Optional, Literal
from app.functions.base import Tool, FunctionType, register_function
from app.services.chroma_service import ChromaService
from pydantic import Field, ConfigDict
import logging

logger = logging.getLogger(__name__)


@register_function(
    func_type=FunctionType.TOOL,
    name="add_memory",
    description="Add a text entry to the long-term memory storage, use this to store information that you should remember (e.g. user information, preferences, etc.)",
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text content to store as a memory"
            },
            "metadata": {
                "type": "object",
                "description": "Optional metadata to store with the memory",
                "additionalProperties": True
            }
        },
        "required": ["text"]
    }
)
class AddMemoryTool(Tool):
    """Tool for adding memories to the Chroma database."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chroma_service: ChromaService = Field(
        default_factory=ChromaService,
        exclude=True
    )

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool to add a memory to the Chroma database.

        Args:
            args: Dictionary containing:
                - text: The text content to store
                - metadata: Optional metadata dictionary

        Returns:
            Dictionary containing:
                - memory_id: The ID of the stored memory
                - status: Success message
        """
        try:
            text = args["text"]
            metadata = args.get("metadata")

            # Add memory to Chroma
            memory_id = await self.chroma_service.add_memory(text, metadata)

            return {
                "memory_id": memory_id,
                "status": "Memory successfully stored"
            }

        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            raise
