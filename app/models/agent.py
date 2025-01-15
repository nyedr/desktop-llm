"""Models for the agent system."""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

from app.models.chat import StrictChatMessage


class AgentCapability(str, Enum):
    """Capabilities that an agent can have."""
    FUNCTION_CALLING = "function_calling"


class AgentState(BaseModel):
    """Model for agent state."""
    goal: str = Field(..., description="Current goal")
    capabilities: List[AgentCapability] = Field(default_factory=list)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    current_state: Dict[str, Any] = Field(default_factory=dict)
    progress: float = Field(default=0.0, description="Progress toward goal")
    execution_context: Dict[str, Any] = Field(default_factory=dict)
    last_tool_result: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)

    def update_progress(self, new_progress: float) -> None:
        """Update progress and record in history."""
        self.progress = new_progress
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "progress_update",
            "progress": new_progress
        })

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the current execution state."""
        return {
            "goal": self.goal,
            "progress": self.progress,
            "current_state": self.current_state,
            "capabilities": [cap.value for cap in self.capabilities],
            "constraints": self.constraints,
            "last_tool_result": self.last_tool_result
        }


class AgentRequest(BaseModel):
    """Request model for agent endpoints."""
    messages: List[StrictChatMessage] = Field(
        ...,
        description="List of messages in the conversation"
    )
    goal: str = Field(
        ...,
        description="The goal or task for the agent to accomplish"
    )
    agent_type: str = Field(
        default="general",
        description="Type of agent to use (e.g., 'general', 'supervisor', or custom registered types)"
    )
    agent_name: Optional[str] = Field(
        default=None,
        description="Custom name for the agent instance"
    )
    capabilities: List[AgentCapability] = Field(
        default_factory=lambda: [AgentCapability.FUNCTION_CALLING],
        description="List of capabilities the agent should have"
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Constraints for the agent's operation"
    )
    stream: bool = Field(
        default=True,
        description="Whether to stream the response or return it all at once"
    )
    model: Optional[str] = Field(
        default=None,
        description="The model to use for agent operations"
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Temperature for model responses"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to generate"
    )
    max_turns: int = Field(
        default=5,
        description="Maximum number of turns before forcing completion"
    )
    enable_tools: Optional[bool] = Field(
        default=None,
        description="Whether to enable tool/function calling"
    )
    enable_memory: bool = Field(
        default=True,
        description="Whether to enable memory/context management"
    )
    allowed_tools: Optional[List[str]] = Field(
        default=None,
        description="List of specific tools the agent is allowed to use"
    )
    excluded_tools: Optional[List[str]] = Field(
        default=None,
        description="List of tools the agent should not use"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the agent execution"
    )
