"""Models for the agent system."""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


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
