"""Models for the agent system."""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class AgentCapability(str, Enum):
    """Capabilities that an agent can have."""
    REASONING = "reasoning"
    PLANNING = "planning"
    FUNCTION_CALLING = "function_calling"
    MEMORY = "memory"
    REFLECTION = "reflection"
    LEARNING = "learning"
    SELF_IMPROVEMENT = "self_improvement"


class AgentThought(BaseModel):
    """Model for agent thoughts."""
    type: str = Field(..., description="Type of thought")
    content: str = Field(..., description="Thought content")
    confidence: float = Field(default=0.0, description="Confidence level")
    reasoning_path: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class AgentDecision(BaseModel):
    """Model for agent decisions."""
    action_type: str = Field(..., description="Type of action")
    action_plan: str = Field(..., description="Detailed plan")
    reasoning: str = Field(..., description="Reasoning behind decision")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class AgentState(BaseModel):
    """Model for agent state."""
    goal: str = Field(..., description="Current goal")
    capabilities: List[AgentCapability] = Field(default_factory=list)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    current_state: Dict[str, Any] = Field(default_factory=dict)
    progress: float = Field(default=0.0, description="Progress toward goal")
    thought_process: List[Dict[str, Any]] = Field(default_factory=list)
    execution_context: Dict[str, Any] = Field(default_factory=dict)
    last_tool_result: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)

    def add_thought(self, thought: Dict[str, Any]) -> None:
        """Add a thought to the thought process."""
        self.thought_process.append(thought)
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "thought",
            "thought": thought
        })

    def update_progress(self, new_progress: float) -> None:
        """Update progress and record in history."""
        self.progress = new_progress
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "progress_update",
            "progress": new_progress
        })

    def record_failure(self, failure_info: Dict[str, Any]) -> None:
        """Record a failure in history."""
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "failure",
            **failure_info
        })

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the current execution state."""
        return {
            "goal": self.goal,
            "progress": self.progress,
            "current_state": self.current_state,
            "capabilities": [cap.value for cap in self.capabilities],
            "constraints": self.constraints,
            "thought_count": len(self.thought_process),
            "last_thought": self.thought_process[-1] if self.thought_process else None,
            "last_tool_result": self.last_tool_result
        }
