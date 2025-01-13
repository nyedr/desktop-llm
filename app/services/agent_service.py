"""Service for managing agent operations."""

import logging
from typing import Dict, Any, AsyncGenerator, List, Optional

from app.functions.agent import AgentConfig
from app.functions.agent_registry import agent_registry
from app.services.function_service import FunctionService
from app.services.model_service import ModelService
from app.models.agent import AgentCapability
from app.models.chat import StrictChatMessage
from app.core.config import config

logger = logging.getLogger(__name__)


class AgentService:
    """Service for managing agent operations."""

    def __init__(self, function_service: FunctionService, model_service: ModelService):
        """Initialize the agent service."""
        self.function_service = function_service
        self.model_service = model_service

    def _create_agent_config(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_tools: Optional[bool] = None,
        allowed_tools: Optional[List[str]] = None,
        excluded_tools: Optional[List[str]] = None
    ) -> AgentConfig:
        """Create agent configuration with defaults from config."""
        return AgentConfig(
            model=model or config.llm.model,
            temperature=temperature or config.llm.temperature,
            max_tokens=max_tokens or config.llm.max_tokens,
            enable_tools=enable_tools if enable_tools is not None else config.llm.enable_tools,
            stream=True,
            hooks_enabled=True,
            allowed_tools=allowed_tools or [],
            excluded_tools=excluded_tools or [],
            tool_policies={}
        )

    async def run_agent_flow(
        self,
        goal: str,
        messages: List[StrictChatMessage],
        capabilities: List[AgentCapability],
        constraints: Dict[str, Any],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_tools: Optional[bool] = None,
        enable_memory: bool = True,
        allowed_tools: Optional[List[str]] = None,
        excluded_tools: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Create and run an agent with the specified parameters,
        streaming intermediate steps.
        """
        try:
            # Create agent config
            agent_config = self._create_agent_config(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_tools=enable_tools,
                allowed_tools=allowed_tools,
                excluded_tools=excluded_tools
            )

            # Get agent class from registry
            agent_type = metadata.get(
                "agent_type", "general") if metadata else "general"
            agent_class = agent_registry.get_agent_class(agent_type)

            # Initialize the agent with both services
            agent = agent_class(
                name=metadata.get(
                    "agent_name", f"{agent_type}_agent") if metadata else f"{agent_type}_agent",
                config=agent_config,
                function_service=self.function_service,
                model_service=self.model_service
            )

            # Set agent state
            agent.agent_state.goal = goal
            agent.agent_state.capabilities = capabilities
            agent.agent_state.constraints = constraints

            # Add messages to agent's context
            agent.agent_state.current_state.update({
                "messages": [msg.model_dump() for msg in messages],
                "enable_memory": enable_memory,
                "metadata": metadata or {}
            })

            # Stream each step from the agent loop
            async for step in agent.run_agent_loop():
                yield step

        except Exception as e:
            logger.error(f"Error in agent flow: {e}")
            yield {
                "phase": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }

    async def list_available_agents(self) -> List[Dict[str, Any]]:
        """List all available agents and their capabilities."""
        return agent_registry.list_agents()

    async def register_agent(self, name: str, agent_class: Any) -> None:
        """Register a new agent type."""
        agent_registry.register(name, agent_class)

    async def set_default_agent(self, agent_class: Any) -> None:
        """Set the default agent type."""
        agent_registry.set_default_agent(agent_class)
