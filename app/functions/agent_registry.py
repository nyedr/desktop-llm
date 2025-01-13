"""Registry for managing custom agents."""

import logging
from typing import Dict, Type, Optional, List, Any, Callable, TypeVar
from pathlib import Path
from functools import wraps

from app.functions.agent import BaseAgent, GeneralAgent, SupervisorAgent
from app.models.agent import AgentCapability

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=Type[BaseAgent])


class AgentRegistry:
    """Registry for managing all available agents."""

    _instance = None
    _agents: Dict[str, Type[BaseAgent]] = {}
    _default_agent = GeneralAgent

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentRegistry, cls).__new__(cls)
            cls._instance._agents = {}
            logger.info("Created new AgentRegistry instance")
        return cls._instance

    def register(self, name: str, agent_class: Type[BaseAgent]) -> None:
        """Register a new agent class."""
        try:
            self._agents[name.lower()] = agent_class
            logger.info(f"Registered agent: {name}")
            self._log_agent_summary()

        except Exception as e:
            logger.error(f"Failed to register agent {name}: {str(e)}")
            raise

    def get_agent_class(self, name: Optional[str] = None) -> Type[BaseAgent]:
        """Get an agent class by name, or return default agent if name is None."""
        if not name:
            return self._default_agent
        return self._agents.get(name.lower(), self._default_agent)

    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents with their capabilities."""
        agents = []
        for name, agent_class in self._agents.items():
            try:
                # Create a temporary instance to get capabilities
                temp_agent = agent_class(name=name)
                capabilities = [cap.value for cap in temp_agent.agent_state.capabilities] if hasattr(
                    temp_agent, 'agent_state') else []

                agents.append({
                    "name": name,
                    "class": agent_class.__name__,
                    "capabilities": capabilities,
                    "description": agent_class.__doc__ or "No description available"
                })
            except Exception as e:
                logger.error(
                    f"Error getting metadata for agent {name}: {str(e)}")

        return agents

    def _log_agent_summary(self) -> None:
        """Log a summary of all registered agents."""
        if not self._agents:
            logger.info("No agents registered")
            return

        summary = "\nRegistered Agents Summary:"
        for agent in self.list_agents():
            summary += f"\n- {agent['name']} ({agent['class']})"
            if agent['capabilities']:
                summary += f"\n  Capabilities: {', '.join(agent['capabilities'])}"

        logger.info(summary)

    async def discover_agents(self, directory: Path) -> None:
        """Discover and load custom agents from a directory."""
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return

        try:
            logger.info(f"Discovering agents in {directory}")
            # Implementation similar to function_registry's discover_functions
            # This can be implemented later based on needs
            pass

        except Exception as e:
            logger.error(f"Error discovering agents: {str(e)}")
            raise

    def set_default_agent(self, agent_class: Type[BaseAgent]) -> None:
        """Set the default agent class."""
        self._default_agent = agent_class
        logger.info(f"Set default agent to: {agent_class.__name__}")

    def register_builtin_agents(self) -> None:
        """Register built-in agent types."""
        # Register GeneralAgent
        general_agent = type('_GeneralAgent', (GeneralAgent,), {
            '__doc__': "General purpose agent for most tasks"
        })
        self.register("general", general_agent)
        self.set_default_agent(general_agent)

        # Register SupervisorAgent
        supervisor_agent = type('_SupervisorAgent', (SupervisorAgent,), {
            '__doc__': "Agent that can orchestrate multiple worker agents"
        })
        self.register("supervisor", supervisor_agent)


# Create the global registry instance
agent_registry = AgentRegistry()


def register_agent(
    name: str,
    description: Optional[str] = None,
    capabilities: Optional[List[AgentCapability]] = None,
    is_default: bool = False
) -> Callable[[T], T]:
    """Decorator to register an agent class."""
    def decorator(agent_class: T) -> T:
        # Update agent class documentation if provided
        if description:
            agent_class.__doc__ = description

        # Update agent capabilities if provided
        if capabilities:
            original_init = agent_class.__init__

            @wraps(original_init)
            def new_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                self.agent_state.capabilities = capabilities

            agent_class.__init__ = new_init

        # Register the agent
        agent_registry.register(name, agent_class)

        # Set as default if specified
        if is_default:
            agent_registry.set_default_agent(agent_class)

        return agent_class
    return decorator


# Initialize built-in agents
agent_registry.register_builtin_agents()

__all__ = ['AgentRegistry', 'agent_registry', 'register_agent']
