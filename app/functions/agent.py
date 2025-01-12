"""Agent class for handling autonomous behaviors in pipelines."""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, AsyncGenerator, Callable, Tuple, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import random

from app.functions.chat_helper import ChatHelper
from app.core.prompts import PROMPTS, get_agent_thought_prompt
from app.models.agent import (
    AgentState,
    AgentThought,
    AgentDecision,
    AgentCapability,
)
from app.models.function import (
    FunctionResponse,
    ToolResponse,
    AgentResponse,
    PipelineResponse
)
from langchain.schema.messages import SystemMessage, HumanMessage, BaseMessage

logger = logging.getLogger(__name__)


class BackoffStrategy(str, Enum):
    """Available backoff strategies for retry logic."""
    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exponential_jitter"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    backoff_factor: float = 1.5
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter_factor: float = 0.1
    custom_backoff_func: Optional[Callable[[int], float]] = None
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    # If True, only retry on specified exceptions
    retry_on_exceptions_only: bool = True

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the current attempt using the configured strategy."""
        if self.custom_backoff_func:
            delay = self.custom_backoff_func(attempt)
        else:
            delay = self._get_strategy_delay(attempt)

        # Apply jitter if using exponential with jitter
        if self.backoff_strategy == BackoffStrategy.EXPONENTIAL_JITTER:
            jitter = random.uniform(-self.jitter_factor, self.jitter_factor)
            delay = delay * (1 + jitter)

        # Ensure delay is within bounds
        return min(max(delay, self.base_delay), self.max_delay)

    def _get_strategy_delay(self, attempt: int) -> float:
        """Get delay based on the selected strategy."""
        if self.backoff_strategy == BackoffStrategy.CONSTANT:
            return self.base_delay
        elif self.backoff_strategy == BackoffStrategy.LINEAR:
            return self.base_delay + (attempt * self.backoff_factor)
        else:  # EXPONENTIAL or EXPONENTIAL_JITTER
            return self.base_delay * (self.backoff_factor ** attempt)


@dataclass
class AgentConfig:
    """Configuration for customizing agent behavior."""
    model: str = "deepseek/deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False

    # Tool configuration
    enable_tools: bool = True
    allowed_tools: Optional[List[str]] = None  # List of allowed tool names
    excluded_tools: Optional[List[str]] = None  # List of tool names to exclude
    custom_tools: Optional[List[Dict[str, Any]]
                           ] = None  # Additional tool schemas
    # Per-tool configuration like rate limits, retries, etc.
    tool_policies: Dict[str, Dict[str, Any]] = None

    # Hook configuration
    hooks_enabled: bool = True
    disabled_phases: List[str] = None
    hook_callbacks: Dict[str, List[Callable]] = None

    # Thought configuration
    custom_thought_prompts: Dict[str, str] = None
    custom_thought_types: Dict[str, Dict[str, Any]] = None

    # Retry configuration
    retry_config: Optional[RetryConfig] = None

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.retry_config is None:
            self.retry_config = RetryConfig()
        if self.disabled_phases is None:
            self.disabled_phases = []
        if self.hook_callbacks is None:
            self.hook_callbacks = {}
        if self.custom_thought_prompts is None:
            self.custom_thought_prompts = {}
        if self.custom_thought_types is None:
            self.custom_thought_types = {}
        if self.tool_policies is None:
            self.tool_policies = {}
        if self.allowed_tools is None:
            self.allowed_tools = []
        if self.excluded_tools is None:
            self.excluded_tools = []
        if self.custom_tools is None:
            self.custom_tools = []


class BaseAgent(ABC):
    """Base class that defines an interface for all agentic workflows."""

    @abstractmethod
    async def think(self, context: Dict[str, Any], thought_type: str = "reason") -> AsyncGenerator[AgentThought, None]:
        """Generate thoughts based on current context."""
        pass

    @abstractmethod
    async def decide(self, thoughts: List[AgentThought], context: Dict[str, Any]) -> AgentDecision:
        """Make a decision based on thoughts and context."""
        pass

    @abstractmethod
    async def act(self, decision: AgentDecision, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the decided action."""
        pass

    @abstractmethod
    async def reflect(self, execution_result: Dict[str, Any], context: Dict[str, Any]) -> AgentThought:
        """Reflect on execution results and update state."""
        pass

    @abstractmethod
    async def run_agent_loop(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Defines the main agent loop or pipeline."""
        pass


class GeneralAgent(BaseAgent):
    """General purpose agent implementation with configurable behavior."""

    def __init__(
        self,
        name: str,
        config: Optional[AgentConfig] = None,
        function_service=None,
        *args,
        **kwargs
    ):
        self.name = name
        self.agent_config = config or AgentConfig()
        self._agent_state: Optional[AgentState] = None
        self._chat_helper: Optional[ChatHelper] = None
        self._execution_lock = asyncio.Lock()
        self._function_service = function_service
        self._available_tools: Dict[str, Dict[str, Any]] = {}
        self._tool_usage_stats: Dict[str, Dict[str, Any]] = {}

        # Initialize thought prompts from core prompts and custom prompts
        self._thought_prompts = {}

        # Add default thought types
        for thought_type in PROMPTS["agent_thought_types"].keys():
            self._thought_prompts[thought_type] = get_agent_thought_prompt(
                thought_type)

        # Add custom prompts
        if self.agent_config.custom_thought_prompts:
            self._thought_prompts.update(
                self.agent_config.custom_thought_prompts)

        # Add custom thought types
        if self.agent_config.custom_thought_types:
            for thought_type, config in self.agent_config.custom_thought_types.items():
                if "prompt" in config:
                    self._thought_prompts[thought_type] = config["prompt"]

    @property
    def agent_state(self) -> AgentState:
        """Get the current agent state."""
        if not self._agent_state:
            self._agent_state = AgentState(
                goal="Complete the task",
                capabilities=[
                    AgentCapability.REASONING,
                    AgentCapability.PLANNING,
                    AgentCapability.FUNCTION_CALLING
                ],
                constraints={}
            )
        return self._agent_state

    @property
    def chat_helper(self) -> ChatHelper:
        """Get the chat helper instance."""
        if not self._chat_helper:
            self._chat_helper = ChatHelper()
        return self._chat_helper

    async def run_agent_loop(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Main entry point for running the agent."""
        async for result in self.execute_with_retry():
            yield result

    async def execute_with_retry(
        self,
        max_retries: Optional[int] = None,
        retry_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
        on_retry: Optional[Callable[[int, Exception], Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute agent with flexible retry logic and improved error handling."""
        retry_config = self.agent_config.retry_config
        max_retries = max_retries or retry_config.max_retries
        retry_exceptions = retry_exceptions or retry_config.retry_exceptions

        attempt = 0
        last_error = None

        while attempt < max_retries:
            try:
                async with self._execution_lock:
                    async for result in self._execute_agent_loop():
                        yield result
                    return
            except Exception as e:
                attempt += 1
                last_error = e

                # Check if we should retry this exception
                should_retry = (
                    not retry_config.retry_on_exceptions_only or
                    isinstance(e, retry_exceptions)
                )

                if not should_retry:
                    logger.error(
                        f"Non-retryable error in agent {self.name}: {str(e)}")
                    raise

                # Log retry attempt
                if attempt < max_retries:
                    logger.warning(
                        f"Retry attempt {attempt}/{max_retries} for agent {self.name} "
                        f"due to error: {str(e)}"
                    )
                else:
                    logger.error(
                        f"Final retry attempt {attempt}/{max_retries} for agent {self.name} "
                        f"failed with error: {str(e)}"
                    )

                # Record failure in agent state
                self.agent_state.record_failure({
                    "attempt": attempt,
                    "error": str(e),
                    "error_type": e.__class__.__name__,
                    "state": self.agent_state.get_execution_summary(),
                    "retry_info": {
                        "max_retries": max_retries,
                        "backoff_strategy": retry_config.backoff_strategy,
                        "should_retry": should_retry
                    }
                })

                # Call retry callback if provided
                if on_retry:
                    try:
                        on_retry(attempt, e)
                    except Exception as callback_error:
                        logger.error(
                            f"Error in retry callback for agent {self.name}: {str(callback_error)}")

                if attempt >= max_retries:
                    logger.error(
                        f"Agent {self.name} exceeded maximum retry attempts ({max_retries})")
                    raise last_error

                # Calculate and apply backoff delay
                delay = retry_config.calculate_delay(attempt)
                logger.info(
                    f"Agent {self.name} backing off for {delay:.2f} seconds before retry {attempt}")
                await asyncio.sleep(delay)

    async def _execute_agent_loop(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Core agent execution loop with enhanced hook system."""
        context = await self._build_context()

        # Run start hooks
        if "think" not in self.agent_config.disabled_phases:
            await self._run_hook("on_start", context)

        while not await self._is_goal_complete():
            # Think phase
            if "think" not in self.agent_config.disabled_phases:
                await self._run_hook("before_think", context)
            thoughts = []
            async for thought in self.think(context):
                thoughts.append(thought)
                self.agent_state.add_thought(thought.model_dump())
                yield {"phase": "thinking", "thought": thought.model_dump()}
                await self._run_hook("after_think", thoughts, context)

            # Decide phase
            if "decide" not in self.agent_config.disabled_phases:
                await self._run_hook("before_decide", thoughts, context)
            decision = await self.decide(thoughts, context)
            yield {"phase": "decision", "decision": decision.model_dump()}
            await self._run_hook("after_decide", decision, context)

            # Act phase
            if "act" not in self.agent_config.disabled_phases:
                await self._run_hook("before_act", decision, context)
            result = await self.act(decision, context)
            self.agent_state.last_tool_result = result
            yield {"phase": "action", "result": result}
            await self._run_hook("after_act", result, context)

            # Reflect phase
            if "reflect" not in self.agent_config.disabled_phases:
                await self._run_hook("before_reflect", result, context)
            reflection = await self.reflect(result, context)
            self.agent_state.add_thought(reflection.model_dump())
            yield {"phase": "reflection", "reflection": reflection.model_dump()}
            await self._run_hook("after_reflect", reflection, context)

            await self._run_hook("on_iteration_end", context)
            context = await self._build_context()

        await self._run_hook("on_finish", context)
        yield {
            "success": True,
            "final_state": self.agent_state.get_execution_summary()
        }

    async def _run_hook(self, hook_name: str, *args, **kwargs) -> None:
        """Run all callbacks for a given hook."""
        if not self.agent_config.hooks_enabled:
            return

        # Run the default hook method
        hook_method = getattr(self, hook_name, None)
        if hook_method:
            await hook_method(*args, **kwargs)

        # Run additional callbacks
        callbacks = self.agent_config.hook_callbacks.get(hook_name, [])
        for callback in callbacks:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)

    async def think(
        self,
        context: Dict[str, Any],
        thought_type: str = "reason"
    ) -> AsyncGenerator[AgentThought, None]:
        """Generate thoughts based on current context with support for custom types."""
        if thought_type not in self._thought_prompts:
            logger.warning(
                f"Unknown thought type: {thought_type}, falling back to 'reason'")
            thought_type = "reason"

        messages = self._create_thought_messages(
            thought_type, self._thought_prompts, context)

        # Get custom configuration for thought type if available
        custom_config = self.agent_config.custom_thought_types.get(
            thought_type, {})
        temperature = custom_config.get(
            "temperature") or self._get_temperature_for_thought(thought_type)
        max_tokens = custom_config.get(
            "max_tokens") or self.agent_config.max_tokens

        # Get available tools based on configuration
        tools = await self._get_available_tools() if self.agent_config.enable_tools else None

        async for response in self.chat_helper.generate_completion(
            messages=messages,
            model=self.agent_config.model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=self.agent_config.stream,
            tools=tools,
            enable_tools=self.agent_config.enable_tools
        ):
            yield AgentThought(
                type=thought_type,
                content=response.get("content", ""),
                confidence=response.get("metadata", {}).get("confidence", 0.0),
                reasoning_path=response.get(
                    "metadata", {}).get("reasoning_path", []),
                alternatives=response.get(
                    "metadata", {}).get("alternatives", []),
                metadata={
                    **response.get("metadata", {}),
                    "thought_config": custom_config,
                    "available_tools": [t.get("function", {}).get("name") for t in (tools or [])]
                }
            )

    async def decide(
        self,
        thoughts: List[AgentThought],
        context: Dict[str, Any]
    ) -> AgentDecision:
        """Make a decision based on thoughts and context."""
        messages = [
            SystemMessage(content=PROMPTS["agent_decision"]),
            HumanMessage(content=self._format_decision_context(
                thoughts, context))
        ]

        async for response in self.chat_helper.generate_completion(
            messages=messages,
            model=self.agent_config.model,
            temperature=0.3,
            max_tokens=self.agent_config.max_tokens,
            stream=self.agent_config.stream,
            tools=self.agent_config.tools,
            enable_tools=self.agent_config.enable_tools
        ):
            return AgentDecision(
                action_type=response.get("metadata", {}).get(
                    "action_type", "default"),
                action_plan=response.get("content", ""),
                confidence=response.get("metadata", {}).get("confidence", 0.0),
                reasoning=response.get("metadata", {}).get("reasoning", ""),
                alternatives=response.get(
                    "metadata", {}).get("alternatives", [])
            )

    async def act(self, decision: AgentDecision, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the decided action."""
        return await self._execute_action(decision, context)

    async def reflect(
        self,
        execution_result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> AgentThought:
        """Reflect on execution results and update state."""
        messages = [
            SystemMessage(content=PROMPTS["agent_reflection"]),
            HumanMessage(content=self._format_reflection_context(
                execution_result, context))
        ]

        async for response in self.chat_helper.generate_completion(
            messages=messages,
            model=self.agent_config.model,
            temperature=0.4,
            max_tokens=self.agent_config.max_tokens,
            stream=self.agent_config.stream,
            tools=self.agent_config.tools,
            enable_tools=self.agent_config.enable_tools
        ):
            reflection = AgentThought(
                type="reflection",
                content=response.get("content", ""),
                confidence=response.get("metadata", {}).get("confidence", 0.0),
                metadata={
                    "progress_evaluation": response.get("metadata", {}).get("progress", 0.0),
                    "state_updates": response.get("metadata", {}).get("state_updates", {}),
                    "next_steps": response.get("metadata", {}).get("next_steps", [])
                }
            )
            self._update_agent_state(reflection)
            return reflection

    # Hook methods (empty by default)
    async def on_start(self, context: Dict[str, Any]): pass
    async def on_finish(self, context: Dict[str, Any]): pass
    async def before_think(self, context: Dict[str, Any]): pass

    async def after_think(
        self, thoughts: List[AgentThought], context: Dict[str, Any]): pass

    async def before_decide(
        self, thoughts: List[AgentThought], context: Dict[str, Any]): pass

    async def after_decide(self, decision: AgentDecision,
                           context: Dict[str, Any]): pass

    async def before_act(self, decision: AgentDecision,
                         context: Dict[str, Any]): pass

    async def after_act(
        self, result: Dict[str, Any], context: Dict[str, Any]): pass

    async def before_reflect(
        self, result: Dict[str, Any], context: Dict[str, Any]): pass
    async def after_reflect(self, reflection: AgentThought,
                            context: Dict[str, Any]): pass

    async def on_iteration_end(self, context: Dict[str, Any]): pass

    # Helper methods
    async def _build_context(self) -> Dict[str, Any]:
        """Build execution context."""
        return {
            "current_state": self.agent_state.current_state,
            "execution_context": self.agent_state.execution_context,
            "last_tool_result": self.agent_state.last_tool_result,
            "progress": self.agent_state.progress,
            "capabilities": self.agent_state.capabilities,
            "constraints": self.agent_state.constraints
        }

    async def _is_goal_complete(self) -> bool:
        """Check if the current goal is complete."""
        return self.agent_state.progress >= 1.0

    async def _execute_action(self, decision: AgentDecision, context: Dict[str, Any]) -> FunctionResponse:
        """Execute the decided action."""
        action_type = decision.action_type

        # Handle function calls
        if action_type == "function_call":
            if not self._function_service:
                raise ValueError(
                    "Function service not provided for function calls")

            function_name = decision.metadata.get("function_name")
            function_args = decision.metadata.get("arguments", {})

            if not function_name:
                raise ValueError(
                    "Function name not provided in decision metadata")

            logger.info(f"Agent {self.name} calling function: {function_name}")
            try:
                result = await self._function_service.execute_function(
                    function_name,
                    function_args
                )
                return ToolResponse(
                    success=True,
                    tool_name=function_name,
                    result=result,
                    execution_time=0.0,  # TODO: Add timing
                    metadata={
                        "action_type": action_type,
                        "args": function_args
                    }
                )
            except Exception as e:
                logger.error(
                    f"Error executing function {function_name}: {str(e)}")
                return ToolResponse(
                    success=False,
                    tool_name=function_name,
                    result=None,
                    error=str(e),
                    execution_time=0.0,  # TODO: Add timing
                    metadata={
                        "action_type": action_type,
                        "args": function_args,
                        "error_type": type(e).__name__
                    }
                )

        # Handle self-calls (agent calling itself)
        elif action_type == "self_call":
            sub_goal = decision.metadata.get("goal")
            sub_context = decision.metadata.get("context", {})

            if not sub_goal:
                raise ValueError("Goal not provided for self call")

            # Create a new agent state for this sub-goal
            original_state = self._agent_state
            self._agent_state = AgentState(
                goal=sub_goal,
                capabilities=original_state.capabilities,
                constraints=original_state.constraints,
                current_state={**original_state.current_state, **sub_context}
            )

            try:
                result = {}
                async for output in self.run_agent_loop():
                    result = output
                return AgentResponse(
                    success=True,
                    agent_name=self.name,
                    state=self._agent_state,
                    thoughts=self._agent_state.thought_process,
                    decisions=[],  # TODO: Track decisions
                    actions_taken=[],  # TODO: Track actions
                    final_output=result,
                    metadata={
                        "action_type": action_type,
                        "sub_goal": sub_goal,
                        "sub_context": sub_context
                    }
                )
            except Exception as e:
                logger.error(f"Error in self-call execution: {str(e)}")
                return AgentResponse(
                    success=False,
                    agent_name=self.name,
                    state=self._agent_state,
                    error=str(e),
                    metadata={
                        "action_type": action_type,
                        "sub_goal": sub_goal,
                        "error_type": type(e).__name__
                    }
                )
            finally:
                # Restore original state
                self._agent_state = original_state

        # Handle chain-of-thought execution
        elif action_type == "chain_of_thought":
            try:
                thoughts = []
                for thought_type in decision.metadata.get("thought_sequence", ["reason"]):
                    async for thought in self.think(context, thought_type):
                        thoughts.append(thought)

                return PipelineResponse(
                    success=True,
                    pipeline_name="chain_of_thought",
                    results=[t.model_dump() for t in thoughts],
                    steps_completed=len(thoughts),
                    total_steps=len(decision.metadata.get(
                        "thought_sequence", ["reason"])),
                    metadata={
                        "action_type": action_type,
                        "thought_sequence": decision.metadata.get("thought_sequence", ["reason"])
                    }
                )
            except Exception as e:
                logger.error(f"Error in chain-of-thought execution: {str(e)}")
                return PipelineResponse(
                    success=False,
                    pipeline_name="chain_of_thought",
                    error=str(e),
                    steps_completed=len(thoughts),
                    total_steps=len(decision.metadata.get(
                        "thought_sequence", ["reason"])),
                    metadata={
                        "action_type": action_type,
                        "error_type": type(e).__name__
                    }
                )

        raise NotImplementedError(f"Action type {action_type} not implemented")

    def _get_temperature_for_thought(self, thought_type: str) -> float:
        """Get appropriate temperature for different thought types."""
        if self.agent_config.thought_config:
            return self.agent_config.thought_config.get(thought_type, 0.7)

        temperatures = {
            "reason": 0.7,
            "plan": 0.5,
            "evaluate": 0.3,
            "reflect": 0.6
        }
        return temperatures.get(thought_type, 0.7)

    def _create_thought_messages(
        self,
        thought_type: str,
        prompts: Dict[str, str],
        context: Dict[str, Any]
    ) -> List[BaseMessage]:
        """Create messages for thought generation."""
        # Use get_agent_thought_prompt for default prompts
        prompt = prompts.get(
            thought_type) or get_agent_thought_prompt(thought_type)
        return [
            SystemMessage(content=prompt),
            HumanMessage(content=self._format_context(context))
        ]

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for LLM consumption."""
        if not self._agent_state:
            return str(context)

        return f"""
Goal: {self._agent_state.goal}
Current State: {self._agent_state.current_state}
Context: {context}
History: {self._agent_state.history[-5:] if self._agent_state.history else 'No history'}
Capabilities: {[cap.value for cap in self._agent_state.capabilities]}
Constraints: {self._agent_state.constraints}
"""

    def _update_agent_state(self, reflection: AgentThought) -> None:
        """Update agent state based on reflection."""
        if not self._agent_state:
            return

        metadata = reflection.metadata
        if "state_updates" in metadata:
            self._agent_state.current_state.update(metadata["state_updates"])
        if "progress_evaluation" in metadata:
            self._agent_state.update_progress(metadata["progress_evaluation"])

        self._agent_state.history.append({
            "timestamp": datetime.now().isoformat(),
            "reflection": reflection.model_dump(),
            "state_snapshot": self._agent_state.model_dump()
        })

    def add_hook_callback(self, hook_name: str, callback: Callable) -> None:
        """Add a callback to a specific hook."""
        if hook_name not in self.agent_config.hook_callbacks:
            self.agent_config.hook_callbacks[hook_name] = []
        self.agent_config.hook_callbacks[hook_name].append(callback)

    def remove_hook_callback(self, hook_name: str, callback: Callable) -> None:
        """Remove a callback from a specific hook."""
        if hook_name in self.agent_config.hook_callbacks:
            try:
                self.agent_config.hook_callbacks[hook_name].remove(callback)
            except ValueError:
                pass

    def add_thought_type(
        self,
        name: str,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> None:
        """Add a new thought type with custom configuration."""
        self._thought_prompts[name] = prompt
        self.agent_config.custom_thought_types[name] = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

    def update_thought_prompt(self, thought_type: str, prompt: str) -> None:
        """Update the prompt for an existing thought type."""
        if thought_type in self._thought_prompts:
            self._thought_prompts[thought_type] = prompt
            if thought_type in self.agent_config.custom_thought_types:
                self.agent_config.custom_thought_types[thought_type]["prompt"] = prompt

    async def _format_decision_context(
        self,
        thoughts: List[AgentThought],
        context: Dict[str, Any]
    ) -> str:
        """Format context for decision making with available functions."""
        base_context = self._format_context(context)

        # Add available functions if function service is present
        if self._function_service:
            available_functions = await self._function_service.get_available_functions()
            functions_str = "\nAvailable Functions:\n"
            for func in available_functions:
                functions_str += f"- {func['name']}: {func['description']}\n"
            base_context += functions_str

        # Add thoughts summary
        thoughts_str = "\nThought Process:\n"
        for thought in thoughts:
            thoughts_str += f"- {thought.type}: {thought.content}\n"

        return base_context + thoughts_str

    def register_function(self, function_name: str, function_impl: Callable) -> None:
        """Register a new function that can be called by the agent."""
        if self._function_service:
            self._function_service.register_function(
                function_name, function_impl)
        else:
            logger.warning(
                "No function service available to register function")

    def configure_tools(
        self,
        allowed_tools: Optional[List[str]] = None,
        excluded_tools: Optional[List[str]] = None,
        custom_tools: Optional[List[Dict[str, Any]]] = None,
        tool_policies: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> None:
        """Configure tool access and policies for the agent."""
        if allowed_tools is not None:
            self.agent_config.allowed_tools = allowed_tools
        if excluded_tools is not None:
            self.agent_config.excluded_tools = excluded_tools
        if custom_tools is not None:
            self.agent_config.custom_tools = custom_tools
        if tool_policies is not None:
            self.agent_config.tool_policies = tool_policies

    def add_tool_policy(self, tool_name: str, policy: Dict[str, Any]) -> None:
        """Add or update policy for a specific tool."""
        self.agent_config.tool_policies[tool_name] = {
            **(self.agent_config.tool_policies.get(tool_name, {})),
            **policy
        }

    def get_tool_policy(self, tool_name: str) -> Dict[str, Any]:
        """Get policy for a specific tool."""
        return self.agent_config.tool_policies.get(tool_name, {})

    async def _get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools based on configuration."""
        if not self._function_service:
            return []

        all_tools = self._function_service.get_function_schemas()
        filtered_tools = []

        for tool in all_tools:
            tool_name = tool.get("function", {}).get("name")
            if not tool_name:
                continue

            # Skip if tool is not in allowed_tools (if specified)
            if self.agent_config.allowed_tools and tool_name not in self.agent_config.allowed_tools:
                continue

            # Skip if tool is in excluded_tools
            if tool_name in self.agent_config.excluded_tools:
                continue

            # Add tool policy metadata if exists
            if tool_name in self.agent_config.tool_policies:
                tool["function"]["metadata"] = {
                    **(tool.get("function", {}).get("metadata", {})),
                    "policy": self.agent_config.tool_policies[tool_name]
                }

            filtered_tools.append(tool)

        # Add custom tools
        if self.agent_config.custom_tools:
            filtered_tools.extend([
                {"type": "function", "function": tool}
                for tool in self.agent_config.custom_tools
            ])

        return filtered_tools

    async def _update_tool_stats(self, tool_name: str, result: Dict[str, Any]) -> None:
        """Update usage statistics for a tool."""
        if tool_name not in self._tool_usage_stats:
            self._tool_usage_stats[tool_name] = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_execution_time": 0.0,
                "last_used": None,
                "errors": {}
            }

        stats = self._tool_usage_stats[tool_name]
        stats["total_calls"] += 1
        stats["last_used"] = datetime.now().isoformat()

        if result.get("success"):
            stats["successful_calls"] += 1
        else:
            stats["failed_calls"] += 1
            error_type = result.get("error_type", "unknown")
            stats["errors"][error_type] = stats["errors"].get(
                error_type, 0) + 1

        if "execution_time" in result:
            stats["total_execution_time"] += result["execution_time"]

    def get_tool_stats(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Get usage statistics for one or all tools."""
        if tool_name:
            return self._tool_usage_stats.get(tool_name, {})
        return self._tool_usage_stats


class SupervisorAgent(GeneralAgent):
    """A specialized agent that can orchestrate multiple worker agents."""

    def __init__(
        self,
        name: str,
        worker_agents: Optional[Dict[str, GeneralAgent]] = None,
        config: Optional[AgentConfig] = None,
        *args,
        **kwargs
    ):
        super().__init__(name, config, *args, **kwargs)
        self.worker_agents = worker_agents or {}
        self.task_queue = asyncio.Queue()
        self.results = {}

    async def delegate_task(
        self,
        worker_name: str,
        task: Dict[str, Any]
    ) -> AgentResponse:
        """Delegate a task to a worker agent."""
        if worker_name not in self.worker_agents:
            raise ValueError(f"No worker agent found with name: {worker_name}")

        worker = self.worker_agents[worker_name]
        worker.agent_state.goal = task.get("goal", "Complete delegated task")
        worker.agent_state.constraints.update(task.get("constraints", {}))

        try:
            result = {}
            async for output in worker.run_agent_loop():
                result = output

            return AgentResponse(
                success=True,
                agent_name=worker_name,
                state=worker.agent_state,
                thoughts=worker.agent_state.thought_process,
                decisions=[],  # TODO: Track decisions
                actions_taken=[],  # TODO: Track actions
                final_output=result,
                metadata={
                    "task": task,
                    "supervisor": self.name
                }
            )
        except Exception as e:
            logger.error(f"Error in worker {worker_name} execution: {str(e)}")
            return AgentResponse(
                success=False,
                agent_name=worker_name,
                state=worker.agent_state,
                error=str(e),
                metadata={
                    "task": task,
                    "supervisor": self.name,
                    "error_type": type(e).__name__
                }
            )

    async def _execute_action(
        self,
        decision: AgentDecision,
        context: Dict[str, Any]
    ) -> FunctionResponse:
        """Execute actions, potentially delegating to worker agents."""
        action_type = decision.action_type

        if action_type == "delegate":
            # Extract worker and task from decision
            worker_name = decision.metadata.get("worker")
            task = decision.metadata.get("task")
            if not worker_name or not task:
                raise ValueError("Delegation requires worker name and task")

            return await self.delegate_task(worker_name, task)

        elif action_type == "parallel_delegate":
            # Handle parallel delegation to multiple workers
            tasks = decision.metadata.get("tasks", [])
            results = []
            total_tasks = len(tasks)
            completed_tasks = 0

            try:
                # Create tasks for each worker
                for task in tasks:
                    worker_name = task.get("worker")
                    if worker_name in self.worker_agents:
                        self.task_queue.put_nowait((worker_name, task))

                # Process queue until empty
                while not self.task_queue.empty():
                    worker_name, task = await self.task_queue.get()
                    try:
                        result = await self.delegate_task(worker_name, task)
                        results.append(result)
                        completed_tasks += 1
                    except Exception as e:
                        results.append(AgentResponse(
                            success=False,
                            agent_name=worker_name,
                            state=self.worker_agents[worker_name].agent_state,
                            error=str(e),
                            metadata={
                                "task": task,
                                "supervisor": self.name,
                                "error_type": type(e).__name__
                            }
                        ))
                    finally:
                        self.task_queue.task_done()

                return PipelineResponse(
                    success=all(r.success for r in results),
                    pipeline_name="parallel_delegation",
                    results=[r.model_dump() for r in results],
                    steps_completed=completed_tasks,
                    total_steps=total_tasks,
                    metadata={
                        "action_type": action_type,
                        "worker_count": len(self.worker_agents),
                        "task_count": total_tasks
                    }
                )
            except Exception as e:
                logger.error(f"Error in parallel delegation: {str(e)}")
                return PipelineResponse(
                    success=False,
                    pipeline_name="parallel_delegation",
                    error=str(e),
                    steps_completed=completed_tasks,
                    total_steps=total_tasks,
                    results=[r.model_dump() for r in results],
                    metadata={
                        "action_type": action_type,
                        "error_type": type(e).__name__,
                        "worker_count": len(self.worker_agents),
                        "task_count": total_tasks
                    }
                )

        else:
            # Default to standard action execution
            return await super()._execute_action(decision, context)

    async def on_start(self, context: Dict[str, Any]):
        """Initialize worker states and task queue."""
        self.task_queue = asyncio.Queue()
        self.results = {}

        # Initialize each worker
        for name, worker in self.worker_agents.items():
            worker.agent_state.current_state["supervisor"] = self.name
            worker.agent_state.current_state["role"] = "worker"

    async def before_think(self, context: Dict[str, Any]):
        """Update context with worker states."""
        worker_states = {
            name: worker.agent_state.get_execution_summary()
            for name, worker in self.worker_agents.items()
        }
        context["worker_states"] = worker_states

    async def after_act(self, result: Dict[str, Any], context: Dict[str, Any]):
        """Process and store results from worker agents."""
        if result.get("type") == "parallel_results":
            for worker_result in result["results"]:
                worker_name = worker_result["worker"]
                self.results[worker_name] = worker_result

    async def on_iteration_end(self, context: Dict[str, Any]):
        """Clean up completed tasks and update progress."""
        # Update overall progress based on worker progress
        if self.worker_agents:
            total_progress = sum(
                worker.agent_state.progress
                for worker in self.worker_agents.values()
            ) / len(self.worker_agents)

            self.agent_state.update_progress(total_progress)
