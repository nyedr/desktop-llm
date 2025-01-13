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
from app.models.agent import (
    AgentState,
    AgentCapability,
)
from app.models.function import (
    FunctionResponse,
    ToolResponse,
    AgentResponse,
    PipelineResponse
)
from langchain.schema.messages import SystemMessage, HumanMessage, BaseMessage
from app.core.prompts import PROMPTS
from app.utils.utils import parse_attempt_completion

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
    allowed_tools: Optional[List[str]] = None
    excluded_tools: Optional[List[str]] = None
    custom_tools: Optional[List[Dict[str, Any]]] = None
    tool_policies: Dict[str, Dict[str, Any]] = None

    # Hook configuration
    hooks_enabled: bool = True
    hook_callbacks: Dict[str, List[Callable]] = None

    # Retry configuration
    retry_config: Optional[RetryConfig] = None

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.retry_config is None:
            self.retry_config = RetryConfig()
        if self.hook_callbacks is None:
            self.hook_callbacks = {}
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
    async def run_agent_loop(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Defines the main agent loop that runs until task completion."""
        pass


class GeneralAgent(BaseAgent):
    """General purpose agent implementation with configurable behavior."""

    def __init__(
        self,
        name: str,
        config: Optional[AgentConfig] = None,
        function_service=None,
        model_service=None,
        *args,
        **kwargs
    ):
        self.name = name
        self.agent_config = config or AgentConfig()
        self._agent_state: Optional[AgentState] = None
        self._chat_helper: Optional[ChatHelper] = None
        self._execution_lock = asyncio.Lock()
        self._function_service = function_service
        self._model_service = model_service
        self._available_tools: Dict[str, Dict[str, Any]] = {}
        self._tool_usage_stats: Dict[str, Dict[str, Any]] = {}

    @property
    def agent_state(self) -> AgentState:
        """Get the current agent state."""
        if not self._agent_state:
            self._agent_state = AgentState(
                goal="Complete the task",
                capabilities=[AgentCapability.FUNCTION_CALLING],
                constraints={}
            )
        return self._agent_state

    @property
    def chat_helper(self) -> ChatHelper:
        """Get the chat helper instance."""
        if not self._chat_helper:
            if not self._model_service:
                raise ValueError("Model service not provided for chat helper")
            self._chat_helper = ChatHelper(model_service=self._model_service)
        return self._chat_helper

    async def run_agent_loop(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Core agent execution loop."""
        context = await self._build_context()
        retry_config = self.agent_config.retry_config or RetryConfig()
        retry_count = 0

        # Run start hook
        await self._run_hook("on_start", context)

        while not await self._is_goal_complete():
            try:
                # Get next action from LLM
                messages = await self._build_messages(context)
                tools = await self._get_available_tools() if self.agent_config.enable_tools else None

                # Track if we got a successful completion
                success = False

                async for response in self.chat_helper.generate_completion(
                    messages=messages,
                    model=self.agent_config.model,
                    temperature=self.agent_config.temperature,
                    max_tokens=self.agent_config.max_tokens,
                    stream=self.agent_config.stream,
                    tools=tools,
                    enable_tools=self.agent_config.enable_tools
                ):
                    # Handle function calls or text responses
                    if isinstance(response, dict) and "function_call" in response:
                        result = await self._execute_action(response, context)
                        self.agent_state.last_tool_result = result
                        # Record the action in history
                        self.agent_state.history.append({
                            "timestamp": datetime.now().isoformat(),
                            "action": response.get("function_call", {}),
                            "result": result,
                            "state_snapshot": self.agent_state.get_execution_summary()
                        })
                        yield {"type": "tool_call", "result": result}
                        success = True
                    else:
                        # Handle regular text response
                        content = response if isinstance(
                            response, str) else response.get("content", "")

                        # Check for completion XML
                        has_completion, result, command, answer = parse_attempt_completion(
                            content)
                        if has_completion:
                            # Update agent state with completion signal
                            self.agent_state.update_progress(1.0)
                            self.agent_state.current_state["completion_result"] = result
                            if command:
                                self.agent_state.current_state["final_command"] = command
                            if answer:
                                self.agent_state.current_state["final_answer"] = answer
                            success = True

                        # Record the message in history
                        self.agent_state.history.append({
                            "timestamp": datetime.now().isoformat(),
                            "message": content,
                            "state_snapshot": self.agent_state.get_execution_summary()
                        })
                        yield {"type": "message", "content": content}

                # Only reset retry count if we got a successful completion
                if success:
                    retry_count = 0
                else:
                    # If we got here without success, it means we got partial responses but no completion
                    raise Exception("Incomplete response from model")

            except Exception as e:
                retry_count += 1
                logger.error(
                    f"Error in agent loop (attempt {retry_count}/{retry_config.max_retries}): {str(e)}")

                if retry_count >= retry_config.max_retries:
                    logger.error(
                        f"Max retries exceeded ({retry_count}/{retry_config.max_retries}), exiting agent loop")
                    self.agent_state.update_progress(1.0)
                    self.agent_state.current_state[
                        "completion_result"] = f"Failed after {retry_count} attempts: {str(e)}"
                    self.agent_state.current_state["completion_error"] = True
                    yield {
                        "type": "error",
                        "error": f"Max retries exceeded ({retry_count}/{retry_config.max_retries}): {str(e)}",
                        "retry_count": retry_count
                    }
                    break

                # Calculate and apply retry delay
                delay = retry_config.calculate_delay(retry_count)
                await asyncio.sleep(delay)
                continue

            await self._run_hook("on_iteration_end", context)
            context = await self._build_context()

        await self._run_hook("on_finish", context)
        yield {
            "type": "complete",
            "success": not self.agent_state.current_state.get("completion_error", False),
            "final_state": self.agent_state.get_execution_summary()
        }

    async def _build_messages(self, context: Dict[str, Any]) -> List[BaseMessage]:
        """Build messages for LLM interaction."""
        # Create system message with agent's goal and capabilities
        system_content = PROMPTS["agent_system"].format(
            goal=self.agent_state.goal,
            capabilities=[cap.value for cap in self.agent_state.capabilities],
            constraints=self.agent_state.constraints
        )

        # Add history summary if available
        history_summary = self._format_history_summary()
        if history_summary:
            system_content += f"\n\nPrevious Actions:\n{history_summary}"

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=self._format_context(context))
        ]
        return messages

    def _format_history_summary(self) -> str:
        """Format a summary of recent actions from history."""
        if not self.agent_state.history:
            return ""

        # Get last 5 actions for context
        recent_history = self.agent_state.history[-5:]
        summary = []

        for entry in recent_history:
            if "action" in entry:
                func_call = entry["action"]
                result = entry["result"]
                summary.append(
                    f"- Called {func_call.get('name')} with args {func_call.get('arguments')}")
                if isinstance(result, dict) and result.get('error'):
                    summary.append(f"  Result: Error - {result['error']}")
                else:
                    summary.append(f"  Result: Success")
            elif "message" in entry:
                summary.append(f"- Responded: {entry['message'][:100]}...")

        return "\n".join(summary)

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

    # Hook methods (empty by default)
    async def on_start(self, context: Dict[str, Any]): pass
    async def on_finish(self, context: Dict[str, Any]): pass
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

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for LLM consumption."""
        if not self._agent_state:
            return str(context)

        return f"""Current State: {context.get('current_state', {})}
Last Action Result: {context.get('last_tool_result', 'None')}
Progress: {context.get('progress', 0.0)}"""

    async def _is_goal_complete(self) -> bool:
        """Check if the current goal is complete."""
        # Check for explicit completion signal
        if "completion_result" in self.agent_state.current_state:
            return True
        # Check progress-based completion
        return self.agent_state.progress >= 1.0

    async def _execute_action(
        self,
        decision: Dict[str, Any],
        context: Dict[str, Any]
    ) -> FunctionResponse:
        """Execute actions, potentially delegating to worker agents."""
        action_type = decision.get("action_type", "default")

        # Check for function calls in metadata
        if "function_call" in decision:
            if not self._function_service:
                raise ValueError(
                    "Function service not provided for function calls")

            function_name = decision.get("function_name")
            function_args = decision.get("arguments", {})

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

        # Handle default action type (when no function call is present)
        else:
            response = AgentResponse(
                success=True,
                agent_name=self.name,
                state=self.agent_state,
                final_output={"content": decision.get(
                    "content", "No response generated")},
                metadata=decision.get("metadata", {})
            )
            # Convert to dict for JSON serialization
            return response.model_dump() if hasattr(response, 'model_dump') else vars(response)

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
        decision: Dict[str, Any],
        context: Dict[str, Any]
    ) -> FunctionResponse:
        """Execute actions, potentially delegating to worker agents."""
        action_type = decision.get("action_type", "default")

        if action_type == "delegate":
            # Extract worker and task from decision
            worker_name = decision.get("worker")
            task = decision.get("task")
            if not worker_name or not task:
                raise ValueError("Delegation requires worker name and task")

            return await self.delegate_task(worker_name, task)

        elif action_type == "parallel_delegate":
            # Handle parallel delegation to multiple workers
            tasks = decision.get("tasks", [])
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
