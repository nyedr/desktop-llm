### Agent System

The agent system provides autonomous capabilities for complex workflows and decision making. Agents can plan, reason, and execute multi-step processes using available tools and memory.

#### Core Components

1. **BaseAgent** (`app/functions/agent.py`):
   - Abstract base class defining the interface for all agentic workflows
   - Core phases: think, decide, act, reflect
   - Standardized agent loop implementation

```python
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
```

2. **GeneralAgent** (`app/functions/agent.py`):
   - Concrete implementation of BaseAgent
   - Configurable behavior through AgentConfig
   - Enhanced retry logic with multiple backoff strategies
   - Comprehensive hook system for customization
   - Tool management and usage statistics

```python
class GeneralAgent(BaseAgent):
    """General purpose agent implementation with configurable behavior."""

    def __init__(
        self,
        name: str,
        config: Optional[AgentConfig] = None,
        function_service=None
    ):
        self.name = name
        self.agent_config = config or AgentConfig()
        self._agent_state = None
        self._chat_helper = None
        self._execution_lock = asyncio.Lock()
        self._function_service = function_service
```

3. **SupervisorAgent** (`app/functions/agent.py`):
   - Extends GeneralAgent for orchestrating multiple worker agents
   - Task delegation capabilities
   - Parallel execution support
   - Worker state management

```python
class SupervisorAgent(GeneralAgent):
    """A specialized agent that can orchestrate multiple worker agents."""

    def __init__(
        self,
        name: str,
        worker_agents: Optional[Dict[str, GeneralAgent]] = None,
        config: Optional[AgentConfig] = None
    ):
        super().__init__(name, config)
        self.worker_agents = worker_agents or {}
        self.task_queue = asyncio.Queue()
        self.results = {}
```

#### Agent Registration and Management

The agent system provides a robust registration system through `AgentRegistry` that allows for easy agent discovery and management:

1. **Registering Custom Agents**:

   ```python
   from app.functions.agent_registry import register_agent
   from app.models.agent import AgentCapability

   @register_agent(
       name="research_agent",
       description="Specialized agent for research tasks",
       capabilities=[
           AgentCapability.REASONING,
           AgentCapability.MEMORY,
           AgentCapability.FUNCTION_CALLING
       ]
   )
   class ResearchAgent(GeneralAgent):
       async def think(self, context: Dict[str, Any], thought_type: str = "research") -> AsyncGenerator[AgentThought, None]:
           # Custom research-focused thinking implementation
           pass
   ```

2. **Using the Agent Service**:

   ```python
   from app.services.agent_service import AgentService

   # Initialize service
   agent_service = AgentService(function_service)

   # Run an agent flow
   async for step in agent_service.run_agent_flow(
       goal="Research quantum computing advances",
       messages=[],
       capabilities=[AgentCapability.REASONING, AgentCapability.MEMORY],
       constraints={"max_sources": 5},
       metadata={"agent_type": "research_agent"}
   ):
       print(f"Step: {step}")
   ```

3. **Creating a Supervisor with Workers**:

   ```python
   # Create worker agents
   workers = {
       "researcher": ResearchAgent("researcher"),
       "writer": WriterAgent("writer"),
       "reviewer": ReviewerAgent("reviewer")
   }

   # Create supervisor
   supervisor = SupervisorAgent(
       name="research_supervisor",
       worker_agents=workers,
       config=AgentConfig(
           model="deepseek/deepseek-chat",
           temperature=0.7
       )
   )

   # Delegate tasks
   result = await supervisor.delegate_task(
       "researcher",
       {
           "goal": "Find recent papers on quantum computing",
           "constraints": {"year_range": [2022, 2023]}
       }
   )
   ```

#### Configuration System

1. **AgentConfig**:

   ```python
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
       disabled_phases: List[str] = None
       hook_callbacks: Dict[str, List[Callable]] = None

       # Thought configuration
       custom_thought_prompts: Dict[str, str] = None
       custom_thought_types: Dict[str, Dict[str, Any]] = None

       # Retry configuration
       retry_config: Optional[RetryConfig] = None
   ```

2. **RetryConfig and BackoffStrategy**:

   ```python
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
       retry_on_exceptions_only: bool = True
   ```

#### Hook System

The agent system provides a comprehensive hook system for customizing behavior:

1. **Available Hooks**:

   - `on_start`: Called when agent starts
   - `on_finish`: Called when agent completes
   - `before_think/after_think`: Around thought generation
   - `before_decide/after_decide`: Around decision making
   - `before_act/after_act`: Around action execution
   - `before_reflect/after_reflect`: Around reflection
   - `on_iteration_end`: Called at the end of each iteration

2. **Hook Management**:

   ```python
   # Adding hooks
   agent.add_hook_callback("before_think", my_callback)

   # Removing hooks
   agent.remove_hook_callback("before_think", my_callback)

   # Disabling phases
   agent_config = AgentConfig(disabled_phases=["reflect"])
   ```

3. **Custom Hook Implementation**:

   ```python
   async def log_thought_hook(thoughts: List[AgentThought], context: Dict[str, Any]):
       for thought in thoughts:
           logger.info(f"Agent thought: {thought.content}")

   # Register the hook
   agent.add_hook_callback("after_think", log_thought_hook)
   ```

#### Standardized Responses

The system defines standard response types for all agent actions:

1. **Tool Response**:

   ```python
   class ToolResponse(FunctionResponse):
       """Response from tool execution."""
       result: Any
       tool_name: str
       execution_time: float = 0.0
   ```

2. **Agent Response**:

   ```python
   class AgentResponse(FunctionResponse):
       """Response from agent execution."""
       agent_name: str
       state: AgentState
       thoughts: List[Dict[str, Any]]
       decisions: List[Dict[str, Any]]
       actions_taken: List[Dict[str, Any]]
       final_output: Dict[str, Any]
   ```

3. **Pipeline Response**:
   ```python
   class PipelineResponse(FunctionResponse):
       """Response from pipeline execution."""
       results: List[Dict[str, Any]]
       pipeline_name: str
       steps_completed: int
       total_steps: int
   ```

#### Tool Management

The agent system provides comprehensive tool management capabilities:

1. **Tool Configuration**:

   ```python
   agent.configure_tools(
       allowed_tools=["tool1", "tool2"],
       excluded_tools=["dangerous_tool"],
       custom_tools=[custom_tool_schema],
       tool_policies={
           "tool1": {
               "rate_limit": 10,
               "max_retries": 3
           }
       }
   )
   ```

2. **Tool Usage Statistics**:

   ```python
   # Get stats for specific tool
   stats = agent.get_tool_stats("tool1")

   # Get stats for all tools
   all_stats = agent.get_tool_stats()
   ```

3. **Tool Policy Validation**:

   ```python
   from app.functions.utils import validate_tool_policy

   policy = {
       "rate_limit": 10,
       "max_retries": 3,
       "timeout": 30.0,
       "cache_results": True
   }

   validated_policy = validate_tool_policy(policy)
   ```

#### Advanced Usage Examples

1. **Custom Agent with Memory Integration**:

   ```python
   @register_agent(
       name="memory_agent",
       capabilities=[AgentCapability.MEMORY, AgentCapability.REASONING]
   )
   class MemoryAwareAgent(GeneralAgent):
       def __init__(self, name: str, config: Optional[AgentConfig] = None):
           super().__init__(name, config)
           self.memory_store = {}

       async def think(self, context: Dict[str, Any], thought_type: str = "reason"):
           # Access memory before thinking
           relevant_memories = self.memory_store.get(thought_type, [])
           context["memories"] = relevant_memories

           async for thought in super().think(context, thought_type):
               # Store new thoughts in memory
               if thought.content not in relevant_memories:
                   self.memory_store.setdefault(thought_type, []).append(thought.content)
               yield thought
   ```

2. **Parallel Task Processing with Supervisor**:

   ```python
   async def process_parallel_tasks():
       supervisor = SupervisorAgent("task_supervisor")

       # Define parallel tasks
       tasks = [
           {"worker": "researcher", "task": {"goal": "Research topic A"}},
           {"worker": "writer", "task": {"goal": "Write summary of topic B"}},
           {"worker": "reviewer", "task": {"goal": "Review document C"}}
       ]

       # Execute parallel delegation
       decision = AgentDecision(
           action_type="parallel_delegate",
           action_plan="Execute research, writing, and review in parallel",
           metadata={"tasks": tasks}
       )

       result = await supervisor._execute_action(decision, {})
       return result
   ```

3. **Custom Retry Strategy**:

   ```python
   def custom_backoff(attempt: int) -> float:
       # Custom exponential backoff with maximum delay of 30 seconds
       return min(5 * (2 ** attempt), 30)

   retry_config = RetryConfig(
       max_retries=5,
       custom_backoff_func=custom_backoff,
       retry_exceptions=(NetworkError, TimeoutError)
   )

   agent_config = AgentConfig(retry_config=retry_config)
   agent = GeneralAgent("resilient_agent", config=agent_config)
   ```
