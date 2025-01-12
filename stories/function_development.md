# Function Development Guide

## Overview

This guide provides comprehensive documentation for developing functions in the Desktop LLM system. The function system is designed to be modular, extensible, and type-safe, with clear separation of concerns between different function types.

## System Architecture

### Core Components

1. **Base Classes** (`app/functions/base.py`):

   - Defines the foundational types and interfaces
   - Provides error handling classes and retry mechanisms
   - Implements parameter normalization
   - Implements the function registration decorator

2. **Registry** (`app/functions/registry.py`):

   - Manages function registration and discovery
   - Handles dynamic loading of functions
   - Maintains function metadata
   - Provides Pydantic v1/v2 compatibility layer

3. **Executor** (`app/functions/executor.py`):

   - Executes functions with validation
   - Handles tool calls from the LLM
   - Provides error handling and logging

4. **Utilities** (`app/functions/utils.py`):

   - Common utilities for function development
   - Message handling helpers
   - Model interaction utilities
   - Application constants and settings
   - Message type conversion and validation

5. **Chat Helper** (`app/functions/chat.py`):
   - Streamlined interface for LLM interactions
   - Model management utilities
   - Chat completion generation
   - Proper error handling and logging

### Important Implementation Notes

#### Function Registration and Parameters

When creating new functions, parameters are now defined in the `@register_function` decorator rather than as class variables:

```python
from typing import Dict, Any, Literal
from app.models.function import Tool, FunctionType
from pydantic import Field, ConfigDict

@register_function(
    func_type=FunctionType.TOOL,
    name="my_tool",
    description="Description of what the tool does",
    parameters={  # Parameters defined here
        "type": "object",
        "properties": {
            "param_name": {
                "type": "string",
                "description": "Parameter description"
            }
        },
        "required": ["param_name"]
    }
)
class MyTool(Tool):
    """Tool documentation."""

    # Required: Allow arbitrary types if using custom service classes
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Required: Define type, name, and description using Field
    type: Literal[FunctionType.TOOL] = Field(
        default=FunctionType.TOOL,
        description="Tool type"
    )
    name: str = Field(
        default="my_tool",
        description="Name of the tool"
    )
    description: str = Field(
        default="Description of what the tool does",
        description="Description of what the tool does"
    )

    # Important: Any instance variables must be declared as Fields
    my_service: MyService = Field(
        default_factory=MyService,  # Use default_factory for class instances
        exclude=True  # Exclude from schema if not needed in API
    )
```

**Common Mistakes to Avoid:**

- ❌ Don't define parameters as a class variable
- ❌ Don't use ClassVar for parameters
- ❌ Don't initialize instance variables without Field declarations
- ❌ Don't forget model_config for custom service types

**Correct Pattern:**

- ✅ Define parameters in the register_function decorator
- ✅ Use Pydantic's Field for all attributes
- ✅ Include proper type annotations
- ✅ Provide default values and descriptions
- ✅ Declare all instance variables as Fields

### Function Types

#### 1. Tools (`class Tool(BaseFunction)`)

Tools extend the LLM's capabilities through function calling:

```python
@register_function(
    func_type=FunctionType.TOOL,
    name="web_scrape",
    description="Scrape and process web pages",
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to scrape"
            },
            "raw": {
                "type": "boolean",
                "description": "Get raw content",
                "default": False
            }
        },
        "required": ["url"]
    }
)
class WebScrapeTool(Tool):
    """Tool for web scraping with retry and error handling."""

    async def _execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Implement the actual tool logic."""
        try:
            return await self.scrape_url(args["url"], args.get("raw", False))
        except Exception as e:
            raise ExecutionError(f"Scraping failed: {str(e)}")

    def normalize_parameters(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Optional: Normalize parameters before execution."""
        return args

    async def _handle_error(self, error: Exception, args: Dict[str, Any]) -> Dict[str, Any]:
        """Optional: Custom error handling logic."""
        return args
```

#### 2. Filters (`class Filter(BaseFunction)`)

Filters modify data flow with priority-based execution:

```python
@register_function(
    func_type=FunctionType.FILTER,
    name="text_modifier",
    description="Modifies text content",
    priority=1,  # Lower number = higher priority
    config={
        "prefix": "[Modified] ",
        "suffix": " [End]"
    }
)
class TextModifierFilter(Filter):
    """Filter that processes both incoming and outgoing data."""

    async def inlet(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming data (high to low priority)."""
        messages = data.get("messages", [])
        if not messages:
            return data

        modified_messages = []
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                message = message.copy()
                message["content"] = self._modify_content(message["content"])
            modified_messages.append(message)

        data["messages"] = modified_messages
        return data

    async def outlet(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process outgoing data (low to high priority)."""
        if data.get("role") == "assistant":
            data = data.copy()
            if isinstance(data.get("content"), str):
                data["content"] = self._modify_content(data["content"])
        return data
```

#### 3. Pipelines (`class Pipeline(BaseFunction)`)

Pipelines orchestrate complex multi-step workflows:

```python
@register_function(
    func_type=FunctionType.PIPELINE,
    name="multi_step_processor",
    description="Processes data through multiple steps",
    config={
        "max_steps": 3,
        "timeout_per_step": 30
    }
)
class MultiStepPipeline(Pipeline):
    """Pipeline with multiple processing steps."""

    async def pipe(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through defined steps."""
        result = data

        # Step 1: Message normalization
        messages = [ensure_strict_message(msg) for msg in result.get("messages", [])]

        # Step 2: Content processing
        processed_messages = []
        for msg in messages:
            if msg.role == "user":
                processed_content = await self._process_user_message(msg.content)
                processed_messages.append(
                    UserMessage(
                        role="user",
                        content=processed_content
                    )
                )
            else:
                processed_messages.append(msg)

        # Step 3: Generate summary
        summary = await self._generate_summary(processed_messages)

        return {
            "messages": processed_messages,
            "summary": summary
        }

    async def _process_user_message(self, content: str) -> str:
        """Custom processing for user messages."""
        pass

    async def _generate_summary(self, messages: List[StrictChatMessage]) -> Dict[str, Any]:
        """Generate summary of processed messages."""
        pass
```

### Error Handling System

The system provides a comprehensive error hierarchy:

```python
class FunctionError(Exception):
    """Base class for all function errors."""
    pass

class ValidationError(FunctionError):
    """Base for validation errors."""
    pass

class InputValidationError(ValidationError):
    """Input validation failure."""
    pass

class OutputValidationError(ValidationError):
    """Output validation failure."""
    pass

class TimeoutError(FunctionError):
    """Execution timeout."""
    pass

class ExecutionError(FunctionError):
    """General execution failure."""
    pass

class SecurityError(FunctionError):
    """Security violation."""
    pass

class ModuleImportError(FunctionError):
    """Module import failure."""
    pass
```

### Parameter Normalization

The system includes a high-priority filter for parameter normalization:

```python
@register_function(
    func_type=FunctionType.FILTER,
    name="parameter_normalizer",
    description="Normalizes function parameters",
    priority=1
)
class ParameterNormalizerFilter(Filter):
    """System-wide parameter normalization."""

    COMMON_NORMALIZATIONS = {
        "temperature_units": {
            "celsius": ["Celsius", "CELSIUS", "C", "c"],
            "fahrenheit": ["Fahrenheit", "FAHRENHEIT", "F", "f"]
        },
        "boolean_values": {
            True: ["true", "True", "TRUE", "1", "yes", "Yes", "YES"],
            False: ["false", "False", "FALSE", "0", "no", "No", "NO"]
        }
    }

    async def inlet(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize parameters in tool calls."""
        if "tool_calls" in data:
            for tool_call in data["tool_calls"]:
                if "function" in tool_call:
                    args = tool_call["function"].get("arguments", {})
                    tool_call["function"]["arguments"] = self._normalize_parameters(
                        args,
                        tool_call["function"].get("name")
                    )
        return data
```

### Utilities

The system provides several utility functions:

```python
# Message handling
def get_last_user_message(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Get the last user message."""
    pass

def get_last_assistant_message(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Get the last assistant message."""
    pass

def get_system_message(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Get the system message."""
    pass

def ensure_strict_message(msg: Any) -> StrictChatMessage:
    """Convert to StrictChatMessage with validation."""
    pass

# Response validation
def validate_function_response(response: FunctionResult) -> bool:
    """Validate a function response."""
    pass

def validate_tool_response(response: ToolResponse) -> bool:
    """Validate a tool response."""
    pass

def validate_filter_response(response: FilterResponse) -> bool:
    """Validate a filter response."""
    pass

def validate_pipeline_response(response: PipelineResponse) -> bool:
    """Validate a pipeline response."""
    pass

def ensure_response_type(response: Any, expected_type: Type[FunctionResult]) -> FunctionResult:
    """Ensure a response matches the expected type."""
    pass

def create_error_response(error: Exception, function_type: str, function_name: str, **kwargs) -> FunctionResult:
    """Create an error response of the appropriate type."""
    pass

# Model interaction
async def generate_chat_completion(
    messages: List[StrictChatMessage],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    tools: Optional[List[Dict[str, Any]]] = None,
    enable_tools: bool = True,
    function_service=None
) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
    """Generate chat completions with streaming support."""
    pass

async def get_all_models() -> List[str]:
    """Get available models."""
    pass
```

### Using the Chat Helper

The `ChatHelper` class provides a streamlined interface for functions that need to interact with language models:

```python
from app.functions.chat import ChatHelper
from app.models.chat import UserMessage, SystemMessage

class MyLLMTool(Tool):
    async def _execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        chat_helper = ChatHelper()

        # Create messages
        messages = [
            SystemMessage(content="System prompt"),
            UserMessage(content=args["user_input"])
        ]

        # Generate completion
        responses = []
        async for response in chat_helper.generate_completion(
            messages=messages,
            model="llama2",  # Optional, uses default if not specified
            temperature=0.7,  # Optional
            stream=True      # Optional
        ):
            responses.append(response)

        return {"responses": responses}
```

Key features of the ChatHelper:

1. **Model Management**:

   - Get available models with `get_available_models()`
   - Automatic fallback to default model
   - Proper error handling for model issues

2. **Chat Generation**:

   - Streaming support with async iteration
   - Configurable parameters (temperature, max_tokens)
   - Tool integration capabilities
   - Proper error handling and logging

3. **Integration with Function System**:
   - Compatible with Tool, Filter, and Pipeline types
   - Supports function calling in chat completions
   - Handles both streaming and non-streaming responses

### Response Types

The system defines standard response types for all functions:

```python
class FunctionResponse(BaseModel):
    """Base class for function responses."""
    success: bool = Field(default=True, description="Whether the function executed successfully")
    error: Optional[str] = Field(default=None, description="Error message if execution failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the execution")

class ToolResponse(FunctionResponse):
    """Response from tool execution."""
    result: Any = Field(..., description="The result of the tool execution")
    tool_name: str = Field(..., description="Name of the tool that was executed")
    execution_time: float = Field(default=0.0, description="Time taken to execute the tool in seconds")

class FilterResponse(FunctionResponse):
    """Response from filter execution."""
    modified_data: Dict[str, Any] = Field(..., description="The modified data after filtering")
    filter_name: str = Field(..., description="Name of the filter that was executed")
    changes_made: bool = Field(default=False, description="Whether any changes were made to the data")

class PipelineResponse(FunctionResponse):
    """Response from pipeline execution."""
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Results from each step in the pipeline")
    pipeline_name: str = Field(..., description="Name of the pipeline that was executed")
    steps_completed: int = Field(default=0, description="Number of steps completed in the pipeline")
    total_steps: int = Field(default=0, description="Total number of steps in the pipeline")
```

### Response Validation

The system provides comprehensive validation for function responses:

```python
# Validate a tool response
try:
    result = await tool.execute(args)
    validate_tool_response(result)
except ValidationError as e:
    logger.error(f"Invalid tool response: {e}")
    result = create_error_response(e, "tool", tool.name)

# Validate a filter response
try:
    result = await filter.execute(args)
    validate_filter_response(result)
except ValidationError as e:
    logger.error(f"Invalid filter response: {e}")
    result = create_error_response(e, "filter", filter.name)

# Validate a pipeline response
try:
    result = await pipeline.execute(args)
    validate_pipeline_response(result)
except ValidationError as e:
    logger.error(f"Invalid pipeline response: {e}")
    result = create_error_response(e, "pipeline", pipeline.name)

# Ensure response type matches expectation
try:
    result = ensure_response_type(response, ToolResponse)
except ValidationError as e:
    logger.error(f"Response type mismatch: {e}")
    result = create_error_response(e, "tool", "unknown")
```

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

### Utility Functions

The system provides a comprehensive set of utility functions in `app/functions/utils.py` to support function development and execution:

#### Message Handling Utilities

```python
def get_last_user_message(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Get the last user message from conversation history."""
    pass

def get_last_assistant_message(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Get the last assistant message from conversation history."""
    pass

def get_system_message(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Get the system message from conversation history."""
    pass

def ensure_strict_message(msg: Any) -> StrictChatMessage:
    """Convert input to StrictChatMessage with validation."""
    pass
```

#### Response Validation Utilities

```python
def validate_function_response(response: FunctionResult) -> bool:
    """Validate a function response."""
    pass

def validate_tool_response(response: ToolResponse) -> bool:
    """Validate a tool response."""
    pass

def validate_filter_response(response: FilterResponse) -> bool:
    """Validate a filter response."""
    pass

def validate_pipeline_response(response: PipelineResponse) -> bool:
    """Validate a pipeline response."""
    pass

def ensure_response_type(response: Any, expected_type: Type[FunctionResult]) -> FunctionResult:
    """Ensure response matches expected type."""
    pass

def create_error_response(error: Exception, function_type: str, function_name: str, **kwargs) -> FunctionResult:
    """Create an error response of appropriate type."""
    pass
```

#### Tool Management Utilities

```python
def get_registered_tools() -> List[Dict[str, Any]]:
    """Get all registered tools from function service."""
    pass

def verify_registered_tool(tool_name: str) -> Optional[Dict[str, Any]]:
    """Verify if a tool is registered and get its schema."""
    pass

def validate_tool_names(tool_names: List[str]) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Validate tool names against registered tools."""
    pass

def validate_tool_policy(policy: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a tool policy configuration."""
    pass

def get_safe_tool_list(
    tool_names: Optional[List[str]] = None,
    required_capabilities: Optional[List[str]] = None,
    validate_policies: bool = True
) -> List[Dict[str, Any]]:
    """Get a filtered and validated list of registered tools."""
    pass
```

#### Application Constants

The module also provides access to important application constants:

```python
APP_CONSTANTS = {
    "DEFAULT_MODEL": config.llm.model,
    "MODEL_TEMPERATURE": config.llm.temperature,
    "MAX_TOKENS": config.llm.max_tokens,
    "FUNCTION_CALLS_ENABLED": config.llm.enable_tools,
    "ENABLE_MODEL_FILTER": config.functions.enable_model_filter,
    "MODEL_FILTER_LIST": config.functions.model_filter_list,
    "BASE_URL": config.llm.base_url,
    "MODEL_REQUEST_TIMEOUT": config.llm.timeout,
    "GENERATION_REQUEST_TIMEOUT": config.llm.timeout
}
```

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

### Response Validation Rules

1. **Common Rules for All Responses**:

   - Must be an instance of the correct response type
   - Failed responses must include an error message
   - Success flag must be boolean
   - Metadata must be a dictionary

2. **Tool Response Rules**:

   - Successful responses must include a result
   - Execution time cannot be negative
   - Tool name must be provided

3. **Filter Response Rules**:

   - Modified data must be a dictionary
   - Changes made flag must be boolean
   - Filter name must be provided

4. **Pipeline Response Rules**:
   - Completed steps cannot exceed total steps
   - Step counts cannot be negative
   - Results must be a list
   - Pipeline name must be provided

### Error Response Creation

The system provides a utility to create appropriate error responses:

```python
# Create a tool error response
error_response = create_error_response(
    error=ValueError("Invalid input"),
    function_type="tool",
    function_name="my_tool",
    execution_time=1.5
)

# Create a filter error response
error_response = create_error_response(
    error=RuntimeError("Processing failed"),
    function_type="filter",
    function_name="my_filter"
)

# Create a pipeline error response
error_response = create_error_response(
    error=Exception("Step 2 failed"),
    function_type="pipeline",
    function_name="my_pipeline",
    steps_completed=1,
    total_steps=3
)
```

### Pydantic Compatibility

The system supports both Pydantic v1 and v2:

```python
def pydantic_field_exists(func_cls, field_name: str) -> bool:
    """Check field existence in v1/v2."""
    if hasattr(func_cls, "model_fields"):  # v2
        return field_name in func_cls.model_fields
    else:  # v1
        return field_name in func_cls.__fields__

def get_field_default(func_cls, field_name: str):
    """Get field default in v1/v2."""
    if hasattr(func_cls, "model_fields"):  # v2
        return func_cls.model_fields[field_name].default
    else:  # v1
        return func_cls.__fields__[field_name].default

def set_field_default(func_cls, field_name: str, value):
    """Set field default in v1/v2."""
    if hasattr(func_cls, "model_fields"):  # v2
        func_cls.model_fields[field_name].default = value
    else:  # v1
        func_cls.__fields__[field_name].default = value
        func_cls.__fields__[field_name].field_info.default = value
```

## Version Compatibility

### Python Version Requirements

- Minimum Python version: 3.9
- Required for:
  - Type hints
  - Async/await syntax
  - Dict union operations

### Dependencies

Specify dependencies in `requirements.txt`:

```txt
pydantic>=2.0.0
jsonschema>=4.0.0
asyncio>=3.4.3
```

### Breaking Changes

Document any breaking changes in your function:

```python
class VersionedTool(Tool):
    """A tool with version compatibility notes.

    Version History:
    - 1.0.0: Initial release
    - 1.1.0: Added streaming support
    - 2.0.0: Breaking change - New parameter format

    Minimum Requirements:
    - Python 3.9+
    - Pydantic 2.0+
    """
    pass
```

## Security Considerations

### 1. Input Validation

Always validate inputs thoroughly:

```python
class SecureTool(Tool):
    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        # Validate input types
        if not isinstance(args.get("input"), str):
            raise InputValidationError("Input must be a string")

        # Sanitize inputs
        sanitized_input = sanitize_input(args["input"])

        # Check for malicious content
        if contains_malicious_content(sanitized_input):
            raise SecurityError("Malicious content detected")
```

### 2. Resource Management

Implement proper resource management:

```python
class ResourceAwareTool(Tool):
    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        # Set timeouts
        timeout = self.config.get("timeout", 30)

        try:
            async with asyncio.timeout(timeout):
                # Resource-intensive operation
                result = await self.process(args)

        except asyncio.TimeoutError:
            # Clean up resources
            await self.cleanup()
            raise TimeoutError(f"Operation timed out after {timeout}s")
```

## Testing

Create comprehensive tests for your functions:

```python
# test_calculator_tool.py
import pytest
from app.functions.types.tools.calculator import CalculatorTool

@pytest.mark.asyncio
async def test_calculator_addition():
    tool = CalculatorTool()
    result = await tool.execute({
        "operation": "add",
        "a": 5,
        "b": 3
    })
    assert result["result"] == 8

@pytest.mark.asyncio
async def test_calculator_invalid_input():
    tool = CalculatorTool()
    with pytest.raises(InputValidationError):
        await tool.execute({
            "operation": "invalid",
            "a": 5,
            "b": 3
        })
```

### Testing Error Handling

Create comprehensive tests for your error handling:

```python
@pytest.mark.asyncio
async def test_weather_tool_parameter_normalization():
    tool = WeatherTool()

    # Test unit normalization
    args = {"location": "New York", "unit": "Celsius"}
    normalized = tool.normalize_parameters(args)
    assert normalized["unit"] == "celsius"

    # Test error handling
    with pytest.raises(InputValidationError):
        await tool.execute({"location": "New York", "unit": "invalid"})

    # Test retry mechanism
    result = await tool.execute({"location": "New York", "unit": "C"})
    assert result["unit"] == "celsius"
```

## Debugging and Monitoring

### 1. Logging

Implement proper logging:

```python
import logging

logger = logging.getLogger(__name__)

class LoggingTool(Tool):
    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Starting execution with args: {args}")

        try:
            result = await self.process(args)
            logger.debug(f"Processing result: {result}")
            return result

        except Exception as e:
            logger.error(f"Error during execution: {e}", exc_info=True)
            raise
```

### 2. Performance Monitoring

Monitor execution times and resource usage:

```python
import time
import psutil

class MonitoredTool(Tool):
    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        try:
            result = await self.process(args)

            # Log performance metrics
            execution_time = time.time() - start_time
            memory_used = psutil.Process().memory_info().rss - start_memory

            logger.info(f"Execution time: {execution_time:.2f}s")
            logger.info(f"Memory used: {memory_used / 1024 / 1024:.2f}MB")

            return result

        except Exception as e:
            logger.error(f"Error during execution: {e}", exc_info=True)
            raise
```

## Best Practices for Error Handling

1. **Layer Your Defenses**:

   - Use the parameter normalizer filter for common cases
   - Implement function-specific normalization
   - Add retry logic for transient failures

2. **Normalize Early**:

   ```python
   def normalize_parameters(self, args: Dict[str, Any]) -> Dict[str, Any]:
       normalized = args.copy()
       # Normalize at the start of execution
       normalized["param"] = self._normalize_param(normalized.get("param"))
       return normalized
   ```

3. **Handle Errors Gracefully**:

   ```python
   async def _handle_error(self, error: Exception, args: Dict[str, Any]) -> Dict[str, Any]:
       if isinstance(error, InputValidationError):
           # Try to fix the input
           return self._fix_validation_error(error, args)
       elif isinstance(error, TimeoutError):
           # Maybe reduce the scope of the request
           return self._reduce_request_scope(args)
       return args
   ```

4. **Provide Helpful Error Messages**:

   ```python
   def _fix_validation_error(self, error: InputValidationError, args: Dict[str, Any]) -> Dict[str, Any]:
       fixed = args.copy()
       if "unit" in error.details.get("invalid_params", []):
           logger.info(f"Converting invalid unit '{args['unit']}' to 'fahrenheit'")
           return {"unit": "fahrenheit", **{k:v for k,v in args.items() if k != "unit"}}
       return args
   ```

5. **Log Extensively**:
   ```python
   async def _execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
       try:
           logger.debug(f"Executing {self.name} with args: {args}")
           result = await self._do_execute(args)
           logger.debug(f"Execution result: {result}")
           return result
       except Exception as e:
           logger.error(f"Error in {self.name}: {e}", exc_info=True)
           raise
   ```

## Function-Specific Parameter Handling

Individual functions can implement custom parameter normalization and error handling:

```python
class WeatherTool(Tool):
    # Parameter normalization mappings
    UNIT_MAPPINGS = {
        "celsius": ["Celsius", "CELSIUS", "C", "c", "centigrade"],
        "fahrenheit": ["Fahrenheit", "FAHRENHEIT", "F", "f"]
    }

    def normalize_parameters(self, args: Dict[str, Any]) -> Dict[str, Any]:
        normalized = args.copy()

        # Normalize temperature unit
        if "unit" in normalized:
            unit = str(normalized["unit"]).lower()
            for standard, variants in self.UNIT_MAPPINGS.items():
                if unit in [v.lower() for v in variants]:
                    normalized["unit"] = standard
                    break

        return normalized

    def _fix_validation_error(self, error: InputValidationError, args: Dict[str, Any]) -> Dict[str, Any]:
        fixed = args.copy()
        if "unit" in error.details.get("invalid_params", []):
            fixed["unit"] = "fahrenheit"  # Default to fahrenheit
        return fixed
```
