### **App Requirements and Expectations**

#### **1. Ensure Proper Utilization of Layers**

The application must use the three defined layers—**Preprocessing**, **Postprocessing**, and **Function Execution**—to manage interactions between user inputs, the LLM, and function executions effectively. Each layer must have specific responsibilities:

- **Preprocessing Layer**:
  - Handles user inputs and applies all filter functions.
  - Prepares the prompt for the LLM, ensuring it aligns with the agent's expectations.
- **Postprocessing Layer**:
  - Processes the LLM output, including formatting, validation, and application of pipe functions to adjust or enrich the results.
  - Handles any output transformations or updates needed before the response is returned to the user.
- **Function Execution Layer**:
  - Executes all action functions defined by the application, such as interacting with external APIs, running calculations, or managing data storage.
  - Provides results to the postprocessing layer for integration into the response.

#### **2. Update Function Schema**

Functions must adhere to a unified schema with the following attributes:

- **`name`**: A unique identifier for the function.
- **`type`**: Specifies whether the function is a **filter**, **pipe**, or **action**:
  - `filter`: Operates in the preprocessing layer.
  - `pipe`: Operates in the postprocessing layer, affecting how the LLM is called or behaves.
  - `action`: Executes logic at the function execution layer.
- **`parameters`**: Describes the inputs required for the function, defined in a structured format.
- **`description`**: A brief explanation of the function's purpose.
- **`dependencies`** (optional): Specifies any dependencies the function requires (e.g., third-party libraries or APIs).
- **`valves`**: Metadata describing dynamic toggles or configurations for the function.

This schema ensures consistency across the app, simplifying function integration and management.

#### **3. Enforce Responsibility Separation**

Each layer and component must have clearly defined responsibilities to promote modularity and prevent cross-layer contamination:

- **Filters**:
  - Modify or validate user inputs.
  - Must not depend on LLM outputs.
- **Pipes**:
  - Enrich or adjust LLM outputs, such as applying pipelines to process raw responses.
  - Should not perform external API calls or execute heavy logic.
- **Actions**:
  - Handle complex execution, including API calls, data manipulation, or auxiliary computations.
  - Should not affect LLM behavior or outputs directly.

#### **4. Dynamic Handling of Functions**

Functions must be dynamically discoverable and configurable:

- Use a registry or plugin system to dynamically load, register, and manage functions.
- Allow runtime toggling of specific functions (e.g., enabling/disabling filters or actions).
- Provide a utility to categorize and organize functions by type, dependencies, and context.
- Incorporate error handling for scenarios where a function fails to execute properly (e.g., fallback mechanisms or retries).

#### **5. Compatibility Across Layers**

The system must ensure compatibility and seamless interaction between the layers:

- **Agent Integration**:
  - Centralize all agent-related features into a unified `Agent` class (e.g., `Agent.generate_response()`), providing a standardized interface for interacting with the LLM and executing tasks.
  - The `Agent` class must encapsulate core functionalities such as prompt validation, LLM calls, tool invocations, and pipeline handling.
- **Shared Utilities**:
  - Provide app utilities (e.g., `AgentUtils.create_prompt()`) that allow functions like pipes and filters to interact with the `Agent` without duplicating logic.
- **Standardized API**:
  - Ensure filters, pipes, and actions use a common interface to interact with the system, enabling reuse and consistent error handling.

#### **6. Documentation and Developer Guidelines**

Comprehensive documentation must be provided to ensure developers can easily extend and maintain the system:

- **Function Development**:
  - Provide clear guidelines for creating new functions, including a detailed explanation of the schema, supported types, and examples.
- **Layer Usage**:
  - Document the purpose and boundaries of each layer, with practical examples to illustrate their usage.
- **Agent Utilities**:
  - Document the `Agent` class and related utilities, specifying how to use them for LLM interaction, prompt handling, and tool execution.
- **Error Handling**:
  - Provide examples of how to handle common errors in each layer, including fallback mechanisms and debugging tools.
- **Testing and Debugging**:
  - Include testing guidelines to ensure functions and layers behave as expected.
  - Provide tools for logging and debugging, with configurable verbosity for developers.
- **Versioning and Compatibility**:
  - Outline versioning requirements for functions and their dependencies, ensuring compatibility with the app and other layers.

### **Function System Revamp**

### **System Architecture**

#### **1. Function Types**

##### **Filters**
- **Purpose:** Intercept and modify data flow
- **Execution Points:**
  - `inlet`: Pre-processing before LLM
  - `outlet`: Post-processing after LLM
- **Use Cases:**
  - Content moderation
  - Format conversion
  - Dynamic routing
  - Context injection
  - Image preprocessing

##### **Tools (Pipes)**
- **Purpose:** Extend LLM capabilities via function calling
- **Execution:** Triggered by LLM function calls
- **Use Cases:**
  - External API access
  - Calculations
  - Data retrieval
  - File operations

##### **Pipelines**
- **Purpose:** Custom processing sequences
- **Execution:** Full control over interaction flow
- **Use Cases:**
  - Custom LLM providers
  - Multi-step workflows
  - Complex data processing

### **2. Execution Flow**

1. **Inlet Filters**
   - Execute in priority order
   - Modify incoming requests

2. **Main Processing**
   - Pipeline execution if defined
   - Tool execution if called by LLM
   - Default LLM interaction

3. **Outlet Filters**
   - Execute in reverse priority order
   - Modify outgoing responses

### **3. Function Schema**

```python
class BaseFunction:
    name: str
    type: Literal["filter", "pipe", "pipeline"]
    description: str
    priority: Optional[int] = None
    config: Optional[Dict[str, Any]] = None

class Filter(BaseFunction):
    async def inlet(self, data: Dict[str, Any]) -> Dict[str, Any]: ...
    async def outlet(self, data: Dict[str, Any]) -> Dict[str, Any]: ...

class Tool(BaseFunction):
    parameters: Dict[str, Any]
    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]: ...

class Pipeline(BaseFunction):
    async def pipe(self, data: Dict[str, Any]) -> Dict[str, Any]: ...
```

### **4. Utility Functions**

#### **Core Utilities**
- Message manipulation
- Payload conversion
- Response formatting
- Event handling

#### **Helper Functions**
- Stream handling
- Error management
- Context management
- File operations

### **Implementation Requirements**

#### **1. Core Components**

- [ ] Function Registry
  - Dynamic loading
  - Type-based organization
  - Priority management

- [ ] Execution Engine
  - Filter chain execution
  - Tool dispatch
  - Pipeline routing

- [ ] Context Management
  - Message history
  - Function state
  - Stream handling

#### **2. Development Guidelines**

- Clear separation between types
- Consistent error handling
- Proper stream management
- Efficient state handling

#### **3. Testing Strategy**

- Unit tests per function
- Integration tests per type
- End-to-end flow tests
- Performance benchmarks
