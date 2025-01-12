# Desktop LLM Todos

## Core Features

### Random todos

- [x] Make the application services non-blocking to improve partition tolerance
- [x] Test the chroma vector database
- [x] Create tools that control the chroma vector database
- [x] Ensure filters and pipelines are implemented correctly
- [x] Update application types to be more strict and consistent
- [x] Update the types of the function files
- [ ] Revamp all tests
- [x] Review datastore and manager memory files
- [x] Update the memory retrieval pipeline: Prompt -> Query -> Embed -> Retrieve -> (Store memory as side process) -> Context
- [x] No llm memory cache
- [x] Only one messages is being stored in a conversation
- [x] Fix filters
- [x] Massively optimize the memory retrieval pipeline
- [x] Get memory working with the new lightrag
- [x] Fix model function calling (function calls work but the model is not being reprompted with their responses)
- [x] Centralize and selectively use profiling to measure performance and identify bottlenecks
- [ ] Get a fast and mostly-consistent response time (first chunk response, ideally 2-3 seconds)
- [ ] Make the memory retrieval pipeline more performant?
- [x] Ensure pipelines are properly implemented as intended
- [ ] Make function and tool handling more robust
- [ ] Implement more agentic capabilities. The AI should be able to call itself to achive a goal, this includes using a function to get data and then using that data to call another function. Should be a toggle for this behavior.
- [ ] Handle this error; "ERROR:lightrag:Failed to process document doc-2e2e68d738a4036e31740ef8a70abe8f: 'NoneType' object is not subscriptable"
- [x] Add api documentation
- [ ] Remove function call caching for real-time tools
- [ ] Create specialized entity extraction llm function that directly uses openrouter with specified providers
- [ ] Check tokenizer initialization for context service, it should be initialized on server start
- [ ] Integrate agents into the system

### Function calling

1. you ask something the AI thinks it can use a function to answer
2. the AI calls the function chart_hits_per_artist(genre, search_years)
3. you give another AI call back your raw retrieved data in a functions role
4. the AI writes a response augmented by the knowledge

### Streaming and Tool Execution

- [x] Implement single continuous streaming
  - [x] Real-time word-by-word streaming
  - [x] Tool call handling within stream
  - [x] Optimize stream continuation after tool execution
- [x] Test function executions
  - [x] Add comprehensive test suite
  - [x] Test edge cases and error handling

### Function System Revamp

- [x] Implement OpenWebUI-style Function Types

  - [x] Create Filter base class with inlet/outlet methods
  - [x] Create Tool (Pipe) base class with execute method
  - [x] Create Pipeline base class with pipe method
  - [x] Implement priority-based execution order

- [x] Core Infrastructure

  - [x] Build dynamic Function Registry
  - [x] Create Execution Engine
  - [x] Implement Context Management system
  - [x] Add Stream Management utilities

- [x] Function Utilities

  - [x] Message manipulation helpers
  - [x] Payload conversion utilities
  - [x] Response formatting tools
  - [x] Error handling framework

### Memory and Context

- [ ] Implement long term LLM memory
  - [x] Integrate Chroma for vector storage
  - [x] Design memory management system
  - [x] Implement memory retrieval and storage
  - [x] Add integration testing for memory retrieval with LLM

### Custom Functions

- [x] Create Example Functions
  - [x] Content moderation filter
  - [x] Text modifier filter
  - [x] Multi-step pipeline

### Testing and Documentation

- [ ] Comprehensive Testing

  - [x] Test streaming functionality
  - [x] Test tool execution
  - [x] Test filter and pipeline system
  - [ ] Add more test cases
  - [x] Test error handling
  - [ ] Deeply test memory retrieval and storage

- [ ] Documentation
  - [ ] Write API documentation
  - [ ] Document deployment process
  - [ ] Add usage examples
  - [x] Create function development guidelines
  - [x] Add more example functions

### Future Enhancements

- [ ] Performance Optimization

  - [ ] Optimize streaming performance
  - [ ] Improve function execution speed
  - [ ] Memory usage optimization

- [ ] UI/UX Improvements
  - [ ] Add progress indicators
  - [ ] Improve error messages
  - [ ] Add debug mode
  - [ ] Enhance logging system
