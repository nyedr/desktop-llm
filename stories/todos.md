# Desktop LLM Todos

## Core Features

### Random todos

- [x] Make the application services non-blocking to improve partition tolerance
- [ ] Test the chroma vector database
- [ ] Create tools that control the chroma vector database

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
  - [ ] Implement Context Management system
  - [x] Add Stream Management utilities

- [x] Function Utilities

  - [x] Message manipulation helpers
  - [x] Payload conversion utilities
  - [x] Response formatting tools
  - [x] Error handling framework

### Memory and Context

- [ ] Implement long term LLM memory
  - [x] Integrate Chroma for vector storage
  - [ ] Set up Redis/Redis Search for fast retrieval
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
  - [ ] Test error handling
  - [ ] Deeply test memory retrieval and storage

- [ ] Documentation
  - [ ] Write API documentation
  - [ ] Document deployment process
  - [ ] Add usage examples
  - [ ] Create function development guidelines
  - [ ] Add more example functions

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
