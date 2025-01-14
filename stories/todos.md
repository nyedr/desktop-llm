# Desktop LLM Todos

## Core Features

- [ ] Low latency memory retrieval (1-2 seconds)
- [ ] Low latency tool execution (1-2 seconds, tool dependent)
- [ ] Get a fast and mostly-consistent response time (first chunk response, ideally 2-3 seconds)
- [ ] File and image handling
- [ ] Agentic capabilities
- [ ] Improve audio data extraction
  - [ ] Prosodic Features
  - [ ] Voice Quality Features
  - [ ] Temporal Dynamics
  - [ ] Emotional Markers
  - [ ] Turn-taking Signals
- [ ] Specialized Memory solution (profiles,audio, rag data switching, base agent memory, etc.)

### Completed todos

- [x] Make the application services non-blocking to improve partition tolerance
- [x] Test the chroma vector database
- [x] Create tools that control the chroma vector database
- [x] Ensure filters and pipelines are implemented correctly
- [x] Update application types to be more strict and consistent
- [x] Update the types of the function files
- [x] Review datastore and manager memory files
- [x] Update the memory retrieval pipeline: Prompt -> Query -> Embed -> Retrieve -> (Store memory as side process) -> Context
- [x] No llm memory cache
- [x] Only one messages is being stored in a conversation
- [x] Fix filters
- [x] Massively optimize the memory retrieval pipeline
- [x] Get memory working with the new lightrag
- [x] Fix model function calling (function calls work but the model is not being reprompted with their responses)
- [x] Centralize and selectively use profiling to measure performance and identify bottlenecks
- [x] Ensure pipelines are properly implemented as intended
- [x] Add api documentation
- [x] Integrate agents into the system

### Incomplete Todos

- [ ] Revamp all tests
- [ ] Make the memory retrieval pipeline more performant?
- [ ] Make function and tool handling more robust
- [ ] Implement more agentic capabilities. The AI should be able to call itself to achive a goal, this includes using a function to get data and then using that data to call another function. Should be a toggle for this behavior.
- [ ] Handle this error; "ERROR:lightrag:Failed to process document doc-2e2e68d738a4036e31740ef8a70abe8f: 'NoneType' object is not subscriptable"
- [ ] Remove function call caching for real-time tools
- [ ] Create specialized entity extraction llm function that directly uses openrouter with specified providers
- [ ] Check tokenizer initialization for context service, it should be initialized on server start
- [ ] Test agentic capabilities (multi-turn function calls, and step-by-step problem solving)
- [ ] Get agent tools working
- [ ] Check for unneeded dependencies

### Function calling

1. you ask something the AI thinks it can use a function to answer
2. the AI calls the function chart_hits_per_artist(genre, search_years)
3. you give another AI call back your raw retrieved data in a functions role
4. the AI writes a response augmented by the knowledge
