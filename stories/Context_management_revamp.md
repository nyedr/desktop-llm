Below is an **updated implementation guide** tailored to your new plan:

1. You will have a single “old conversations” table in your vector store (Chroma or otherwise) that stores _all user and assistant messages_ for ephemeral conversation logs.
2. You will have a **separate “model memory”** collection for longer-term or personal user data, which the model can also update if needed (like ChatGPT’s memory system).
3. Large messages (i.e., single user or assistant messages) are **chunked** to improve retrieval accuracy and to avoid storing huge embeddings for entire paragraphs. Each chunk is associated with `(conversation_id, response_id)` so you can piece them back together if needed.
4. **Summaries** are generated **only** when context gets too big, or optionally on idle. They may be stored in the ephemeral conversations table or not, depending on your preference.
5. **Multiple conversation IDs** let you keep old conversation logs accessible if they’re relevant. The user can jump back or reference them.

Below are **code improvement suggestions**, an **updated guide**, and a **sample database schema**.

---

## A. Proposed Database Schema Examples

### A.1 Ephemeral Conversation Logs (Single Table)

```python
# Table: ephemeral_conversations (or conversation_logs)
# Purpose: Store user/assistant messages, chunked for large content.

{
  "id": <string, unique, e.g. UUID>,
  "conversation_id": <string>,          # Ties logs to a conversation
  "response_id": <string>,             # Groups chunks of a single message/response
  "chunk_index": <int>,                # If a single response is split into multiple chunks
  "content": <string>,                 # The chunked text itself
  "metadata": <dict>,                  # { "role": "user/assistant", "timestamp", etc. }
  "embedding": <vector>,               # Vector embedding if stored
  "created_at": <datetime>
}
```

**Implementation Tips**

- If the logs can be very large, chunk them so each row has a smaller text. Each chunk still references the same `(conversation_id, response_id)`.
- You can reconstruct the entire message by querying all chunks with the same `(conversation_id, response_id)`, ordered by `chunk_index`.

### A.2 Model Memory Collection

```python
# Table: model_memory
# Purpose: Store curated, longer-term knowledge or preferences.

{
  "id": <string, e.g. UUID>,
  "content": <string>,                 # Possibly chunked or not, up to you
  "metadata": <dict>,                  # { "category": "preferences", "importance", etc. }
  "embedding": <vector>,
  "created_at": <datetime>
}
```

**Implementation Tips**

- If “model memory” can get very large, chunk it similarly, or store it in bigger documents if you expect it to remain small.
- The user can manipulate or remove items from this table to refine the model’s personalized memory.

---

## B. Updated Implementation Guide

### B.1 Storing Conversation Logs in a Single Table

1. **Chunk Large Responses**
   - Whenever a user or assistant message is too large (e.g., more than 500 tokens), split it into smaller pieces.
   - Assign them the same `response_id`.
   - Insert them into `ephemeral_conversations`.
2. **Metadata**
   - Include `metadata={"role": "assistant", "timestamp": "...", "some_other_info": "..."}`.
   - Possibly store `archived: bool` to mark older data.

**Implementation Tip**  
In your `ChromaService.add_memory`, you can do:

```python
def chunk_text(text: str, chunk_size=500) -> List[str]:
    # Pseudocode: break text into smaller pieces
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks
```

Then store each chunk with `(conversation_id, response_id, chunk_index)` in the metadata.

### B.2 Summarization Only On Overflow or Idle

- **Overflow**: If your context manager sees the total token usage is beyond `max_context_tokens`, it can do an on-demand summarization of older conversation logs, replacing multiple logs with a summary chunk.
- **Idle**: If the user has been inactive or if you have a background job, you can also choose to compress older logs to avoid indefinite growth.

**Implementation Tip**

- In `_manage_context_size` (or `_teardown()`), once you detect your conversation’s logs are too large to remain in context, call a summarization method.
- Store the summary in `ephemeral_conversations` as well if you want future references to that condensed version. Mark it with `metadata={"type": "summary"}`.

### B.3 On-Demand Memory vs. Automatic

- **Model Memory** is stored in `model_memory`. The model can add new entries with a function call, or you can allow the user to manually store important data.
- **Conversation Logs** automatically store every user/assistant turn.
- If the user wants to finalize something into “model memory,” they can call a function like `save_to_model_memory` to push it from ephemeral logs to the curated table.

---

## C. Code Improvement Suggestions

Below are some specific improvement suggestions in your codebase, referencing the files you provided.

1. **Use More Fine-Grained Collections**

   - Right now, `ChromaService` references a single collection for “long-term memory.”
   - Create a second `ChromaService` instance or extend the class to handle two collections:
     - `ephemeral_logs_collection_name`
     - `model_memory_collection_name`

2. **Add a `response_id` & `chunk_index` to Metadata**

   - In `_add_single_memory` and `_add_batch_memories`, incorporate `metadata["response_id"]` and `metadata["chunk_index"]` if provided.
   - This helps you tie chunks together.

3. **Chunking on Input**

   - Before calling `add_memory`, chunk the text if it’s large. Then do `add_batch_memories`.
   - Example in `_process_text_message` or in the code that adds conversation logs:

   ```python
   chunks = chunk_text(msg["content"], chunk_size=500)
   for idx, chunk in enumerate(chunks):
       metadata = {
           "conversation_id": current_convo_id,
           "response_id": msg_id,
           "chunk_index": idx,
           "role": msg["role"],
           # ...
       }
       await ephemeral_chroma_service.add_memory(chunk, metadata=metadata)
   ```

4. **`LLMContextManager`: Summarization Moved to Overflow**

   - You currently do summarization in `_teardown()`. To adopt the new approach, you might want to do it in `_manage_context_size` or upon noticing you exceed the token limit.
   - Or keep `_teardown()` for a final summary if the conversation is truly ending.

5. **Refine `_manage_context_size`**

   - Instead of discarding older messages, call a helper function `_summarize_oldest_logs()`, then store that summary chunk.
   - Example:

   ```python
   if total_tokens > self.max_context_tokens:
       # Summarize oldest conversation logs
       summary = self._summarize_logs(old_logs)
       # Replace them with summary chunk or mark them archived
       # Insert summary chunk in ephemeral table with type="summary"
   ```

6. **Database vs. In-Memory**

   - If you want to keep partial ephemeral logs in memory for fast iteration, that’s fine. Just ensure you also store them in the DB if you want them persisted.

7. **Add a Model “Memory Retrieval” Function**
   - If the user wants to ask “What do you know about me?” you can create a function “retrieve_user_info” that calls the `model_memory_service.retrieve_memories(...)`.
   - The model then decides when to call it (like ChatGPT does with function calling).

---

## D. Tradeoffs and Rationale

1. **Single Table for Ephemeral**:
   - Minimizes complexity vs. splitting ephemeral logs into multiple smaller specialized tables.
   - `(conversation_id, response_id, chunk_index)` is enough to reconstruct or do chunk-level retrieval.
2. **Separate Model Memory**:
   - Minimizes noise from ephemeral logs. The curated data can remain stable and quickly retrieved without wading through massive conversation logs.
3. **Optional Summaries**:
   - Summaries can degrade fidelity if you do them too often. Only do them at overflow or on idle.
4. **Maintaining Relevance**:
   - You can adopt a scoring system that slightly boosts logs from the same conversation, or let the raw vector similarity handle it.

Overall, these changes should keep your system clean, **optimal**, and easy to expand. They **do not** contradict your end goal, which is to maintain the **most relevant content** in the model’s context while letting older content remain accessible when truly relevant.

---

## E. Sample (Pseudo) Updated Code Excerpt

Here’s a short example illustrating chunking with `response_id`:

```python
def chunk_text(text: str, chunk_size: int = 300) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i : i + chunk_size]))
    return chunks

async def store_user_message(chroma_service, conversation_id: str, user_message: str):
    # If user_message is large, chunk it
    chunks = chunk_text(user_message)
    response_id = str(uuid.uuid4())  # or use time-based ID
    for idx, c in enumerate(chunks):
        await chroma_service.add_memory(
            text=c,
            metadata={
                "conversation_id": conversation_id,
                "response_id": response_id,
                "chunk_index": idx,
                "role": "user",
                "timestamp": datetime.now().isoformat()
            }
        )
```

Then you can retrieve with:

```python
# Reassemble the chunks for a single response if needed
async def get_response_text(chroma_service, conversation_id: str, response_id: str) -> str:
    results = await chroma_service.retrieve_with_metadata(
        query="",  # might pass a blank or a short dummy
        metadata_filter={
            "conversation_id": conversation_id,
            "response_id": response_id
        },
        top_k=50  # or large enough to get all chunks
    )
    # Sort by chunk_index
    sorted_chunks = sorted(results, key=lambda x: x["metadata"].get("chunk_index", 0))
    full_text = " ".join([res["document"] for res in sorted_chunks])
    return full_text
```

---

## Final Summary

- Storing everything in **one ephemeral table** for old conversation logs (chunked) plus a separate **model_memory** is a great approach.
- **Summaries** can be optionally stored or ephemeral. If you want them for future reference, store them as well. If they’re only to reduce immediate prompt size, you can skip storing them.
- The main code changes revolve around **chunking** large messages, adopting `(conversation_id, response_id, chunk_index)` for ephemeral logs, and **moving** summarization to an overflow/idle approach.
- Retrieving memories from ephemeral conversations should be optional, but default to true.

With these refinements, you’ll have a **robust**, **scalable**, and **clean** system that ensures the **most relevant** data remains accessible to the LLM—fulfilling your goal of maximizing user experience and context usage!

# Context Management Migration Plan

## Overview

This document outlines the detailed plan for implementing the new context management system. The implementation involves changes to the Chroma database schema, context management logic, and related services.

## Implementation Phases

### Phase 1: Environment Setup

1. **Create New Collections**
   - Create Chroma collections: `ephemeral_logs` and `model_memory`
   - Update configuration files with new collection names

### Phase 2: Service Updates

1. **ChromaService Modifications**

   - File: `app/services/chroma_service.py`
   - Add support for multiple collections
   - Implement chunking functionality
   - Add methods for handling `response_id` and `chunk_index`

2. **LLMContextManager Updates**
   - File: `app/context/llm_context.py`
   - Implement new context management logic
   - Add summarization on overflow/idle
   - Update context size management

### Phase 3: API and Function Updates

1. **Chat Router Changes**

   - File: `app/routers/chat.py`
   - Modify message handling to use new context system
   - Add support for multiple conversation IDs

2. **Memory Management Functions**
   - File: `app/functions/types/tools/memory_tool.py`
   - Implement `save_to_model_memory` function
   - Add `retrieve_user_info` function

### Phase 4: Testing and Validation

1. **Unit Tests**

   - Update existing tests in `tests/services/test_chroma_service.py`
   - Add new tests in `tests/context/test_llm_context.py`

2. **Integration Testing**

   - Test end-to-end functionality
   - Verify backward compatibility

3. **Performance Testing**
   - Test chunking and summarization performance
   - Verify memory usage with large conversations

### Phase 5: Deployment

1. **Deployment Plan**

   - File: `scripts/deploy_context_revamp.sh`
   - Verify service availability during update

2. **Monitoring**
   - Set up monitoring for new context system
   - Track performance metrics

## Implementation Timeline

| Week | Tasks                             |
| ---- | --------------------------------- |
| 1    | Phase 1: Environment Setup        |
| 2    | Phase 2: Service Updates          |
| 3    | Phase 3: API and Function Updates |
| 4    | Phase 4: Testing and Validation   |
| 5    | Phase 5: Deployment               |

## Risk Management

### Potential Risks

1. Performance degradation
2. Backward compatibility issues

### Mitigation Strategies

1. Perform implementation in staging environment first
2. Maintain old API endpoints during transition

## Rollback Plan

1. Revert to previous Chroma collection structure
2. Roll back code changes using Git

## Documentation Updates

1. Update API documentation in `app/routers/chat.py`
2. Add developer guide for new context system

## Post-Implementation Tasks

1. Monitor system performance
2. Gather user feedback
3. Optimize based on usage patterns

This implementation plan provides a structured approach to implementing the context management revamp while minimizing disruption to existing functionality.
