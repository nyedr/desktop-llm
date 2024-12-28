Below is an **implementation guide** for a **more optimal** context manager that supports **long-term memory** in a **FastAPI** or **desktop LLM** project. This design addresses your desire to avoid summarizing every conversation turn, perform a thorough and targeted **relevancy check**, minimize storage duplication, and provide a more **structured prompt** to the model.

---

# High-Level Goals

1. **Efficient Memory Storage**: Store conversation context in Chroma without repeatedly adding nearly identical summaries.
2. **Targeted Retrieval**: Dynamically retrieve only the **most relevant chunks** of old context, limiting overhead while preserving continuity.
3. **Optional Summaries**: Summaries should happen **periodically** (e.g., every N messages) or on demand, rather than every single request.
4. **Structured Prompt**: Provide the LLM with labeled memories in a more **schema-driven** manner so it clearly understands each piece of context.
5. **Prevent Duplication**: Implement stricter checks (metadata-based) to avoid storing multiple near-duplicate entries.
6. **Better Prompt Engineering**: Instead of a single “system” memory dump, consider a more structured approach to let the model parse memory effectively.

---

# Proposed Architecture

```
+-----------------+   (1) conversation messages  +----------------------+
|    FRONTEND     | --------------------------->  |        FastAPI       |
+-----------------+                                +----------------------+
                                                    |
                                                    | (2) enters
                                                    v  LLMContextManager
                                         +---------------------------------+
                                         |  1) Periodic Summaries (optional)|
                                         |  2) Memory Storage              |
                                         |  3) Memory Retrieval            |
                                         |  4) Prompt Engineering          |
                                         |  5) Output final context        |
                                         +---------------------------------+
                                                    |
                                                    | (3) final structured
                                                    v    prompt messages
                                             +-----------------+
                                             |  model_service  |
                                             |  (stream LLM)   |
                                             +-----------------+
                                                    |
                                                    v
                                            +------------------+
                                            |  SSE to Client   |
                                            +------------------+
```

---

## Step 1: Restructure Memory Storage

### 1.1 Use Clear Metadata for Conversation Identification

- **ConversationID**: Include a conversation ID (unique per session or per user) in every memory you store. This ensures retrieval only focuses on relevant conversation items.
- **Chunking**: Store messages in smaller chunks if they’re large. For instance, chunk each user/assistant turn or any large text.
- **No Summaries by Default**: Instead of summarizing on every request, store each user/assistant turn in Chroma as a separate **Document** with metadata like `{"conversation_id": ..., "role": ..., "timestamp": ...}`.

**Implementation Tip**

```python
# When storing a user message:
await chroma_service.add_memory(
    text=user_message,
    metadata={
        "conversation_id": unique_convo_id,
        "role": "user",
        "timestamp": datetime.now().isoformat()
    }
)
```

Store the assistant response similarly.

### 1.2 Periodic Summaries

Instead of summarizing each request, **summarize only after N messages** or if the conversation context is too large. For example:

1. After every 5 user messages (configurable), generate a summary that merges these 5 into a shorter chunk.
2. Replace or “retire” old chunks: You can remove them or mark them as “archived” if your conversation is extremely large.

**Implementation Tip**  
In `_teardown()`, **check** if `(message_count % 5) == 0`. If yes, summarize the last 5 messages into a single summary chunk, then store that chunk with metadata. Optionally, remove or mark the original 5 messages so they’re no longer retrieved.

---

## Step 2: More Targeted Retrieval

### 2.1 Vector Similarity + Metadata Filters

- Filter by `conversation_id` to only pull data from the current conversation.
- Retrieve top `k` documents with the highest similarity to the **current user query** (or last user message).
- Optionally filter out “archived” chunks or older summaries if they’re not relevant.

**Implementation Tip**

```python
memories = await self.chroma_service.retrieve_with_metadata(
    query=query_text,
    metadata_filter={"conversation_id": current_convo_id},
    top_k=self.top_k
)
```

You can also pass a **score_threshold** to ignore low-scoring results.

### 2.2 Summaries vs. Raw Chunks

If you store periodic summaries, your retrieval might see both summary chunks and raw message chunks for the same time period. You can:

- Always include the most recent summary chunk.
- Then retrieve raw messages from the last 10 minutes or last N messages to maintain immediate context.
- Avoid duplication by skipping raw messages that the summary already covers if the summary is highly relevant.

---

## Step 3: Prompt Engineering

### 3.1 Structured “Memory Blocks”

Rather than a single system message containing all memories, adopt a multi-part approach. For example:

1. **System**: Base instructions: “You are a helpful assistant… etc.”
2. **System**: `[Memory #1: Some relevant chunk]`
3. **System**: `[Memory #2: Another chunk]`
4. **User**: Actual user query.

This avoids having the LLM see everything as one giant text block. It can parse them as separate messages with distinct content.

#### Example of `_engineer_prompt()` Pseudocode

```python
def _engineer_prompt(self):
    # Basic system instruction
    final_messages = [{
        "role": "system",
        "content": (
            "You are a helpful assistant. Use the following memory chunks, if relevant, "
            "when answering the user. Do not repeat them verbatim unless necessary."
        )
    }]

    # Insert each memory as a separate system message
    for i, memory in enumerate(self.retrieved_memories):
        final_messages.append({
            "role": "system",
            "content": f"[Memory Chunk {i+1}]\n{memory['content']}",
            "metadata": {"relevance_score": memory["relevance_score"]}
        })

    # Then append the processed conversation
    final_messages.extend(self.processed_inputs)

    self.final_context_messages = final_messages
```

**Why This Helps**: The LLM can handle each chunk distinctly and it’s less likely to get “confused” by a single massive system message.

### 3.2 Additional Metadata in the Prompt

Optionally, you can label each memory chunk with timestamps, user roles, or summary indicators:

```
[Memory Chunk 1]
From user at 2024-12-20 13:05:45
Relevance: 0.88
Text: "Last time we..."

[Memory Chunk 2]
(Summary) covers messages from 2024-12-15 to 2024-12-18...
```

---

## Step 4: Summarization Strategy

### 4.1 Summarize Infrequently (Every N or on Demand)

**Approach**:

1. Keep raw messages in the DB for the last N turns.
2. Summaries happen only after N user turns or when the conversation is idle.
3. Store the summary chunk in Chroma with metadata, possibly marking older messages as “archived” or “summarized.”

### 4.2 Summaries for Speed

If the conversation is massive, retrieving 50+ messages from Chroma for every user request can slow down your system. Summaries reduce retrieval to fewer, bigger chunks. You can also store multiple levels of summaries (e.g., daily summary, weekly summary, etc.) if your conversation is extremely long.

**Implementation Tip**

```python
if (total_user_messages % SUMMARY_INTERVAL) == 0:
    # Summarize the last SUMMARY_INTERVAL messages
    short_summary = await self.langchain_service.summarize_text(conversation_text)
    # Store summary in Chroma
    await self.chroma_service.add_memory(short_summary, metadata=...)
    # Optional: mark the original messages as archived
```

---

## Step 5: Preventing Duplication

### 5.1 Store Conversation-ID in Metadata

Every chunk: `{"conversation_id": <ID>, "turn_index": <int>, ...}`. This ensures you only see data from the correct conversation.

### 5.2 Fuzzy Duplicate Check

- Before storing a new summary or chunk, do a quick similarity check with existing items in the conversation.
- If the new chunk is ~90% similar (or some threshold) to an existing one, skip storing or merge them.

**Implementation Tip**  
Use `retrieve_memories(query=potential_new_chunk, score_threshold=0.9)` to see if a near-duplicate exists.

---

## Step 6: Implementation Sketch

Below is a skeleton code snippet focusing on the **key differences** from your current manager:

```python
class OptimalLLMContextManager:
    def __init__(
        self,
        chroma_service: ChromaService,
        langchain_service: 'LangChainService',
        conversation_history: List[Dict[str, Any]],
        conversation_id: str,
        max_context_tokens: int = 2048,
        top_k: int = 5,
        summary_interval: int = 5
    ):
        self.chroma_service = chroma_service
        self.langchain_service = langchain_service
        self.conversation_history = conversation_history
        self.conversation_id = conversation_id
        self.max_context_tokens = max_context_tokens
        self.top_k = top_k
        self.summary_interval = summary_interval

        self.processed_inputs = []
        self.retrieved_memories = []
        self.final_context_messages = []

        # If you want a tokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    async def __aenter__(self):
        # 1) Process the user input
        self.processed_inputs = self._process_input(self.conversation_history)

        # 2) Retrieve top-k relevant memories for the last user query
        self.retrieved_memories = await self._retrieve_topk_memories()

        # 3) Engineer prompt to better structure the memory
        self._engineer_prompt()

        # 4) Manage context size
        self._manage_context_size()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Optional: only summarize if enough messages since last summary
        if self._should_summarize():
            await self._summarize_and_store()
        # Clean up if needed

    def get_context_messages(self):
        return self.final_context_messages

    def _process_input(self, messages):
        # Similar to your existing logic for images, files, etc.
        # Return a list of standardized message dicts
        return messages

    async def _retrieve_topk_memories(self):
        # Only use the last user message as the query
        user_query = ""
        for msg in reversed(self.processed_inputs):
            if msg["role"] == "user" and msg.get("content"):
                user_query = msg["content"]
                break
        if not user_query:
            return []

        # Retrieve from Chroma with conversation_id filter
        results = await self.chroma_service.retrieve_with_metadata(
            user_query,
            {"conversation_id": self.conversation_id},
            top_k=self.top_k
        )
        # Sort by relevance_score desc
        memories = sorted(results, key=lambda x: x["relevance_score"], reverse=True)
        return memories

    def _engineer_prompt(self):
        system_message = {
            "role": "system",
            "content": (
                "You are a helpful assistant with a memory. Below are some relevant memory chunks.\n"
                "Use them if they help you answer the user's query, but do not repeat them verbatim.\n"
            )
        }
        final_messages = [system_message]

        for i, mem in enumerate(self.retrieved_memories):
            chunk_text = mem["content"]
            final_messages.append({
                "role": "system",
                "content": f"[Memory {i+1}]\n{chunk_text}"
            })

        final_messages.extend(self.processed_inputs)
        self.final_context_messages = final_messages

    def _manage_context_size(self):
        # Token counting logic if desired
        # Example placeholder
        pass

    def _should_summarize(self) -> bool:
        # Suppose we only summarize if the user had X messages since last summary
        # This code depends on how you track summary intervals
        return False  # placeholder

    async def _summarize_and_store(self):
        # Summarize only the newly arrived messages or the chunk since last summary
        pass
```

**Implementation Tips**

1. **Use `conversation_id`**: If you have multi-user sessions or separate chat sessions, this is critical.
2. **Periodic Summaries**: Possibly keep a `summary_count` or track how many messages have been added to the DB since the last summary. Summarize only if it exceeds a threshold.
3. **Prevent Over-Frequent Summaries**: Summaries can be slow and cause extra model calls, so keep them minimal.

---

## Step 7: QA and Edge Cases

1. **Empty Conversations**: If the user never typed anything, skip memory retrieval.
2. **Large Summaries**: If your summary is long, chunk it or re-summarize it. Consider hierarchical summaries.
3. **Concurrent Requests**: Make sure your `chroma_service` can handle concurrent calls or use a concurrency lock if needed.
4. **Max Tokens**: If you want to be strict, integrate an actual token counting library (like `transformers`). Remove older memory chunks or less relevant ones if you exceed `max_context_tokens`.

---

## Answers to Your Questions

1. **How Should Conversation Context Be Stored as Memory?**

   - **Store each message** in Chroma as a separate Document, labeled by `conversation_id`, `role`, and a timestamp. Periodically store “summary” documents with clear metadata like `{"type": "summary"}`.

2. **How Should Relevant Data Be Accessed?**

   - Always filter by `conversation_id`, then do a vector similarity search on the last user query (or possibly the entire last user turn). Retrieve top-K. You can also combine “recent messages” with “summaries” to get a broader or hierarchical context.

3. **How to Update `_engineer_prompt`**
   - Provide smaller, labeled chunks, each as a separate system message. Possibly group them by role or timestamp to give the model better structure.
   - Start with a short system directive: “Here are memory snippets. Use them if relevant.” Then list them, followed by user messages.

---

# Conclusion

By **storing** individual user/assistant messages (plus occasional summaries) and **retrieving** them with a thorough similarity search—while **reducing** redundant summarization—you’ll maintain an **efficient** long-term memory system. Combining those memory chunks in a structured prompt ensures the LLM can more clearly interpret each piece of context.

This approach will help you achieve a **consistent “long-term memory” feel**, reduce **duplicate entries**, and **avoid** overloading the model with repeated data, all while staying within the **limited context window** of the model.
