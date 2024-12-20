import os
import sys
import pytest
import asyncio
import json
from typing import AsyncGenerator, Dict, Any

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import config
from app.services.model_service import ModelService

@pytest.mark.asyncio
async def test_chat_streaming():
    """Test that chat streaming works correctly."""
    # Initialize model service
    model_service = ModelService()
    
    # Test message
    messages = [
        {"role": "user", "content": "What's the weather like in New York?"}
    ]
    
    # Test streaming chat
    chunks_received = []
    async for chunk in model_service.chat(
        messages=messages,
        model=config.DEFAULT_MODEL,
        stream=True,
        enable_tools=True,
        tools=[{
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        }]
    ):
        print(f"Received chunk: {chunk}")
        chunks_received.append(chunk)
        
        # If we got a tool call, simulate the function response
        if isinstance(chunk, dict) and 'tool_calls' in chunk:
            print("\nReceived tool call, sending weather data...\n")
            
            # First, add the assistant's message with the tool call
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": chunk['tool_calls']
            })
            
            # Then add the function response
            weather_data = {
                "location": "New York",
                "temperature": 72,
                "unit": "fahrenheit",
                "conditions": "sunny",
                "humidity": 65,
                "wind_speed": 8
            }
            
            messages.append({
                "role": "tool",
                "tool_call_id": chunk['tool_calls'][0]['id'] if 'id' in chunk['tool_calls'][0] else "call_1",
                "name": "get_current_weather",
                "content": json.dumps(weather_data)
            })
            
            print(f"\nSending messages to model: {json.dumps(messages, indent=2)}\n")
            
            # Get the response with the function result
            async for response_chunk in model_service.chat(
                messages=messages,
                model=config.DEFAULT_MODEL,
                stream=True
            ):
                print(f"Received response chunk: {response_chunk}")
                chunks_received.append(response_chunk)
        
    # Verify we got some chunks
    assert len(chunks_received) > 0, "No chunks received from streaming"
    print(f"\nTotal chunks received: {len(chunks_received)}")
    
    # Print all chunks in order
    print("\nAll chunks in order:")
    for i, chunk in enumerate(chunks_received, 1):
        print(f"\nChunk {i}:")
        print(chunk)
    
    # Verify we got a tool call
    tool_calls = [chunk for chunk in chunks_received if isinstance(chunk, dict) and 'tool_calls' in chunk]
    assert len(tool_calls) > 0, "No tool calls received"
    print(f"\nTool calls received: {len(tool_calls)}")
    
    # Verify we got content after tool call
    content_chunks = [chunk for chunk in chunks_received if isinstance(chunk, dict) and 'content' in chunk and chunk['content']]
    assert len(content_chunks) > 0, "No content chunks received"
    print(f"\nContent chunks received: {len(content_chunks)}")
    
    # Print the final response
    print("\nFinal content chunks:")
    for chunk in content_chunks:
        print(chunk['content'])

if __name__ == "__main__":
    asyncio.run(test_chat_streaming())