"""Agent router for handling agent-based flows with streaming responses."""

import json
import logging
import uuid
from typing import AsyncGenerator, Dict, Any, List, Optional

from fastapi import APIRouter, Request, Depends
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field

from app.services.agent_service import AgentService
from app.dependencies.providers import Providers
from app.utils.profiling import profile_request
from app.models.chat import StrictChatMessage
from app.models.agent import AgentCapability
from app.utils.utils import parse_attempt_completion

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/agent",
    tags=["agent"],
    responses={
        400: {"description": "Bad request - Invalid input parameters"},
        500: {"description": "Internal server error"},
        429: {"description": "Too many requests - Rate limit exceeded"},
    }
)


class AgentRequest(BaseModel):
    """Request model for agent endpoints."""
    messages: List[StrictChatMessage] = Field(
        ...,
        description="List of messages in the conversation"
    )
    goal: str = Field(
        ...,
        description="The goal or task for the agent to accomplish"
    )
    agent_type: str = Field(
        default="general",
        description="Type of agent to use (e.g., 'general', 'supervisor', or custom registered types)"
    )
    agent_name: Optional[str] = Field(
        default=None,
        description="Custom name for the agent instance"
    )
    capabilities: List[AgentCapability] = Field(
        default_factory=lambda: [AgentCapability.FUNCTION_CALLING],
        description="List of capabilities the agent should have"
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Constraints for the agent's operation"
    )
    stream: bool = Field(
        default=True,
        description="Whether to stream the response or return it all at once"
    )
    model: Optional[str] = Field(
        default=None,
        description="The model to use for agent operations"
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Temperature for model responses"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to generate"
    )
    max_turns: int = Field(
        default=5,
        description="Maximum number of turns before forcing completion"
    )
    enable_tools: Optional[bool] = Field(
        default=None,
        description="Whether to enable tool/function calling"
    )
    enable_memory: bool = Field(
        default=True,
        description="Whether to enable memory/context management"
    )
    allowed_tools: Optional[List[str]] = Field(
        default=None,
        description="List of specific tools the agent is allowed to use"
    )
    excluded_tools: Optional[List[str]] = Field(
        default=None,
        description="List of tools the agent should not use"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the agent execution"
    )


@router.get("/types",
            summary="List Available Agent Types",
            description="Get a list of all registered agent types and their capabilities")
async def list_agent_types(
    agent_service: AgentService = Depends(
        lambda: AgentService(Providers.get_function_service()))
) -> List[Dict[str, Any]]:
    """List all available agent types and their capabilities."""
    return await agent_service.list_available_agents()


async def stream_agent_steps(
    request: Request,
    agent_request: AgentRequest,
    agent_service: AgentService = Depends(Providers.get_agent_service)
) -> AsyncGenerator[Dict[str, Any], None]:
    """Generate streaming agent steps."""
    request_id = str(uuid.uuid4())
    logger.info(
        f"[{request_id}] Starting agent flow with goal: {agent_request.goal}")

    async with profile_request(request_id) as profiler:
        while True:  # Main retry loop
            try:
                # Update metadata with agent type and name
                metadata = agent_request.metadata or {}
                metadata.update({
                    "agent_type": agent_request.agent_type,
                    "agent_name": agent_request.agent_name or f"{agent_request.agent_type}_agent",
                    "max_turns": agent_request.max_turns  # Pass max_turns to agent
                })

                # Start streaming if this is first attempt
                if profiler.retry_count == 0:
                    yield {
                        "event": "start",
                        "data": json.dumps({
                            "status": "streaming",
                            "request_id": request_id,
                            "goal": agent_request.goal,
                            "agent_type": agent_request.agent_type,
                            "capabilities": [cap.value for cap in agent_request.capabilities],
                            "max_turns": agent_request.max_turns
                        })
                    }

                # Record when we're about to start agent flow
                profiler.record_model_request()

                # Track turn count and message accumulation
                turn_count = 0
                current_message = []
                last_complete_message = None

                # Stream agent steps
                async for step in agent_service.run_agent_flow(
                    goal=agent_request.goal,
                    messages=agent_request.messages,
                    capabilities=agent_request.capabilities,
                    constraints=agent_request.constraints,
                    model=agent_request.model,
                    temperature=agent_request.temperature,
                    max_tokens=agent_request.max_tokens,
                    enable_tools=agent_request.enable_tools,
                    enable_memory=agent_request.enable_memory,
                    allowed_tools=agent_request.allowed_tools,
                    excluded_tools=agent_request.excluded_tools,
                    metadata=metadata
                ):
                    if await request.is_disconnected():
                        logger.info(f"[{request_id}] Client disconnected")
                        return

                    # Record first response timing
                    if not profiler.first_response_time:
                        profiler.record_first_response()

                    # Convert step to event data
                    if isinstance(step, dict):
                        event_data = step
                    else:
                        # Handle response objects by converting to dict
                        event_data = step.model_dump() if hasattr(step, 'model_dump') else vars(step)

                    # Handle model responses and tool calls
                    if isinstance(event_data, dict):
                        if "content" in event_data:
                            content = event_data["content"]
                            current_message.append(content)

                            # Only try parsing when we see a potential end tag
                            accumulated = "".join(current_message)
                            if "</attempt_completion>" in accumulated:
                                is_complete, result, command, answer = parse_attempt_completion(
                                    accumulated)
                                if is_complete:
                                    logger.info(
                                        f"[{request_id}] Completion detected:")
                                    logger.info(
                                        f"[{request_id}] - Result: {result}")
                                    if command:
                                        logger.info(
                                            f"[{request_id}] - Command: {command}")
                                    if answer:
                                        logger.info(
                                            f"[{request_id}] - Answer: {answer}")

                                    # Send the final token before completing
                                    yield {
                                        "event": "message",
                                        "data": json.dumps(event_data)
                                    }

                                    # Send the answer as a message if present
                                    if answer:
                                        yield {
                                            "event": "message",
                                            "data": json.dumps({
                                                "type": "message",
                                                "content": answer
                                            })
                                        }

                                    yield {
                                        "event": "complete",
                                        "data": json.dumps({
                                            "status": "complete",
                                            "request_id": request_id,
                                            "metrics": profiler.get_summary(),
                                            "forced_completion": False,
                                            "final_turn_count": turn_count,
                                            "completion_result": result,
                                            "completion_command": command,
                                            "completion_answer": answer
                                        })
                                    }
                                    return

                            # Handle normal message accumulation for turn counting
                            elif content.strip() in ["\n", "."]:
                                complete_message = accumulated.strip()
                                if complete_message and not any(tag in complete_message for tag in ["<attempt_completion>", "</attempt_completion>"]):
                                    logger.info(
                                        f"[{request_id}] Model response (turn {turn_count}): {complete_message}")
                                    if complete_message != last_complete_message:
                                        turn_count += 1
                                        last_complete_message = complete_message
                                current_message = []

                        elif "type" in event_data and event_data["type"] == "tool_call":
                            result = event_data.get("result", {})
                            if isinstance(result, dict) and "tool_name" in result:
                                profiler.record_tool_execution(
                                    tool_name=result["tool_name"],
                                    duration=result.get("execution_time", 0.0),
                                    success=result.get("success", False),
                                    metadata={
                                        "error": result.get("error"),
                                        "tool_type": result.get("tool_type")
                                    }
                                )
                                logger.info(
                                    f"[{request_id}] Tool execution: {result['tool_name']}")
                                turn_count += 1  # Count tool calls as turns

                    # Stream the event data
                    yield {
                        "event": "message",
                        "data": json.dumps(event_data)
                    }

                # If we get here, execution completed successfully
                # Log any remaining message
                if current_message:
                    complete_message = "".join(current_message).strip()
                    if complete_message:
                        logger.info(
                            f"[{request_id}] Final model response: {complete_message}")

                yield {
                    "event": "complete",
                    "data": json.dumps({
                        "status": "complete",
                        "request_id": request_id,
                        "metrics": profiler.get_summary(),
                        "forced_completion": False,
                        "final_turn_count": turn_count
                    })
                }
                return  # Exit retry loop on success

            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)

                logger.error(
                    f"[{request_id}] Error in turn {turn_count}: {error_type} - {error_msg}")

                # Record error and check if we should retry
                should_retry = profiler.record_error(error_msg, error_type, {
                    "goal": agent_request.goal,
                    "model": agent_request.model,
                    "turn_count": turn_count
                })

                if not should_retry:
                    # If we've exceeded retries, send error event and exit
                    yield {
                        "event": "error",
                        "data": json.dumps({
                            "error": error_msg,
                            "error_type": error_type,
                            "request_id": request_id,
                            "turn_count": turn_count,
                            "metrics": profiler.get_summary()
                        })
                    }
                    return  # Exit retry loop on max retries exceeded

                # Reset first response timing for retry
                profiler.reset_first_response()
                # Continue to next iteration of retry loop


@router.post("/run",
             response_class=EventSourceResponse,
             summary="Run Agent Flow",
             description="""
    Start an agent-based workflow with a specific goal and context, streaming the agent's progress.
    
    Key Features:
    - Streaming response using Server-Sent Events (SSE)
    - Function/tool calling capabilities
    - Progress tracking and state management
    - Real-time status updates
    
    The response is streamed as a series of events:
    - 'start': Indicates the start of the agent flow
    - 'message': Contains content or tool call results
    - 'error': Contains error information if something fails
    - 'complete': Indicates successful completion
    """,
             response_description="Server-Sent Events stream of agent execution steps",
             responses={
                 200: {
                     "description": "Successful response",
                     "content": {
                         "text/event-stream": {
                             "example": 'event: message\ndata: {"type": "message", "content": "..."}\n\n'
                         }
                     }
                 }
             }
             )
async def run_agent(
    request: Request,
    agent_request: AgentRequest
) -> EventSourceResponse:
    """Run an agent-based workflow with streaming updates.

    The response is streamed as Server-Sent Events (SSE) with the following event types:
    - 'start': Indicates the start of the agent flow
    - 'message': Contains content or tool call results
    - 'error': Contains error information if something fails
    - 'complete': Indicates successful completion

    Features:
    - Streaming response using SSE
    - Function/tool calling capabilities
    - Progress tracking and state management
    - Real-time status updates

    Args:
        request: The FastAPI request object
        agent_request: The agent request parameters including goal and context

    Returns:
        An EventSourceResponse that streams the agent's execution steps

    Raises:
        HTTPException: If there are errors in request processing
    """
    return EventSourceResponse(
        stream_agent_steps(
            request=request,
            agent_request=agent_request,
            agent_service=Providers.get_agent_service()
        ),
        media_type="text/event-stream"
    )
