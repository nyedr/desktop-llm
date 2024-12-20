"""Server-Sent Events (SSE) implementation for streaming responses."""

import asyncio
import json
from typing import AsyncGenerator, Any, Optional, Dict
from dataclasses import dataclass
from datetime import datetime

import logging
logger = logging.getLogger(__name__)

@dataclass
class SSEConfig:
    """Configuration for SSE streaming."""
    retry_timeout: int = 3000  # Milliseconds to wait before reconnection
    keep_alive_interval: float = 15.0  # Seconds between keep-alive messages
    event_queue_size: int = 100
    
class SSEStream:
    """Manages Server-Sent Events streaming with keep-alive and reconnection support."""
    
    def __init__(self, config: SSEConfig = SSEConfig()):
        self.config = config
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=config.event_queue_size)
        self._keep_alive_task: Optional[asyncio.Task] = None
        self._last_event_time = datetime.now()
        
    async def _keep_alive(self):
        """Send periodic keep-alive messages."""
        try:
            while True:
                await asyncio.sleep(self.config.keep_alive_interval)
                time_since_last = (datetime.now() - self._last_event_time).total_seconds()
                
                if time_since_last >= self.config.keep_alive_interval:
                    await self.event_queue.put({
                        "event": "keep-alive",
                        "data": "",
                        "id": f"ka-{int(datetime.now().timestamp())}"
                    })
        except asyncio.CancelledError:
            logger.debug("Keep-alive task cancelled")
            
    def start_keep_alive(self):
        """Start the keep-alive task."""
        if not self._keep_alive_task:
            self._keep_alive_task = asyncio.create_task(self._keep_alive())
            
    def stop_keep_alive(self):
        """Stop the keep-alive task."""
        if self._keep_alive_task:
            self._keep_alive_task.cancel()
            self._keep_alive_task = None
            
    async def format_sse(self, data: Any, event: Optional[str] = None, id: Optional[str] = None) -> str:
        """Format data as SSE message."""
        message = []
        
        if id is not None:
            message.append(f"id: {id}")
            
        if event is not None:
            message.append(f"event: {event}")
            
        if isinstance(data, (dict, list)):
            data = json.dumps(data)
            
        for chunk in str(data).split('\n'):
            message.append(f"data: {chunk}")
            
        return '\n'.join(message) + '\n\n'
        
    async def process_stream(
        self,
        input_stream: AsyncGenerator[Any, None],
        event_type: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Process an input stream and yield SSE formatted messages."""
        self.start_keep_alive()
        
        try:
            message_counter = 0
            async for data in input_stream:
                message_counter += 1
                self._last_event_time = datetime.now()
                
                event = await self.format_sse(
                    data=data,
                    event=event_type,
                    id=f"evt-{message_counter}"
                )
                
                await self.event_queue.put(event)
                
            # Send completion event
            completion = await self.format_sse(
                data={"status": "complete"},
                event="completion",
                id=f"evt-{message_counter + 1}"
            )
            await self.event_queue.put(completion)
            
        except Exception as e:
            logger.error(f"Error processing stream: {e}")
            error_event = await self.format_sse(
                data={"error": str(e)},
                event="error",
                id=f"err-{int(datetime.now().timestamp())}"
            )
            await self.event_queue.put(error_event)
            raise
            
        finally:
            self.stop_keep_alive()
            
    async def __aiter__(self):
        """Make SSEStream an async iterator."""
        try:
            while True:
                try:
                    event = await self.event_queue.get()
                    if event is None:  # Stream end sentinel
                        break
                    yield event
                    self.event_queue.task_done()
                except asyncio.CancelledError:
                    break
        finally:
            self.stop_keep_alive()
