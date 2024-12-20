"""Stream processor implementation for handling buffered streaming with flow control."""

import asyncio
from typing import AsyncGenerator, Optional, Any, Callable
from datetime import datetime
import logging

from .stream_config import StreamConfig

logger = logging.getLogger(__name__)

class StreamBuffer:
    """Manages buffered streaming with flow control."""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.buffer = bytearray()
        self.buffer_lock = asyncio.Lock()
        self.flow_control = asyncio.Event()
        self.flow_control.set()  # Start allowing data flow
        self.last_yield_time = datetime.now()
        
    async def add_data(self, data: bytes):
        """Add data to the buffer with flow control."""
        async with self.buffer_lock:
            self.buffer.extend(data)
            
            # Check if we need to pause data flow
            if len(self.buffer) >= self.config.buffer_size * self.config.flow_control_threshold:
                self.flow_control.clear()
                
    async def get_chunk(self) -> Optional[bytes]:
        """Get a chunk of data from the buffer."""
        async with self.buffer_lock:
            if not self.buffer:
                return None
                
            # Get chunk
            chunk_size = min(len(self.buffer), self.config.chunk_size)
            chunk = bytes(self.buffer[:chunk_size])
            del self.buffer[:chunk_size]
            
            # Resume data flow if buffer is low enough
            if len(self.buffer) < self.config.buffer_size * self.config.flow_control_threshold:
                self.flow_control.set()
                
            self.last_yield_time = datetime.now()
            return chunk

class StreamProcessor:
    """Processes streaming data with buffering and flow control."""
    
    def __init__(
        self,
        config: StreamConfig = StreamConfig(),
        preprocessor: Optional[Callable[[bytes], bytes]] = None,
        postprocessor: Optional[Callable[[bytes], bytes]] = None
    ):
        self.config = config
        self.buffer = StreamBuffer(config)
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        
    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass
        
    async def process_stream(
        self,
        input_stream: AsyncGenerator[Any, None]
    ) -> AsyncGenerator[bytes, None]:
        """Process a stream with buffering and flow control."""
        try:
            async for data in input_stream:
                if not data:
                    continue
                    
                # Preprocess if needed
                if self.preprocessor:
                    try:
                        data = self.preprocessor(data)
                    except Exception as e:
                        raise ValueError(f"Preprocessing failed: {str(e)}")
                
                # Convert to bytes if necessary
                if isinstance(data, str):
                    data = data.encode()
                elif not isinstance(data, bytes):
                    data = str(data).encode()
                
                # Wait for flow control if necessary
                await self.buffer.flow_control.wait()
                
                # Add to buffer
                await self.buffer.add_data(data)
                
                # Yield complete chunks
                while True:
                    chunk = await self.buffer.get_chunk()
                    if not chunk:
                        break
                        
                    # Postprocess if needed
                    if self.postprocessor:
                        try:
                            chunk = self.postprocessor(chunk)
                        except Exception as e:
                            raise ValueError(f"Postprocessing failed: {str(e)}")
                            
                    yield chunk
                    
        except Exception as e:
            logger.error(f"Error processing stream: {str(e)}")
            raise
            
        # Yield any remaining data in buffer
        async with self.buffer.buffer_lock:
            if self.buffer.buffer:
                remaining = bytes(self.buffer.buffer)
                if self.postprocessor:
                    remaining = self.postprocessor(remaining)
                yield remaining
                self.buffer.buffer.clear()
