"""System monitoring service."""

import gc
import logging
import psutil
import asyncio
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class SystemMonitor:
    """Monitor system resources and perform cleanup."""
    
    def __init__(self):
        self._monitor_task: Optional[asyncio.Task] = None
        self._last_cleanup: datetime = datetime.now()
        self._cleanup_interval: int = 300  # 5 minutes
        
    async def start_monitoring(self):
        """Start the monitoring task."""
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            logger.info("System monitoring started")
    
    async def stop_monitoring(self):
        """Stop the monitoring task."""
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            logger.info("System monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while True:
            try:
                await self._check_resources()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _check_resources(self):
        """Check system resources and perform cleanup if needed."""
        try:
            memory = psutil.virtual_memory()
            
            # Log current resource usage
            logger.debug(f"Memory usage: {memory.percent}%")
            logger.debug(f"CPU usage: {psutil.cpu_percent()}%")
            
            # Perform cleanup if memory usage is high or cleanup interval has passed
            if (memory.percent > 90 or 
                (datetime.now() - self._last_cleanup).total_seconds() > self._cleanup_interval):
                await self._perform_cleanup()
                
        except Exception as e:
            logger.error(f"Error checking resources: {e}", exc_info=True)
    
    async def _perform_cleanup(self):
        """Perform memory cleanup."""
        try:
            # Collect garbage
            gc.collect()
            
            # Log cleanup results
            memory_after = psutil.virtual_memory()
            logger.info(f"Cleanup completed. Memory usage: {memory_after.percent}%")
            
            self._last_cleanup = datetime.now()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)

# Create a global instance
monitor = SystemMonitor()
