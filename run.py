import sys
import asyncio
import uvicorn
from uvicorn.config import Config
from uvicorn.server import Server

if __name__ == "__main__":
    # Configure event loop policy for Windows
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        loop = asyncio.ProactorEventLoop()
        asyncio.set_event_loop(loop)

    # Create uvicorn config
    config = Config(
        app="app.main:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
        log_level="debug",
        loop="none"  # Let us handle the event loop
    )

    # Create and run server
    server = Server(config=config)
    server.run()
