import asyncio
import json
import anyio
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.shared.session import JSONRPCRequest, JSONRPCResponse, JSONRPCError
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def main():
    # Configure server parameters
    server_params = StdioServerParameters(
        command="node",
        args=["src/filesystem/dist/index.js"],
        cwd="C:\\Users\\Eidan Garcia\\code\\desktop-llm"
    )

    logger.info("Starting test client...")

    async with anyio.create_task_group() as tg:
        async with stdio_client(server_params) as (receive_stream, send_stream):
            logger.info("Connected to server")

            async def receive_loop():
                try:
                    while True:
                        message = await receive_stream.receive()
                        logger.info(f"[Python] Received raw: {message}")

                        # Extract the actual message from the root wrapper
                        if hasattr(message, 'root'):
                            message = message.root

                        if isinstance(message, (JSONRPCResponse, JSONRPCError)):
                            logger.info(f"[Python] Parsed response: {message}")
                        elif isinstance(message, dict):
                            try:
                                if "error" in message:
                                    response = JSONRPCError.model_validate(
                                        message)
                                else:
                                    response = JSONRPCResponse(
                                        jsonrpc=message.get("jsonrpc", "2.0"),
                                        id=message.get("id"),
                                        result=message.get("result")
                                    )
                                logger.info(
                                    f"[Python] Parsed response: {response}")
                            except Exception as e:
                                logger.error(
                                    f"[Python] Error parsing response: {e}")
                except Exception as e:
                    logger.error(f"[Python] Error in receive loop: {e}")

            async def send_test_message():
                try:
                    # Create a proper JSONRPCRequest object
                    test_message = JSONRPCRequest(
                        jsonrpc="2.0",
                        method="ping",
                        id=1,
                        params={}
                    )
                    logger.info(
                        f"[Python] Sending: {test_message.model_dump_json()}")
                    await send_stream.send(test_message)
                    logger.info("[Python] Test message sent")
                except Exception as e:
                    logger.error(f"[Python] Error sending message: {e}")

            # Start receive loop
            tg.start_soon(receive_loop)

            # Wait a bit for server to start
            await asyncio.sleep(1)

            # Send test message
            await send_test_message()

            # Keep running for a bit to receive response
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())
