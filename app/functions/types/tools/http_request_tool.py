from typing import Dict, Any, Optional, Literal
import aiohttp
import asyncio
import logging
from pydantic import Field, ConfigDict
from app.models.function import Tool, FunctionType
from app.models.function import register_function, ExecutionError, InputValidationError, TimeoutError

logger = logging.getLogger(__name__)


@register_function(
    func_type=FunctionType.TOOL,
    name="http_request",
    description="Make HTTP requests to URLs with support for different methods, headers, and data",
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to send the request to"
            },
            "method": {
                "type": "string",
                "description": "HTTP method to use",
                "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"],
                "default": "GET"
            },
            "headers": {
                "type": "object",
                "description": "Optional headers to include in the request",
                "additionalProperties": {
                    "type": "string"
                }
            },
            "data": {
                "type": "object",
                "description": "Optional data to send with the request (for POST/PUT/PATCH)",
                "additionalProperties": True
            },
            "timeout": {
                "type": "number",
                "description": "Request timeout in seconds",
                "default": 30
            }
        },
        "required": ["url"]
    }
)
class HttpRequestTool(Tool):
    """Tool for making HTTP requests with proper error handling and security measures."""

    # Required: Allow arbitrary types for aiohttp.ClientSession
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Required: Define type, name, and description
    type: Literal[FunctionType.TOOL] = Field(
        default=FunctionType.TOOL,
        description="Tool type"
    )
    name: str = Field(
        default="http_request",
        description="Name of the tool"
    )
    description: str = Field(
        default="Make HTTP requests to URLs with support for different methods, headers, and data",
        description="Description of what the tool does"
    )

    # Instance variables
    session: Optional[aiohttp.ClientSession] = Field(
        default=None,
        exclude=True
    )

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    def normalize_parameters(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize request parameters."""
        normalized = args.copy()

        # Normalize method to uppercase
        if "method" in normalized:
            normalized["method"] = str(normalized["method"]).upper()

        # Ensure headers is a dictionary
        if "headers" in normalized and not isinstance(normalized["headers"], dict):
            normalized["headers"] = {}

        # Set default timeout if not provided
        if "timeout" not in normalized:
            normalized["timeout"] = 30

        return normalized

    async def _execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the HTTP request."""
        try:
            session = await self._get_session()
            timeout = aiohttp.ClientTimeout(
                total=float(args.get("timeout", 30)))

            async with session.request(
                method=args.get("method", "GET"),
                url=args["url"],
                headers=args.get("headers", {}),
                json=args.get("data"),
                timeout=timeout,
                ssl=True  # Enable SSL verification for security
            ) as response:
                # Read response content
                content = await response.text()

                # Prepare response data
                result = {
                    "status": response.status,
                    "headers": dict(response.headers),
                    "content": content,
                    "url": str(response.url)
                }

                # Log response info
                logger.debug(
                    f"Request to {args['url']} completed with status {response.status}")

                # Clean up session after request
                await self.cleanup()

                return result

        except aiohttp.ClientError as e:
            logger.error(f"HTTP request failed: {str(e)}", exc_info=True)
            await self.cleanup()
            raise ExecutionError(f"HTTP request failed: {str(e)}")
        except asyncio.TimeoutError:
            logger.error(f"Request timed out after {args.get('timeout', 30)}s")
            await self.cleanup()
            raise TimeoutError(
                f"Request timed out after {args.get('timeout', 30)}s")
        except Exception as e:
            logger.error(
                f"Unexpected error during HTTP request: {str(e)}", exc_info=True)
            await self.cleanup()
            raise ExecutionError(
                f"Unexpected error during HTTP request: {str(e)}")

    async def _handle_error(self, error: Exception, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle errors during execution."""
        if isinstance(error, InputValidationError):
            # Try to fix validation errors
            fixed_args = args.copy()
            if "method" in args and args["method"] not in ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"]:
                fixed_args["method"] = "GET"
            return fixed_args
        return args

    async def cleanup(self):
        """Clean up resources."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    def __del__(self):
        """Ensure session is closed when object is garbage collected."""
        if self.session and not self.session.closed:
            if asyncio.get_event_loop().is_running():
                asyncio.create_task(self.cleanup())
            else:
                logger.warning(
                    "Session cleanup skipped: event loop not running")
