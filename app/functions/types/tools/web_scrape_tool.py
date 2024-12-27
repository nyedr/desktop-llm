"""Web scraping tool using Jina Reader for extracting text content from web pages."""

import re
import logging
import requests
from typing import Dict, Any, Literal, Optional
from pydantic import Field

from app.functions.base import Tool, FunctionType, register_function, ExecutionError

logger = logging.getLogger(__name__)


def extract_title(text: str) -> Optional[str]:
    """Extract the title from structured text."""
    match = re.search(r'Title: (.*)\n', text)
    return match.group(1).strip() if match else None


def clean_urls(text: str) -> str:
    """Clean URLs from structured text."""
    return re.sub(r'\((http[^)]+)\)', '', text)


@register_function(
    func_type=FunctionType.TOOL,
    name="web_scrape",
    description="Scrape and process web pages using Jina Reader (r.jina.ai)"
)
class WebScrapeTool(Tool):
    """Tool for scraping web content using Jina Reader."""
    type: Literal[FunctionType.TOOL] = Field(
        default=FunctionType.TOOL, description="Tool type")
    name: str = Field(default="web_scrape",
                      description="Name of the web scrape tool")
    description: str = Field(
        default="Scrape and process web pages using Jina Reader with configurable cleaning and caching",
        description="Description of what the tool does"
    )
    parameters: Dict[str, Any] = Field(
        default={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL of the web page to scrape"
                }
            },
            "required": ["url"]
        },
        description="Parameters schema for the web scrape tool"
    )

    async def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the web scraping command."""
        try:
            if not args:
                logger.error("No arguments provided to web_scrape")
                raise ValueError("No arguments provided")

            if "url" not in args:
                logger.error("URL argument missing in web_scrape")
                raise ValueError("URL argument is required")

            url = args["url"]
            # clean_content = args.get("clean_content", True)
            clean_content = True
            # disable_caching = args.get("disable_caching", False)
            disable_caching = False

            # Construct Jina Reader URL
            jina_url = f"https://r.jina.ai/{url}"
            logger.info(f"Scraping URL: {url} via Jina Reader")

            # Set up headers
            headers = {
                "X-No-Cache": "true" if disable_caching else "false",
                "X-With-Generated-Alt": "true"
            }

            # Make the request
            try:
                response = requests.get(jina_url, headers=headers)
                response.raise_for_status()
                content = response.text

                # Clean content if requested
                if clean_content:
                    logger.debug("Cleaning URLs from content")
                    content = clean_urls(content)

                # Extract title for logging
                title = extract_title(content)
                logger.info(f"Successfully scraped: {title if title else url}")

                return {
                    "content": content,
                    "title": title,
                    "url": url,
                    "cleaned": clean_content
                }

            except requests.RequestException as e:
                error_msg = f"Error scraping web page: {str(e)}"
                logger.error(error_msg)
                raise ExecutionError(error_msg)

        except Exception as e:
            logger.error(f"Error in web scraping tool: {e}", exc_info=True)
            raise ExecutionError(f"Web scraping failed: {str(e)}")
