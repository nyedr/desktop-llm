"""Web scraping tool using Jina Reader with markdown conversion."""

import re
import logging
import requests
from typing import Dict, Any, Optional
import markdownify

from app.models.function import Tool, FunctionType, register_function, ExecutionError

logger = logging.getLogger(__name__)


def extract_title(text: str) -> Optional[str]:
    """Extract the title from structured text."""
    match = re.search(r'Title: (.*)\n', text)
    return match.group(1).strip() if match else None


def clean_urls(text: str) -> str:
    """Clean URLs from structured text."""
    return re.sub(r'\((http[^)]+)\)', '', text)


def convert_to_markdown(content: str) -> str:
    """Convert the content to markdown format."""
    # First clean up the content
    content = clean_urls(content)

    # Remove metadata lines and prefix
    content = re.sub(r'^(?:Title:|URL Source:|Markdown Content:|Contents in markdown format:).*\n',
                     '', content, flags=re.MULTILINE)
    # Extra check for this specific prefix
    content = re.sub(r'^Contents in markdown format:\n', '', content)

    # Fix headers (remove duplicate dashes under headers)
    content = re.sub(r'\n-+\n(?=[A-Za-z])', '\n', content)

    # Fix any remaining headers that use dashes
    content = re.sub(r'([^\n]+)\n-+', r'# \1', content)

    # Fix table formatting
    def fix_table_row(match):
        row = match.group(0)
        # Remove all newlines and extra spaces within cells
        row = re.sub(r'\s*\n+\s*', ' ', row)
        # Clean up cell formatting
        row = re.sub(r'\|\s+', '| ', row)  # Space after |
        row = re.sub(r'\s+\|', ' |', row)  # Space before |
        row = re.sub(r'\s{2,}', ' ', row)  # Multiple spaces to single space
        row = re.sub(r'\| +\|', '| |', row)  # Empty cells
        return row.strip()

    # Find and fix tables (more aggressive pattern)
    table_pattern = r'\|[^|]*\|[^\n]*(?:\n+\|[^|]*\|[^\n]*)*'
    content = re.sub(table_pattern, lambda m: fix_table_row(
        m).replace('\n', ''), content)

    # Fix number formatting
    def fix_number(match):
        num = match.group(0)
        # Handle decimal numbers
        if '.' in num:
            parts = num.split('.')
            # Format the integer part if it's large enough
            if len(parts[0]) >= 4:
                parts[0] = ' '.join(parts[0][i:i+3]
                                    for i in range(0, len(parts[0]), 3)).strip()
            return '.'.join(parts)
        # Format large integers
        if len(num) >= 4:
            return ' '.join(num[i:i+3] for i in range(0, len(num), 3)).strip()
        return num

    # Find and format numbers
    content = re.sub(r'\b\d+(?:\.\d+)?\b', fix_number, content)

    # Convert to markdown (only if it contains HTML)
    if '<' in content and '>' in content:
        content = markdownify.markdownify(content, heading_style="ATX")

    # Clean up the markdown
    content = re.sub(r'\n{3,}', '\n\n', content)  # Remove excess newlines
    # Fix header spacing
    content = re.sub(r'(\n#{1,6} [^\n]+)\n+(?=#{1,6} )', r'\1\n', content)
    # Normalize paragraph spacing
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)

    # Remove any remaining extra whitespace
    # Replace multiple spaces with single space
    content = re.sub(r' +', ' ', content)
    # Remove leading spaces after newlines
    content = re.sub(r'\n +', '\n', content)
    content = content.strip()

    return content


@register_function(
    func_type=FunctionType.TOOL,
    name="web_scrape",
    description="Scrape and process web pages using Jina Reader (r.jina.ai)",
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL of the web page to scrape"
            },
            "raw": {
                "type": "boolean",
                "description": "Get the actual content without markdown conversion",
                "default": False
            }
        },
        "required": ["url"]
    }
)
class WebScrapeTool(Tool):
    """Tool for scraping web content using Jina Reader."""

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
            raw = args.get("raw", False)

            # Construct Jina Reader URL
            jina_url = f"https://r.jina.ai/{url}"
            logger.info(f"Scraping URL: {url} via Jina Reader")

            # Set up headers
            headers = {
                "X-No-Cache": "false",
                "X-With-Generated-Alt": "true"
            }

            # Make the request
            try:
                response = requests.get(jina_url, headers=headers, timeout=30)
                response.raise_for_status()
                content = response.text

                if not content:
                    raise ExecutionError("Received empty response from server")

                # Extract title for logging
                title = extract_title(content)

                if raw:
                    content = content
                else:
                    content = convert_to_markdown(content)

                logger.info(f"Successfully scraped: {title if title else url}")

                return {
                    "content": content,
                    "title": title,
                    "url": url,
                    "is_raw": raw
                }

            except requests.RequestException as e:
                error_msg = f"Error scraping web page: {str(e)}"
                logger.error(error_msg)
                raise ExecutionError(error_msg)

        except Exception as e:
            logger.error(f"Error in web scraping tool: {e}", exc_info=True)
            raise ExecutionError(f"Web scraping failed: {str(e)}")
