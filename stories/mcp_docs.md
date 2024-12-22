Quickstart
For Server Developers
Get started building your own server to use in Claude for Desktop and other clients.

In this tutorial, we’ll build a simple MCP weather server and connect it to a host, Claude for Desktop. We’ll start with a basic setup, and then progress to more complex use cases.

​
What we’ll be building
Many LLMs (including Claude) do not currently have the ability to fetch the forecast and severe weather alerts. Let’s use MCP to solve that!

We’ll build a server that exposes two tools: get-alerts and get-forecast. Then we’ll connect the server to an MCP host (in this case, Claude for Desktop):

Servers can connect to any client. We’ve chosen Claude for Desktop here for simplicity, but we also have guides on building your own client as well as a list of other clients here.

Why Claude for Desktop and not Claude.ai?

​
Core MCP Concepts
MCP servers can provide three main types of capabilities:

Resources: File-like data that can be read by clients (like API responses or file contents)
Tools: Functions that can be called by the LLM (with user approval)
Prompts: Pre-written templates that help users accomplish specific tasks
This tutorial will primarily focus on tools.

Python
Node
Let’s get started with building our weather server! You can find the complete code for what we’ll be building here.

Prerequisite knowledge
This quickstart assumes you have familiarity with:

Python
LLMs like Claude
System requirements
For Python, make sure you have Python 3.9 or higher installed.

Set up your environment
First, let’s install uv and set up our Python project and environment:

MacOS/Linux

Windows

curl -LsSf https://astral.sh/uv/install.sh | sh
Make sure to restart your terminal afterwards to ensure that the uv command gets picked up.

Now, let’s create and set up our project:

MacOS/Linux

Windows

# Create a new directory for our project

uv init weather
cd weather

# Create virtual environment and activate it

uv venv
source .venv/bin/activate

# Install dependencies

uv add mcp httpx

# Remove template file

rm hello.py

# Create our files

mkdir -p src/weather
touch src/weather/**init**.py
touch src/weather/server.py
Add this code to pyproject.toml:

...rest of config

[build-system]
requires = [ "hatchling",]
build-backend = "hatchling.build"

[project.scripts]
weather = "weather:main"
Add this code to **init**.py:

src/weather/**init**.py

from . import server
import asyncio

def main():
"""Main entry point for the package."""
asyncio.run(server.main())

# Optionally expose other important items at package level

**all** = ['main', 'server']
Now let’s dive into building your server.

Building your server
Importing packages
Add these to the top of your server.py:

from typing import Any
import asyncio
import httpx
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio
Setting up the instance
Then initialize the server instance and the base URL for the NWS API:

NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"

server = Server("weather")
Implementing tool listing
We need to tell clients what tools are available. The list_tools() decorator registers this handler:

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
"""
List available tools.
Each tool specifies its arguments using JSON Schema validation.
"""
return [
types.Tool(
name="get-alerts",
description="Get weather alerts for a state",
inputSchema={
"type": "object",
"properties": {
"state": {
"type": "string",
"description": "Two-letter state code (e.g. CA, NY)",
},
},
"required": ["state"],
},
),
types.Tool(
name="get-forecast",
description="Get weather forecast for a location",
inputSchema={
"type": "object",
"properties": {
"latitude": {
"type": "number",
"description": "Latitude of the location",
},
"longitude": {
"type": "number",
"description": "Longitude of the location",
},
},
"required": ["latitude", "longitude"],
},
),
]

This defines our two tools: get-alerts and get-forecast.

Helper functions
Next, let’s add our helper functions for querying and formatting the data from the National Weather Service API:

async def make_nws_request(client: httpx.AsyncClient, url: str) -> dict[str, Any] | None:
"""Make a request to the NWS API with proper error handling."""
headers = {
"User-Agent": USER_AGENT,
"Accept": "application/geo+json"
}

    try:
        response = await client.get(url, headers=headers, timeout=30.0)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None

def format_alert(feature: dict) -> str:
"""Format an alert feature into a concise string."""
props = feature["properties"]
return (
f"Event: {props.get('event', 'Unknown')}\n"
f"Area: {props.get('areaDesc', 'Unknown')}\n"
f"Severity: {props.get('severity', 'Unknown')}\n"
f"Status: {props.get('status', 'Unknown')}\n"
f"Headline: {props.get('headline', 'No headline')}\n"
"---"
)
Implementing tool execution
The tool execution handler is responsible for actually executing the logic of each tool. Let’s add it:

@server.call_tool()
async def handle_call_tool(
name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
"""
Handle tool execution requests.
Tools can fetch weather data and notify clients of changes.
"""
if not arguments:
raise ValueError("Missing arguments")

    if name == "get-alerts":
        state = arguments.get("state")
        if not state:
            raise ValueError("Missing state parameter")

        # Convert state to uppercase to ensure consistent format
        state = state.upper()
        if len(state) != 2:
            raise ValueError("State must be a two-letter code (e.g. CA, NY)")

        async with httpx.AsyncClient() as client:
            alerts_url = f"{NWS_API_BASE}/alerts?area={state}"
            alerts_data = await make_nws_request(client, alerts_url)

            if not alerts_data:
                return [types.TextContent(type="text", text="Failed to retrieve alerts data")]

            features = alerts_data.get("features", [])
            if not features:
                return [types.TextContent(type="text", text=f"No active alerts for {state}")]

            # Format each alert into a concise string
            formatted_alerts = [format_alert(feature) for feature in features[:20]] # only take the first 20 alerts
            alerts_text = f"Active alerts for {state}:\n\n" + "\n".join(formatted_alerts)

            return [
                types.TextContent(
                    type="text",
                    text=alerts_text
                )
            ]
    elif name == "get-forecast":
        try:
            latitude = float(arguments.get("latitude"))
            longitude = float(arguments.get("longitude"))
        except (TypeError, ValueError):
            return [types.TextContent(
                type="text",
                text="Invalid coordinates. Please provide valid numbers for latitude and longitude."
            )]

        # Basic coordinate validation
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            return [types.TextContent(
                type="text",
                text="Invalid coordinates. Latitude must be between -90 and 90, longitude between -180 and 180."
            )]

        async with httpx.AsyncClient() as client:
            # First get the grid point
            lat_str = f"{latitude}"
            lon_str = f"{longitude}"
            points_url = f"{NWS_API_BASE}/points/{lat_str},{lon_str}"
            points_data = await make_nws_request(client, points_url)

            if not points_data:
                return [types.TextContent(type="text", text=f"Failed to retrieve grid point data for coordinates: {latitude}, {longitude}. This location may not be supported by the NWS API (only US locations are supported).")]

            # Extract forecast URL from the response
            properties = points_data.get("properties", {})
            forecast_url = properties.get("forecast")

            if not forecast_url:
                return [types.TextContent(type="text", text="Failed to get forecast URL from grid point data")]

            # Get the forecast
            forecast_data = await make_nws_request(client, forecast_url)

            if not forecast_data:
                return [types.TextContent(type="text", text="Failed to retrieve forecast data")]

            # Format the forecast periods
            periods = forecast_data.get("properties", {}).get("periods", [])
            if not periods:
                return [types.TextContent(type="text", text="No forecast periods available")]

            # Format each period into a concise string
            formatted_forecast = []
            for period in periods:
                forecast_text = (
                    f"{period.get('name', 'Unknown')}:\n"
                    f"Temperature: {period.get('temperature', 'Unknown')}°{period.get('temperatureUnit', 'F')}\n"
                    f"Wind: {period.get('windSpeed', 'Unknown')} {period.get('windDirection', '')}\n"
                    f"{period.get('shortForecast', 'No forecast available')}\n"
                    "---"
                )
                formatted_forecast.append(forecast_text)

            forecast_text = f"Forecast for {latitude}, {longitude}:\n\n" + "\n".join(formatted_forecast)

            return [types.TextContent(
                type="text",
                text=forecast_text
            )]
    else:
        raise ValueError(f"Unknown tool: {name}")

Running the server
Finally, implement the main function to run the server:

async def main(): # Run the server using stdin/stdout streams
async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
await server.run(
read_stream,
write_stream,
InitializationOptions(
server_name="weather",
server_version="0.1.0",
capabilities=server.get_capabilities(
notification_options=NotificationOptions(),
experimental_capabilities={},
),
),
)

# This is needed if you'd like to connect to a custom client

if **name** == "**main**":
asyncio.run(main())
Your server is complete! Run uv run src/weather/server.py to confirm that everything’s working.

Let’s now test your server from an existing MCP host, Claude for Desktop.

Testing your server with Claude for Desktop
Claude for Desktop is not yet available on Linux. Linux users can proceed to the Building a client tutorial to build an MCP client that connects to the server we just built.

First, make sure you have Claude for Desktop installed. You can install the latest version here. If you already have Claude for Desktop, make sure it’s updated to the latest version.

We’ll need to configure Claude for Desktop for whichever MCP servers you want to use. To do this, open your Claude for Desktop App configuration at ~/Library/Application Support/Claude/claude_desktop_config.json in a text editor. Make sure to create the file if it doesn’t exist.

For example, if you have VS Code installed:

MacOS/Linux
Windows

code ~/Library/Application\ Support/Claude/claude_desktop_config.json
You’ll then add your servers in the mcpServers key. The MCP UI elements will only show up in Claude for Desktop if at least one server is properly configured.

In this case, we’ll add our single weather server like so:

MacOS/Linux
Windows
Python

{
"mcpServers": {
"weather": {
"command": "uv",
"args": [
"--directory",
"/ABSOLUTE/PATH/TO/PARENT/FOLDER/weather",
"run",
"weather"
]
}
}
}
Make sure you pass in the absolute path to your server.

This tells Claude for Desktop:

There’s an MCP server named “weather”
To launch it by running uv --directory /ABSOLUTE/PATH/TO/PARENT/FOLDER/weather run weather
Save the file, and restart Claude for Desktop.

​
Test with commands
Let’s make sure Claude for Desktop is picking up the two tools we’ve exposed in our weather server. You can do this by looking for the hammer icon:

After clicking on the hammer icon, you should see two tools listed:

If your server isn’t being picked up by Claude for Desktop, proceed to the Troubleshooting section for debugging tips.

If the hammer icon has shown up, you can now test your server by running the following commands in Claude for Desktop:

What’s the weather in Sacramento?
What are the active weather alerts in Texas?

Since this is the US National Weather service, the queries will only work for US locations.

​
What’s happening under the hood
When you ask a question:

The client sends your question to Claude
Claude analyzes the available tools and decides which one(s) to use
The client executes the chosen tool(s) through the MCP server
The results are sent back to Claude
Claude formulates a natural language response
The response is displayed to you!

Example Servers
A list of example servers and implementations

This page showcases various Model Context Protocol (MCP) servers that demonstrate the protocol’s capabilities and versatility. These servers enable Large Language Models (LLMs) to securely access tools and data sources.

​
Reference implementations
These official reference servers demonstrate core MCP features and SDK usage:

​
Data and file systems
Filesystem - Secure file operations with configurable access controls
PostgreSQL - Read-only database access with schema inspection capabilities
SQLite - Database interaction and business intelligence features
Google Drive - File access and search capabilities for Google Drive
​
Development tools
Git - Tools to read, search, and manipulate Git repositories
GitHub - Repository management, file operations, and GitHub API integration
GitLab - GitLab API integration enabling project management
Sentry - Retrieving and analyzing issues from Sentry.io
​
Web and browser automation
Brave Search - Web and local search using Brave’s Search API
Fetch - Web content fetching and conversion optimized for LLM usage
Puppeteer - Browser automation and web scraping capabilities
​
Productivity and communication
Slack - Channel management and messaging capabilities
Google Maps - Location services, directions, and place details
Memory - Knowledge graph-based persistent memory system
​
AI and specialized tools
EverArt - AI image generation using various models
Sequential Thinking - Dynamic problem-solving through thought sequences
AWS KB Retrieval - Retrieval from AWS Knowledge Base using Bedrock Agent Runtime
​
Official integrations
These MCP servers are maintained by companies for their platforms:

Axiom - Query and analyze logs, traces, and event data using natural language
Browserbase - Automate browser interactions in the cloud
Cloudflare - Deploy and manage resources on the Cloudflare developer platform
E2B - Execute code in secure cloud sandboxes
Neon - Interact with the Neon serverless Postgres platform
Obsidian Markdown Notes - Read and search through Markdown notes in Obsidian vaults
Qdrant - Implement semantic memory using the Qdrant vector search engine
Raygun - Access crash reporting and monitoring data
Search1API - Unified API for search, crawling, and sitemaps
Tinybird - Interface with the Tinybird serverless ClickHouse platform
​
Community highlights
A growing ecosystem of community-developed servers extends MCP’s capabilities:

Docker - Manage containers, images, volumes, and networks
Kubernetes - Manage pods, deployments, and services
Linear - Project management and issue tracking
Snowflake - Interact with Snowflake databases
Spotify - Control Spotify playback and manage playlists
Todoist - Task management integration
Note: Community servers are untested and should be used at your own risk. They are not affiliated with or endorsed by Anthropic.

For a complete list of community servers, visit the MCP Servers Repository.

​
Getting started
​
Using reference servers
TypeScript-based servers can be used directly with npx:

npx -y @modelcontextprotocol/server-memory
Python-based servers can be used with uvx (recommended) or pip:

# Using uvx

uvx mcp-server-git

# Using pip

pip install mcp-server-git
python -m mcp_server_git
​
Configuring with Claude
To use an MCP server with Claude, add it to your configuration:

{
"mcpServers": {
"memory": {
"command": "npx",
"args": ["-y", "@modelcontextprotocol/server-memory"]
},
"filesystem": {
"command": "npx",
"args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/files"]
},
"github": {
"command": "npx",
"args": ["-y", "@modelcontextprotocol/server-github"],
"env": {
"GITHUB_PERSONAL_ACCESS_TOKEN": "<YOUR_TOKEN>"
}
}
}
}
​
Additional resources
MCP Servers Repository - Complete collection of reference implementations and community servers
Awesome MCP Servers - Curated list of MCP servers
MCP CLI - Command-line inspector for testing MCP servers
MCP Get - Tool for installing and managing MCP servers
Visit our GitHub Discussions to engage with the MCP community.

Building MCP with LLMs
Speed up your MCP development using LLMs such as Claude!

This guide will help you use LLMs to help you build custom Model Context Protocol (MCP) servers and clients. We’ll be focusing on Claude for this tutorial, but you can do this with any frontier LLM.

​
Preparing the documentation
Before starting, gather the necessary documentation to help Claude understand MCP:

Visit https://modelcontextprotocol.io/llms-full.txt and copy the full documentation text
Navigate to either the MCP TypeScript SDK or Python SDK repository
Copy the README files and other relevant documentation
Paste these documents into your conversation with Claude
​
Describing your server
Once you’ve provided the documentation, clearly describe to Claude what kind of server you want to build. Be specific about:

What resources your server will expose
What tools it will provide
Any prompts it should offer
What external systems it needs to interact with
For example:

Build an MCP server that:

- Connects to my company's PostgreSQL database
- Exposes table schemas as resources
- Provides tools for running read-only SQL queries
- Includes prompts for common data analysis tasks
  ​
  Working with Claude
  When working with Claude on MCP servers:

Start with the core functionality first, then iterate to add more features
Ask Claude to explain any parts of the code you don’t understand
Request modifications or improvements as needed
Have Claude help you test the server and handle edge cases
Claude can help implement all the key MCP features:

Resource management and exposure
Tool definitions and implementations
Prompt templates and handlers
Error handling and logging
Connection and transport setup
​
Best practices
When building MCP servers with Claude:

Break down complex servers into smaller pieces
Test each component thoroughly before moving on
Keep security in mind - validate inputs and limit access appropriately
Document your code well for future maintenance
Follow MCP protocol specifications carefully
​
Next steps
After Claude helps you build your server:

Review the generated code carefully
Test the server with the MCP Inspector tool
Connect it to Claude.app or other MCP clients
Iterate based on real usage and feedback
Remember that Claude can help you modify and improve your server as requirements change over time.

Need more guidance? Just ask Claude specific questions about implementing MCP features or troubleshooting issues that arise.
