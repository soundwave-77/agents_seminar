import sys

from agents.mcp import MCPServerStdio, MCPServerStreamableHttp


def create_pilotplans_server() -> MCPServerStreamableHttp:
    """Create a public PilotPlans MCP connection without an API key."""
    return MCPServerStreamableHttp(
        name="pilotplans",
        params={
            "url": "https://mcp.pilotplans.com/mcp",
        },
        cache_tools_list=True,
        client_session_timeout_seconds=60,
        max_retry_attempts=5,
        retry_backoff_seconds_max=60,
    )


def create_fli_server() -> MCPServerStdio:
    """Create a Fli MCP connection to open with an async context manager."""
    return MCPServerStdio(
        name="fli",
        params={
            "command": "fli-mcp",
        },
        cache_tools_list=True,
        client_session_timeout_seconds=60,
        max_retry_attempts=5,
        retry_backoff_seconds_max=60,
    )


def create_keenable_server() -> MCPServerStreamableHttp:
    """Create a public Keenable MCP connection without an API key."""
    return MCPServerStreamableHttp(
        name="keenable",
        params={
            "url": "https://api.keenable.ai/mcp-public",
            "timeout": 60,
        },
        cache_tools_list=True,
        client_session_timeout_seconds=60,
        max_retry_attempts=5,
        retry_backoff_seconds_max=60,
    )


def create_weather_server() -> MCPServerStdio:
    """Create a weather MCP connection to open with an async context manager."""
    return MCPServerStdio(
        name="weather",
        params={
            "command": sys.executable,
            "args": ["-m", "mcp_weather_server"],
        },
        cache_tools_list=True,
        client_session_timeout_seconds=60,
        max_retry_attempts=5,
        retry_backoff_seconds_max=60,
    )
