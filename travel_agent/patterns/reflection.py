from agents import Agent
from agents.mcp import MCPServer


def create_reflection_agent(
    servers: list[MCPServer],
    model_name: str,
) -> Agent:
    """Create a travel agent that critiques and revises its draft."""
    raise NotImplementedError("Implement the Reflection pattern.")
