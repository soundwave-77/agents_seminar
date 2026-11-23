from agents import Agent
from agents.mcp import MCPServer


def create_multi_agent(
    servers: list[MCPServer],
    model_name: str,
) -> Agent:
    """Create a travel agent that coordinates specialists."""
    raise NotImplementedError("Implement the multi-agent pattern.")
