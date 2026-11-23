import pathlib

import agents
import openai.types.shared
from agents import Agent
from agents.mcp import MCPServer

from travel_agent import tools


def create_react_agent(
    servers: list[MCPServer],
    model_name: str,
) -> Agent:
    """Create one travel agent using the already connected MCP servers."""
    model = agents.OpenAIProvider(use_responses=False).get_model(model_name)

    instructions_path = pathlib.Path(__file__).parent / "instructions" / "react.md"
    instructions = instructions_path.read_text(encoding="utf-8")

    return Agent(
        name="react_travel_agent",
        instructions=instructions,
        model=model,
        model_settings=agents.ModelSettings(
            top_p=1,
            temperature=0,
            reasoning=openai.types.shared.Reasoning(effort="medium"),
        ),
        mcp_servers=servers,
        tools=[tools.get_current_date],
    )
