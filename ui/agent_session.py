import os
import uuid
from collections.abc import Callable

import agents
import agents.mcp
import streamlit as st
from agents import Agent
from agents.mcp import MCPServer

from travel_agent import mcp_servers, tracing
from travel_agent.patterns import multi_agent, plan_and_execute, react, reflection

DEFAULT_AGENT_NAME = "ReAct"
type AgentFactory = Callable[[list[MCPServer], str], Agent]


def _agent_factories() -> dict[str, AgentFactory]:
    return {
        DEFAULT_AGENT_NAME: react.create_react_agent,
        "Reflection": reflection.create_reflection_agent,
        "Plan-and-Execute": plan_and_execute.create_plan_and_execute_agent,
        "Multi-agent": multi_agent.create_multi_agent,
    }


def select_agent(key: str) -> str:
    """Render a selector and retain its value across page navigation."""
    names = tuple(_agent_factories())
    selected = st.selectbox(
        "Агент",
        names,
        index=names.index(st.session_state.get(key, DEFAULT_AGENT_NAME)),
        key=f"_{key}",
    )
    st.session_state[key] = selected
    return selected


async def initialize(agent_name: str = DEFAULT_AGENT_NAME) -> Agent:
    """Get the named agent, sharing MCP connections across agents and pages."""
    active_agents: dict[str, Agent] = st.session_state.setdefault("agents", {})
    if agent_name not in active_agents:
        active_agents[agent_name] = await _create_agent(_agent_factories()[agent_name])
    return active_agents[agent_name]


async def _create_agent(create_agent: AgentFactory) -> Agent:
    model = os.environ["AGENT_MODEL"]
    if "manager" in st.session_state:
        return create_agent(st.session_state.manager.active_servers, model)

    tracing_enabled = tracing.setup_tracing()
    manager = agents.mcp.MCPServerManager(
        [
            mcp_servers.create_pilotplans_server(),
            mcp_servers.create_fli_server(),
            mcp_servers.create_keenable_server(),
            mcp_servers.create_weather_server(),
        ],
        strict=True,
        connect_in_parallel=True,
        connect_timeout_seconds=60,
    )
    await manager.connect_all()
    try:
        # Finish MCP initialization before Streamlit pauses the event loop.
        for server in manager.active_servers:
            await server.list_tools()
        agent = create_agent(manager.active_servers, model)
    except Exception:
        await manager.cleanup_all()
        raise
    st.session_state.manager = manager
    st.session_state.tracing_enabled = tracing_enabled
    return agent


def new_dialogue() -> None:
    """Replace conversation memory while keeping the agent and MCP servers."""
    session = agents.SQLiteSession(str(uuid.uuid4()))
    if "session" in st.session_state:
        st.session_state.session.close()
    st.session_state.session = session
    st.session_state.messages = []
