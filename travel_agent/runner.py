import agents
from agents import Agent, Session
from agents.run import DEFAULT_MAX_TURNS


async def run_agent(
    agent: Agent,
    message: str,
    session: Session,
    *,
    max_turns: int | None = DEFAULT_MAX_TURNS,
    tracing_disabled: bool = False,
) -> str:
    """Run one user turn with the given agent and dialogue session."""
    result = await agents.Runner.run(
        agent,
        message,
        session=session,
        max_turns=max_turns,
        run_config=agents.RunConfig(
            tracing_disabled=tracing_disabled,
            group_id=session.session_id,
        ),
    )
    return str(result.final_output)
