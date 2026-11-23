import os

import langfuse
import nest_asyncio
from openinference.instrumentation import openai_agents


def setup_tracing() -> bool:
    """Initialize optional Langfuse tracing once per process."""
    if not all(
        os.environ.get(name, "").strip()
        for name in (
            "LANGFUSE_SECRET_KEY",
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_BASE_URL",
        )
    ):
        return False

    instrumentor = openai_agents.OpenAIAgentsInstrumentor()
    if not instrumentor.is_instrumented_by_opentelemetry:
        nest_asyncio.apply()
        instrumentor.instrument()
        langfuse.get_client()

    return True
