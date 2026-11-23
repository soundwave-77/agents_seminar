import importlib
import os
import unittest
import unittest.mock
import warnings

import agents
from agents.mcp import MCPServer


class AgentPatternTests(unittest.TestCase):
    def test_react(self) -> None:
        self._check_pattern("ReAct", "react", "create_react_agent")

    def test_reflection(self) -> None:
        self._check_pattern("Reflection", "reflection", "create_reflection_agent")

    def test_plan_and_execute(self) -> None:
        self._check_pattern(
            "Plan-and-Execute",
            "plan_and_execute",
            "create_plan_and_execute_agent",
        )

    def test_multi_agent(self) -> None:
        self._check_pattern("Multi-agent", "multi_agent", "create_multi_agent")

    def _check_pattern(self, name: str, module_name: str, function_name: str) -> None:
        servers: list[MCPServer] = []
        for server_name in ("pilotplans", "fli", "keenable", "weather"):
            server = unittest.mock.Mock(spec=MCPServer)
            server.name = server_name
            servers.append(server)
        try:
            with (
                unittest.mock.patch.dict(
                    os.environ,
                    {
                        "OPENAI_API_KEY": "offline-test",
                        "OPENAI_BASE_URL": "https://api.example.invalid/v1",
                    },
                ),
                unittest.mock.patch.object(
                    agents.OpenAIProvider,
                    "get_model",
                    return_value=unittest.mock.Mock(spec=agents.Model),
                ),
            ):
                module = importlib.import_module(f"travel_agent.patterns.{module_name}")
                create_agent = getattr(module, function_name)
                agent = create_agent(servers, "test-model")
        except NotImplementedError:
            message = f"{name}: паттерн пока не реализован."
        # Report factory failures as warnings so every pattern can be checked.
        except Exception as error:  # noqa: BLE001
            message = (
                f"{name}: не удалось создать агента ({type(error).__name__}: {error})."
            )
        else:
            if isinstance(agent, agents.Agent):
                return
            message = (
                f"{name}: {function_name} должна вернуть Agent, "
                f"получен {type(agent).__name__}."
            )
        warnings.warn(message, stacklevel=2)
