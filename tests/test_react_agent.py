import contextlib
import datetime
import os
import pathlib
import tempfile
import unittest
import unittest.mock

import agents
import agents.mcp
import agents.models.openai_provider
import agents.tool_context
import httpx2
import mcp.types
import openai
import openai.types.responses
from agents import ModelResponse

from travel_agent import tools
from travel_agent.patterns import react


def _text_response(text: str) -> ModelResponse:
    return ModelResponse(
        output=[
            openai.types.responses.ResponseOutputMessage(
                id="message-1",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    openai.types.responses.ResponseOutputText(
                        type="output_text",
                        text=text,
                        annotations=[],
                    ),
                ],
            ),
        ],
        usage=agents.Usage(),
        response_id=None,
    )


class ReactAgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        environment = unittest.mock.patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": ""},
        )
        environment.start()
        self.addCleanup(environment.stop)
        self.model = unittest.mock.create_autospec(agents.Model, instance=True)
        self.model.get_retry_advice.return_value = None
        self.server = agents.mcp.MCPServerStdio(
            params={"command": "unused-offline-test"},
            name="fake-weather",
        )
        self.server.list_tools = unittest.mock.AsyncMock(
            return_value=[
                mcp.types.Tool(
                    name="get_current_weather",
                    description="Get current weather for a city.",
                    inputSchema={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                ),
            ],
        )
        self.server.call_tool = unittest.mock.AsyncMock(
            return_value=mcp.types.CallToolResult(
                content=[mcp.types.TextContent(type="text", text="18 C")],
            ),
        )
        with unittest.mock.patch.object(
            agents.OpenAIProvider,
            "get_model",
            return_value=self.model,
        ):
            self.agent = react.create_react_agent(
                [self.server],
                "test-model",
            )

    async def test_instructions_load_from_another_working_directory(
        self,
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            unittest.mock.patch.object(
                agents.OpenAIProvider,
                "get_model",
                return_value=self.model,
            ),
            contextlib.chdir(directory),
        ):
            agent = react.create_react_agent(
                [self.server],
                "test-model",
            )

        self.assertEqual(
            agent.instructions,
            self.agent.instructions,
        )

    async def test_date_tool_reads_date_on_each_call_without_system_prefix(
        self,
    ) -> None:
        self.assertIn(tools.get_current_date, self.agent.tools)
        instructions_path = (
            pathlib.Path(react.__file__).parent / "instructions" / "react.md"
        )
        self.assertEqual(
            self.agent.instructions,
            instructions_path.read_text(encoding="utf-8"),
        )
        dates = [datetime.date(2026, 11, 15), datetime.date(2026, 11, 16)]
        context = agents.tool_context.ToolContext(
            context=None,
            tool_name="get_current_date",
            tool_call_id="date-test",
            tool_arguments="{}",
        )
        with unittest.mock.patch.object(tools.datetime, "date") as date:
            date.today.side_effect = dates
            first = await tools.get_current_date.on_invoke_tool(context, "{}")
            second = await tools.get_current_date.on_invoke_tool(context, "{}")

        self.assertEqual(first, "2026-11-15")
        self.assertEqual(second, "2026-11-16")

    async def test_tool_observation_and_history_reach_next_model_calls(
        self,
    ) -> None:
        self.model.get_response.side_effect = [
            agents.ModelResponse(
                output=[
                    openai.types.responses.ResponseFunctionToolCall(
                        type="function_call",
                        call_id="weather-1",
                        name="get_current_weather",
                        arguments='{"city": "Lisbon"}',
                    ),
                ],
                usage=agents.Usage(),
                response_id=None,
            ),
            _text_response("В Лиссабоне 18 C."),
            _text_response("Для завтра нужен прогноз."),
        ]
        config = agents.RunConfig(tracing_disabled=True)
        with contextlib.closing(
            agents.SQLiteSession("test", db_path=":memory:"),
        ) as session:
            first = await agents.Runner.run(
                self.agent,
                "Погода в Лиссабоне?",
                session=session,
                run_config=config,
            )
            second = await agents.Runner.run(
                self.agent,
                "А завтра?",
                session=session,
                run_config=config,
            )

        self.server.call_tool.assert_awaited_once()
        self.assertEqual(
            self.server.call_tool.await_args.args[:2],
            ("get_current_weather", {"city": "Lisbon"}),
        )
        observation_input = self.model.get_response.await_args_list[1].kwargs["input"]
        self.assertTrue(
            any(
                item.get("type") == "function_call_output"
                and "18 C" in str(item.get("output"))
                for item in observation_input
            )
        )
        continuation = self.model.get_response.await_args_list[2].kwargs["input"]
        self.assertEqual(continuation[-1]["content"], "А завтра?")
        self.assertIn("Погода в Лиссабоне?", str(continuation))
        self.assertIn("В Лиссабоне 18 C.", str(continuation))
        self.assertEqual(first.final_output, "В Лиссабоне 18 C.")
        self.assertEqual(second.final_output, "Для завтра нужен прогноз.")
        self.assertTrue(
            all(
                call.kwargs["tracing"] == agents.ModelTracing.DISABLED
                for call in self.model.get_response.await_args_list
            )
        )

    async def test_sdk_uses_chat_api_model_key_and_base_url(
        self,
    ) -> None:
        cases = (
            (None, "https://api.openai.com/v1/"),
            (
                "https://api.example.test/v1",
                "https://api.example.test/v1/",
            ),
        )
        for base_url, expected_url in cases:
            with self.subTest(base_url=base_url):
                if base_url is None:
                    os.environ.pop("OPENAI_BASE_URL", None)
                else:
                    os.environ["OPENAI_BASE_URL"] = base_url
                http = unittest.mock.Mock(side_effect=AssertionError("No HTTP"))
                async with openai.DefaultAsyncHttpx2Client(
                    transport=httpx2.MockTransport(http),
                ) as http_client:
                    with unittest.mock.patch.object(
                        agents.models.openai_provider,
                        "shared_http_client",
                        return_value=http_client,
                    ):
                        agent = react.create_react_agent(
                            [],
                            "qwen/qwen3.8-27b",
                        )
                        model = agent.model
                        self.assertIsInstance(
                            model,
                            agents.OpenAIChatCompletionsModel,
                        )
                        self.assertEqual(model.model, "qwen/qwen3.8-27b")
                        self.assertEqual(
                            str(model._client.base_url),
                            expected_url,
                        )
                        self.assertEqual(model._client.api_key, "test-key")
                        await model._client.close()
                http.assert_not_called()


if __name__ == "__main__":
    unittest.main()
