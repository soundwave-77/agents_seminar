import asyncio
import contextlib
import os
import pathlib
import unittest
import unittest.mock

import agents
import agents.mcp
import openai.types.responses
import streamlit.testing.v1
from agents import ModelResponse

from travel_agent import tracing
from travel_agent.patterns import plan_and_execute, react, reflection


def _text_response(text: str) -> ModelResponse:
    return ModelResponse(
        output=[
            openai.types.responses.ResponseOutputMessage(
                id=text,
                type="message",
                role="assistant",
                status="completed",
                content=[
                    openai.types.responses.ResponseOutputText(
                        type="output_text",
                        text=text,
                        annotations=[],
                    )
                ],
            )
        ],
        usage=agents.Usage(),
        response_id=None,
    )


class StreamlitChatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(
            unittest.mock.patch.dict(
                os.environ,
                {"AGENT_MODEL": "test"},
            )
        )
        self.stack.enter_context(
            unittest.mock.patch.object(
                tracing,
                "setup_tracing",
                return_value=False,
            )
        )
        self.manager_factory = self.stack.enter_context(
            unittest.mock.patch.object(agents.mcp, "MCPServerManager"),
        )
        self.manager = self.manager_factory.return_value
        self.manager.connect_all = unittest.mock.AsyncMock()
        self.manager.cleanup_all = unittest.mock.AsyncMock()
        self.server = unittest.mock.Mock(spec=agents.mcp.MCPServer)
        self.server.list_tools.return_value = []
        self.manager.active_servers = [self.server]
        self.create_agent_function = react.create_react_agent
        self.create_agent = self.stack.enter_context(
            unittest.mock.patch.object(
                react,
                "create_react_agent",
            )
        )
        self.model = unittest.mock.create_autospec(agents.Model, instance=True)
        self.model.get_retry_advice.return_value = None
        self.create_agent.return_value = agents.Agent(
            name="test",
            model=self.model,
        )
        session_class = agents.SQLiteSession
        self.session_class = session_class

        def create_session(session_id):
            session = session_class(session_id)
            self.stack.callback(session.close)
            return session

        self.stack.enter_context(
            unittest.mock.patch.object(
                agents,
                "SQLiteSession",
                side_effect=create_session,
            )
        )
        self.application = streamlit.testing.v1.AppTest.from_file(
            str(pathlib.Path(__file__).resolve().parents[1] / "app.py"),
        )

    def test_startup_reuses_agent_connections_and_history(self) -> None:
        self.model.get_response.side_effect = [
            _text_response("Ответ 1"),
            _text_response("Ответ 2"),
        ]
        self.application.run()
        self.manager.connect_all.assert_awaited_once()
        self.server.list_tools.assert_awaited_once()
        self.model.get_response.assert_not_awaited()
        session = self.application.session_state["session"]
        self.assertEqual(session.db_path, ":memory:")
        self.application.chat_input[0].set_value("Париж").run()
        self.application.chat_input[0].set_value("А музеи?").run()
        self.assertFalse(self.application.exception)
        self.create_agent.assert_called_once()
        self.manager.connect_all.assert_awaited_once()
        self.manager.cleanup_all.assert_not_awaited()
        self.server.list_tools.assert_awaited_once()
        self.assertIs(self.application.session_state["session"], session)
        self.assertEqual(len(self.application.chat_message), 4)
        history = self.model.get_response.await_args_list[1].kwargs["input"]
        self.assertIn("Париж", str(history))
        self.assertIn("Ответ 1", str(history))

    def test_new_dialogue_resets_history_and_preserves_resources(self) -> None:
        self.model.get_response.side_effect = [
            _text_response("Ответ 1"),
            _text_response("Ответ 2"),
        ]
        self.application.run()
        self.application.chat_input[0].set_value("Париж").run()
        previous = self.application.session_state["session"]
        old_id = previous.session_id
        with unittest.mock.patch.object(
            previous,
            "close",
            wraps=previous.close,
        ) as close:
            self.application.button[0].click().run()
            close.assert_called_once()
        self.assertFalse(self.application.exception)
        self.assertNotEqual(
            self.application.session_state["session"].session_id,
            old_id,
        )
        self.assertEqual(len(self.application.chat_message), 0)
        self.assertIsNot(self.application.session_state["session"], previous)
        self.application.chat_input[0].set_value("Рим").run()
        self.assertFalse(self.application.exception)
        self.assertEqual(
            self.model.get_response.await_args_list[1].kwargs["input"],
            [{"role": "user", "content": "Рим"}],
        )
        self.create_agent.assert_called_once()
        self.manager.connect_all.assert_awaited_once()
        self.manager.cleanup_all.assert_not_awaited()
        with (
            asyncio.Runner() as runner,
            contextlib.closing(
                self.session_class(old_id),
            ) as restored,
        ):
            self.assertEqual(runner.run(restored.get_items()), [])

    def test_switching_agent_keeps_page_choice_and_starts_new_history(self) -> None:
        self.model.get_response.return_value = _text_response("Ответ ReAct")
        other_model = unittest.mock.create_autospec(agents.Model, instance=True)
        other_model.get_retry_advice.return_value = None
        other_model.get_response.return_value = _text_response("Ответ Reflection")
        other_agent = agents.Agent(name="reflection", model=other_model)
        with unittest.mock.patch.object(
            reflection,
            "create_reflection_agent",
            return_value=other_agent,
        ) as reflection_factory:
            self.application.run()
            self.application.chat_input[0].set_value("Париж").run()
            self.application.selectbox[0].select("Reflection").run()
            self.assertEqual(len(self.application.chat_message), 0)

            self.application.switch_page("ui/pages/dialogue_generation.py").run()
            self.assertEqual(self.application.selectbox[0].value, "ReAct")
            self.application.selectbox[0].select("Multi-agent").run()
            self.application.switch_page("ui/pages/chat.py").run()
            self.assertEqual(self.application.selectbox[0].value, "Reflection")
            self.application.chat_input[0].set_value("Рим").run()
            self.assertFalse(self.application.exception)
            self.application.switch_page("ui/pages/dialogue_generation.py").run()
            self.assertEqual(self.application.selectbox[0].value, "Multi-agent")

        self.model.get_response.assert_awaited_once()
        other_model.get_response.assert_awaited_once()
        self.assertEqual(
            other_model.get_response.await_args.kwargs["input"],
            [{"role": "user", "content": "Рим"}],
        )
        reflection_factory.assert_called_once_with([self.server], "test")
        self.manager.connect_all.assert_awaited_once()

    def test_unimplemented_agent_can_be_deselected(self) -> None:
        self.application.run()
        with unittest.mock.patch.object(
            plan_and_execute,
            "create_plan_and_execute_agent",
            side_effect=NotImplementedError,
        ):
            self.application.selectbox[0].select("Plan-and-Execute").run()
        self.assertEqual(
            self.application.info[0].value,
            "Агент Plan-and-Execute пока не реализован.",
        )
        self.assertFalse(self.application.chat_input)
        self.model.get_response.assert_not_awaited()
        self.application.selectbox[0].select("ReAct").run()
        self.assertEqual(len(self.application.chat_input), 1)
        self.create_agent.assert_called_once()
        self.manager.connect_all.assert_awaited_once()

    def test_other_browser_sessions_have_separate_history(self) -> None:
        self.application.run()
        other = streamlit.testing.v1.AppTest.from_file(
            str(pathlib.Path(__file__).resolve().parents[1] / "app.py"),
        ).run()
        self.assertNotEqual(
            self.application.session_state["session"].session_id,
            other.session_state["session"].session_id,
        )
        self.assertEqual(self.create_agent.call_count, 2)

    def test_slash_model_name_uses_chat_completions_without_prefix_routing(
        self,
    ) -> None:
        self.create_agent.side_effect = self.create_agent_function
        with (
            unittest.mock.patch.dict(
                os.environ,
                {
                    "AGENT_MODEL": "qwen/test-model",
                    "OPENAI_API_KEY": "test-key",
                    "OPENAI_BASE_URL": "https://api.example.test/v1",
                },
            ),
            unittest.mock.patch.object(
                agents.Runner,
                "run",
                new_callable=unittest.mock.AsyncMock,
                return_value=unittest.mock.Mock(final_output="Ответ"),
            ) as run,
        ):
            self.application.run()
            self.application.chat_input[0].set_value("Привет").run()
            self.assertFalse(self.application.exception)
            model = run.await_args.args[0].model
            self.assertIsInstance(model, agents.OpenAIChatCompletionsModel)
            self.assertEqual(model.model, "qwen/test-model")
            self.assertEqual(
                str(model._client.base_url),
                "https://api.example.test/v1/",
            )
            with asyncio.Runner() as runner:
                runner.run(model._client.close())

    def test_failed_request_allows_another_message(self) -> None:
        self.model.get_response.side_effect = [
            RuntimeError("private-payload"),
            _text_response("Готово"),
        ]
        self.application.run()
        with self.assertLogs(level="ERROR"):
            self.application.chat_input[0].set_value("Париж").run()
        self.application.chat_input[0].set_value("Повтори").run()
        self.assertFalse(self.application.exception)
        text = " ".join(item.value for item in self.application.markdown)
        self.assertIn("Не удалось завершить запрос", text)
        self.assertIn("Готово", text)
        self.assertNotIn("private-payload", text)
        self.create_agent.assert_called_once()

    def test_tool_discovery_failure_cleans_connections_and_can_retry(self) -> None:
        self.server.list_tools.side_effect = RuntimeError("transport closed")
        with self.assertLogs(level="ERROR"):
            self.application.run()
        self.manager.cleanup_all.assert_awaited_once()
        self.create_agent.assert_not_called()
        self.assertNotIn("ReAct", self.application.session_state["agents"])
        self.server.list_tools.side_effect = None
        self.application.run()
        self.assertFalse(self.application.exception)
        self.assertEqual(len(self.application.chat_input), 1)

    def test_agent_creation_failure_cleans_connections_and_can_retry(self) -> None:
        self.create_agent.side_effect = RuntimeError("invalid agent")
        with self.assertLogs(level="ERROR"):
            self.application.run()
        self.manager.cleanup_all.assert_awaited_once()
        self.assertNotIn("ReAct", self.application.session_state["agents"])
        self.create_agent.side_effect = None
        self.application.run()
        self.assertFalse(self.application.exception)
        self.assertEqual(len(self.application.chat_input), 1)


class ManagerLifecycleTests(unittest.TestCase):
    def test_connections_survive_separate_loop_runs_and_close_in_owner_tasks(
        self,
    ) -> None:
        events = []
        servers = []
        for name in ("first", "second"):
            server = unittest.mock.Mock(spec=agents.mcp.MCPServer)
            server.name = name

            async def connect(name=name):
                events.append((name, "connect", asyncio.current_task()))

            async def cleanup(name=name):
                events.append((name, "cleanup", asyncio.current_task()))

            server.connect.side_effect = connect
            server.cleanup.side_effect = cleanup
            servers.append(server)

        manager = agents.mcp.MCPServerManager(
            servers,
            strict=True,
            connect_in_parallel=True,
        )
        with asyncio.Runner() as runner:
            runner.run(manager.connect_all())
            runner.run(asyncio.sleep(0))
            runner.run(asyncio.sleep(0))
            self.assertEqual(len(events), 2)
            runner.run(manager.cleanup_all())
        self.assertEqual(len(events), 4)
        for name in ("first", "second"):
            lifecycle = [event for event in events if event[0] == name]
            self.assertIs(lifecycle[0][2], lifecycle[1][2])


if __name__ == "__main__":
    unittest.main()
