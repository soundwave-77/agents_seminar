import contextlib
import os
import unittest
import unittest.mock

import agents
import agents.tracing
import agents.tracing.provider
import opentelemetry.sdk.trace
import opentelemetry.sdk.trace.export
import opentelemetry.sdk.trace.export.in_memory_span_exporter
import opentelemetry.trace

from travel_agent import tracing

LANGFUSE_ENV = {
    "LANGFUSE_SECRET_KEY": "test-secret",
    "LANGFUSE_PUBLIC_KEY": "test-public",
    "LANGFUSE_BASE_URL": "https://langfuse.example.test",
}


class AgentTracingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(
            unittest.mock.patch.dict(os.environ, LANGFUSE_ENV, clear=True),
        )
        self.client = self.stack.enter_context(
            unittest.mock.patch.object(tracing.langfuse, "get_client"),
        )
        self.nest = self.stack.enter_context(
            unittest.mock.patch.object(tracing.nest_asyncio, "apply"),
        )

    def test_missing_partial_or_blank_settings_disable_tracing(self) -> None:
        cases = (
            {},
            {"LANGFUSE_SECRET_KEY": "test-secret"},
            {**LANGFUSE_ENV, "LANGFUSE_PUBLIC_KEY": "   "},
        )
        for settings in cases:
            with (
                self.subTest(settings=list(settings)),
                unittest.mock.patch.dict(os.environ, settings, clear=True),
                unittest.mock.patch.object(
                    tracing.openai_agents,
                    "OpenAIAgentsInstrumentor",
                ) as instrumentor,
            ):
                self.assertFalse(tracing.setup_tracing())
                instrumentor.assert_not_called()
        self.client.assert_not_called()
        self.nest.assert_not_called()

    def test_initialization_happens_once_in_the_guide_order(
        self,
    ) -> None:
        with (
            unittest.mock.patch.object(
                tracing.openai_agents,
                "OpenAIAgentsInstrumentor",
            ) as factory,
        ):
            instrumentor = factory.return_value
            instrumentor.is_instrumented_by_opentelemetry = False
            instrumentor.instrument.side_effect = lambda: setattr(
                instrumentor,
                "is_instrumented_by_opentelemetry",
                True,
            )
            calls = unittest.mock.Mock()
            calls.attach_mock(self.nest, "apply")
            calls.attach_mock(self.client, "client")
            calls.attach_mock(instrumentor.instrument, "instrument")

            self.assertTrue(tracing.setup_tracing())
            self.assertTrue(tracing.setup_tracing())

        self.assertEqual(
            calls.mock_calls,
            [
                unittest.mock.call.apply(),
                unittest.mock.call.instrument(),
                unittest.mock.call.client(),
            ],
        )

    def test_native_agent_model_and_tool_spans_reach_the_otel_provider(
        self,
    ) -> None:
        previous = agents.tracing.get_trace_provider()
        sdk_provider = agents.tracing.provider.DefaultTraceProvider()
        agents.tracing.set_trace_provider(sdk_provider)
        self.addCleanup(agents.tracing.set_trace_provider, previous)
        previous_processor = unittest.mock.Mock(spec=agents.TracingProcessor)
        sdk_provider.register_processor(previous_processor)

        provider = opentelemetry.sdk.trace.TracerProvider(
            shutdown_on_exit=False,
        )
        exporter = opentelemetry.sdk.trace.export.in_memory_span_exporter.InMemorySpanExporter()
        provider.add_span_processor(
            opentelemetry.sdk.trace.export.SimpleSpanProcessor(exporter),
        )
        self.addCleanup(provider.shutdown)
        instrumentor = tracing.openai_agents.OpenAIAgentsInstrumentor()
        self.addCleanup(instrumentor.uninstrument)

        with unittest.mock.patch.object(
            opentelemetry.trace,
            "get_tracer_provider",
            return_value=provider,
        ):
            self.assertTrue(tracing.setup_tracing())
            with (
                agents.trace("offline-test", group_id="dialogue"),
                agents.agent_span(name="TravelAgent"),
            ):
                with agents.generation_span(model="test-model"):
                    pass
                with agents.function_span(name="get_current_date"):
                    pass

        spans = exporter.get_finished_spans()
        kinds = {span.attributes.get("openinference.span.kind") for span in spans}
        self.assertTrue({"AGENT", "LLM", "TOOL"}.issubset(kinds))
        root = next(span for span in spans if span.name == "offline-test")
        self.assertEqual(root.attributes["session.id"], "dialogue")
        previous_processor.on_trace_start.assert_not_called()
        previous_processor.on_span_start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
