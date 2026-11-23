import asyncio
import contextlib
import io
import os
import types
import unittest
import unittest.mock

import tenacity
from agents.run import DEFAULT_MAX_TURNS

from ui.pages import dialogue_generation


class DialogueGenerationTests(unittest.TestCase):
    def test_all_30_dialogues_overlap_and_keep_separate_histories(self) -> None:
        jobs = [f"scenario-{index:02d}" for index in range(30)]
        sessions = {}
        failed = set()
        assistant = object()
        saved = []
        observations = {}
        save_retried = set()
        other_saved = asyncio.Event()
        ui = unittest.mock.MagicMock()
        ui.file_uploader.return_value = io.StringIO(
            "job_id,job_text\n"
            + "\n".join(f"{index:03d},{job}" for index, job in enumerate(jobs))
        )
        ui.button.return_value = True
        ui.text_input.return_value = "  dialogues/test-dataset  "
        ui.session_state.tracing_enabled = True
        client = unittest.mock.Mock()

        def start_observation(**kwargs):
            session_id = kwargs["metadata"]["session_id"]
            self.assertNotIn(session_id, observations)
            observation = types.SimpleNamespace(
                trace_id=f"trace-{session_id}",
                id=f"observation-{session_id}",
            )
            observations[session_id] = {**kwargs, "observation": observation}
            return contextlib.nullcontext(observation)

        client.start_as_current_observation = unittest.mock.Mock(
            side_effect=start_observation,
        )

        async def synthetic_user(model, prompt, schema, **kwargs):
            job = next(job for job in jobs if job in prompt)
            await asyncio.sleep(0)
            return schema(simulated_input=job), 0.0

        async def save_item(
            *,
            dataset_name,
            id,
            input,
            metadata,
            source_trace_id,
            source_observation_id,
        ):
            self.assertEqual(dataset_name, "dialogues/test-dataset")
            observation = observations[id]["observation"]
            self.assertEqual(source_trace_id, observation.trace_id)
            self.assertEqual(source_observation_id, observation.id)
            if metadata["job_id"] == "000":
                await asyncio.wait_for(other_saved.wait(), timeout=5)
            else:
                other_saved.set()
            if metadata["job_id"] == "001" and id not in save_retried:
                save_retried.add(id)
                raise RuntimeError("Temporary dataset failure")
            self.assertEqual(id, metadata["session_id"])
            saved.append({"input": input, "metadata": metadata})

        client.async_api.dataset_items.create = unittest.mock.AsyncMock(
            side_effect=save_item,
        )
        with asyncio.Runner() as runner, contextlib.ExitStack() as stack:
            loop = runner.get_loop()
            all_started = asyncio.Event()

            async def respond(agent, input, *, session, max_turns, run_config):
                self.assertIs(agent, assistant)
                self.assertEqual(max_turns, DEFAULT_MAX_TURNS)
                self.assertEqual(run_config.group_id, session.session_id)
                self.assertEqual(session.db_path, ":memory:")
                history = await session.get_items()
                self.assertTrue(all(item["content"] == input for item in history))
                sessions[input] = session
                if len(sessions) == len(jobs):
                    all_started.set()
                await asyncio.wait_for(all_started.wait(), timeout=5)
                if input == jobs[0] and input not in failed:
                    failed.add(input)
                    await session.add_items(
                        [
                            {"role": "user", "content": "failed attempt"},
                        ]
                    )
                    raise RuntimeError("Temporary model failure")
                await session.add_items(
                    [
                        {"role": "user", "content": input},
                        {"role": "assistant", "content": input},
                    ]
                )
                return types.SimpleNamespace(final_output=input)

            stack.enter_context(
                unittest.mock.patch.dict(
                    os.environ,
                    {
                        "SYNTHETIC_USER_MODEL": "test-user",
                        "OPENAI_API_KEY": "test-key",
                        "OPENAI_BASE_URL": "https://example.test/v1",
                    },
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    dialogue_generation,
                    "st",
                    ui,
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    dialogue_generation.agent_session,
                    "select_agent",
                    return_value="Multi-agent",
                )
            )
            initialize = stack.enter_context(
                unittest.mock.patch.object(
                    dialogue_generation.agent_session,
                    "initialize",
                    return_value=assistant,
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    dialogue_generation.langfuse,
                    "get_client",
                    return_value=client,
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    dialogue_generation.asyncio,
                    "get_event_loop",
                    return_value=loop,
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    dialogue_generation.agents.Runner,
                    "run",
                    side_effect=respond,
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    dialogue_generation.deepeval.models.LocalModel,
                    "a_generate",
                    synthetic_user,
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    dialogue_generation,
                    "RETRY_REQUEST",
                    tenacity.retry(stop=tenacity.stop_after_attempt(2)),
                )
            )
            dialogue_generation.main()
            for session in sessions.values():
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    runner.run(session.get_items())

        ui.error.assert_not_called()
        initialize.assert_awaited_once_with("Multi-agent")
        self.assertEqual(ui.session_state.dialogue_dataset_agent_name, "Multi-agent")
        client.create_dataset.assert_called_once_with(
            name="dialogues/test-dataset",
        )
        self.assertEqual(len(sessions), 30)
        session_ids = {session.session_id for session in sessions.values()}
        self.assertEqual(len(session_ids), 30)
        self.assertEqual(len(saved), 30)
        self.assertEqual(client.start_as_current_observation.call_count, 30)
        client.flush.assert_called_once()
        self.assertEqual(len(save_retried), 1)
        self.assertEqual(ui.session_state.dialogue_dataset_count, 30)
        self.assertEqual(failed, {jobs[0]})
        for result in saved:
            job = result["metadata"]["job_text"]
            observation = observations[result["metadata"]["session_id"]]
            self.assertEqual(observation["name"], "generated_dialogue")
            self.assertEqual(observation["as_type"], "span")
            self.assertEqual(observation["input"], {"job_text": job})
            self.assertEqual(observation["output"], result["input"])
            self.assertEqual(
                result["metadata"],
                {
                    "job_id": f"{jobs.index(job):03d}",
                    "job_text": job,
                    "session_id": sessions[job].session_id,
                },
            )
            self.assertEqual(len(result["input"]), 20)
            self.assertEqual(
                [turn["role"] for turn in result["input"]],
                ["user", "assistant"] * 10,
            )
            self.assertTrue(all(turn["content"] == job for turn in result["input"]))

    def test_completed_dialogue_is_saved_even_if_generation_later_fails(self):
        ui = unittest.mock.MagicMock()
        ui.session_state.__contains__.return_value = True
        ui.session_state.tracing_enabled = True
        client = unittest.mock.Mock()
        client.async_api.dataset_items.create = unittest.mock.AsyncMock()
        client.start_as_current_observation = unittest.mock.Mock(
            return_value=contextlib.nullcontext(
                types.SimpleNamespace(
                    trace_id="trace-completed",
                    id="observation-completed",
                )
            ),
        )
        completed = dialogue_generation.deepeval.test_case.ConversationalTestCase(
            turns=[
                dialogue_generation.deepeval.test_case.Turn(
                    role="user",
                    content="First",
                ),
                dialogue_generation.deepeval.test_case.Turn(
                    role="assistant",
                    content="Done",
                    metadata={"session_id": "completed"},
                ),
            ],
        )

        def simulate(*, on_simulation_complete, **kwargs):
            on_simulation_complete(completed, 0)
            raise RuntimeError("Generation failed after one conversation")

        simulator = unittest.mock.Mock()
        simulator.simulate.side_effect = simulate

        with (
            asyncio.Runner() as runner,
            unittest.mock.patch.object(dialogue_generation, "st", ui),
            unittest.mock.patch.object(dialogue_generation.agent_session, "initialize"),
            unittest.mock.patch.object(
                dialogue_generation.asyncio,
                "get_event_loop",
                return_value=runner.get_loop(),
            ),
            unittest.mock.patch.object(
                dialogue_generation.langfuse,
                "get_client",
                return_value=client,
            ),
            unittest.mock.patch.object(
                dialogue_generation.deepeval.simulator,
                "ConversationSimulator",
                return_value=simulator,
            ),
            unittest.mock.patch.object(
                dialogue_generation.deepeval.models,
                "LocalModel",
            ),
            self.assertRaisesRegex(RuntimeError, "Generation failed"),
        ):
            dialogue_generation.generate_dialogues(
                [
                    {"job_id": "1", "job_text": "First"},
                    {"job_id": "2", "job_text": "Second"},
                ],
                "dialogues/partial",
            )

        client.async_api.dataset_items.create.assert_awaited_once_with(
            dataset_name=ui.session_state.dialogue_dataset_name,
            id="completed",
            input=[
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Done"},
            ],
            metadata={
                "job_id": "1",
                "job_text": "First",
                "session_id": "completed",
            },
            source_trace_id="trace-completed",
            source_observation_id="observation-completed",
        )
        client.start_as_current_observation.assert_called_once()
        client.flush.assert_called_once()
        self.assertEqual(ui.session_state.dialogue_dataset_count, 1)


if __name__ == "__main__":
    unittest.main()
