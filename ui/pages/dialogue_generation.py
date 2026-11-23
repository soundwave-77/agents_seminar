import asyncio
import logging
import os
import uuid
from asyncio import Task
from typing import Any

import agents
import coolname
import deepeval.dataset
import deepeval.models
import deepeval.simulator
import langfuse
import pandas as pd
import streamlit as st
import tenacity
from agents import SQLiteSession
from deepeval.test_case import ConversationalTestCase, Turn

from travel_agent import runner
from ui import agent_session

logger = logging.getLogger(__name__)

RETRY_REQUEST = tenacity.retry(
    stop=tenacity.stop_never,
    retry=tenacity.retry_if_not_exception_type(NotImplementedError),
    wait=tenacity.wait_exponential_jitter(multiplier=2, max=60),
    before_sleep=tenacity.before_sleep_log(
        logger,
        logging.WARNING,
    ),
)


def generate_dialogues(
    jobs: list[dict[str, str]],
    dataset_name: str,
    *,
    agent_name: str = agent_session.DEFAULT_AGENT_NAME,
) -> None:
    """Generate all scenarios concurrently and save completed dialogues."""
    loop = asyncio.get_event_loop()
    try:
        agent = loop.run_until_complete(
            RETRY_REQUEST(agent_session.initialize)(agent_name),
        )
    except NotImplementedError:
        st.info(f"Агент {agent_name} пока не реализован.")
        return
    if not st.session_state.tracing_enabled:
        st.error("Для сохранения диалогов настрой Langfuse в .env.")
        return

    client = langfuse.get_client()
    RETRY_REQUEST(client.create_dataset)(name=dataset_name)
    st.session_state.dialogue_dataset_name = dataset_name
    st.session_state.dialogue_dataset_agent_name = agent_name
    st.session_state.dialogue_dataset_count = 0
    st.session_state.dialogue_dataset_total = len(jobs)
    progress = st.progress(0, text=f"Сгенерировано 0 из {len(jobs)} диалогов")
    sessions: dict[str, SQLiteSession] = {}
    saves: list[Task[None]] = []

    @RETRY_REQUEST
    async def reply(input: str, thread_id: str) -> Turn:
        if thread_id not in sessions:
            sessions[thread_id] = agents.SQLiteSession(thread_id)
        session = sessions[thread_id]
        history = await session.get_items()
        try:
            answer = await runner.run_agent(
                agent,
                input,
                session=session,
            )
        except Exception:
            await session.clear_session()
            await session.add_items(history)
            raise
        return Turn(
            role="assistant",
            content=answer,
            metadata={"session_id": thread_id},
        )

    @RETRY_REQUEST
    async def save_dialogue(dialogue: dict[str, Any]) -> None:
        await client.async_api.dataset_items.create(
            dataset_name=dataset_name,
            id=dialogue["session_id"],
            input=dialogue["dialogue"],
            metadata={
                "job_id": dialogue["job_id"],
                "job_text": dialogue["job_text"],
                "session_id": dialogue["session_id"],
            },
            source_trace_id=dialogue["source_trace_id"],
            source_observation_id=dialogue["source_observation_id"],
        )
        st.session_state.dialogue_dataset_count += 1

    def on_dialogue_complete(
        conversation: ConversationalTestCase,
        index: int,
    ) -> None:
        session_id = str(uuid.uuid4())
        for turn in conversation.turns:
            if turn.role == "assistant" and turn.metadata:
                session_id = turn.metadata["session_id"]
                break
        dialogue = {
            "job_id": jobs[index]["job_id"],
            "job_text": jobs[index]["job_text"],
            "session_id": session_id,
            "dialogue": [
                turn.model_dump(
                    mode="json",
                    exclude_none=True,
                    exclude={"metadata"},
                )
                for turn in conversation.turns
            ],
        }
        with (
            langfuse.propagate_attributes(session_id=session_id),
            client.start_as_current_observation(
                name="generated_dialogue",
                as_type="span",
                input={"job_text": dialogue["job_text"]},
                output=dialogue["dialogue"],
                metadata={
                    "job_id": dialogue["job_id"],
                    "session_id": session_id,
                },
            ) as observation,
        ):
            dialogue["source_trace_id"] = observation.trace_id
            dialogue["source_observation_id"] = observation.id
        saves.append(loop.create_task(save_dialogue(dialogue)))
        progress.progress(
            len(saves) / len(jobs),
            text=f"Сгенерировано {len(saves)} из {len(jobs)} диалогов",
        )

    simulator = deepeval.simulator.ConversationSimulator(
        model_callback=reply,
        simulator_model=deepeval.models.LocalModel(
            model=os.environ["SYNTHETIC_USER_MODEL"],
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL"),
            generation_kwargs={"reasoning_effort": "none"},
        ),
        async_mode=True,
        max_concurrent=len(jobs),
        language="Russian",
    )
    simulator.a_generate_schema = RETRY_REQUEST(simulator.a_generate_schema)
    try:
        simulator.simulate(
            conversational_goldens=[
                deepeval.dataset.ConversationalGolden(scenario=job["job_text"])
                for job in jobs
            ],
            max_user_simulations=10,
            on_simulation_complete=on_dialogue_complete,
        )
    finally:
        for session in sessions.values():
            session.close()
        try:
            if saves:
                loop.run_until_complete(asyncio.gather(*saves))
        finally:
            client.flush()


def main() -> None:
    """Generate conversations from JTBD and save them directly to Langfuse."""
    st.title("Генерация диалогов по JTBD")
    agent_name = agent_session.select_agent("generation_agent_name")
    uploaded = st.file_uploader(
        "CSV с колонками job_id и job_text",
        type="csv",
    )
    if uploaded is not None:
        if "dialogue_dataset_input" not in st.session_state:
            st.session_state.dialogue_dataset_input = coolname.generate_slug(2)
        dataset_name = st.text_input(
            "Название датасета в Langfuse",
            key="dialogue_dataset_input",
        ).strip()
        if st.button("Сгенерировать диалоги", disabled=not dataset_name):
            jobs = pd.read_csv(uploaded, dtype=str, keep_default_na=False)
            if not {"job_id", "job_text"}.issubset(jobs.columns):
                st.error("В CSV нужны колонки job_id и job_text.")
                return
            if jobs.empty:
                st.error("В CSV нужны строки с JTBD.")
                return
            try:
                with st.spinner("Генерируем и сохраняем диалоги в Langfuse…"):
                    generate_dialogues(
                        jobs.to_dict("records"),
                        dataset_name,
                        agent_name=agent_name,
                    )
            except Exception:
                logger.exception("Не удалось завершить генерацию и сохранение")
                st.error("Генерация остановлена. Проверь подключение и логи.")

    if "dialogue_dataset_count" in st.session_state:
        dataset_agent_name = st.session_state.get(
            "dialogue_dataset_agent_name",
            agent_session.DEFAULT_AGENT_NAME,
        )
        st.caption(
            f"Датасет Langfuse — {st.session_state.dialogue_dataset_name}. "
            f"Агент — {dataset_agent_name}.",
        )
        st.write(
            f"Сохранено диалогов — {st.session_state.dialogue_dataset_count} "
            f"из {st.session_state.dialogue_dataset_total}",
        )


if __name__ == "__main__":
    main()
