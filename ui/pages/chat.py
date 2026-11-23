import asyncio
import logging
import time
from collections.abc import Iterator

import streamlit as st

from travel_agent import runner
from ui import agent_session

logger = logging.getLogger(__name__)


def _stream_answer(answer: str) -> Iterator[str]:
    """Yield the completed answer in small chunks for a smooth reveal."""
    for offset in range(0, len(answer), 3):
        yield answer[offset : offset + 3]
        time.sleep(0.04)


def main() -> None:
    """Forward chat messages to the travel agent and display its answers."""
    st.title("Помощник по путешествиям")
    agent_name = agent_session.select_agent("chat_agent_name")
    loop = asyncio.get_event_loop()
    try:
        with st.spinner("Подключаем инструменты…"):
            agent = loop.run_until_complete(agent_session.initialize(agent_name))
    except NotImplementedError:
        st.info(f"Агент {agent_name} пока не реализован.")
        return
    except Exception:
        logger.exception("Не удалось запустить агента")
        st.error("Не удалось запустить агента. Проверь подключение и логи.")
        return

    history_agent_name = st.session_state.get(
        "chat_history_agent_name",
        agent_session.DEFAULT_AGENT_NAME,
    )
    if "session" not in st.session_state or history_agent_name != agent_name:
        agent_session.new_dialogue()
    st.session_state.chat_history_agent_name = agent_name

    messages = st.container()
    for message in st.session_state.messages:
        with messages.chat_message(message["role"]):
            st.markdown(message["content"])

    st.html(
        "<style>.st-key-new_dialogue button { height: calc(3.5rem + 2px); }</style>"
    )
    input_column, button_column = st.bottom.columns(
        [4, 1.2],
        vertical_alignment="bottom",
    )
    with input_column:
        prompt = st.chat_input("Напиши запрос")
    with button_column:
        st.button(
            "Новый диалог",
            key="new_dialogue",
            type="primary",
            on_click=agent_session.new_dialogue,
            width="stretch",
        )
    if not prompt:
        return
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    with messages.chat_message("user"):
        st.markdown(prompt)

    with messages.chat_message("assistant"):
        try:
            with st.spinner("Агент отвечает…"):
                answer = loop.run_until_complete(
                    runner.run_agent(
                        agent,
                        prompt,
                        session=st.session_state.session,
                        max_turns=None,
                        tracing_disabled=not st.session_state.tracing_enabled,
                    ),
                )
        except Exception:
            logger.exception("Не удалось выполнить запрос агента")
            answer = "Не удалось завершить запрос. Попробуй ещё раз."
            st.markdown(answer)
        else:
            st.write_stream(_stream_answer(answer), cursor="▌")
    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
