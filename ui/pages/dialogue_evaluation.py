import logging

import langfuse
import pandas as pd
import streamlit as st
from deepeval import test_case

from evaluation import dialogues

logger = logging.getLogger(__name__)


def main() -> None:
    """Load a Langfuse dataset and display local scores and their means."""
    st.title("Оценка качества")
    dataset_name = st.text_input(
        "Название датасета в Langfuse",
        value=st.session_state.get("dialogue_dataset_name", ""),
    ).strip()
    if st.button("Оценить диалоги", disabled=not dataset_name):
        try:
            with st.spinner("Загружаем датасет и оцениваем диалоги…"):
                client = langfuse.get_client()
                dataset = [
                    test_case.ConversationalTestCase(
                        name=item.id,
                        scenario=item.metadata["job_text"],
                        turns=item.input,
                    )
                    for item in client.get_dataset(dataset_name).items
                    if item.status == "ACTIVE"
                ]
                if not dataset:
                    raise ValueError("В датасете нет активных диалогов")
                progress = st.progress(0, text="Оцениваем диалоги…")

                def on_progress(completed: int, total: int) -> None:
                    progress.progress(
                        completed / total,
                        text=f"Выполнено {completed} из {total} оценок",
                    )

                scores = dialogues.evaluate_dialogues(
                    dataset,
                    on_progress=on_progress,
                )
                st.session_state.dialogue_evaluation_result = {
                    "dataset_name": dataset_name,
                    "scores": scores,
                    "score_scale": "0-1",
                }
        except Exception as error:
            logger.exception("Не удалось оценить диалоги")
            st.error(f"Не удалось оценить диалоги — {error}")

    if "dialogue_evaluation_result" in st.session_state:
        result = st.session_state.dialogue_evaluation_result
        if result.get("score_scale") != "0-1":
            st.info("Сохранены оценки по прежней шкале. Оцените диалоги заново.")
            return
        scores = pd.DataFrame(result["scores"]).set_index("id")
        reasons = scores.pop("reasons")
        summary = scores.agg(["mean", "count"]).T
        st.caption(
            f"Датасет — {result['dataset_name']}. Диалогов — {len(scores)}. "
            "Оценки GEval от 0 до 1: чем выше, тем лучше. "
            "Баллы судьи 1, 2, 3, 4, 5 соответствуют "
            "метрикам 0, 0.25, 0.5, 0.75, 1.",
        )
        st.dataframe(
            summary.rename(
                columns={
                    "mean": "Средняя оценка",
                    "count": "Оценено",
                }
            ),
            width="stretch",
        )
        st.dataframe(scores, width="stretch")
        for dialogue_id, judge_reasons in reasons.items():
            with st.expander(f"Обоснования — {dialogue_id}"):
                for judge_name, reason in judge_reasons.items():
                    st.markdown(
                        f"**{judge_name}: {scores.loc[dialogue_id, judge_name]:.2f}**"
                    )
                    st.write(reason)


if __name__ == "__main__":
    main()
