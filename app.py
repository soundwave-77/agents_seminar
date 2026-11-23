import streamlit as st


def main() -> None:
    """Run the selected application page."""
    page = st.navigation(
        [
            st.Page("ui/pages/chat.py", title="Чат", default=True),
            st.Page("ui/pages/dialogue_generation.py", title="Генерация диалогов"),
            st.Page("ui/pages/dialogue_evaluation.py", title="Оценка качества"),
        ]
    )
    page.run()


if __name__ == "__main__":
    main()
