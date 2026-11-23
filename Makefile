.PHONY: test run lint format

test:
	uv run python -m pytest -v tests

run:
	uv run --env-file .env streamlit run app.py

lint:
	uv run ruff check .

format:
	uv run ruff format .
