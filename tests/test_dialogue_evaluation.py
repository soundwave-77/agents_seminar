import json
import os
from unittest import mock

import httpx
import openai
import pytest
from deepeval import metrics
from deepeval.test_case import ConversationalTestCase, Turn
from streamlit.testing.v1 import AppTest

from evaluation import dialogues


@pytest.fixture
def judge_client(monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL", "test-judge")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    send = mock.Mock()
    with (
        openai.OpenAI(
            http_client=httpx.Client(transport=httpx.MockTransport(send)),
        ) as client,
        mock.patch.object(
            dialogues.models.LocalModel, "load_model", return_value=client
        ),
    ):
        yield send


def response(score, reason="Обоснование"):
    return httpx.Response(
        200,
        json={
            "id": "test-completion",
            "object": "chat.completion",
            "created": 0,
            "model": "test-judge",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"score": score, "reason": reason}),
                    },
                }
            ],
        },
    )


def conversation(name="dialogue-1"):
    return ConversationalTestCase(
        name=name,
        scenario="Найти жильё на двоих",
        turns=[
            Turn(role="user", content="Нужен отель в Риме до 300 евро."),
            Turn(role="assistant", content="На какие даты?"),
            Turn(role="user", content="На 20–22 ноября."),
            Turn(role="assistant", content="Доступных вариантов не нашёл."),
        ],
    )


@pytest.mark.parametrize(
    ("score", "expected"), [(1, 0), (2, 0.25), (3, 0.5), (4, 0.75), (5, 1)]
)
def test_native_geval_normalizes_the_rubric(judge_client, score, expected):
    judge_client.return_value = response(score)
    with mock.patch.object(dialogues.metrics, "GEval", wraps=metrics.GEval) as factory:
        result = dialogues.evaluate_dialogues([conversation()])[0]
    assert factory.call_count == 3
    for call in factory.call_args_list:
        rubric = call.kwargs["rubric"]
        assert [item.score_range for item in rubric] == [(i, i) for i in range(1, 6)]
        assert all(item.expected_outcome for item in rubric)
    for name in ("dialogue_quality", "response_format", "task_completion"):
        assert result[name] == expected
        assert result["reasons"][name] == "Обоснование"
    assert judge_client.call_count == 3
    for call in judge_client.call_args_list:
        request = call.args[0]
        body = json.loads(request.content)
        assert str(request.url) == "https://example.test/v1/chat/completions"
        assert body["model"] == "test-judge"
        assert body["temperature"] == 0
        assert body["reasoning_effort"] == "medium"
        response_format = body["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        schema = response_format["json_schema"]["schema"]
        assert set(schema["required"]) == {"score", "reason"}
        assert schema["additionalProperties"] is False
        assert schema["properties"]["score"]["type"] == "integer"
        assert schema["properties"]["score"]["minimum"] == 1
        assert schema["properties"]["score"]["maximum"] == 5
        prompt = body["messages"][0]["content"]
        assert conversation().scenario in prompt
        for turn in conversation().turns:
            assert turn.content in prompt


def test_results_stay_with_their_dialogue_and_progress_counts_checks(judge_client):
    judge_client.side_effect = [
        response(score, f"Evidence {i}") for i, score in enumerate([5, 4, 3, 2, 1, 2])
    ]
    progress = mock.Mock()
    results = dialogues.evaluate_dialogues(
        [conversation(), conversation("dialogue-2")], progress
    )
    assert [row["id"] for row in results] == ["dialogue-1", "dialogue-2"]
    assert [row["dialogue_quality"] for row in results] == [1, 0.75]
    assert [row["response_format"] for row in results] == [0.5, 0.25]
    assert [row["task_completion"] for row in results] == [0, 0.25]
    assert results[1]["reasons"]["response_format"] == "Evidence 3"
    assert progress.call_args_list == [mock.call(i, 6) for i in range(1, 7)]


@pytest.mark.parametrize(
    ("score", "reason"), [(0, "Wrong"), (6, "Wrong"), (3, ""), (3, " ")]
)
def test_invalid_metric_is_not_added_to_results(judge_client, score, reason):
    judge_client.return_value = response(score, reason)
    progress = mock.Mock()
    with pytest.raises(ValueError, match="dialogue_quality, диалог dialogue-1"):
        dialogues.evaluate_dialogues([conversation()], progress)
    progress.assert_not_called()


def test_unsupported_structured_output_does_not_fall_back(judge_client):
    judge_client.return_value = httpx.Response(
        400,
        json={
            "error": {
                "message": "json_schema is not supported",
                "type": "invalid_request_error",
            }
        },
    )
    with pytest.raises(openai.BadRequestError, match="json_schema is not supported"):
        dialogues.evaluate_dialogues([conversation()])
    assert judge_client.call_count == 1


def test_empty_dataset_needs_no_model_configuration():
    with mock.patch.dict(os.environ, {}, clear=True):
        assert dialogues.evaluate_dialogues([]) == []


def test_page_displays_normalized_means_counts_and_reasons(judge_client):
    judge_client.side_effect = [response(score) for score in [1, 5, 2, 4, 3, 5]]
    client = mock.Mock()
    client.get_dataset.return_value.items = [
        mock.Mock(
            id=f"dialogue-{i}",
            status="ACTIVE",
            metadata={"job_text": "Найти жильё"},
            input=conversation().turns,
        )
        for i in range(2)
    ] + [mock.Mock(status="ARCHIVED")]
    with mock.patch("langfuse.get_client", return_value=client):
        app = AppTest.from_file("../ui/pages/dialogue_evaluation.py").run()
        app.text_input[0].set_value("test-dataset").run()
        app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert len(app.dataframe) == 2
    summary = app.dataframe[0].value
    assert list(summary["Средняя оценка"]) == [0.5, 0.5, 0.75]
    assert list(summary["Оценено"]) == [2, 2, 2]
    assert len(app.expander) == 2
    assert "Обоснование" in [m.value for m in app.markdown]
    assert judge_client.call_count == 6


def test_page_does_not_present_old_scores_on_the_new_scale():
    app = AppTest.from_file("../ui/pages/dialogue_evaluation.py")
    app.session_state["dialogue_evaluation_result"] = {
        "dataset_name": "old-dataset",
        "scores": [{"id": "old-dialogue", "task_completion": 5, "reasons": {}}],
    }
    app.run()
    assert not app.exception
    assert not app.dataframe
    assert "прежней шкале" in app.info[0].value
