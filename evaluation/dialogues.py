import json
import os
import pathlib
from collections.abc import Callable

import yaml
from deepeval import metrics, models
from deepeval.metrics.g_eval import Rubric
from deepeval.metrics.g_eval.schema import ReasonScore
from deepeval.test_case import ConversationalTestCase, LLMTestCase, SingleTurnParams


def evaluate_dialogues(
    dataset: list[ConversationalTestCase],
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, object]]:
    """Return GEval scores in 0–1 and reasons for each full dialogue."""
    if not dataset:
        return []
    schema = ReasonScore.model_json_schema()
    schema["additionalProperties"] = False
    schema["properties"]["score"].update(type="integer", minimum=1, maximum=5)
    schema["properties"]["reason"]["minLength"] = 1
    model = models.LocalModel(
        model=os.environ["JUDGE_MODEL"],
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
        temperature=0,
        generation_kwargs={
            "reasoning_effort": "medium",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ReasonScore",
                    "strict": True,
                    "schema": schema,
                },
            },
        },
    )
    test_cases = [
        LLMTestCase(
            name=dialogue.name,
            input=dialogue.scenario or "",
            actual_output=json.dumps(
                {
                    "scenario": dialogue.scenario,
                    "turns": [
                        {"turn": index, "role": turn.role, "content": turn.content}
                        for index, turn in enumerate(dialogue.turns, start=1)
                    ],
                },
                ensure_ascii=False,
            ),
        )
        for dialogue in dataset
    ]
    results = [{"id": dialogue.name, "reasons": {}} for dialogue in dataset]
    config_files = sorted((pathlib.Path(__file__).parent / "judges").glob("*.yaml"))
    total = len(dataset) * len(config_files)
    completed = 0
    for config_file in config_files:
        config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        config["rubric"] = [Rubric(**item) for item in config["rubric"]]
        judge = metrics.GEval(
            **config,
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
            model=model,
            threshold=None,
            async_mode=False,
        )
        for test_case, result in zip(test_cases, results):
            score = judge.measure(test_case, _show_indicator=False)
            if not 0 <= score <= 1 or not judge.reason or not judge.reason.strip():
                raise ValueError(
                    f"Судья {judge.name}, диалог {test_case.name}: "
                    "нужна оценка 0–1 с непустым обоснованием."
                )
            result[judge.name] = score
            result["reasons"][judge.name] = judge.reason
            completed += 1
            if on_progress is not None:
                on_progress(completed, total)
    return results
