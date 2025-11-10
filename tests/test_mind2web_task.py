"""Tests for Mind2WebTask prompt/prediction flow."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from src.llm.base import LLMProvider
from src.mind2web.dataset import Mind2WebDataset
from src.mind2web.prompting import Mind2WebPromptBuilder
from src.mind2web.task import Mind2WebTask, parse_action_output


class DummyLLM(LLMProvider):
    def __init__(self, responses: list[str]):
        self._responses = responses

    async def chat(self, messages, temperature=0.0, max_tokens=512, **kwargs):
        if not self._responses:
            raise RuntimeError("No more dummy responses")
        return self._responses.pop(0)

    def get_provider_name(self) -> str:
        return "dummy-llm"


def _build_dataset(tmp_path: Path) -> Mind2WebDataset:
    payload = [
        {
            "task_id": "task_flow",
            "confirmed_task": "Click the login button",
            "website": "example.com",
            "actions": [
                {
                    "action_type": "CLICK",
                    "cleaned_html": "<button id='login'>Login</button>",
                    "history": [],
                    "pos_candidates": [
                        {"backend_node_id": 1001, "repr": "<button id='login'>Login</button>"}
                    ],
                    "neg_candidates": [
                        {"backend_node_id": 1002, "repr": "<button id='signup'>Signup</button>"}
                    ],
                }
            ],
        }
    ]
    data_root = tmp_path / "test_task.json"
    data_root.write_text(json.dumps(payload), encoding="utf-8")

    scores = {
        "task_flow:0": {
            "1001": {"score": 0.9, "rank": 0},
            "1002": {"score": 0.1, "rank": 1},
        }
    }
    scores_path = tmp_path / "scores.pkl"
    with scores_path.open("wb") as handle:
        pickle.dump(scores, handle)

    return Mind2WebDataset(
        data_root=tmp_path,
        split="test_task",
        top_k=5,
        scores_path=scores_path,
    )


@pytest.mark.asyncio
async def test_mind2web_task_runs_end_to_end(tmp_path):
    dataset = _build_dataset(tmp_path)
    llm = DummyLLM(["Answer: B.\nAction: CLICK.\nValue: N/A."])
    prompt_builder = Mind2WebPromptBuilder()

    task = Mind2WebTask(
        dataset=dataset,
        llm=llm,
        prompt_builder=prompt_builder,
        temperature=0.0,
        max_tokens=64,
        log_every=0,
    )

    result = await task.evaluate()

    assert result.metrics["step_micro"] == 1.0
    assert result.metrics["operation_micro"] == 1.0
    assert result.metrics["task_success"] == 1.0
    assert len(result.predictions) == 1
    assert result.predictions[0].candidate_id == "1001"


def test_parse_action_output_handles_aliases():
    output = "Answer: AA.\nAction: press button.\nValue: \"Seattle\""
    choice, action, value = parse_action_output(output)
    assert choice == "AA"
    assert action == "CLICK"
    assert value == "Seattle"


@pytest.mark.asyncio
async def test_operation_f1_requires_matching_value(tmp_path):
    payload = [
        {
            "task_id": "task_type",
            "confirmed_task": "Type the city",
            "website": "example.com",
            "actions": [
                {
                    "action_type": "TYPE",
                    "value": "Seattle",
                    "cleaned_html": "<input id='city'>",
                    "pos_candidates": [{"backend_node_id": 2001, "repr": "<input id='city'>"}],
                    "neg_candidates": [{"backend_node_id": 2002, "repr": "<input id='state'>"}],
                }
            ],
        }
    ]
    data_root = tmp_path / "type_task.json"
    data_root.write_text(json.dumps(payload), encoding="utf-8")

    dataset = Mind2WebDataset(data_root=data_root, split="type_task", top_k=5)
    llm = DummyLLM(["Answer: B.\nAction: TYPE.\nValue: Seattle"])
    prompt_builder = Mind2WebPromptBuilder()

    task = Mind2WebTask(
        dataset=dataset,
        llm=llm,
        prompt_builder=prompt_builder,
        temperature=0.0,
        max_tokens=64,
        log_every=0,
    )

    result = await task.evaluate()
    assert result.metrics["operation_micro"] == 1.0
    assert result.predictions[0].operation_f1 == 1.0
