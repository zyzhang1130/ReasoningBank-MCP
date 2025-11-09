"""Mind2Web dataset loader tests."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

from src.mind2web.dataset import Mind2WebDataset


def _write_sample_json(path: Path, *, pos_rank: int, neg_rank: int):
    payload = [
        {
            "task_id": "task_1",
            "confirmed_task": "Click the login button",
            "website": "example.com",
            "actions": [
                {
                    "action_type": "CLICK",
                    "cleaned_html": "<button id='login'>Login</button>",
                    "pos_candidates": [
                        {"backend_node_id": 101, "repr": "<button id='login'>Login</button>"}
                    ],
                    "neg_candidates": [
                        {"backend_node_id": 102, "repr": "<button id='signup'>Signup</button>"}
                    ],
                }
            ],
        }
    ]

    scores = {
        "task_1:0": {
            "101": {"score": 0.95, "rank": pos_rank},
            "102": {"score": 0.25, "rank": neg_rank},
        }
    }

    data_root = path / "test_task.json"
    data_root.write_text(json.dumps(payload), encoding="utf-8")

    score_path = path / "scores.pkl"
    with score_path.open("wb") as handle:
        pickle.dump(scores, handle)

    return data_root.parent, score_path


def test_dataset_keeps_positive_after_topk(tmp_path):
    root, scores = _write_sample_json(tmp_path, pos_rank=0, neg_rank=5)

    dataset = Mind2WebDataset(
        data_root=root,
        split="test_task",
        top_k=1,
        scores_path=scores,
    )

    assert len(dataset.samples) == 1
    sample = dataset.samples[0]
    assert sample.positive_ids == ["101"]
    assert sample.candidates[0].rank == 0


def test_dataset_skips_when_positive_pruned(tmp_path):
    root, scores = _write_sample_json(tmp_path, pos_rank=10, neg_rank=0)

    dataset = Mind2WebDataset(
        data_root=root,
        split="test_task",
        top_k=1,
        scores_path=scores,
    )

    assert len(dataset.samples) == 0
    assert len(dataset.skipped_samples) == 1
