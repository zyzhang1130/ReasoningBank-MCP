#!/usr/bin/env python3
"""CLI helper to evaluate LLM agents on Mind2Web using ReasoningBank utilities."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mind2web.task import evaluate_from_cli, save_eval_output


DEFAULT_CONFIG = "config.yaml"
DEFAULT_DATA_ROOT = "/home/zy1130/datasets/Mind2Web/test_task"
DEFAULT_SCORES = "/home/zy1130/datasets/Mind2Web/scores_all_data.pkl"
DEFAULT_OUTPUT_DIR = "outputs/mind2web_test_task"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model on Mind2Web using ReasoningBank.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to ReasoningBank config file.")
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="Directory containing Mind2Web JSON files.",
    )
    parser.add_argument("--split", default="test_task", help="Mind2Web split (e.g., test_task, test_website).")
    parser.add_argument(
        "--scores-pickle",
        default=DEFAULT_SCORES,
        help="Optional scores_all_data.pkl path for candidate ranks.",
    )
    parser.add_argument("--top-k", type=int, default=50, help="Top-K candidates to keep per step.")
    parser.add_argument("--prompt-template", help="Custom prompt template JSON (defaults to bundled file).")
    parser.add_argument("--max-examples", type=int, help="Only evaluate the first N samples.")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature (paper uses 0).")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max generation tokens.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save metrics/predictions JSON files.",
    )
    return parser.parse_args()


async def _async_main(args: argparse.Namespace):
    result = await evaluate_from_cli(
        config_path=args.config,
        data_root=args.data_root,
        split=args.split,
        scores_path=args.scores_pickle,
        top_k=args.top_k,
        prompt_template=args.prompt_template,
        max_examples=args.max_examples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print("=== Mind2Web Evaluation Summary ===")
    for key, value in result.metrics.items():
        print(f"{key:>15}: {value:.4f}")
    for key, value in result.totals.items():
        print(f"{key:>15}: {value}")

    if args.output_dir:
        save_eval_output(result, Path(args.output_dir))


def main():
    args = parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
