"""Mind2Web task orchestration, prompting, and evaluation."""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from ..llm import LLMFactory
from ..llm.base import LLMProvider
from ..config import Config
from .dataset import Mind2WebDataset, Mind2WebSample, CandidateElement
from .prompting import Mind2WebPromptBuilder

logger = logging.getLogger(__name__)

ANSWER_RE = re.compile(r"Answer\s*[:\-]\s*([A-Z]+)", re.IGNORECASE)
ACTION_RE = re.compile(r"Action\s*[:\-]\s*([A-Z_ ]+)", re.IGNORECASE)
VALUE_RE = re.compile(r"Value\s*[:\-]\s*(.+)", re.IGNORECASE)
TOKEN_RE = re.compile(r"\w+")

ACTION_ALIASES = {
    "CLICK": "CLICK",
    "PRESS": "CLICK",
    "OPEN": "CLICK",
    "SELECT": "SELECT",
    "SELECTOPTION": "SELECT",
    "SELECTOPTIONAL": "SELECT",
    "CHOOSE": "SELECT",
    "TYPE": "TYPE",
    "INPUT": "TYPE",
    "ENTER": "TYPE",
}


@dataclass
class Mind2WebPrediction:
    sample_id: str
    task_id: str
    step_index: int
    choice: Optional[str]
    candidate_id: Optional[str]
    action: Optional[str]
    predicted_value: Optional[str]
    raw_output: str
    prompt: str
    success_element: bool
    success_action: bool
    step_success: bool
    operation_f1: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        base = asdict(self)
        base["metadata"] = dict(self.metadata)
        return base


@dataclass
class Mind2WebEvalResult:
    metrics: Dict[str, float]
    totals: Dict[str, int]
    predictions: List[Mind2WebPrediction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"metrics": self.metrics, "totals": self.totals}


class _MetricAccumulator:
    def __init__(self):
        self.total_steps = 0
        self.element_correct = 0
        self.step_correct = 0
        self.operation_sum = 0.0

        self.task_element: Dict[str, List[bool]] = {}
        self.task_operation: Dict[str, List[float]] = {}
        self.task_steps: Dict[str, List[bool]] = {}

        self.skipped_steps = 0

    def add(self, prediction: Mind2WebPrediction):
        self.total_steps += 1
        if prediction.success_element:
            self.element_correct += 1
        if prediction.step_success:
            self.step_correct += 1
        self.operation_sum += prediction.operation_f1

        self.task_element.setdefault(prediction.task_id, []).append(prediction.success_element)
        self.task_operation.setdefault(prediction.task_id, []).append(prediction.operation_f1)
        self.task_steps.setdefault(prediction.task_id, []).append(prediction.step_success)

    def skip(self):
        self.skipped_steps += 1

    def finalize(self) -> Dict[str, float]:
        metrics = {
            "element_micro": self.element_correct / self.total_steps if self.total_steps else 0.0,
            "operation_micro": self.operation_sum / self.total_steps if self.total_steps else 0.0,
            "step_micro": self.step_correct / self.total_steps if self.total_steps else 0.0,
        }
        metrics["element_macro"] = self._macro_average(self.task_element)
        metrics["operation_macro"] = self._macro_average(self.task_operation)
        metrics["step_macro"] = self._macro_average(self.task_steps)
        metrics["task_success"] = self._task_success_rate(self.task_steps)
        return metrics

    @staticmethod
    def _macro_average(table: Dict[str, List[bool]]) -> float:
        if not table:
            return 0.0
        per_task = []
        for values in table.values():
            if not values:
                continue
            per_task.append(sum(values) / len(values))
        if not per_task:
            return 0.0
        return mean(per_task)

    @staticmethod
    def _task_success_rate(table: Dict[str, List[bool]]) -> float:
        if not table:
            return 0.0
        successes = sum(1 for values in table.values() if values and all(values))
        total = len(table)
        return successes / total if total else 0.0


class Mind2WebTask:
    """High-level Mind2Web evaluation loop."""

    def __init__(
        self,
        dataset: Mind2WebDataset,
        llm: LLMProvider,
        prompt_builder: Mind2WebPromptBuilder,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        log_every: int = 25,
    ):
        self.dataset = dataset
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.log_every = log_every

    async def predict_single(self, sample: Mind2WebSample) -> Mind2WebPrediction:
        prompt, choice_map = self.prompt_builder.build(sample)

        messages = [
            {"role": "system", "content": self.prompt_builder.system_prompt},
            {"role": "user", "content": prompt},
        ]

        raw_output = await self.llm.chat(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        choice, action, value = parse_action_output(raw_output)
        candidate = choice_map.get(choice) if choice else None
        candidate_id = candidate.backend_node_id if candidate else None

        success_element = bool(candidate_id and candidate_id in sample.positive_ids)
        success_action = action == sample.action_type
        operation_f1 = compute_operation_f1(
            sample_action_type=sample.action_type,
            sample_action_value=sample.action_value,
            predicted_action=action,
            predicted_value=value,
        )
        step_success = success_element and operation_f1 == 1.0

        metadata = {
            "split": sample.split,
            "website": sample.website,
            "domain": sample.domain,
            "positive_ids": sample.positive_ids,
            "negative_ids": sample.negative_ids,
            "gold_action_value": sample.action_value,
        }

        return Mind2WebPrediction(
            sample_id=sample.sample_id,
            task_id=sample.task_id,
            step_index=sample.step_index,
            choice=choice,
            candidate_id=candidate_id,
            action=action,
            predicted_value=value,
            raw_output=raw_output,
            prompt=prompt,
            success_element=success_element,
            success_action=success_action,
            step_success=step_success,
            operation_f1=operation_f1,
            metadata=metadata,
        )

    async def evaluate(
        self,
        *,
        max_examples: Optional[int] = None,
    ) -> Mind2WebEvalResult:
        predictions: List[Mind2WebPrediction] = []
        accumulator = _MetricAccumulator()

        samples = list(self.dataset)
        if max_examples:
            samples = samples[:max_examples]

        for idx, sample in enumerate(samples, start=1):
            if not sample.has_positive:
                logger.debug(
                    "Evaluating sample %s even though no positive candidates remain (will count as failure)",
                    sample.sample_id,
                )
            prediction = await self.predict_single(sample)
            predictions.append(prediction)
            accumulator.add(prediction)

            if self.log_every and idx % self.log_every == 0:
                logger.info(
                    "Evaluated %s/%s samples (element micro=%.3f, step micro=%.3f)",
                    idx,
                    len(samples),
                    accumulator.element_correct / accumulator.total_steps if accumulator.total_steps else 0.0,
                    accumulator.step_correct / accumulator.total_steps if accumulator.total_steps else 0.0,
                )

        metrics = accumulator.finalize()
        totals = {
            "evaluated_steps": accumulator.total_steps,
            "skipped_steps": accumulator.skipped_steps,
            "total_samples": len(samples),
            "total_dataset_samples": len(self.dataset),
            "unique_tasks": len(accumulator.task_steps),
        }

        return Mind2WebEvalResult(metrics=metrics, totals=totals, predictions=predictions)


def parse_action_output(output: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse 'Answer: X', 'Action: CLICK', and optional 'Value: ...' from the LLM output."""
    if not output:
        return None, None, None

    answer_match = ANSWER_RE.search(output)
    action_match = ACTION_RE.search(output)
    value_match = VALUE_RE.search(output)

    choice = answer_match.group(1).strip().upper() if answer_match else None
    action = action_match.group(1).strip().upper().replace(" ", "").replace("_", "") if action_match else None
    value = _normalize_value_field(value_match.group(1)) if value_match else None

    if action:
        action = ACTION_ALIASES.get(action, action)

    return choice, action, value


def _normalize_value_field(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return ""
    if value.upper() in {"N/A", "NA", "NONE", "NULL"}:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1].strip()
    return value


def _tokenize_value(value: Optional[str]) -> List[str]:
    if not value:
        return []
    tokens = TOKEN_RE.findall(value.lower())
    return tokens


def compute_operation_f1(
    *,
    sample_action_type: Optional[str],
    sample_action_value: Optional[str],
    predicted_action: Optional[str],
    predicted_value: Optional[str],
) -> float:
    """Token-level F1 over operation outputs following the Mind2Web protocol."""
    gold_action = (sample_action_type or "").upper()
    pred_action = (predicted_action or "").upper()

    if not gold_action:
        return 0.0

    if gold_action == "CLICK":
        return 1.0 if pred_action == "CLICK" else 0.0

    if gold_action in {"TYPE", "SELECT"}:
        if pred_action != gold_action:
            return 0.0
        gold_tokens = _tokenize_value(sample_action_value)
        pred_tokens = _tokenize_value(predicted_value)
        if not gold_tokens and not pred_tokens:
            return 1.0
        if not gold_tokens or not pred_tokens:
            return 0.0

        gold_counts = Counter(gold_tokens)
        pred_counts = Counter(pred_tokens)
        overlap = 0
        for token, count in gold_counts.items():
            if token in pred_counts:
                overlap += min(count, pred_counts[token])

        if overlap == 0:
            return 0.0

        precision = overlap / sum(pred_counts.values())
        recall = overlap / sum(gold_counts.values())
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    # Fall back to exact match for less common operations.
    return 1.0 if pred_action == gold_action else 0.0


async def evaluate_from_cli(
    config_path: str,
    data_root: str,
    split: str,
    *,
    scores_path: Optional[str] = None,
    top_k: Optional[int] = 50,
    prompt_template: Optional[str] = None,
    max_examples: Optional[int] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> Mind2WebEvalResult:
    """Utility entrypoint for CLI scripts."""
    config = Config(config_path)
    llm = LLMFactory.create(config.get_llm_config())
    dataset = Mind2WebDataset(
        data_root=data_root,
        split=split,
        top_k=top_k,
        scores_path=scores_path,
    )
    prompt_builder = Mind2WebPromptBuilder(prompt_template)
    task = Mind2WebTask(
        dataset=dataset,
        llm=llm,
        prompt_builder=prompt_builder,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await task.evaluate(max_examples=max_examples)


def save_eval_output(result: Mind2WebEvalResult, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "mind2web_metrics.json"
    preds_path = output_dir / "mind2web_predictions.json"

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump({"metrics": result.metrics, "totals": result.totals}, handle, indent=2, ensure_ascii=False)

    with preds_path.open("w", encoding="utf-8") as handle:
        json.dump([pred.to_dict() for pred in result.predictions], handle, indent=2, ensure_ascii=False)

    logger.info("Saved metrics to %s", metrics_path)
    logger.info("Saved predictions to %s", preds_path)
