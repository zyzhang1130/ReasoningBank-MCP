"""Mind2Web dataset loader and helpers."""
from __future__ import annotations

import json
import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

logger = logging.getLogger(__name__)


def _normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _candidate_repr(candidate: Dict[str, Any]) -> str:
    for key in ("repr", "text", "cleaned_html", "outer_html", "description", "snippet"):
        if key in candidate and candidate[key]:
            return _normalize_text(str(candidate[key]))
    # fall back to concatenating tag + text attrs if present
    tag = candidate.get("tag") or candidate.get("tag_name")
    text = candidate.get("text") or candidate.get("content")
    if tag or text:
        return _normalize_text(f"{tag or ''} {text or ''}")
    return ""


class ScoreLookup:
    """Attach rank/score metadata for candidate elements."""

    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as handle:
            raw = pickle.load(handle)
        self._table = self._normalize_raw_scores(raw)
        logger.info("Loaded Mind2Web score table with %s entries from %s", len(self._table), path)

    def _normalize_raw_scores(self, raw: Any) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Normalize a wide range of pickle formats into:
            sample_key -> backend_node_id -> {"score": float, "rank": int}
        """
        table: Dict[str, Dict[str, Dict[str, Any]]] = {}

        if isinstance(raw, dict):
            items = raw.items()
        elif isinstance(raw, list):
            # treat list entries as {sample_id: ..., candidates: {...}}
            items = []
            for entry in raw:
                if isinstance(entry, dict):
                    key = entry.get("sample_id") or entry.get("uid") or entry.get("id")
                    candidates = entry.get("candidates") or entry.get("scores")
                    if key and candidates:
                        items.append((key, candidates))
        else:
            raise TypeError("Unsupported score pickle format")

        for key, value in items:
            if isinstance(value, dict):
                candidate_map = self._normalize_candidate_map(value)
            elif isinstance(value, list):
                candidate_map = self._normalize_candidate_sequence(value)
            else:
                continue

            if candidate_map:
                table[str(key)] = candidate_map

        return table

    def _normalize_candidate_sequence(self, seq: Sequence[Any]) -> Dict[str, Dict[str, Any]]:
        candidate_map: Dict[str, Dict[str, Any]] = {}
        for item in seq:
            if isinstance(item, dict):
                cid = item.get("backend_node_id") or item.get("node_id") or item.get("id")
                if cid is None:
                    continue
                candidate_map[str(cid)] = {
                    "score": item.get("score"),
                    "rank": item.get("rank"),
                }
            elif isinstance(item, (tuple, list)) and len(item) >= 3:
                cid = item[0]
                score = item[1]
                rank = item[2]
                candidate_map[str(cid)] = {"score": score, "rank": rank}
        return candidate_map

    def _normalize_candidate_map(self, data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        candidate_map: Dict[str, Dict[str, Any]] = {}
        for key, value in data.items():
            # nested entries may sit under "candidates"
            if key in {"candidates", "scores"} and isinstance(value, dict):
                for cid, packed in value.items():
                    candidate_map[str(cid)] = self._unpack_score_entry(packed)
                continue
            candidate_map[str(key)] = self._unpack_score_entry(value)
        return candidate_map

    @staticmethod
    def _unpack_score_entry(entry: Any) -> Dict[str, Any]:
        if isinstance(entry, dict):
            return {"score": entry.get("score"), "rank": entry.get("rank")}
        if isinstance(entry, (list, tuple)):
            score = entry[0] if len(entry) > 0 else None
            rank = entry[1] if len(entry) > 1 else None
            return {"score": score, "rank": rank}
        if isinstance(entry, (int, float)):
            # when only score is provided
            return {"score": entry, "rank": None}
        return {"score": None, "rank": None}

    def lookup(self, sample_keys: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        for key in sample_keys:
            if key in self._table:
                return self._table[key]
        return {}


@dataclass
class CandidateElement:
    """Structured representation of a candidate DOM node."""

    backend_node_id: str
    text: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None
    rank: Optional[int] = None
    is_positive: bool = False

    def for_prompt(self, max_length: int = 220) -> str:
        snippet = self.text or self.attributes.get("text") or self.attributes.get("outer_html") or ""
        snippet = _normalize_text(str(snippet))
        if not snippet:
            snippet = "<empty>"
        if len(snippet) > max_length:
            snippet = f"{snippet[: max_length - 3]}..."
        meta_bits = []
        tag = self.attributes.get("tag") or self.attributes.get("tag_name")
        if tag:
            meta_bits.append(tag)
        aria = self.attributes.get("aria_label") or self.attributes.get("aria-label")
        if aria:
            meta_bits.append(f'aria-label="{aria}"')
        if self.rank is not None:
            meta_bits.append(f"rank={self.rank}")
        if self.score is not None:
            meta_bits.append(f"score={self.score:.3f}")
        meta = f" ({', '.join(meta_bits)})" if meta_bits else ""
        return f"{snippet}{meta}"


@dataclass
class Mind2WebSample:
    """One Mind2Web step (task + action)."""

    sample_id: str
    task_id: str
    split: str
    website: Optional[str]
    domain: Optional[str]
    step_index: int
    total_steps: int
    confirmed_task: str
    history: List[str]
    page_html: str
    action_type: str
    action_value: Optional[str]
    candidates: List[CandidateElement]
    positive_ids: List[str]
    negative_ids: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def step_number(self) -> int:
        return self.step_index + 1

    @property
    def has_positive(self) -> bool:
        return len(self.positive_ids) > 0

    @property
    def history_text(self) -> str:
        if not self.history:
            return "None"
        return "\n".join(self.history)

    def score_lookup_keys(self) -> List[str]:
        keys = [
            self.metadata.get("score_id"),
            self.metadata.get("action_uid"),
            self.metadata.get("step_uid"),
            f"{self.task_id}:{self.step_index}",
            f"{self.task_id}_{self.step_index}",
            f"{self.split}:{self.task_id}:{self.step_index}",
        ]
        return [str(key) for key in keys if key]


class Mind2WebDataset(Iterable[Mind2WebSample]):
    """Eager Mind2Web dataset loader."""

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        *,
        top_k: Optional[int] = 50,
        scores_path: Optional[str | Path] = None,
        skip_samples_without_positive: bool = False,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.top_k = top_k
        self.skip_samples_without_positive = skip_samples_without_positive
        self.score_lookup = ScoreLookup(Path(scores_path)) if scores_path else None

        self.samples: List[Mind2WebSample] = []
        self.skipped_samples: List[Mind2WebSample] = []

        if not self.data_root.exists():
            raise FileNotFoundError(f"Mind2Web data root does not exist: {self.data_root}")

        self._load_split()

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Mind2WebSample]:
        return iter(self.samples)

    def summary(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "data_root": str(self.data_root),
            "total_samples": len(self.samples),
            "skipped_samples": len(self.skipped_samples),
            "top_k": self.top_k,
            "has_scores": self.score_lookup is not None,
        }

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _load_split(self):
        logger.info("Loading Mind2Web split '%s' from %s", self.split, self.data_root)
        files = list(self._resolve_split_files())
        if not files:
            raise FileNotFoundError(
                f"No Mind2Web JSON files found for split '{self.split}' under {self.data_root}"
            )

        for file_path in files:
            tasks = self._load_task_file(file_path)
            for task in tasks:
                for sample in self._build_samples_for_task(task, file_path):
                    if not sample.has_positive and self.skip_samples_without_positive:
                        self.skipped_samples.append(sample)
                        continue
                    self.samples.append(sample)

        logger.info(
            "Loaded %s Mind2Web samples (skipped %s without positive targets)",
            len(self.samples),
            len(self.skipped_samples),
        )

    def _resolve_split_files(self) -> Iterator[Path]:
        if self.data_root.is_file():
            yield self.data_root
            return

        candidate = self.data_root / f"{self.split}.json"
        if candidate.exists():
            yield candidate
            return

        split_dir = self.data_root / self.split
        if split_dir.exists():
            for file_path in sorted(split_dir.glob("*.json")):
                yield file_path
            return

        for file_path in sorted(self.data_root.glob(f"*{self.split}*.json")):
            yield file_path

    def _load_task_file(self, path: Path) -> List[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if isinstance(data, dict):
            for key in ("tasks", "data", "entries", "samples"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            if isinstance(data.get("actions"), list):
                return [data]
            raise ValueError(f"Unsupported Mind2Web JSON structure in {path}")

        if isinstance(data, list):
            return data

        raise ValueError(f"Unexpected JSON payload type in {path}: {type(data)}")

    def _build_samples_for_task(self, task: Dict[str, Any], source_path: Path) -> Iterator[Mind2WebSample]:
        task_id = task.get("task_id") or task.get("id") or task.get("meta", {}).get("task_id")
        confirmed_task = (
            task.get("confirmed_task")
            or task.get("task_description")
            or task.get("instruction")
            or task.get("goal")
            or ""
        )

        actions = task.get("actions") or []
        total_steps = len(actions)
        for step_index, action in enumerate(actions):
            yield self._build_sample(
                task_id=task_id or f"{source_path.stem}",
                confirmed_task=confirmed_task,
                actions=actions,
                action=action,
                step_index=step_index,
                total_steps=total_steps,
                source_path=source_path,
                task=task,
            )

    def _build_sample(
        self,
        *,
        task_id: str,
        confirmed_task: str,
        actions: List[Dict[str, Any]],
        action: Dict[str, Any],
        step_index: int,
        total_steps: int,
        source_path: Path,
        task: Dict[str, Any],
    ) -> Mind2WebSample:
        action_type = (
            action.get("action_type")
            or action.get("action")
            or action.get("operation")
            or action.get("type")
            or "CLICK"
        )
        action_type = action_type.upper()

        action_value = action.get("value") or action.get("input_text") or action.get("text")

        history = []
        for prior in actions[:step_index]:
            history.append(
                prior.get("action_repr")
                or prior.get("action_reprs")
                or prior.get("action")
                or prior.get("operation")
                or "N/A"
            )

        page_html = action.get("cleaned_html") or action.get("raw_html") or ""

        positive_raw = action.get("pos_candidates") or []
        negative_raw = action.get("neg_candidates") or []

        score_map = (
            self.score_lookup.lookup(
                [
                    *(action.get(key) for key in ("score_id", "sample_id", "uid") if action.get(key)),
                    f"{task_id}:{step_index}",
                    f"{task_id}_{step_index}",
                    f"{self.split}:{task_id}:{step_index}",
                ]
            )
            if self.score_lookup
            else {}
        )

        candidates = self._build_candidates(
            positive_raw,
            negative_raw,
            score_map,
        )

        candidates = self._filter_candidates(candidates)
        positive_ids = [c.backend_node_id for c in candidates if c.is_positive]
        negative_ids = [c.backend_node_id for c in candidates if not c.is_positive]

        sample_id = action.get("sample_id") or f"{task_id}:{step_index}"

        metadata = {
            "source_file": str(source_path),
            "website": task.get("website"),
            "domain": task.get("domain"),
            "action_uid": action.get("uid") or action.get("action_uid"),
            "step_uid": action.get("step_uid"),
            "score_id": action.get("score_id"),
        }

        return Mind2WebSample(
            sample_id=str(sample_id),
            task_id=str(task_id),
            split=self.split,
            website=task.get("website"),
            domain=task.get("domain"),
            step_index=step_index,
            total_steps=total_steps,
            confirmed_task=confirmed_task,
            history=history,
            page_html=page_html,
            action_type=action_type,
            action_value=action_value,
            candidates=candidates,
            positive_ids=positive_ids,
            negative_ids=negative_ids,
            metadata=metadata,
        )

    def _build_candidates(
        self,
        positive_raw: List[Dict[str, Any]],
        negative_raw: List[Dict[str, Any]],
        score_map: Dict[str, Dict[str, Any]],
    ) -> List[CandidateElement]:
        candidates: List[CandidateElement] = []
        for raw, is_positive in ((positive_raw, True), (negative_raw, False)):
            for entry in raw:
                backend_node_id = entry.get("backend_node_id") or entry.get("node_id")
                if backend_node_id is None:
                    continue

                cid = str(backend_node_id)
                score_info = score_map.get(cid, {})
                candidate = CandidateElement(
                    backend_node_id=cid,
                    text=_candidate_repr(entry),
                    attributes={k: v for k, v in entry.items() if k not in {"backend_node_id", "node_id"}},
                    score=score_info.get("score"),
                    rank=score_info.get("rank"),
                    is_positive=is_positive,
                )
                candidates.append(candidate)

        # ensure positives come first to avoid pruning them when ranks are missing
        candidates.sort(key=lambda c: (0 if c.is_positive else 1, c.rank if c.rank is not None else 10**6))
        return candidates

    def _filter_candidates(self, candidates: List[CandidateElement]) -> List[CandidateElement]:
        if self.top_k is None or self.top_k <= 0:
            return candidates

        with_rank = [c for c in candidates if c.rank is not None]
        if with_rank:
            filtered = [c for c in candidates if c.rank is None or c.rank < self.top_k]
            if filtered:
                return filtered

        # fallback: keep first top_k candidates while preserving positives
        filtered: List[CandidateElement] = []
        for candidate in candidates:
            if len(filtered) >= self.top_k:
                # ensure positives survive even if exceeding top_k
                if candidate.is_positive and candidate not in filtered:
                    filtered.append(candidate)
                continue
            filtered.append(candidate)
        return filtered
