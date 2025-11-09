"""Prompt construction utilities for Mind2Web evaluation."""
from __future__ import annotations

import json
import logging
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .dataset import CandidateElement, Mind2WebSample

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "prompts" / "mind2web_llm_prompt.json"


def _alphabet_label(index: int) -> str:
    """Convert 0-based index into spreadsheet-style labels (A, B, ..., Z, AA, AB, ...)."""
    letters = []
    while True:
        index, remainder = divmod(index, 26)
        letters.append(string.ascii_uppercase[remainder])
        if index == 0:
            break
        index -= 1
    return "".join(reversed(letters))


class Mind2WebPromptBuilder:
    """Render few-shot prompts that follow the Mind2Web multi-choice format."""

    def __init__(
        self,
        template_path: str | Path | None = None,
        *,
        max_html_chars: int = 6000,
        max_candidates: Optional[int] = 50,
    ):
        path = Path(template_path) if template_path else DEFAULT_TEMPLATE_PATH
        if not path.exists():
            raise FileNotFoundError(f"Mind2Web prompt template not found: {path}")

        with path.open("r", encoding="utf-8") as handle:
            self.template = json.load(handle)

        self.system_prompt = self.template.get(
            "system_prompt",
            "You are an expert web agent that reasons about DOM snapshots to pick the next action.",
        )
        self.examples = self.template.get("examples", [])
        self.question = self.template.get(
            "question",
            "What should be the next action? Select from the options below. "
            "If no option matches, choose option A ('None of the above').",
        )
        self.answer_instructions = self.template.get(
            "answer_instructions",
            "Respond with two lines:\nAnswer: <LETTER>.\nAction: <CLICK/TYPE/SELECT>.",
        )
        self.max_html_chars = max_html_chars
        self.max_candidates = max_candidates

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def build(
        self,
        sample: Mind2WebSample,
    ) -> Tuple[str, Dict[str, Optional[CandidateElement]]]:
        """Build the final prompt string and return the choice map."""
        sections: List[str] = []
        if self.examples:
            sections.append(self._format_examples())

        sections.append(self._format_current_step(sample))
        prompt = "\n\n".join(sections)
        prompt += f"\n\n{self.answer_instructions}"

        choice_map = self._build_choice_map(sample)
        return prompt, choice_map

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _format_examples(self) -> str:
        rendered = []
        for idx, example in enumerate(self.examples, start=1):
            rendered.append(
                "\n".join(
                    [
                        f"### Example {idx}",
                        f"Task: {example.get('task')}",
                        f"Previous actions: {example.get('history', 'None')}",
                        "Page HTML:",
                        f"\"\"\"\n{example.get('context')}\n\"\"\"",
                        f"Question: {example.get('question')}",
                        example.get("choices", ""),
                        example.get("answer", ""),
                    ]
                )
            )
        return "\n\n".join(rendered)

    def _clip_html(self, html: str) -> str:
        if not html:
            return ""
        html = html.strip()
        if len(html) <= self.max_html_chars:
            return html
        return f"{html[: self.max_html_chars]}...\n<!-- truncated -->"

    def _format_current_step(self, sample: Mind2WebSample) -> str:
        context = self._clip_html(sample.page_html)
        history = sample.history_text
        lines = [
            "### Mind2Web Evaluation Sample",
            f"Task instruction: {sample.confirmed_task}",
            f"Step: {sample.step_number}/{sample.total_steps}",
            f"Previous actions:\n{history}",
            "Current page HTML snapshot:",
            f"\"\"\"\n{context}\n\"\"\"",
            f"Question: {self.question}",
        ]
        choice_block, _ = self._format_choices(sample)
        lines.append(choice_block)
        return "\n\n".join(lines)

    def _format_choices(
        self,
        sample: Mind2WebSample,
    ) -> Tuple[str, Dict[str, Optional[CandidateElement]]]:
        choice_lines = ["Options:", "A. None of the above (no correct element is visible)."]
        choice_map: Dict[str, Optional[CandidateElement]] = {"A": None}

        candidates = self._ordered_candidates(sample)
        token_budget = self.max_candidates or len(candidates)

        for idx, candidate in enumerate(candidates):
            if idx >= token_budget:
                break
            label = _alphabet_label(idx + 1)  # +1 because 0 -> A (reserved)
            choice_map[label] = candidate
            choice_lines.append(f"{label}. {candidate.for_prompt()}")

        return "\n".join(choice_lines), choice_map

    def _build_choice_map(
        self,
        sample: Mind2WebSample,
    ) -> Dict[str, Optional[CandidateElement]]:
        _, choice_map = self._format_choices(sample)
        return choice_map

    def _ordered_candidates(self, sample: Mind2WebSample) -> List[CandidateElement]:
        candidates = list(sample.candidates)
        candidates.sort(
            key=lambda c: (
                c.rank if c.rank is not None else 10**6,
                0 if c.is_positive else 1,
                c.backend_node_id,
            )
        )
        return candidates
