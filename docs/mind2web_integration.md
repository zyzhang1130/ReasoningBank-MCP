# Mind2Web Integration Guide

This document explains how to reproduce the Mind2Web benchmark inside the ReasoningBank framework. The integration is intentionally modular: the MCP server remains unchanged, while the new `src/mind2web` package and the `scripts/eval_mind2web.py` helper cover dataset parsing, prompting, action prediction, and metric computation.

## 2. Dataset Loader (`src/mind2web/dataset.py`)

1. **Loading JSON tasks** – Point the loader to the directory that contains the official JSON files. For each task the loader iterates over every action/step and creates an evaluation sample.
2. **Context extraction** – Each action stores both `raw_html` and `cleaned_html`. The loader prefers `cleaned_html` so that the candidate list stays aligned with the DOM snapshot shipped with Mind2Web.
3. **Candidate gathering** – Both `pos_candidates` and `neg_candidates` are parsed, normalised, and converted into `CandidateElement` objects. Metadata such as tag names, visible texts, or any extra attributes are preserved in `candidate.attributes`.
4. **scores_all_data.pkl support** – If you downloaded the official pickle, pass its location to the dataset. The loader builds a flexible lookup map (`backend_node_id -> {score, rank}`) so the step-level samples can carry their probability scores and candidate ranks.
5. **Top-K filtering** – Set `top_k` (paper uses 50). Candidates whose rank is greater than or equal to `top_k` are dropped. When rank metadata is missing the loader keeps the first `top_k` elements but still ensures every positive candidate survives the pruning. Samples that end up with zero positives are automatically skipped (configurable).

The dataset summary (`Mind2WebDataset.summary()`) reports how many steps were loaded and how many were dropped because their gold element disappeared after filtering.

## 3. Prompting & Prediction (`src/mind2web/prompting.py`, `src/mind2web/task.py`)

1. **Few-shot prompt** – The default template lives in `src/prompts/mind2web_llm_prompt.json`, mirroring the paper’s 3-shot multiple-choice format. Override it with `--prompt-template` if you want to bring your own examples.
2. **Prompt contents** – Each inference prompt contains:
   - Task instruction (`confirmed_task`) and the step number.
   - Previous actions (`action_repr` strings from earlier steps).
   - The cleaned HTML context wrapped inside `"""` blocks (auto-truncated to avoid giant payloads).
   - Options list. Option `A` is always “None of the above”, while `B`, `C`, … point to actual DOM candidates. For transparency, every option includes snippet text plus rank/score metadata (when available).
   - An explicit answer format reminder (`Answer: X.\nAction: CLICK`).
3. **LLM decoding** – `Mind2WebTask` relies on the standard ReasoningBank LLM providers. Temperature defaults to `0.0` (deterministic decoding) and can be controlled per run.
4. **Parsing outputs** – The helper `parse_action_output` extracts both the option letter and the action type via regex. Action names are normalised (`CLICK`, `TYPE`, `SELECT`) before evaluating against the ground truth.
5. **Step-by-step loop** – `Mind2WebTask.evaluate()` walks through the dataset, builds prompts, calls the model, and records per-step predictions. Each prediction stores the raw LLM output, the prompt, and success flags (element/action/step).

## 4. Metrics (`src/mind2web/task.py`)

For each sample we compute:

- **Element Accuracy** – Whether the predicted candidate ID is inside the post-filter gold list.
- **Action Accuracy** – Whether the predicted operation (CLICK/TYPE/SELECT) matches the step label.
- **Step Success** – Element and action must both be correct.

Aggregations:

- **Micro** averages (overall accuracy across all steps) for element/action/step.
- **Macro** averages (per-task accuracy averaged across tasks) to match the paper’s protocol.
- **Task success rate** – A task counts as solved only when every step is successful.

The accumulator also tracks how many steps were skipped (e.g., after top-k filtering removed the positive element).

## 5. Evaluation Script (`scripts/eval_mind2web.py`)

Use the CLI helper to run a full evaluation:

```bash
python scripts/eval_mind2web.py \
  --config config.yaml \
  --data-root /path/to/Mind2Web/test \
  --split test_task \
  --scores-pickle /path/to/scores_all_data.pkl \
  --top-k 50 \
  --temperature 0.0 \
  --output-dir outputs/mind2web_test_task
```

- `--data-root` can be a single JSON file, a split directory (e.g., `test_task/`), or a parent directory that contains files named `*test_task*.json`.
- `--scores-pickle` is optional but recommended to faithfully follow the benchmark (attach candidate ranks).
- `--max-examples` is handy for quick smoke tests.
- `--prompt-template` allows swapping in an alternative few-shot prompt.
- Results are printed to stdout and, when `--output-dir` is set, dumped into `mind2web_metrics.json` and `mind2web_predictions.json`.

## 6. Downloading Mind2Web Assets (Hugging Face)

You can fetch the data and helper assets using Hugging Face utilities:

```python
from datasets import load_dataset
ds = load_dataset("osunlp/Mind2Web", "default")

from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="osunlp/Mind2Web", filename="scores_all_data.pkl", repo_type="dataset")
hf_hub_download(repo_id="osunlp/Mind2Web", filename="test.zip", repo_type="dataset")
```

After downloading `test.zip` from the dataset page, unzip it with the password provided by the authors to obtain files such as `test_task.json`, `test_website.json`, etc. Place these under a directory referenced by `--data-root`. The same Hugging Face hub also hosts fine-tuned checkpoints (`osunlp/flan-t5-large-mind2web`, `osunlp/deberta-v3-large-mind2web`), which can be loaded through `transformers` if you want to reproduce the paper’s exact models.

## 7. Testing & Reproducibility

1. Start with a handful of samples (`--max-examples 5`) to ensure prompts look correct and the parsing step recognises the “Answer/Action” fields.
2. Once satisfied, run the full `test_task`, `test_website`, and `test_domain` splits. Compare macro step accuracy and task success with the paper’s reported results.
3. The integration is read-only with respect to ReasoningBank’s memory. During benchmarking you can keep MCP memory disabled, or optionally route trajectories into the MCP server if you want to study continual-learning effects—nothing in the new modules depends on it.
4. Deterministic decoding (`temperature=0`) plus rank-based candidate filtering make runs reproducible; still, set environment seeds for any local decoder (e.g., Flan-T5) if you leave sampling on.

Following the steps above gives you a faithful recreation of the Mind2Web benchmark pipeline while keeping the ReasoningBank memory server untouched.
