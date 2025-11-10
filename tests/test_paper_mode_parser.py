"""Tests for paper-faithful memory extraction helpers."""
from types import SimpleNamespace
from pathlib import Path

import importlib.util

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "tools" / "extract_memory.py"
spec = importlib.util.spec_from_file_location("extract_memory", MODULE_PATH)
extract_memory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract_memory)
ExtractMemoryTool = extract_memory.ExtractMemoryTool


class _StubConfig:
    def __init__(self, preset: str = "paper_faithful"):
        self._data = {
            "mode": {"preset": preset},
            "extraction": {}
        }

    def get(self, *keys, default=None):
        value = self._data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def get_extraction_config(self):  # pragma: no cover - compatibility helper
        return self._data.get("extraction", {})

    def is_paper_faithful_mode(self):
        return True


def _make_tool():
    config = _StubConfig()
    dummy = SimpleNamespace(get_provider_name=lambda: "dummy")
    return ExtractMemoryTool(
        config,
        storage_backend=None,
        llm_provider=dummy,
        embedding_provider=dummy,
        memory_manager=None,
    )


def test_markdown_parser_extracts_items():
    tool = _make_tool()
    response = """
```markdown
# Memory Item 1
## Title Title Navigation Strategy
## Description When searching for specific information within history ...
## Content Detect pagination mode and examine all items before responding.

# Memory Item 2
## Title Fallback After Pagination Failure
## Description Avoid infinite scroll traps when the primary strategy fails.
## Content Switch to alternative sections or search filters when the default workflow hits dead ends.
```
"""

    items = tool._parse_markdown_memories(response)
    assert len(items) == 2
    assert items[0]["title"] == "Title Navigation Strategy"
    assert "fallback" in items[1]["title"].lower()


def test_normalize_enforces_required_fields():
    tool = _make_tool()
    raw_items = [
        {"title": "A", "description": "B", "content": "C"},
        {"title": "", "description": "missing", "content": "text"},
    ]

    normalized = tool._normalize_memory_items(raw_items)
    assert len(normalized) == 1
    assert normalized[0]["title"] == "A"
