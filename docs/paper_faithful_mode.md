# Paper-faithful Mode

This guide explains how to run the MCP server with the exact configuration used in the ReasoningBank paper (cosine retrieval, no automated consolidation, and fully synchronous extraction).

## Enabling the Mode

1. Copy the default config:
   ```bash
   cp src/default_config.yaml ~/.reasoningbank/paper_config.yaml
   ```
2. In the copied file set:
   ```yaml
   mode:
     preset: "paper_faithful"
   ```
3. Start the server with the new config:
   ```bash
   python -m src.server --config ~/.reasoningbank/paper_config.yaml
   ```

When `preset` is `paper_faithful` the server automatically:

- Switches retrieval to pure cosine similarity (no success/recency boosts).
- Disables the memory manager so every extracted memory is stored verbatim.
- Runs extraction synchronously and keeps the original paper prompts/temperature settings.

No other manual tweaks are needed; even if the config file still contains `memory_manager.enabled: true` or `retrieval.strategy: hybrid`, the overrides take priority.

## Suggested Evaluation Flow

To mirror the paper’s experiments:

1. Run tasks sequentially. After each task, call `extract_memory` with the trajectory and query. The synchronous mode ensures the new memory is available before the next task.
2. Before starting the next task, call `retrieve_memory` (top_k=1). The returned snippet goes into the system prompt of your agent controller (e.g., BrowserGym agent or SWE-Bench runner).
3. Record success/failure and number of steps externally (the MCP server simply stores memories). A minimal evaluation loop looks like this:
   ```python
   for task in task_stream:
       retrieved = mcp_call("retrieve_memory", {"query": task.query, "agent_id": agent})
       context = base_system_prompt + retrieved["formatted_prompt"]
       trajectory, success = run_agent(task, system_prompt=context)
       mcp_call("extract_memory", {
           "trajectory": trajectory,
           "query": task.query,
           "agent_id": agent,
           "async_mode": False
       })
       log_metrics(task, success, steps=len(trajectory))
   ```
4. Compare against a no-memory baseline by skipping the `retrieve_memory` call.

This is the same closed-loop process described in the paper (retrieve → act → reflect). Use different `agent_id` values to isolate runs for ablations, or to reset memory between datasets.

## Troubleshooting

- **Memories accumulating across runs**: delete `~/.reasoningbank/data/memories.json` or use a unique `agent_id`.
- **Need hybrid retrieval again**: switch the preset back to `default` in the config.
- **Async pipeline required**: paper mode forces synchronous extraction; for asynchronous ingestion use the default preset instead.
