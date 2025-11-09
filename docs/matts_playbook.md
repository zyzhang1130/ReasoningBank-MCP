# Memory-aware Test-Time Scaling (MaTTS) Playbook

The MCP server does not orchestrate MaTTS directly, but you can implement the paper’s parallel and sequential scaling loops with a thin controller. This note sketches the wiring.

## Parallel Scaling (Self-Contrast)

```python
async def run_parallel(task, agent_id, k=3):
    retrieved = await mcp_call("retrieve_memory", {
        "query": task.query,
        "agent_id": agent_id,
        "top_k": 1
    })

    trajectories = []
    for idx in range(k):
        prompt = base_prompt + retrieved["formatted_prompt"]
        traj = await run_agent(task, system_prompt=prompt, seed=idx)
        trajectories.append(traj)

    best = await pick_best_trajectory(task.query, trajectories)
    await ingest_memories(task, trajectories, agent_id)
    return best
```

1. Retrieve memory once per task.
2. Launch `k` agent rollouts (different seeds or decoding temperatures).
3. Use an external judge (LLM-as-a-judge or a rule-based scorer) to choose the best trajectory (Best-of-N) to report back to the user.
4. Feed every trajectory into `extract_memory` so both successes and failures enrich ReasoningBank.

## Sequential Scaling (Self-Refine)

```python
async def run_sequential(task, agent_id, refinement_rounds=3):
    retrieved = await mcp_call("retrieve_memory", {
        "query": task.query,
        "agent_id": agent_id,
        "top_k": 1
    })

    trajectory = await run_agent(task, system_prompt=base_prompt + retrieved["formatted_prompt"])

    for round_idx in range(refinement_rounds):
        critique = await self_check(task, trajectory)
        if critique.is_perfect:
            break
        trajectory = await run_agent(
            task,
            system_prompt=base_prompt + retrieved["formatted_prompt"],
            critique=critique
        )

    await ingest_memories(task, [trajectory], agent_id)
    return trajectory
```

1. Run the task once with the retrieved memory.
2. Ask the agent (or an external LLM) to critique its own steps.
3. Feed the critique as additional context to rerun the task, producing a refined trajectory.
4. Only the final trajectory needs to be returned to the user, but intermediate attempts can also be stored if you want a richer pool of failure memories.

## Helper Utilities

- `run_agent(...)`: your environment-specific runner (BrowserGym, SWE-Bench shell, etc.).
- `pick_best_trajectory(...)`: prompts an evaluator model to choose the best rollout; see Appendix A.3 prompts in the paper for guidance.
- `ingest_memories(...)`: loops over trajectories and calls `extract_memory` with `async_mode=False` when you need immediate availability.
- Use different `agent_id` values if you want to separate MaTTS experiments from standard runs.

By keeping the orchestration outside the MCP server, you can reproduce the multi-trajectory evaluations from the paper while still benefiting from the shared memory storage and retrieval APIs.
