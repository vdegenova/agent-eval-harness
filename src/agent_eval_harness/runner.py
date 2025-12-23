from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from .agents import Agent
from .models import Action, Outcome, OutcomeStatus, RunTrace, Task, TraceStep
from .tools import ToolRegistry


def evaluate_task(
    task: Task,
    agent: Agent,
    tools: ToolRegistry,
    max_steps: int = 50,
    plan=None,
) -> RunTrace:
    """
    Executes an agent loop for a task and records the trace.
    """
    run = RunTrace(
        run_id=str(uuid.uuid4()),
        task=task,
        agent_name=agent.name,
        plan=plan or [],
    )

    action: Optional[Action] = agent.begin_task(task)
    step_count = 0

    while action and step_count < max_steps:
        observation = tools.invoke(action.tool_name or "", action.input)
        # attach action id to observation for traceability
        observation.action_id = action.id
        run.steps.append(
            TraceStep(
                action=action,
                observation=observation,
            )
        )
        step_count += 1
        action = agent.handle_observation(observation)

    run.outcome = Outcome(
        status=OutcomeStatus.success,
        summary=agent.finalize(task),
        metrics={
            "steps": step_count,
            "total_cost": run.total_cost,
            "total_latency_ms": run.total_latency_ms,
        },
    )
    run.finished_at = datetime.utcnow()
    return run
