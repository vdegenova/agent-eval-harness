from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Budget(BaseModel):
    total: float
    unit: str = "usd"
    spent: float = 0.0

    @property
    def remaining(self) -> float:
        return max(self.total - self.spent, 0.0)


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Optional[Dict[str, Any]] = None
    cost_per_call: Optional[float] = None
    mock: bool = False


class Task(BaseModel):
    id: str
    goal: str
    context: Dict[str, Any] = Field(default_factory=dict)
    constraints: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    budget: Optional[Budget] = None


class PlanStepStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    done = "done"
    failed = "failed"
    skipped = "skipped"


class PlanStep(BaseModel):
    id: str
    description: str
    status: PlanStepStatus = PlanStepStatus.pending
    notes: Optional[str] = None


class ActionType(str, Enum):
    tool = "tool"
    message = "message"
    decision = "decision"


class Action(BaseModel):
    id: str
    task_id: str
    agent_name: str
    type: ActionType
    tool_name: Optional[str] = None
    input: Dict[str, Any] = Field(default_factory=dict)
    thought: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Observation(BaseModel):
    action_id: str
    output: Any
    error: Optional[str] = None
    latency_ms: Optional[int] = None
    cost: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TraceStep(BaseModel):
    action: Action
    observation: Observation


class OutcomeStatus(str, Enum):
    success = "success"
    failure = "failure"
    partial = "partial"
    aborted = "aborted"


class Outcome(BaseModel):
    status: OutcomeStatus
    summary: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    final_answer: Optional[str] = None


class RunTrace(BaseModel):
    run_id: str
    task: Task
    agent_name: str
    steps: List[TraceStep] = Field(default_factory=list)
    plan: List[PlanStep] = Field(default_factory=list)
    outcome: Optional[Outcome] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None

    @property
    def total_cost(self) -> float:
        return sum((step.observation.cost or 0.0) for step in self.steps)

    @property
    def total_latency_ms(self) -> int:
        return sum((step.observation.latency_ms or 0) for step in self.steps)
