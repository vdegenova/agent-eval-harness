from __future__ import annotations

import abc
from typing import Optional

from .models import Action, Observation, Task


class Agent(abc.ABC):
    """
    Minimal agent interface. Agents should decide actions based on observations.
    """

    name: str

    @abc.abstractmethod
    def begin_task(self, task: Task) -> Action:
        """
        Called once at the start of a task.
        """

    @abc.abstractmethod
    def handle_observation(self, observation: Observation) -> Optional[Action]:
        """
        Called after each tool invocation. Return next action or None to stop.
        """

    def finalize(self, task: Task) -> str:
        """
        Final natural language answer or report. Override for richer behavior.
        """
        return "Task finished."


class EchoAgent(Agent):
    """
    Trivial agent used for demonstrations; echoes the goal through a tool call.
    """

    name = "echo-agent"

    def __init__(self, tool_name: str = "echo") -> None:
        self.tool_name = tool_name
        self.task: Task | None = None

    def begin_task(self, task: Task) -> Action:
        self.task = task
        return Action(
            id="action-1",
            task_id=task.id,
            agent_name=self.name,
            type="tool",
            tool_name=self.tool_name,
            input={"message": task.goal, "context": task.context},
            thought="Echoing task goal via tool.",
        )

    def handle_observation(self, observation: Observation) -> Optional[Action]:
        return None

    def finalize(self, task: Task) -> str:
        return f"Echoed: {task.goal}"


class SingleShotOpenAIAgent(Agent):
    """
    One-shot agent that sends the task goal to an OpenAI chat tool.
    """

    name = "openai-single-shot"

    def __init__(self, tool_name: str = "openai_chat", system_prompt: str | None = None) -> None:
        self.tool_name = tool_name
        self.system_prompt = system_prompt or "You are a helpful agent. Solve the user's task."
        self._last_message: str | None = None

    def begin_task(self, task: Task) -> Action:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task.goal},
        ]
        return Action(
            id="action-1",
            task_id=task.id,
            agent_name=self.name,
            type="tool",
            tool_name=self.tool_name,
            input={"messages": messages},
            thought="Send task to OpenAI chat model.",
        )

    def handle_observation(self, observation: Observation) -> Optional[Action]:
        message = observation.output.get("message") if isinstance(observation.output, dict) else None
        if message:
            self._last_message = message.get("content")
        return None

    def finalize(self, task: Task) -> str:
        return self._last_message or "No response from model."
