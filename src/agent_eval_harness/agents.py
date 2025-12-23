from __future__ import annotations

import abc
import json
from typing import Optional

from openai import OpenAI

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


class MultiTurnOpenAIAgent(Agent):
    """
    Multi-step agent that uses OpenAI function calling to decide tool invocations.
    """

    name = "openai-multi"

    def __init__(
        self,
        tools_registry,
        tool_names,
        model: str = "gpt-4o-mini",
        system_prompt: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        from .openai_tools import build_openai_tools_from_registry

        self.tool_names = tool_names
        self.tools_registry = tools_registry
        self.tool_definitions = build_openai_tools_from_registry(tools_registry, tool_names)
        self.system_prompt = system_prompt or "You are an agent that plans and calls tools to solve the task."
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.messages: list[dict] = []
        self._last_answer: str | None = None
        self._task: Task | None = None
        self._pending_tool_call_id: str | None = None
        self._pending_tool_name: str | None = None
        self._counter = 0
        self.model = model

    def begin_task(self, task: Task) -> Action | None:
        self._task = task
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task.goal},
        ]
        if task.context:
            self.messages.append({"role": "user", "content": f"Context: {json.dumps(task.context)}"})
        if task.constraints:
            self.messages.append({"role": "user", "content": f"Constraints: {task.constraints}"})
        return self._next_action()

    def handle_observation(self, observation: Observation) -> Optional[Action]:
        # Attach tool result and ask the model what to do next
        content = observation.output
        if observation.error:
            content = {"error": observation.error, "output": observation.output}
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": self._pending_tool_call_id or "tool-call",
                "name": self._pending_tool_name or "",
                "content": json.dumps(content),
            }
        )
        self._pending_tool_call_id = None
        self._pending_tool_name = None
        return self._next_action()

    def _next_action(self) -> Optional[Action]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.tool_definitions,
            parallel_tool_calls=False,
        )
        message = response.choices[0].message
        msg_dict = message.model_dump(exclude_none=True)

        # If multiple tool calls are returned, keep only the first to avoid missing responses.
        tool_calls = msg_dict.get("tool_calls") or []
        if tool_calls:
            msg_dict["tool_calls"] = [tool_calls[0]]
        self.messages.append(msg_dict)

        if tool_calls:
            call = tool_calls[0]
            # call may be a dict or a Pydantic object; normalize to dict access.
            fn = call.get("function") if isinstance(call, dict) else getattr(call, "function", None)
            call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", "")
            args_raw = ""
            fn_name = ""
            if fn:
                if isinstance(fn, dict):
                    args_raw = fn.get("arguments") or "{}"
                    fn_name = fn.get("name") or ""
                else:
                    args_raw = getattr(fn, "arguments", "") or "{}"
                    fn_name = getattr(fn, "name", "") or ""
            args = {}
            try:
                args = json.loads(args_raw)
            except Exception:  # noqa: BLE001
                args = {}
            self._pending_tool_call_id = call_id or "tool-call"
            self._pending_tool_name = fn_name or ""
            self._counter += 1
            return Action(
                id=f"action-{self._counter}",
                task_id=self._task.id if self._task else "unknown",
                agent_name=self.name,
                type="tool",
                tool_name=fn_name,
                input=args,
                thought=f"Invoke tool {fn_name or 'unknown'}",
            )

        # No more tool calls; finalize
        self._last_answer = message.content or "No response."
        return None

    def finalize(self, task: Task) -> str:
        return self._last_answer or "No response."


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
