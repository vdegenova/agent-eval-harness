from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import typer
import yaml

from .agents import EchoAgent, SingleShotOpenAIAgent
from .models import Budget, Task, ToolSpec
from .openai_tools import register_openai_chat_tool
from .runner import evaluate_task
from .tools import ToolRegistry

app = typer.Typer(help="Agentic evaluation harness CLI")


def _load_task_from_file(path: Path) -> Task:
    payload: Dict[str, Any] = yaml.safe_load(path.read_text())
    budget = None
    if b := payload.get("budget"):
        budget = Budget(**b)
    return Task(
        id=payload.get("id") or path.stem,
        goal=payload["goal"],
        context=payload.get("context", {}),
        constraints=payload.get("constraints", []),
        tools=payload.get("tools", []),
        budget=budget,
    )


@app.command()
def demo(goal: str = "Say hello world") -> None:
    """
    Runs a tiny demo using the built-in EchoAgent and echo tool.
    """
    task = Task(id="demo-task", goal=goal, tools=["echo"])
    tools = ToolRegistry()
    tools.register(
        ToolSpec(
            name="echo",
            description="Returns the message it receives.",
            input_schema={"type": "object", "properties": {"message": {"type": "string"}}},
            mock=True,
        )
    )
    agent = EchoAgent(tool_name="echo")
    trace = evaluate_task(task=task, agent=agent, tools=tools)
    typer.echo(json.dumps(trace.model_dump(), indent=2, default=str))


@app.command("demo-openai")
def demo_openai(
    goal: str = "Summarize the benefits of exercise",
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
) -> None:
    """
    Runs a one-shot OpenAI chat demo. Requires OPENAI_API_KEY in the environment.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise typer.BadParameter("OPENAI_API_KEY is not set.")

    task = Task(id="openai-demo", goal=goal, tools=["openai_chat"])
    tools = ToolRegistry()
    register_openai_chat_tool(
        tools,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    agent = SingleShotOpenAIAgent(tool_name="openai_chat")
    trace = evaluate_task(task=task, agent=agent, tools=tools)
    typer.echo(json.dumps(trace.model_dump(), indent=2, default=str))


@app.command()
def run_scenario(file: Path) -> None:
    """
    Load a YAML scenario describing a task and run it with EchoAgent.
    Replace EchoAgent with your own implementation to evaluate real agents.
    """
    task = _load_task_from_file(file)
    tools = ToolRegistry()

    # For now, every tool is mocked unless a handler is registered by a real agent.
    for tool_name in task.tools:
        tools.register(
            ToolSpec(
                name=tool_name,
                description=f"Mock tool {tool_name}",
                mock=True,
            )
        )
    agent = EchoAgent(tool_name=task.tools[0] if task.tools else "echo")
    trace = evaluate_task(task=task, agent=agent, tools=tools)
    typer.echo(json.dumps(trace.model_dump(), indent=2, default=str))


if __name__ == "__main__":
    app()
