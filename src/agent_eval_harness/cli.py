from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import typer
import yaml

from .agents import EchoAgent, MultiTurnOpenAIAgent, SingleShotOpenAIAgent
from .models import Budget, RunTrace, Task, ToolSpec
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


def _print_trace(trace: RunTrace, pretty: bool = False) -> None:
    if not pretty:
        typer.echo(json.dumps(trace.model_dump(), indent=2, default=str))
        return

    try:
        from rich import box
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except ImportError:  # pragma: no cover - fallback path
        typer.echo(json.dumps(trace.model_dump(), indent=2, default=str))
        return

    console = Console()
    header = Text(f"Run {trace.run_id}", style="bold") + Text(
        f" | agent: {trace.agent_name} | steps: {len(trace.steps)}", style="dim"
    )
    console.print(header)

    summary = Table(box=box.SIMPLE, show_header=False)
    summary.add_row("Task", trace.task.goal)
    summary.add_row("Outcome", trace.outcome.summary if trace.outcome else "N/A")
    summary.add_row("Status", trace.outcome.status.value if trace.outcome else "unknown")
    summary.add_row("Total cost", f"{trace.total_cost:.4f}")
    summary.add_row("Total latency (ms)", str(trace.total_latency_ms))
    console.print(Panel(summary, title="Summary", expand=False))

    table = Table(title="Steps", box=box.MINIMAL_DOUBLE_HEAD)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Type")
    table.add_column("Tool")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Status")
    table.add_column("Output", overflow="fold")

    for idx, step in enumerate(trace.steps, start=1):
        action = step.action
        obs = step.observation
        action_type = getattr(action.type, "value", str(action.type))
        tool_name = action.tool_name or "-"
        latency = str(obs.latency_ms or 0)
        cost = f"{(obs.cost or 0):.4f}"
        status = "error" if obs.error else "ok"
        output = obs.error or _shorten_output(obs.output)
        table.add_row(str(idx), action_type, tool_name, latency, cost, status, output)

    console.print(table)


def _shorten_output(output: Any, limit: int = 120) -> str:
    text = ""
    try:
        if isinstance(output, (dict, list)):
            text = json.dumps(output)
        else:
            text = str(output)
    except Exception:  # noqa: BLE001
        text = "<unprintable output>"
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


@app.command()
def demo(goal: str = "Say hello world", pretty: bool = typer.Option(False, help="Pretty print the trace")) -> None:
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
    _print_trace(trace, pretty=pretty)


@app.command("demo-openai")
def demo_openai(
    goal: str = "Summarize the benefits of exercise",
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
    pretty: bool = typer.Option(False, help="Pretty print the trace"),
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
    _print_trace(trace, pretty=pretty)


@app.command("demo-openai-multi")
def demo_openai_multi(
    goal: str = "Plan dinner and get a quick recipe idea",
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
    pretty: bool = typer.Option(False, help="Pretty print the trace"),
) -> None:
    """
    Runs a multi-turn OpenAI demo with function calling over mocked tools.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise typer.BadParameter("OPENAI_API_KEY is not set.")

    tools = ToolRegistry()

    # Mock tools for the demo
    tools.register(
        ToolSpec(
            name="get_ingredients",
            description="Return ingredients for a simple dish.",
            input_schema={
                "type": "object",
                "properties": {"dish": {"type": "string"}},
                "required": ["dish"],
            },
            mock=True,
        ),
        handler=lambda payload: {"dish": payload["dish"], "ingredients": ["pasta", "tomato", "garlic"]},
    )
    tools.register(
        ToolSpec(
            name="get_steps",
            description="Return short cooking steps.",
            input_schema={
                "type": "object",
                "properties": {"dish": {"type": "string"}},
                "required": ["dish"],
            },
            mock=True,
        ),
        handler=lambda payload: {"dish": payload["dish"], "steps": ["Boil pasta", "Make sauce", "Combine"]},
    )

    task = Task(id="openai-multi-demo", goal=goal, tools=["get_ingredients", "get_steps"])
    agent = MultiTurnOpenAIAgent(
        tools_registry=tools,
        tool_names=task.tools,
        model=model,
        api_key=api_key,
        base_url=base_url,
        system_prompt="You are a planning assistant. Decide which tools to call to plan a quick meal.",
    )
    trace = evaluate_task(task=task, agent=agent, tools=tools, max_steps=5)
    _print_trace(trace, pretty=pretty)


@app.command()
def run_scenario(file: Path, pretty: bool = typer.Option(False, help="Pretty print the trace")) -> None:
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
    _print_trace(trace, pretty=pretty)


if __name__ == "__main__":
    app()
