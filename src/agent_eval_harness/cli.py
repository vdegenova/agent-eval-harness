from __future__ import annotations

import json
import os
from pathlib import Path
import re
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


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _load_pantry() -> list[dict[str, Any]]:
    pantry_path = _repo_root() / "data" / "pantry.json"
    if not pantry_path.exists():
        return []
    try:
        return json.loads(pantry_path.read_text())
    except Exception:  # noqa: BLE001
        return []


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug or "recipe"


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


@app.command("demo-openai-recipe")
def demo_openai_recipe(
    goal: str = (
        "Plan a 20-minute vegetarian pasta dinner. "
        "Call tools to check pantry, build a shopping list, draft steps, make a timer plan, "
        "and save notes."
    ),
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
    pretty: bool = typer.Option(False, help="Pretty print the trace"),
) -> None:
    """
    Multi-turn recipe planner using real (non-mock) tools plus OpenAI.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise typer.BadParameter("OPENAI_API_KEY is not set.")

    pantry = _load_pantry()
    tools = ToolRegistry()

    def pantry_lookup(payload: Dict[str, Any]) -> Dict[str, Any]:
        dish = payload.get("dish", "")
        matches = pantry
        return {"dish": dish, "available": matches}

    def shopping_list(payload: Dict[str, Any]) -> Dict[str, Any]:
        needed = payload.get("ingredients", []) or []
        have = {item.get("name", "").lower(): item for item in pantry}
        missing = [item for item in needed if item.lower() not in have]
        return {"missing": missing, "have": list(have.keys())}

    def recipe_steps(payload: Dict[str, Any]) -> Dict[str, Any]:
        dish = payload.get("dish") or "dish"
        dietary = payload.get("dietary_notes", "")
        prompt = (
            f"Create concise numbered cooking steps (<=8 steps, <=120 words) for {dish}."
            f" Dietary notes: {dietary or 'none'}."
        )
        from openai import OpenAI

        openai_client = OpenAI(api_key=api_key, base_url=base_url)
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You are a concise recipe writer."}, {"role": "user", "content": prompt}],
        )
        content = resp.choices[0].message.content or ""
        steps = [line.strip(" -") for line in content.split("\n") if line.strip()]
        return {"steps": steps, "raw": content}

    def timer_plan(payload: Dict[str, Any]) -> Dict[str, Any]:
        steps = payload.get("steps") or []
        schedule = []
        current = 0
        for idx, step in enumerate(steps, start=1):
            duration = 3 if idx % 2 == 0 else 4
            schedule.append(
                {"step": step, "start_min": current, "end_min": current + duration, "duration_min": duration}
            )
            current += duration
        return {"schedule": schedule, "total_minutes": current}

    def note_run(payload: Dict[str, Any]) -> Dict[str, Any]:
        dish = payload.get("dish") or "dish"
        notes = payload.get("notes") or ""
        notes_dir = _repo_root() / "recipes_log"
        notes_dir.mkdir(parents=True, exist_ok=True)
        path = notes_dir / f"{_safe_slug(dish)}.md"
        path.write_text(notes)
        return {"saved_to": str(path)}

    tools.register(
        ToolSpec(
            name="pantry_lookup",
            description="Check pantry items available for the dish.",
            input_schema={
                "type": "object",
                "properties": {"dish": {"type": "string"}},
                "required": ["dish"],
            },
            mock=False,
        ),
        handler=pantry_lookup,
    )
    tools.register(
        ToolSpec(
            name="shopping_list",
            description="Compare needed ingredients to pantry and return missing items.",
            input_schema={
                "type": "object",
                "properties": {"ingredients": {"type": "array", "items": {"type": "string"}}},
                "required": ["ingredients"],
            },
            mock=False,
        ),
        handler=shopping_list,
    )
    tools.register(
        ToolSpec(
            name="recipe_steps",
            description="Draft concise cooking steps using OpenAI.",
            input_schema={
                "type": "object",
                "properties": {
                    "dish": {"type": "string"},
                    "dietary_notes": {"type": "string"},
                },
                "required": ["dish"],
            },
            mock=False,
        ),
        handler=recipe_steps,
    )
    tools.register(
        ToolSpec(
            name="timer_plan",
            description="Turn steps into a simple timed schedule.",
            input_schema={
                "type": "object",
                "properties": {"steps": {"type": "array", "items": {"type": "string"}}},
                "required": ["steps"],
            },
            mock=False,
        ),
        handler=timer_plan,
    )
    tools.register(
        ToolSpec(
            name="note_run",
            description="Persist notes for the dish to a local log file.",
            input_schema={
                "type": "object",
                "properties": {"dish": {"type": "string"}, "notes": {"type": "string"}},
                "required": ["dish", "notes"],
            },
            mock=False,
        ),
        handler=note_run,
    )

    task = Task(
        id="openai-recipe-demo",
        goal=goal,
        tools=["pantry_lookup", "shopping_list", "recipe_steps", "timer_plan", "note_run"],
    )
    agent = MultiTurnOpenAIAgent(
        tools_registry=tools,
        tool_names=task.tools,
        model=model,
        api_key=api_key,
        base_url=base_url,
        system_prompt=(
            "You are a cooking planner. Use tools in a sensible order: pantry_lookup, shopping_list, "
            "recipe_steps, timer_plan, note_run. Call tools individually, multiple steps if needed. "
            "Only give the final answer after note_run."
        ),
    )
    trace = evaluate_task(task=task, agent=agent, tools=tools, max_steps=10)
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
