# agent-eval-harness

Prototype framework for evaluating agentic workflows (tasks, plans, actions, outcomes), not single prompts.

## What’s included
- Typed data model (`Task`, `PlanStep`, `Action`, `Observation`, `Outcome`, `RunTrace`, `ToolSpec`, `Budget`).
- Agent interface with reference agents:
  - `EchoAgent` (mock echo tool) for wiring/tests.
  - `SingleShotOpenAIAgent` (OpenAI Chat API) for single-shot tasks.
  - `MultiTurnOpenAIAgent` (OpenAI function calling) for multi-step tool use.
- Tool registry with mock support and an OpenAI chat tool helper.
- Runner that records traces and basic metrics (steps, cost, latency).
- Typer CLI (`agent-eval`) with demo commands and YAML-driven scenarios.

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```
Requires Python 3.11+ (or 3.12 via Homebrew) and network for OpenAI calls.

## CLI commands
- Mock demo (no network):
  ```bash
  agent-eval demo --goal "Plan a weekend hike" --pretty
  ```
- OpenAI demo (requires `OPENAI_API_KEY`):
  ```bash
  export OPENAI_API_KEY=sk-...
  agent-eval demo-openai --goal "Summarize the agent-eval-harness repo" --model gpt-4o-mini --pretty
  ```
- Multi-turn cooking demo with real tools (pantry lookup, shopping list, recipe steps via OpenAI, timer plan, note save):
  ```bash
  agent-eval demo-openai-recipe --goal "Plan a 20-minute vegetarian pasta dinner with a shopping list and timer plan" --model gpt-4o-mini --pretty
  ```
- Multi-turn OpenAI demo with tool calls (requires `OPENAI_API_KEY`):
  ```bash
  agent-eval demo-openai-multi --goal "Plan dinner and get a quick recipe idea" --model gpt-4o-mini --pretty
  ```
- Run a YAML scenario (uses `EchoAgent` by default):
  ```bash
  agent-eval run-scenario scenario.yml --pretty
  ```

Example scenario (`scenario.yml`):
```yaml
id: file-search
goal: Find the error rate in today's logs
context:
  date: 2024-06-01
constraints:
  - Stay within budget
tools: [search_logs, summarize]
budget:
  total: 0.25
  unit: usd
```

## How it works (end-to-end)
1) CLI builds a `Task` (goal/context/tools/budget).
2) Tools are registered in a `ToolRegistry` (mock echo, real OpenAI chat, or your handlers).
3) Agent produces the first `Action` (`EchoAgent` → echo; `SingleShotOpenAIAgent` → OpenAI chat; `MultiTurnOpenAIAgent` → OpenAI with function calling).
4) Runner invokes the tool, records `Action` + `Observation` in `RunTrace`, and loops until the agent stops or `max_steps` is hit.
5) Outcome is marked success with metrics and the agent’s final message. The CLI prints a JSON trace (or a pretty table with `--pretty`).

## Multi-turn recipe demo (real tools)
- Tools:
  - `pantry_lookup`: reads `data/pantry.json` to list available ingredients.
  - `shopping_list`: compares needed ingredients to pantry, returns missing items.
  - `recipe_steps`: calls OpenAI to draft concise steps for a dish.
  - `timer_plan`: turns steps into a timed schedule.
  - `note_run`: saves notes to `recipes_log/<dish>.md`.
- Agent: `MultiTurnOpenAIAgent` with OpenAI function calling; instructed to call tools in sensible order and only finalize after `note_run`.
- Run:
  ```bash
  agent-eval demo-openai-recipe --goal "Plan a 20-minute vegetarian pasta dinner with a shopping list and timer plan" --model gpt-4o-mini --pretty
  ```
- Outputs: rich CLI table plus a saved note file for the dish.

## Extending
- Add tools: register a `ToolSpec` + handler via `ToolRegistry.register`.
- Add agents: subclass `Agent` and implement `begin_task`, `handle_observation`, `finalize`.
- Swap tool backends: use `register_openai_chat_tool` for OpenAI or provide your own handlers.
- Wire scoring/replay: extend the runner to enforce budgets, seed randomness, and store traces.

## Limitations (current)
- Single-step agents only; no planning or multi-action loops yet.
- Tools default to mocks unless you register real handlers.
- No scoring/evaluation heuristics, persistence, or deterministic replay.
- No test suite/CI wired up yet.

## Roadmap ideas
- Multi-step agent loop with plan/status updates and budget enforcement.
- Pluggable scoring (success heuristics, cost/latency caps, safety checks).
- Trace persistence + replay for regressions.
- Adapters for multiple model providers (OpenAI, local, others) behind `Agent`.
