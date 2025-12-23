# agent-eval-harness

Prototype framework for evaluating agentic workflows (tasks, plans, actions, outcomes), not single prompts.

## What’s included
- Typed data model (`Task`, `PlanStep`, `Action`, `Observation`, `Outcome`, `RunTrace`, `ToolSpec`, `Budget`).
- Agent interface with two reference agents:
  - `EchoAgent` (mock echo tool) for wiring/tests.
  - `SingleShotOpenAIAgent` (OpenAI Chat API) for real completions.
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
  agent-eval demo --goal "Plan a weekend hike"
  ```
- OpenAI demo (requires `OPENAI_API_KEY`):
  ```bash
  export OPENAI_API_KEY=sk-...
  agent-eval demo-openai --goal "Summarize the agent-eval-harness repo" --model gpt-4o-mini
  ```
- Run a YAML scenario (uses `EchoAgent` by default):
  ```bash
  agent-eval run-scenario scenario.yml
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
2) Tools are registered in a `ToolRegistry` (`echo` mock or `openai_chat` real API).
3) Agent produces the first `Action` (`EchoAgent` → echo; `SingleShotOpenAIAgent` → OpenAI chat).
4) Runner invokes the tool, records `Action` + `Observation` in `RunTrace`, and stops (single step today).
5) Outcome is marked success with metrics and the agent’s final message. The CLI prints the JSON trace.

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
