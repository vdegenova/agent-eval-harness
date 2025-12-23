from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from .models import Observation, ToolSpec


class ToolRegistry:
    """
    Simple in-memory tool registry with optional mock implementations.
    """

    def __init__(self) -> None:
        self._specs: Dict[str, ToolSpec] = {}
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

    def register(
        self,
        spec: ToolSpec,
        handler: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> None:
        self._specs[spec.name] = spec
        if handler:
            self._handlers[spec.name] = handler

    def get(self, name: str) -> ToolSpec:
        if name not in self._specs:
            raise KeyError(f"tool '{name}' not registered")
        return self._specs[name]

    def list(self) -> Dict[str, ToolSpec]:
        return dict(self._specs)

    def invoke(self, tool_name: str, payload: Dict[str, Any]) -> Observation:
        spec = self.get(tool_name)
        start = time.perf_counter()
        try:
            if handler := self._handlers.get(tool_name):
                output = handler(payload)
            elif spec.mock:
                output = {"mock": True, "input": payload}
            else:
                raise RuntimeError(f"no handler registered for tool '{tool_name}'")
            error = None
        except Exception as exc:  # noqa: BLE001
            output = None
            error = str(exc)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        cost = spec.cost_per_call or 0.0
        return Observation(
            action_id="",  # populated by runner when wiring trace
            output=output,
            error=error,
            latency_ms=elapsed_ms,
            cost=cost,
        )
