from __future__ import annotations

from typing import Any, Dict, List, Optional

from openai import OpenAI

from .models import ToolSpec
from .tools import ToolRegistry


def register_openai_chat_tool(
    tools: ToolRegistry,
    model: str = "gpt-4o-mini",
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    tool_name: str = "openai_chat",
) -> str:
    """
    Register an OpenAI chat completion tool in the registry.
    """
    client = OpenAI(api_key=api_key, base_url=base_url)

    def handler(payload: Dict[str, Any]) -> Dict[str, Any]:
        messages: List[Dict[str, str]] = payload.get("messages", [])
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        message = response.choices[0].message
        usage = response.usage.model_dump() if response.usage else None
        return {
            "message": message.model_dump(),
            "usage": usage,
        }

    tools.register(
        ToolSpec(
            name=tool_name,
            description=f"OpenAI chat completion tool using model {model}",
            input_schema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {"type": "object"},
                    }
                },
                "required": ["messages"],
            },
            mock=False,
        ),
        handler=handler,
    )
    return tool_name
