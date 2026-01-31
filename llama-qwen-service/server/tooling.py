import json
from dataclasses import dataclass
from typing import Any, Optional

TOOL_SCHEMA = {
    "name": "fetch_url",
    "description": "Fetch the raw text of a URL over HTTP(S).",
    "args": {
        "url": "https://example.com"
    },
}


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]


def parse_tool_call(raw_text: str) -> Optional[ToolCall]:
    """Parse a tool call JSON object from a model response."""
    raw_text = raw_text.strip()
    if not raw_text.startswith("{"):
        return None
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("tool") != TOOL_SCHEMA["name"]:
        return None
    args = payload.get("args", {})
    if not isinstance(args, dict) or "url" not in args:
        return None
    return ToolCall(name=payload["tool"], args=args)
