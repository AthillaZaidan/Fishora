"""Read an LLM reply as text and as one JSON object, whatever its wrapping.

Two shapes broke the agent path (findings W21 and W4): with the Responses API
``AIMessage.content`` is a list of content blocks, not a string, and chat
models often wrap JSON in a Markdown code fence. Every orchestrator node reads
replies through these two functions instead of ``json.loads(raw.content)``.
"""

from __future__ import annotations

import json
import re

_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?\s*```\s*$", re.S)


def reply_text(reply) -> str:
    """Plain text of a LangChain message, a list of content blocks, or a string."""
    if isinstance(reply, str):
        return reply
    text = getattr(reply, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(reply, "content", reply)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


def reply_json(reply) -> dict:
    """The first JSON object in a reply. Raises ValueError when there is none."""
    if isinstance(reply, dict):
        return reply
    text = reply_text(reply).strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    decoder = json.JSONDecoder()
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("reply contains no JSON object")
