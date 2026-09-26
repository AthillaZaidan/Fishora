"""Reading replies: content blocks (W21) and fenced JSON (W4)."""

import pytest
from langchain_core.messages import AIMessage

from apps.main_api.services.llm_output import reply_json, reply_text

PAYLOAD = '{"taste": "Rasa ringan", "sources": []}'


def test_responses_api_content_blocks():
    msg = AIMessage(content=[{"type": "text", "text": PAYLOAD, "annotations": []}])
    assert reply_json(msg)["taste"] == "Rasa ringan"


def test_plain_string_and_message():
    assert reply_json(PAYLOAD)["taste"] == "Rasa ringan"
    assert reply_text(AIMessage(content=PAYLOAD)) == PAYLOAD


@pytest.mark.parametrize("wrapped", [
    f"```json\n{PAYLOAD}\n```",
    f"```\n{PAYLOAD}\n```",
    f"Berikut jawabannya:\n{PAYLOAD}\nSemoga membantu.",
])
def test_fenced_or_prose_wrapped_json(wrapped):
    assert reply_json(wrapped)["taste"] == "Rasa ringan"


def test_no_json_raises():
    with pytest.raises(ValueError):
        reply_json("tidak ada data")
