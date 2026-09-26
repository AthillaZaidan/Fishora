"""The chat model one knowledge card runs on."""


def card_llm(deps, settings, session_id: str | None):
    """The injected LLM port when present (tests, alternative providers; W2),
    else the OpenCode Go client built from settings with this card's session id
    (a fresh one when None). None when no key is configured."""
    if getattr(deps, "llm", None) is not None:
        return deps.llm
    if settings is None or not settings.opencode_go_api_key.get_secret_value().strip():
        return None
    from apps.main_api.services.generation import make_opencode_go_llm

    return make_opencode_go_llm(settings, session_id=session_id)
