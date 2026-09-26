"""OpenCode Go client contract (W20) and the blank-key path (W3)."""

import pytest
from pydantic import SecretStr

from apps.main_api.config import MainSettings
from apps.main_api.contracts import RetrievedChunk, SpeciesRecord
from apps.main_api.errors import OpenCodeUnavailable
from apps.main_api.services.generation import (
    USER_AGENT,
    KnowledgeGenerator,
    OpenCodeGoClient,
    make_opencode_go_llm,
)


def _settings(key: str) -> MainSettings:
    return MainSettings(_env_file=None, database_url="postgresql+psycopg://t@localhost/t",
                        opencode_go_api_key=SecretStr(key))


def test_client_sends_session_and_user_agent():
    llm = make_opencode_go_llm(_settings("test-key"), session_id="fishora-card-abc")
    assert llm.default_headers["x-opencode-session"] == "fishora-card-abc"
    assert llm.default_headers["User-Agent"] == USER_AGENT


def test_client_without_session_gets_a_fresh_one():
    a = make_opencode_go_llm(_settings("test-key")).default_headers["x-opencode-session"]
    b = make_opencode_go_llm(_settings("test-key")).default_headers["x-opencode-session"]
    assert a and b and a != b


def test_blank_key_is_a_provider_outage_not_a_crash():
    species = SpeciesRecord("species_nila", "nila", "Nila", "Oreochromis niloticus", "species", "VERIFIED_TAXONOMY", None)
    chunk = RetrievedChunk(
        chunk_id="c1", species_id="species_nila", source_id="s1", source_type="t", category="taste_texture",
        content="mild", distance=0.1, chunk_verification_status="verified", source_verification_status="verified",
        source_title="T", source_publisher=None, source_url=None, source_reviewed_at=None,
    )
    generator = KnowledgeGenerator(lambda: OpenCodeGoClient(_settings("")))
    with pytest.raises(OpenCodeUnavailable):
        generator.generate(species, [chunk])
