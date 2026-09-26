"""Shared fixtures for the Fishora RAG test suite.

Layout (pytest markers in parentheses):

    unit/         pure logic, fake tokenizer/LLM, no model   (fast)
    retrieval/    real E5 + real corpus, quality gates        (e5)
    generation/   critic + full run_graph with scripted LLM   (e5)
    integration/  FastAPI app with in-memory ports            (api, e5)
    e2e/          identify -> verify -> knowledge card, dashboard (e5)

``e5`` tests skip cleanly when the local model cache is missing.
"""

from __future__ import annotations

import io
import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")


def pytest_collection_modifyitems(config, items):
    for item in items:
        layer = item.path.parent.name
        # The repo's own "integration" marker means "needs Postgres"; these
        # in-memory API tests do not, so they are tagged "api" instead.
        marker = {"integration": "api"}.get(layer, layer)
        if marker in {"unit", "retrieval", "generation", "api", "e2e"}:
            item.add_marker(getattr(pytest.mark, marker))


class WordTokenizer:
    """Deterministic whitespace tokenizer implementing the Tokenizer port."""

    def __init__(self):
        self._ids: dict[str, int] = {}
        self._words: dict[int, str] = {}

    def encode(self, text, *, add_special_tokens=True):
        ids = []
        for word in text.split():
            if word not in self._ids:
                self._ids[word] = len(self._ids) + 1
                self._words[self._ids[word]] = word
            ids.append(self._ids[word])
        return ids

    def decode(self, ids, *, skip_special_tokens=True):
        return " ".join(self._words[i] for i in ids)


@pytest.fixture
def word_tokenizer():
    return WordTokenizer()


@pytest.fixture(scope="session")
def e5():
    from apps.main_api.services.embeddings import LocalE5Embedder

    embedder = LocalE5Embedder()
    try:
        embedder.embed_query("warmup")
    except Exception as exc:  # model cache or torch stack missing
        pytest.skip(f"local E5 model unavailable: {type(exc).__name__}")
    return embedder


@pytest.fixture(scope="session")
def store(e5):
    from evals.corpus import build_store

    return build_store(e5)


@pytest.fixture(scope="session")
def species():
    from evals.corpus import species_records

    return species_records()


@pytest.fixture
def scripted_llm():
    from evals.pipeline_eval import _scripted_llm

    return _scripted_llm(delay=0.0, fenced=False)


def png_bytes() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (40, 90, 160)).save(buffer, format="PNG")
    return buffer.getvalue()


def sign_in(client, username: str = "rian", password: str = "demo"):
    """Sign a TestClient in (the operator by default): the fish, jobs and
    quality routes answer 401 without a session."""
    response = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return client


@pytest.fixture
def app_factory(e5, store, species, scripted_llm):
    """Build the real FastAPI app with every external port in memory."""
    from apps.main_api.main import create_main_app
    from apps.main_api.ports import AppDependencies
    from evals.fakes import (
        FixedCVClient,
        InMemoryImageStore,
        InMemoryJobRepository,
        InMemoryPredictionRepository,
        InMemorySpeciesRepository,
    )

    def build(label: str = "nila", llm=scripted_llm):
        deps = AppDependencies(
            cv_client=FixedCVClient(label),
            species_repo=InMemorySpeciesRepository(species),
            prediction_repo=InMemoryPredictionRepository(),
            image_store=InMemoryImageStore(),
            embedder=e5,
            knowledge_repo=store,
            job_repo=InMemoryJobRepository(),
        )
        deps.llm = llm  # the LLM port; ignored by code that has none
        return create_main_app(deps=deps), deps

    return build
