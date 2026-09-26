"""Local E5 embeddings through LangChain's Hugging Face adapter.

Vectors come from the local ``sentence-transformers`` model cache; no remote
embedding endpoint is ever called. The model is loaded lazily on the first
embed/tokenizer access and kept for the lifetime of the embedder instance, so
importing this module or constructing the embedder downloads nothing. The
dependency is a production install; the import is guarded so unrelated
imports and tests stay importable without the (torch-heavy) library.
"""

from __future__ import annotations

from typing import Sequence

from apps.main_api.errors import RetrievalUnavailable

try:  # production dependency; absent only in minimal test environments
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # pragma: no cover - exercised only in minimal environments
    HuggingFaceEmbeddings = None  # type: ignore[assignment,misc]

E5_MODEL_NAME = "intfloat/multilingual-e5-base"
E5_DIMENSION = 768


class E5QueryPassageFormatter:
    """E5 prompting: queries carry ``query: `` and passages ``passage: ``."""

    def query(self, text: str) -> str:
        return f"query: {text}"

    def passage(self, text: str) -> str:
        return f"passage: {text}"


class LocalE5Embedder:
    """768-dimensional normalized E5 vectors from the local model cache."""

    model_name = E5_MODEL_NAME
    dimension = E5_DIMENSION
    _formatter = E5QueryPassageFormatter()

    def __init__(
        self,
        model_name: str = E5_MODEL_NAME,
        device: str | None = None,
        local_files_only: bool = True,
    ):
        if model_name != E5_MODEL_NAME:
            raise ValueError("Fishora ingestion and query embeddings require intfloat/multilingual-e5-base")
        self._device = device
        # Deployment mode: only the locally cached model is used; nothing is
        # downloaded implicitly. Opt out explicitly to fetch from the hub.
        self._local_files_only = local_files_only
        self._model = None  # loaded lazily on first use, once per instance

    def _load(self):
        if HuggingFaceEmbeddings is None:
            # A domain error, not a bare RuntimeError: unhandled it became a 500
            # whose body named an internal package to whoever called the API.
            raise RetrievalUnavailable(
                "langchain-huggingface is not installed; install the fishora production dependencies"
            )
        if self._model is None:
            self._model = HuggingFaceEmbeddings(
                model=self.model_name,
                model_kwargs={
                    "device": self._device,
                    "local_files_only": self._local_files_only,
                },
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._model

    def warmup(self) -> None:
        """Load the weights and run one query, so the first card request does
        not pay the model load (about 11 s cold, W7)."""
        self.embed_query("warmup")

    @property
    def tokenizer(self):
        """Real tokenizer (encode/decode) of the loaded model, for chunking."""
        model = self._load()
        client = getattr(model, "client", None) or model._client
        return client.tokenizer

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        values = self._load().embed_documents(
            [self._formatter.passage(text) for text in texts]
        )
        return [[float(value) for value in row] for row in values]

    def embed_query(self, text: str) -> list[float]:
        value = self._load().embed_query(self._formatter.query(text))
        return [float(item) for item in value]
