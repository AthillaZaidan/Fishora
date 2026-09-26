from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# Both spellings, since tools disagree on which they resolve.
DEFAULT_CORS_ALLOW_ORIGINS = (
    "http://localhost:3000,http://127.0.0.1:3000,"
    "http://localhost:3111,http://127.0.0.1:3111"
)


def parse_origins(value: str) -> list[str]:
    """Split a comma-separated origin list. Blank denies every origin."""
    return [origin.strip() for origin in value.split(",") if origin.strip()]


class MainSettings(BaseSettings):
    """Application settings.

    Secrets are typed as SecretStr and only exposed via get_secret_value(),
    so repr()/serialization never leaks them. The OpenCode API key is blank by
    default; it is required only when the production OpenCode client is
    constructed (a later task), not for settings construction or unit tests.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = Field(validation_alias=AliasChoices("FISHORA_DATABASE_URL", "database_url"))
    cv_service_url: str = Field(default="http://localhost:8001", validation_alias=AliasChoices("FISHORA_CV_SERVICE_URL", "cv_service_url"))
    cv_timeout_seconds: float = Field(default=30.0, validation_alias=AliasChoices("FISHORA_CV_TIMEOUT_SECONDS", "cv_timeout_seconds"))
    cv_max_image_bytes: int = Field(default=10 * 1024 * 1024, validation_alias=AliasChoices("FISHORA_CV_MAX_IMAGE_BYTES", "cv_max_image_bytes"))
    image_storage_dir: Path = Field(default=Path("data/images"), validation_alias=AliasChoices("FISHORA_IMAGE_STORAGE_DIR", "image_storage_dir"))
    embedding_model_name: str = Field(default="intfloat/multilingual-e5-base", validation_alias=AliasChoices("FISHORA_EMBEDDING_MODEL_NAME", "embedding_model_name"))
    embedding_dimension: int = Field(default=768, validation_alias=AliasChoices("FISHORA_EMBEDDING_DIMENSION", "embedding_dimension"))
    embedding_device: str = Field(default="cpu", validation_alias=AliasChoices("FISHORA_EMBEDDING_DEVICE", "embedding_device"))
    opencode_go_base_url: str = Field(default="https://opencode.ai/zen/go/v1", validation_alias=AliasChoices("FISHORA_OPENCODE_GO_BASE_URL", "opencode_go_base_url"))
    # Blank key allowed; the production OpenCode client constructor enforces it.
    opencode_go_api_key: SecretStr = Field(default=SecretStr(""), validation_alias=AliasChoices("OPENCODE_GO_API_KEY", "opencode_go_api_key"))
    opencode_go_model: str = Field(default="gpt-6-luna", validation_alias=AliasChoices("FISHORA_OPENCODE_GO_MODEL", "opencode_go_model"))
    opencode_go_timeout_seconds: float = Field(default=60.0, validation_alias=AliasChoices("FISHORA_OPENCODE_GO_TIMEOUT_SECONDS", "opencode_go_timeout_seconds"))
    # Responses-API reasoning effort for every card LLM call (none|low|medium|high|xhigh|max);
    # "medium" was the knee of the effort sweep (reports/iteration-2/effort_sweep.json).
    opencode_go_reasoning_effort: str | None = Field(default="medium", validation_alias=AliasChoices("FISHORA_OPENCODE_GO_REASONING_EFFORT", "opencode_go_reasoning_effort"))
    # A plain string: pydantic-settings would JSON-parse a list[str] field.
    cors_allow_origins: str = Field(
        default=DEFAULT_CORS_ALLOW_ORIGINS,
        validation_alias=AliasChoices("FISHORA_CORS_ALLOW_ORIGINS", "cors_allow_origins"),
    )

    @property
    def cors_origins(self) -> list[str]:
        return parse_origins(self.cors_allow_origins)
