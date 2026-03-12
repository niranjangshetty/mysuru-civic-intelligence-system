"""Application configuration with environment override support."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Application metadata."""

    model_config = SettingsConfigDict(extra="ignore")
    name: str = "Mysuru Civic Intelligence System"
    version: str = "1.0.0"
    debug: bool = False


class PathsConfig(BaseSettings):
    """Path configuration."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)
    documents_dir: str = Field(default="data/documents", alias="DOCUMENTS_DIR")
    index_dir: str = Field(default="data/indices", alias="INDEX_DIR")


class EmbeddingsConfig(BaseSettings):
    """Embedding model configuration."""

    model_config = SettingsConfigDict(extra="ignore")
    model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    dimension: int = 384


class RetrievalConfig(BaseSettings):
    """Retrieval parameters."""

    model_config = SettingsConfigDict(extra="ignore")
    top_k: int = Field(default=5, alias="RETRIEVAL_TOP_K")
    score_threshold: float = Field(default=0.5, alias="RETRIEVAL_SCORE_THRESHOLD")


class ChunkingConfig(BaseSettings):
    """Document chunking parameters."""

    model_config = SettingsConfigDict(extra="ignore")
    chunk_size: int = 512
    chunk_overlap: int = 64


class LLMConfig(BaseSettings):
    """LLM configuration (Ollama)."""

    model_config = SettingsConfigDict(extra="ignore")
    provider: str = "ollama"
    model: str = Field(default="mistral:7b-instruct", alias="OLLAMA_MODEL")
    base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    temperature: float = 0.3
    max_tokens: int = 1024


class Settings(BaseSettings):
    """Aggregated application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app: AppConfig = Field(default_factory=AppConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    @classmethod
    def from_yaml(cls, path: Optional[Path] = None) -> "Settings":
        """Load settings from YAML file, with env overrides."""
        settings = cls()
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = path or (project_root / "config" / "settings.yaml")

        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            # Merge YAML into settings (env vars still take precedence via pydantic)
            if "app" in data:
                settings.app = AppConfig(**{**settings.app.model_dump(), **data["app"]})
            if "paths" in data:
                settings.paths = PathsConfig(
                    **{**settings.paths.model_dump(), **data["paths"]}
                )
            if "embeddings" in data:
                settings.embeddings = EmbeddingsConfig(
                    **{**settings.embeddings.model_dump(), **data["embeddings"]}
                )
            if "retrieval" in data:
                settings.retrieval = RetrievalConfig(
                    **{**settings.retrieval.model_dump(), **data["retrieval"]}
                )
            if "chunking" in data:
                settings.chunking = ChunkingConfig(
                    **{**settings.chunking.model_dump(), **data["chunking"]}
                )
            if "llm" in data:
                settings.llm = LLMConfig(**{**settings.llm.model_dump(), **data["llm"]})

        return settings

    def resolve_paths(self, base: Path) -> None:
        """Resolve relative paths against project root."""
        base = base.resolve()
        self.paths.documents_dir = str(base / self.paths.documents_dir)
        self.paths.index_dir = str(base / self.paths.index_dir)


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get cached settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings.from_yaml()
        _settings.resolve_paths(Path(__file__).parent.parent.parent)
    return _settings
