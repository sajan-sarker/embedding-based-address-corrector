import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# HF_TOKEN etc. are read by huggingface_hub straight from the environment
load_dotenv(PROJECT_ROOT / ".env")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


CONFIG = load_config()


def _get(*keys: str, default: Any = None) -> Any:
    value: Any = CONFIG
    for key in keys:
        value = (value or {}).get(key)
    return default if value is None else value


def _env(name: str, default: Any) -> Any:
    # Environment variables (and .env) override config.yaml
    return os.getenv(name) or default


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _csv(value: str | list[str]) -> list[str]:
    items = value.split(",") if isinstance(value, str) else value
    return [item.strip() for item in items if item.strip()]


@dataclass(frozen=True)
class Settings:
    model_name: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", _get("embedding", "model")))
    model_path: Path = field(default_factory=lambda: _path(_env("EMBEDDING_MODEL_PATH", _get("embedding", "model_path"))))
    device: str | None = field(default_factory=lambda: _env("EMBEDDING_DEVICE", _get("embedding", "device")))
    chroma_path: Path = field(default_factory=lambda: _path(_env("CHROMA_PATH", _get("chroma_path"))))
    data_path: Path = field(default_factory=lambda: _path(_env("DATA_PATH", _get("data_path"))))
    n_results: int = field(default_factory=lambda: int(_env("N_RESULTS", _get("correction", "n_results"))))
    padding: int = field(default_factory=lambda: int(_env("NGRAM_PADDING", _get("correction", "padding"))))
    sliding_weight: float = field(default_factory=lambda: float(_env("SLIDING_WEIGHT", _get("correction", "sliding_weight", default=0.6))))
    similarity_weight: float = field(default_factory=lambda: float(_env("SIMILARITY_WEIGHT", _get("correction", "similarity_weight", default=0.4))))
    cors_origins: list[str] = field(default_factory=lambda: _csv(_env("CORS_ORIGINS", _get("cors_origins") or [])))


def get_settings() -> Settings:
    return Settings()
