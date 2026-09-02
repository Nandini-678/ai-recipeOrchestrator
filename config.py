"""Central configuration for the recipe orchestrator.

Every secret and tunable is resolved here, once, from the environment. No other
module reads ``os.environ`` or hardcodes a key, so swapping a model or moving to
a hosted deployment is a `.env` change rather than a code change.

Missing keys are *not* an import-time error: the unit tests and the offline
agents (ingredient, safety, retrieval) run fine without any credentials. Agents
that genuinely need a key call :func:`require` and fail loudly at that point.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma"
SQLITE_PATH = DATA_DIR / "memory.sqlite3"
NUTRITION_CACHE_PATH = PROCESSED_DATA_DIR / "nutrition_cache.json"


class ConfigError(RuntimeError):
    """Raised when a setting a code path actually needs is missing."""


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the runtime configuration."""

    groq_api_key: str | None
    groq_model: str
    usda_api_key: str | None
    themealdb_api_key: str

    @property
    def themealdb_base_url(self) -> str:
        """Base URL for TheMealDB, which embeds the key in the path."""
        return f"https://www.themealdb.com/api/json/v1/{self.themealdb_api_key}"


def load_settings() -> Settings:
    """Read the current environment into a :class:`Settings` snapshot."""
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY") or None,
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        usda_api_key=os.getenv("USDA_API_KEY") or None,
        themealdb_api_key=os.getenv("THEMEALDB_API_KEY", "1"),
    )


def require(value: str | None, name: str) -> str:
    """Return ``value``, or raise :class:`ConfigError` naming the missing key.

    Args:
        value: The setting to check, typically off a :class:`Settings` instance.
        name: The environment variable name, used in the error message.

    Raises:
        ConfigError: If ``value`` is unset or empty.
    """
    if not value:
        raise ConfigError(
            f"{name} is not set. Copy .env.example to .env and add your key."
        )
    return value


settings = load_settings()
