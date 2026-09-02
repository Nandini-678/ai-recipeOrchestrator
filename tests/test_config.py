"""Smoke tests for the configuration layer.

These run without a .env file or any API key, which is the point: the offline
agents and their tests must never depend on credentials.
"""

import pytest

import config


def test_settings_load_without_credentials():
    """Missing keys resolve to None rather than blowing up at import time."""
    settings = config.load_settings()
    assert settings.groq_model  # always has a default
    assert settings.themealdb_api_key  # defaults to the public test key


def test_themealdb_base_url_embeds_key():
    settings = config.load_settings()
    assert settings.themealdb_base_url.endswith(f"/{settings.themealdb_api_key}")


def test_require_returns_present_value():
    assert config.require("abc123", "GROQ_API_KEY") == "abc123"


@pytest.mark.parametrize("missing", [None, ""])
def test_require_raises_and_names_the_variable(missing):
    with pytest.raises(config.ConfigError, match="GROQ_API_KEY"):
        config.require(missing, "GROQ_API_KEY")


def test_project_paths_are_absolute():
    assert config.PROJECT_ROOT.is_absolute()
    assert config.DATA_DIR.is_dir()
