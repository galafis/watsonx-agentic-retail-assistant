"""Application configuration loaded from environment variables and settings.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


def _load_yaml_config() -> dict[str, Any]:
    """Load configuration from settings.yaml file."""
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


_yaml = _load_yaml_config()


class WatsonxSettings(BaseSettings):
    """IBM Watsonx API settings."""

    api_key: str = Field(default="", alias="WATSONX_API_KEY")
    project_id: str = Field(default="", alias="WATSONX_PROJECT_ID")
    url: str = Field(
        default="https://us-south.ml.cloud.ibm.com",
        alias="WATSONX_URL",
    )
    generation_model: str = Field(default="ibm/granite-3-8b-instruct")
    embedding_model: str = Field(default="ibm/slate-125m-english-rtrvr")


class RedisSettings(BaseSettings):
    """Redis cache settings."""

    url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")


class DatabaseSettings(BaseSettings):
    """PostgreSQL database settings."""

    url: str = Field(
        default="postgresql://retail:retail_pass@localhost:5432/retail_assistant",
        alias="DATABASE_URL",
    )


class AppSettings(BaseSettings):
    """Application-level settings."""

    host: str = Field(default="0.0.0.0", alias="APP_HOST")
    port: int = Field(default=8080, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    environment: str = Field(default="development", alias="ENVIRONMENT")


class Settings:
    """Aggregated application settings from env vars and YAML config."""

    def __init__(self) -> None:
        self.watsonx = WatsonxSettings()
        self.redis = RedisSettings()
        self.database = DatabaseSettings()
        self.app = AppSettings()
        self.yaml = _yaml

    @property
    def agents_config(self) -> dict[str, Any]:
        """Agent-specific configuration."""
        return self.yaml.get("agents", {})

    @property
    def tools_config(self) -> dict[str, Any]:
        """Tool-specific configuration."""
        return self.yaml.get("tools", {})

    @property
    def governance(self) -> dict[str, Any]:
        """Governance configuration."""
        return self.yaml.get("governance", {})

    @property
    def guardrails_config(self) -> dict[str, Any]:
        """Guardrails configuration within governance."""
        return self.governance.get("guardrails", {})

    @property
    def generation_params(self) -> dict[str, Any]:
        """Watsonx generation parameters from YAML."""
        return self.yaml.get("watsonx", {}).get("generation", {}).get("parameters", {})


settings = Settings()
