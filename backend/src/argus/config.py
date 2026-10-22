"""Settings loaded from the environment and backend/.env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ (this file is backend/src/argus/config.py). An absolute path means .env
# is found no matter which directory pytest or alembic is started from.
BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Real environment variables take precedence over values in .env."""

    # extra="ignore": .env also holds keys we don't model here, like ANTHROPIC_API_KEY.
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    database_url: str
    redis_url: str


@lru_cache
def get_settings() -> Settings:
    # Cached so .env is read once, and every caller shares the same Settings object.
    return Settings()
