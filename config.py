"""
Configuration settings for the application.
"""

import os
from typing import List

from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # CORS settings
    allowed_origins: List[str] = ["*"]

    # Redis cache settings
    redis_enabled: bool = False
    redis_url: str = "redis://localhost:6379/0"

    # Rate limiting
    rate_limit_per_minute: int = 1000
    batch_rate_limit_per_minute: int = 500

    # Translation settings
    max_text_length: int = 5000
    max_batch_size: int = 10
    translation_timeout: int = 10

    # Logging
    log_level: str = "INFO"

    class Config:
        """Pydantic config."""
        env_file = ".env"
        case_sensitive = False


# Environment-specific configurations
def get_settings() -> Settings:
    """Get settings based on environment."""
    env = os.getenv("ENVIRONMENT", "development")

    if env == "production":
        return Settings(
            debug=False,
            redis_enabled=True,
            log_level="WARNING",
            allowed_origins=["*"]
        )
    elif env == "testing":
        return Settings(
            debug=True,
            redis_enabled=False,
            port=8001
        )
    else:  # development
        return Settings(debug=True)
