"""Configuration for bicleaner-service."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Service configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="BICLEANER_")

    # Model settings
    model_type: str = "xlmr"
    model_path: str = "bitextor/bicleaner-ai-full-en-xx"

    # Inference settings
    batch_size: int = 32
    max_batch_size: int = 100
    inference_timeout_sec: int = 60

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8080


settings = Settings()
